"""A reliable, dependency-aware multi-agent orchestrator for Agentaus.

Model 2 adds structured plans, DAG scheduling, verification, repair loops,
provider retries, concurrency and token budgets, SQLite checkpoints, and
provenance-aware synthesis.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sqlite3
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar
from contextlib import contextmanager

import httpx
from pydantic import BaseModel, Field, ValidationError, model_validator

from reporting import RunReporter


DEFAULT_BASE_URL = os.environ.get("AGENTAUS_BASE_URL", "https://agentaus.com.au").rstrip("/")
DEFAULT_PATH = os.environ.get("AGENTAUS_PATH", "/api/v1/chat/completions")
LOGGER = logging.getLogger("divide_et_impera.model2")
DEFAULT_OUT_DIR = ".divide_et_impera/runs"
DONE_STATES = {"verified", "complete"}
T = TypeVar("T", bound=BaseModel)


class Complexity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class SubTask(BaseModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    title: str = Field(min_length=3, max_length=160)
    role: str = Field(min_length=2, max_length=80)
    instruction: str = Field(min_length=10, max_length=12_000)
    depends_on: list[str] = Field(default_factory=list, max_length=20)
    deliverables: list[str] = Field(min_length=1, max_length=20)
    acceptance_criteria: list[str] = Field(min_length=1, max_length=20)
    priority: int = Field(default=3, ge=1, le=5)
    timeout_seconds: int = Field(default=120, ge=10, le=600)
    estimated_complexity: Complexity = Complexity.medium


class TaskPlan(BaseModel):
    summary: str = Field(min_length=10, max_length=2000)
    subtasks: list[SubTask] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_dag(self) -> "TaskPlan":
        ids = [task.id for task in self.subtasks]
        if len(ids) != len(set(ids)):
            raise ValueError("Subtask IDs must be unique")
        known = set(ids)
        deps = {task.id: set(task.depends_on) for task in self.subtasks}
        for task in self.subtasks:
            if task.id in task.depends_on:
                raise ValueError(f"Task {task.id} cannot depend on itself")
            unknown = set(task.depends_on) - known
            if unknown:
                raise ValueError(
                    f"Task {task.id} has unknown dependencies: {sorted(unknown)}"
                )

        # Kahn's algorithm catches cycles.
        remaining = set(ids)
        completed: set[str] = set()
        while remaining:
            ready = {task_id for task_id in remaining if deps[task_id] <= completed}
            if not ready:
                raise ValueError("Subtask dependencies contain a cycle")
            completed |= ready
            remaining -= ready
        return self


class WorkerResult(BaseModel):
    task_id: str
    output: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    claims: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)


class WorkerSubmission(BaseModel):
    output: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    claims: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)


class PlanReview(BaseModel):
    approved: bool
    score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    suggested_changes: list[str] = Field(default_factory=list)


class Verification(BaseModel):
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    missing_deliverables: list[str] = Field(default_factory=list)
    repairs: list[str] = Field(default_factory=list)


class JudgeResult(BaseModel):
    accepted: bool
    score: float = Field(ge=0.0, le=1.0)
    critical_issues: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)


@dataclass
class Artifact:
    task_id: str
    kind: str
    content: str
    confidence: float
    sources: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)


@dataclass
class RunMetrics:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    retries: int = 0
    by_stage: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class TaskState:
    status: str = "pending"
    attempts: int = 0
    repairs: int = 0
    result: WorkerResult | None = None
    verification: Verification | None = None
    error: str | None = None
    seconds: float = 0.0
    resumed: bool = False


class AgentausError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class InvalidOutputError(AgentausError):
    pass


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _validation_summary(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(map(str, error['loc'])) or '<root>'}: {error['msg']}"
        for error in exc.errors()
    )


def _schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()

    def make_strict(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object":
                value["additionalProperties"] = False
            for child in value.values():
                make_strict(child)
        elif isinstance(value, list):
            for child in value:
                make_strict(child)

    make_strict(schema)
    return schema


def _tool(name: str, description: str, model: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": _schema(model),
        },
    }


class AgentausAsyncClient:
    """Async provider client with retries, usage tracking, and bounded SSE."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        path: str = DEFAULT_PATH,
        *,
        timeout: float = 120.0,
        max_retries: int = 3,
        metrics: RunMetrics | None = None,
    ) -> None:
        self.api_key = api_key
        self.url = base_url.rstrip("/") + path
        self.timeout = httpx.Timeout(timeout, connect=min(timeout, 10.0))
        self.max_retries = max_retries
        self.metrics = metrics or RunMetrics()

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "*/*",
            "Content-Type": "application/json",
        }

    def payload(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"messages": messages, "stream": stream}
        if any(message["role"] == "system" for message in messages):
            body["system_prompt_overwrite"] = True
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
        body.update(extra)
        return body

    async def chat(
        self,
        http: httpx.AsyncClient,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stage: str = "request",
        **extra: Any,
    ) -> dict[str, Any]:
        body = self.payload(
            messages, tools=tools, tool_choice=tool_choice, stream=False, **extra
        )
        LOGGER.debug(
            "API request stage=%s messages=%d chars=%d tools=%d extra=%s",
            stage,
            len(messages),
            sum(len(message["content"]) for message in messages),
            len(tools or []),
            extra,
        )
        for attempt in range(self.max_retries + 1):
            self.metrics.requests += 1
            started_at = time.monotonic()
            LOGGER.debug(
                "API call started stage=%s attempt=%d/%d",
                stage,
                attempt + 1,
                self.max_retries + 1,
            )
            try:
                response = await http.post(
                    self.url, headers=self.headers(), json=body
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                elapsed = time.monotonic() - started_at
                LOGGER.warning(
                    "API call network failure stage=%s attempt=%d elapsed=%.1fs "
                    "error=%s",
                    stage,
                    attempt + 1,
                    elapsed,
                    exc,
                )
                if attempt >= self.max_retries:
                    raise AgentausError(
                        f"Provider network failure during {stage} after "
                        f"{elapsed:.1f}s: {exc}",
                        retryable=False,
                    ) from exc
                self.metrics.retries += 1
                await self._pause(stage, attempt)
                continue

            elapsed = time.monotonic() - started_at
            request_id = (
                response.headers.get("x-request-id")
                or response.headers.get("request-id")
                or response.headers.get("traceparent")
            )
            LOGGER.info(
                "API call finished stage=%s status=%d elapsed=%.1fs%s",
                stage,
                response.status_code,
                elapsed,
                f" request_id={request_id}" if request_id else "",
            )
            if response.status_code == 200:
                try:
                    data = response.json()
                except json.JSONDecodeError as exc:
                    raise AgentausError(
                        "Provider returned invalid JSON",
                        status=200,
                        retryable=False,
                    ) from exc
                usage = data.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                self.metrics.prompt_tokens += prompt_tokens
                self.metrics.completion_tokens += completion_tokens
                bucket = stage.split(":", 1)[0]
                self.metrics.by_stage[bucket] = (
                    self.metrics.by_stage.get(bucket, 0)
                    + prompt_tokens
                    + completion_tokens
                )
                LOGGER.debug(
                    "API usage stage=%s prompt=%d completion=%d run_total=%d",
                    stage,
                    prompt_tokens,
                    completion_tokens,
                    self.metrics.total_tokens,
                )
                return data

            retryable = response.status_code in {408, 429, 500, 502, 503, 504}
            if retryable and attempt < self.max_retries:
                self.metrics.retries += 1
                LOGGER.warning(
                    "API call retrying stage=%s status=%d",
                    stage,
                    response.status_code,
                )
                await self._pause(stage, attempt, response)
                continue
            raise AgentausError(
                f"Provider returned HTTP {response.status_code}: "
                f"{response.text[:300]}",
                status=response.status_code,
                retryable=retryable,
            )
        raise RuntimeError("Unreachable retry state")

    async def _pause(
        self,
        stage: str,
        attempt: int,
        response: httpx.Response | None = None,
    ) -> None:
        delay = min(8.0, (2**attempt) + random.random())
        if response is not None:
            try:
                retry_after = float(response.headers.get("retry-after", 0))
            except ValueError:
                retry_after = 0.0
            delay = max(delay, min(60.0, retry_after))
        LOGGER.debug("Backoff stage=%s attempt=%d delay=%.1fs", stage, attempt + 1, delay)
        await asyncio.sleep(delay)

    async def reply_text(
        self,
        http: httpx.AsyncClient,
        messages: list[dict[str, str]],
        **extra: Any,
    ) -> str:
        data = await self.chat(http, messages, **extra)
        choice = (data.get("choices") or [{}])[0]
        content = ((choice.get("message") or {}).get("content"))
        if not isinstance(content, str):
            raise AgentausError("Provider returned no text content")
        return content

    async def tool_call(
        self,
        http: httpx.AsyncClient,
        messages: list[dict[str, str]],
        *,
        tool: dict[str, Any],
        expected_name: str,
        model: type[T],
        stage: str = "structured-output",
        **extra: Any,
    ) -> T:
        data = await self.chat(
            http,
            messages,
            tools=[tool],
            tool_choice="required",
            stage=stage,
            **extra,
        )
        calls = (
            ((data.get("choices") or [{}])[0].get("message") or {}).get("tool_calls")
            or []
        )
        LOGGER.debug("Tool response stage=%s calls=%d", stage, len(calls))
        for call in calls:
            function = call.get("function") or {}
            if function.get("name") != expected_name:
                continue
            raw = function.get("arguments")
            LOGGER.debug(
                "Tool arguments stage=%s name=%s raw=%.1000s",
                stage,
                expected_name,
                raw,
            )
            try:
                arguments = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError as exc:
                raise InvalidOutputError("Tool arguments were not valid JSON") from exc
            if not isinstance(arguments, dict):
                raise InvalidOutputError("Tool arguments were not a JSON object")
            try:
                return model.model_validate(arguments)
            except ValidationError as exc:
                raise InvalidOutputError(
                    f"Invalid {expected_name} arguments: {_validation_summary(exc)}"
                ) from exc
        raise InvalidOutputError(
            f"Provider did not call required tool {expected_name}"
        )


class SQLiteRunStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connection() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    plan_json TEXT,
                    final_output TEXT,
                    metrics_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    verification_json TEXT,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (run_id, task_id)
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_run ON tasks(run_id);
                """
            )
            columns = {row["name"] for row in db.execute("PRAGMA table_info(tasks)")}
            if "error" not in columns:
                db.execute("ALTER TABLE tasks ADD COLUMN error TEXT")

    def create_run(self, run_id: str, goal: str) -> None:
        now = time.time()
        with self.connection() as db:
            db.execute("DELETE FROM tasks WHERE run_id=?", (run_id,))
            db.execute(
                "INSERT OR REPLACE INTO runs "
                "(id, goal, status, created_at, updated_at) "
                "VALUES (?, ?, 'running', ?, ?)",
                (run_id, goal, now, now),
            )

    def save_plan(self, run_id: str, plan: TaskPlan) -> None:
        with self.connection() as db:
            db.execute(
                "UPDATE runs SET plan_json=?, updated_at=? WHERE id=?",
                (plan.model_dump_json(), time.time(), run_id),
            )
            for task in plan.subtasks:
                db.execute(
                    "INSERT OR REPLACE INTO tasks "
                    "(run_id, task_id, status, attempt, updated_at) "
                    "VALUES (?, ?, 'pending', 0, ?)",
                    (run_id, task.id, time.time()),
                )

    def save_task(
        self,
        run_id: str,
        task_id: str,
        status: str,
        *,
        result: WorkerResult | None = None,
        verification: Verification | None = None,
        attempt: int | None = None,
        error: str | None = None,
    ) -> None:
        with self.connection() as db:
            db.execute(
                """
                UPDATE tasks
                SET status=?,
                    result_json=COALESCE(?, result_json),
                    verification_json=COALESCE(?, verification_json),
                    attempt=COALESCE(?, attempt),
                    error=?,
                    updated_at=?
                WHERE run_id=? AND task_id=?
                """,
                (
                    status,
                    result.model_dump_json() if result else None,
                    verification.model_dump_json() if verification else None,
                    attempt,
                    error,
                    time.time(),
                    run_id,
                    task_id,
                ),
            )

    def mark_running(self, run_id: str) -> None:
        with self.connection() as db:
            db.execute(
                "UPDATE runs SET status='running', updated_at=? WHERE id=?",
                (time.time(), run_id),
            )

    def load_run(
        self, run_id: str
    ) -> tuple[str, TaskPlan | None, dict[str, TaskState]] | None:
        with self.connection() as db:
            run = db.execute(
                "SELECT goal, plan_json FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            rows = db.execute(
                "SELECT task_id, status, result_json, verification_json, attempt "
                "FROM tasks WHERE run_id=?",
                (run_id,),
            ).fetchall()
        if run is None:
            return None
        plan = TaskPlan.model_validate_json(run["plan_json"]) if run["plan_json"] else None
        done: dict[str, TaskState] = {}
        for row in rows:
            if row["status"] not in DONE_STATES or not row["result_json"]:
                continue
            verification_json = row["verification_json"]
            done[row["task_id"]] = TaskState(
                status=row["status"],
                attempts=row["attempt"],
                result=WorkerResult.model_validate_json(row["result_json"]),
                verification=(
                    Verification.model_validate_json(verification_json)
                    if verification_json
                    else None
                ),
                resumed=True,
            )
        return run["goal"], plan, done

    def finish(
        self,
        run_id: str,
        status: str,
        *,
        output: str | None = None,
        metrics: RunMetrics | None = None,
    ) -> None:
        with self.connection() as db:
            db.execute(
                "UPDATE runs SET status=?, final_output=?, metrics_json=?, updated_at=? "
                "WHERE id=?",
                (
                    status,
                    output,
                    json.dumps(asdict(metrics)) if metrics else None,
                    time.time(),
                    run_id,
                ),
            )


class Orchestrator:
    def __init__(
        self,
        provider: AgentausAsyncClient,
        *,
        store: SQLiteRunStore,
        concurrency: int = 4,
        max_workers_attempts: int = 2,
        verification_required: bool = True,
        max_verification_rounds: int = 2,
        token_budget: int = 100_000,
        run_id: str | None = None,
        models: dict[str, str] | None = None,
        min_score: float = 0.75,
        plan_review: bool = True,
        max_synthesis_rounds: int = 1,
        fail_fast: bool = False,
        reporter: RunReporter | None = None,
    ) -> None:
        self.provider = provider
        self.store = store
        self.concurrency = max(1, concurrency)
        self.max_workers_attempts = max(1, max_workers_attempts)
        self.verification_required = verification_required
        self.max_verification_rounds = max(1, max_verification_rounds)
        self.token_budget = token_budget
        self.run_id = run_id or new_run_id()
        self.models = models or {}
        self.metrics = provider.metrics
        self.min_score = min_score
        self.plan_review = plan_review
        self.max_synthesis_rounds = max(0, max_synthesis_rounds)
        self.fail_fast = fail_fast
        self.reporter = reporter
        self.states: dict[str, TaskState] = {}
        self.status = "pending"

    def _model_params(
        self, stage: str, complexity: Complexity | None = None
    ) -> dict[str, str]:
        model = self.models.get(stage)
        if complexity is not None:
            model = self.models.get(f"{stage}:{complexity.value}") or model
        return {"model": model} if model else {}

    def _check_budget(self) -> None:
        if self.metrics.total_tokens >= self.token_budget:
            raise RuntimeError(
                f"Token budget exhausted: {self.metrics.total_tokens}/{self.token_budget}"
            )

    def _emit(self, name: str, *args: Any, **kwargs: Any) -> None:
        if self.reporter is None:
            return
        try:
            getattr(self.reporter, name)(*args, **kwargs)
        except OSError as exc:
            LOGGER.warning("Report write failed name=%s error=%s", name, exc)

    def _passes(self, verification: Verification) -> bool:
        return verification.passed and verification.score >= self.min_score

    def _judge_passes(self, judge: JudgeResult) -> bool:
        return judge.accepted and judge.score >= self.min_score

    def _set_state(
        self,
        task_id: str,
        status: str,
        *,
        attempts: int | None = None,
        error: str | None = None,
    ) -> None:
        state = self.states[task_id]
        state.status = status
        state.error = error
        if attempts is not None:
            state.attempts = attempts
        self.store.save_task(
            self.run_id, task_id, status, attempt=attempts, error=error
        )

    async def create_plan(
        self,
        http: httpx.AsyncClient,
        goal: str,
    ) -> TaskPlan:
        plan = await self._request_plan(http, goal)
        if not self.plan_review:
            return plan
        try:
            review = await self.review_plan(http, goal, plan)
        except InvalidOutputError as exc:
            LOGGER.warning("Plan review skipped error=%s", exc)
            return plan
        LOGGER.info(
            "Plan review approved=%s score=%.2f issues=%d",
            review.approved,
            review.score,
            len(review.issues),
        )
        if review.approved or not review.issues:
            return plan
        feedback = (
            "A reviewer rejected your previous plan.\n\n"
            f"Previous plan:\n{plan.model_dump_json()}\n\n"
            f"Issues:\n{json.dumps(review.issues)}\n\n"
            f"Suggested changes:\n{json.dumps(review.suggested_changes)}\n\n"
            "Submit an improved plan that resolves every issue."
        )
        try:
            return await self._request_plan(http, goal, feedback)
        except RuntimeError as exc:
            LOGGER.warning("Plan revision failed; keeping original plan error=%s", exc)
            return plan

    async def _request_plan(
        self,
        http: httpx.AsyncClient,
        goal: str,
        feedback: str | None = None,
    ) -> TaskPlan:
        system = (
            "You are the lead project architect. Convert the goal into a complete "
            "dependency-aware execution plan. Every subtask must be independently "
            "verifiable, have measurable deliverables and acceptance criteria, and "
            "use no more than 12 tasks. Use dependencies only where actual output "
            "is consumed by another task, so independent work can run in parallel. "
            "Workers see only their own instruction and the outputs of their "
            "dependencies, so every instruction must be self-contained. Priority "
            "runs from 1 (least urgent) to 5 (critical path). Return only the "
            "submit_task_plan tool call."
        )
        tool = _tool(
            "submit_task_plan",
            "Submit a strict dependency-aware task plan.",
            TaskPlan,
        )
        base = [
            {"role": "system", "content": system},
            {"role": "user", "content": goal},
        ]
        if feedback:
            base.append({"role": "user", "content": feedback})
        messages = base
        last_error: Exception | None = None
        for attempt in range(1, 4):
            self._check_budget()
            try:
                return await self.provider.tool_call(
                    http,
                    messages,
                    tool=tool,
                    expected_name="submit_task_plan",
                    model=TaskPlan,
                    stage="planning",
                    **self._model_params("planning"),
                )
            except InvalidOutputError as exc:
                last_error = exc
                LOGGER.warning("Plan attempt %d/3 rejected: %s", attempt, exc)
                messages = base + [
                    {
                        "role": "user",
                        "content": (
                            f"Your previous submission was rejected: {exc}. "
                            "Submit a corrected plan."
                        ),
                    }
                ]
        raise RuntimeError(f"Could not obtain a valid task plan: {last_error}")

    async def review_plan(
        self,
        http: httpx.AsyncClient,
        goal: str,
        plan: TaskPlan,
    ) -> PlanReview:
        self._check_budget()
        return await self.provider.tool_call(
            http,
            [
                {
                    "role": "system",
                    "content": (
                        "You are a skeptical plan reviewer. Look for missing steps "
                        "the goal needs, overlapping or redundant tasks, false "
                        "dependencies that serialise parallel work, vague or "
                        "unmeasurable acceptance criteria, and tasks too large for "
                        "one worker. Approve only if the plan is ready to execute. "
                        "Return only the submit_plan_review tool call."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Goal:\n{goal}\n\nPlan:\n{plan.model_dump_json()}",
                },
            ],
            tool=_tool(
                "submit_plan_review",
                "Submit a critical review of the task plan.",
                PlanReview,
            ),
            expected_name="submit_plan_review",
            model=PlanReview,
            stage="plan-review",
            **self._model_params("verification"),
        )

    def _task_prompt(
        self,
        goal: str,
        task: SubTask,
        dependencies: dict[str, WorkerResult],
    ) -> str:
        dependency_context = [
            {
                "task_id": task_id,
                "output": result.output,
                "confidence": result.confidence,
                "claims": result.claims,
                "sources": result.sources,
            }
            for task_id, result in dependencies.items()
            if task_id in task.depends_on
        ]
        return (
            f"Original goal:\n{goal}\n\n"
            f"Task:\n{task.title}\n\n"
            f"Instruction:\n{task.instruction}\n\n"
            f"Required deliverables:\n{json.dumps(task.deliverables)}\n\n"
            f"Acceptance criteria:\n{json.dumps(task.acceptance_criteria)}\n\n"
            f"Dependency artifacts:\n{json.dumps(dependency_context, ensure_ascii=False)}"
        )

    async def run_worker(
        self,
        http: httpx.AsyncClient,
        goal: str,
        task: SubTask,
        dependencies: dict[str, WorkerResult],
    ) -> WorkerResult:
        started_at = time.monotonic()
        LOGGER.info(
            "Worker started task_id=%s title=%r role=%s",
            task.id,
            task.title,
            task.role,
        )
        system = (
            f"You are an expert {task.role}. Complete only this assigned task. "
            "Be specific, evidence-oriented, concise, and separate verified facts "
            "from inference. Return the submit_worker_result tool call. Do not "
            "include or invent a task ID; the orchestrator assigns it."
        )
        prompt = self._task_prompt(goal, task, dependencies)
        tool = _tool(
            "submit_worker_result",
            "Submit the worker's complete, verifiable result. Do not include a task ID.",
            WorkerSubmission,
        )
        base = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        messages = base

        last_error: Exception | None = None
        deadline = asyncio.get_running_loop().time() + task.timeout_seconds
        for attempt in range(1, self.max_workers_attempts + 1):
            remaining_seconds = deadline - asyncio.get_running_loop().time()
            if remaining_seconds <= 0:
                last_error = TimeoutError(
                    f"Task exceeded its {task.timeout_seconds}s time limit"
                )
                break
            self._set_state(task.id, "running", attempts=attempt)
            self._check_budget()
            try:
                async with asyncio.timeout(remaining_seconds):
                    submission = await self.provider.tool_call(
                        http,
                        messages,
                        tool=tool,
                        expected_name="submit_worker_result",
                        model=WorkerSubmission,
                        stage=f"worker:{task.id}",
                        **self._model_params("worker", task.estimated_complexity),
                    )
                assigned_result = WorkerResult(
                    task_id=task.id,
                    **submission.model_dump(),
                )
                LOGGER.info(
                    "Worker finished task_id=%s confidence=%.2f elapsed=%.1fs",
                    task.id,
                    assigned_result.confidence,
                    time.monotonic() - started_at,
                )
                return assigned_result
            except (InvalidOutputError, TimeoutError) as exc:
                last_error = exc
                LOGGER.warning(
                    "Worker failed task_id=%s title=%r attempt=%d/%d error=%s",
                    task.id,
                    task.title,
                    attempt,
                    self.max_workers_attempts,
                    exc,
                )
                if isinstance(exc, InvalidOutputError):
                    messages = base + [
                        {
                            "role": "user",
                            "content": (
                                f"Your previous submission was rejected: {exc}. "
                                "Resubmit a corrected result."
                            ),
                        }
                    ]
                if asyncio.get_running_loop().time() >= deadline:
                    break
        raise RuntimeError(
            f"Worker task_id={task.id} failed after "
            f"{self.max_workers_attempts} attempts; last error: {last_error}"
        ) from last_error

    async def verify_worker(
        self,
        http: httpx.AsyncClient,
        goal: str,
        task: SubTask,
        result: WorkerResult,
        dependencies: dict[str, WorkerResult] | None = None,
    ) -> Verification:
        self._check_budget()
        upstream = [
            {"task_id": task_id, "claims": dep.claims, "sources": dep.sources}
            for task_id, dep in (dependencies or {}).items()
            if task_id in task.depends_on
        ]
        prompt = (
            f"Original goal:\n{goal}\n\n"
            f"Task:\n{task.model_dump_json()}\n\n"
            f"Upstream claims this task relied on:\n"
            f"{json.dumps(upstream, ensure_ascii=False)}\n\n"
            f"Worker result:\n{result.model_dump_json()}\n\n"
            "Independently verify completeness, correctness, evidence quality, "
            "consistency with the upstream claims, and acceptance criteria. Do "
            "not defer to the worker's confidence."
        )
        tool = _tool(
            "submit_verification",
            "Submit an independent verification result.",
            Verification,
        )
        return await self.provider.tool_call(
            http,
            [
                {
                    "role": "system",
                    "content": (
                        "You are a rigorous independent reviewer. Prefer failure "
                        "for unsupported critical claims. Return only the "
                        "submit_verification tool call."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            tool=tool,
            expected_name="submit_verification",
            model=Verification,
            stage=f"verification:{task.id}",
            **self._model_params("verification"),
        )

    async def repair_worker(
        self,
        http: httpx.AsyncClient,
        goal: str,
        task: SubTask,
        dependencies: dict[str, WorkerResult],
        result: WorkerResult,
        verification: Verification,
    ) -> WorkerResult:
        self._check_budget()
        prompt = (
            f"{self._task_prompt(goal, task, dependencies)}\n\n"
            f"Your previous result:\n{result.model_dump_json()}\n\n"
            f"Independent review:\n"
            f"{json.dumps(verification.model_dump(), ensure_ascii=False)}\n\n"
            "Repair the result to satisfy every acceptance criterion."
        )
        submission = await self.provider.tool_call(
            http,
            [
                {
                    "role": "system",
                    "content": (
                        f"You are an expert {task.role}. "
                        "Repair the prior result precisely."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            tool=_tool(
                "submit_worker_result",
                "Submit the repaired worker result. Do not include a task ID.",
                WorkerSubmission,
            ),
            expected_name="submit_worker_result",
            model=WorkerSubmission,
            stage=f"repair:{task.id}",
            **self._model_params("worker", task.estimated_complexity),
        )
        return WorkerResult(task_id=task.id, **submission.model_dump())

    def _log_verification(self, task: SubTask, verification: Verification) -> None:
        LOGGER.info(
            "Verification task_id=%s passed=%s score=%.2f issues=%d",
            task.id,
            verification.passed,
            verification.score,
            len(verification.issues),
        )

    async def verify_and_repair(
        self,
        http: httpx.AsyncClient,
        goal: str,
        task: SubTask,
        dependencies: dict[str, WorkerResult],
        result: WorkerResult,
    ) -> tuple[WorkerResult, Verification]:
        state = self.states[task.id]
        verification = await self.verify_worker(http, goal, task, result, dependencies)
        self._log_verification(task, verification)
        for _ in range(self.max_verification_rounds):
            if self._passes(verification):
                break
            result = await self.repair_worker(
                http, goal, task, dependencies, result, verification
            )
            state.repairs += 1
            state.result = result
            verification = await self.verify_worker(
                http, goal, task, result, dependencies
            )
            self._log_verification(task, verification)
        return result, verification

    async def process_task(
        self,
        http: httpx.AsyncClient,
        goal: str,
        task: SubTask,
        dependencies: dict[str, WorkerResult],
    ) -> WorkerResult:
        state = self.states[task.id]
        result = await self.run_worker(http, goal, task, dependencies)
        state.result = result
        if not self.verification_required:
            state.status = "complete"
            self.store.save_task(self.run_id, task.id, "complete", result=result)
            return result
        result, verification = await self.verify_and_repair(
            http, goal, task, dependencies, result
        )
        passed = self._passes(verification)
        state.result, state.verification = result, verification
        state.status = "verified" if passed else "rework_limit"
        self.store.save_task(
            self.run_id,
            task.id,
            state.status,
            result=result,
            verification=verification,
        )
        if not passed:
            raise RuntimeError(
                f"Task {task.id} failed verification after "
                f"{self.max_verification_rounds} repair rounds: "
                + "; ".join(verification.issues)
            )
        return result

    async def _run_task(
        self,
        semaphore: asyncio.Semaphore,
        http: httpx.AsyncClient,
        goal: str,
        task: SubTask,
        results: dict[str, WorkerResult],
    ) -> WorkerResult:
        async with semaphore:
            started_at = time.monotonic()
            try:
                return await self.process_task(http, goal, task, results)
            finally:
                self.states[task.id].seconds = time.monotonic() - started_at

    def _skip_blocked(self, pending: dict[str, SubTask], dead: set[str]) -> None:
        changed = True
        while changed:
            changed = False
            for task in list(pending.values()):
                blockers = sorted(set(task.depends_on) & dead)
                if not blockers:
                    continue
                reason = (
                    "Skipped because upstream task(s) did not complete: "
                    + ", ".join(blockers)
                )
                self._set_state(task.id, "skipped", error=reason)
                LOGGER.warning(
                    "Task skipped task_id=%s blockers=%s", task.id, blockers
                )
                self._emit("write_task", task, self.states[task.id])
                dead.add(task.id)
                del pending[task.id]
                changed = True

    def _fail_task(self, task: SubTask, error: BaseException) -> None:
        state = self.states[task.id]
        status = "rework_limit" if state.status == "rework_limit" else "failed"
        message = str(error) or type(error).__name__
        state.status, state.error = status, message
        self.store.save_task(self.run_id, task.id, status, error=message)
        LOGGER.error(
            "Task failed task_id=%s status=%s error=%s", task.id, status, message
        )
        self._emit("write_task", task, state)

    def _log_done(self, task: SubTask, finished: int, total: int) -> None:
        state = self.states[task.id]
        score = f"{state.verification.score:.2f}" if state.verification else "n/a"
        LOGGER.info(
            "Task done task_id=%s progress=%d/%d score=%s repairs=%d elapsed=%.1fs",
            task.id,
            finished,
            total,
            score,
            state.repairs,
            state.seconds,
        )
        self._emit("write_task", task, state)

    async def execute_plan(
        self,
        http: httpx.AsyncClient,
        goal: str,
        plan: TaskPlan,
    ) -> dict[str, WorkerResult]:
        semaphore = asyncio.Semaphore(self.concurrency)
        results: dict[str, WorkerResult] = {
            task_id: state.result
            for task_id, state in self.states.items()
            if state.status in DONE_STATES and state.result is not None
        }
        pending = {task.id: task for task in plan.subtasks if task.id not in results}
        running: dict[asyncio.Task[WorkerResult], SubTask] = {}
        dead: set[str] = set()
        total = len(plan.subtasks)
        if results:
            LOGGER.info("Checkpoint reused for %d/%d task(s)", len(results), total)

        try:
            while pending or running:
                self._skip_blocked(pending, dead)
                ready = sorted(
                    (
                        task
                        for task in pending.values()
                        if all(dep in results for dep in task.depends_on)
                    ),
                    key=lambda task: (-task.priority, task.id),
                )
                if ready:
                    LOGGER.info(
                        "Scheduling %d task(s): %s",
                        len(ready),
                        ", ".join(f"{task.id} ({task.title})" for task in ready),
                    )
                for task in ready:
                    del pending[task.id]
                    future = asyncio.create_task(
                        self._run_task(semaphore, http, goal, task, results),
                        name=task.id,
                    )
                    running[future] = task
                if not running:
                    break
                finished, _ = await asyncio.wait(
                    list(running), return_when=asyncio.FIRST_COMPLETED
                )
                for future in finished:
                    task = running.pop(future)
                    error = (
                        asyncio.CancelledError("task cancelled")
                        if future.cancelled()
                        else future.exception()
                    )
                    if error is not None:
                        self._fail_task(task, error)
                        dead.add(task.id)
                        if self.fail_fast:
                            raise RuntimeError(
                                f"Task {task.id} failed (fail-fast): "
                                f"{self.states[task.id].error}"
                            )
                        continue
                    results[task.id] = future.result()
                    self._log_done(task, len(results), total)
        finally:
            for future in running:
                future.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)

        if pending:
            raise RuntimeError("No schedulable tasks remain; dependency graph deadlock")
        if not results:
            raise RuntimeError(
                "All tasks failed: "
                + "; ".join(
                    f"{task_id}: {state.error}"
                    for task_id, state in self.states.items()
                    if state.error
                )
            )
        return results

    async def synthesize(
        self,
        http: httpx.AsyncClient,
        goal: str,
        plan: TaskPlan,
        results: dict[str, WorkerResult],
        missing: list[tuple[str, str]] | None = None,
    ) -> tuple[str, JudgeResult]:
        self._check_budget()
        artifacts = [
            Artifact(
                task_id=task.id,
                kind="worker_result",
                content=results[task.id].output,
                confidence=results[task.id].confidence,
                sources=results[task.id].sources,
                claims=results[task.id].claims,
            )
            for task in plan.subtasks
            if task.id in results
        ]
        artifact_text = "\n\n".join(
            f"## Artifact {artifact.task_id}\n"
            f"Confidence: {artifact.confidence}\n"
            f"Sources: {json.dumps(artifact.sources)}\n"
            f"Claims: {json.dumps(artifact.claims)}\n"
            f"{artifact.content}"
            for artifact in artifacts
        )
        gap_text = "".join(f"- {task_id}: {reason}\n" for task_id, reason in missing or [])
        system = (
            "You are the final editor and integrator. Produce one "
            "cohesive, polished answer. Resolve conflicts conservatively, "
            "preserve provenance and caveats, omit unsupported claims, "
            "and do not invent missing work. Attribute key claims to their "
            "source artifact with [task_id] tags where that aids verification."
        )
        base_prompt = (
            f"Original goal:\n{goal}\n\n"
            f"Plan summary:\n{plan.summary}\n\n"
            f"Verified artifacts:\n{artifact_text}"
        )
        if gap_text:
            base_prompt += (
                "\n\nIncomplete work (state these gaps plainly and do not "
                f"fill them in):\n{gap_text}"
            )
        output = await self.provider.reply_text(
            http,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": base_prompt},
            ],
            stage="synthesis",
            **self._model_params("synthesis"),
        )
        judge = await self.judge_output(http, goal, plan, output, results, gap_text)
        for round_number in range(1, self.max_synthesis_rounds + 1):
            if self._judge_passes(judge):
                break
            LOGGER.info(
                "Synthesis revision %d/%d score=%.2f issues=%d",
                round_number,
                self.max_synthesis_rounds,
                judge.score,
                len(judge.critical_issues),
            )
            self._check_budget()
            output = await self.provider.reply_text(
                http,
                [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": (
                            f"{base_prompt}\n\n"
                            f"Your previous draft:\n{output}\n\n"
                            f"Judge feedback:\n"
                            f"{json.dumps(judge.model_dump(), ensure_ascii=False)}\n\n"
                            "Revise the draft to resolve the critical issues, "
                            "using only the verified artifacts."
                        ),
                    },
                ],
                stage=f"synthesis-revision:{round_number}",
                **self._model_params("synthesis"),
            )
            judge = await self.judge_output(http, goal, plan, output, results, gap_text)
        return output, judge

    async def judge_output(
        self,
        http: httpx.AsyncClient,
        goal: str,
        plan: TaskPlan,
        output: str,
        results: dict[str, WorkerResult],
        gap_text: str = "",
    ) -> JudgeResult:
        self._check_budget()
        claims = {task_id: result.claims for task_id, result in results.items()}
        known_gaps = (
            f"Known incomplete work (already disclosed in the answer):\n{gap_text}\n"
            if gap_text
            else ""
        )
        return await self.provider.tool_call(
            http,
            [
                {
                    "role": "system",
                    "content": (
                        "You are an exacting acceptance judge. Judge against the "
                        "original goal and plan, report critical gaps, and flag "
                        "substantive statements that no verified claim supports. "
                        "Do not penalise incomplete work that the answer discloses."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original goal:\n{goal}\n\n"
                        f"Plan:\n{plan.model_dump_json()}\n\n"
                        f"Verified claims by task:\n"
                        f"{json.dumps(claims, ensure_ascii=False)}\n\n"
                        f"{known_gaps}"
                        f"Final answer:\n{output}"
                    ),
                },
            ],
            tool=_tool(
                "submit_judge_result",
                "Judge whether the final answer satisfies the original goal.",
                JudgeResult,
            ),
            expected_name="submit_judge_result",
            model=JudgeResult,
            stage="final-acceptance-check",
            **self._model_params("verification"),
        )

    def _report(
        self,
        goal: str,
        plan: TaskPlan | None,
        started_at: float,
        judge: JudgeResult | None,
        error: str | None = None,
    ) -> None:
        self._emit(
            "write_report",
            goal=goal,
            plan=plan,
            states=self.states,
            status=self.status,
            metrics=self.metrics,
            elapsed=time.monotonic() - started_at,
            judge=judge,
            error=error,
        )

    async def run(self, goal: str) -> str:
        self.store.create_run(self.run_id, goal)
        return await self._drive(goal, None, {})

    async def resume(self) -> str:
        saved = self.store.load_run(self.run_id)
        if saved is None:
            raise RuntimeError(f"No saved run with id {self.run_id}")
        goal, plan, done = saved
        self.store.mark_running(self.run_id)
        LOGGER.info(
            "Resuming run_id=%s checkpointed_tasks=%d", self.run_id, len(done)
        )
        return await self._drive(goal, plan, done)

    async def _drive(
        self,
        goal: str,
        plan: TaskPlan | None,
        done: dict[str, TaskState],
    ) -> str:
        started_at = time.monotonic()
        judge: JudgeResult | None = None
        LOGGER.info("Run started run_id=%s", self.run_id)
        limits = httpx.Limits(
            max_connections=max(1, self.concurrency * 2),
            max_keepalive_connections=max(1, self.concurrency),
        )
        try:
            async with httpx.AsyncClient(
                timeout=self.provider.timeout,
                limits=limits,
                follow_redirects=False,
            ) as http:
                if plan is None:
                    LOGGER.info("Run stage started run_id=%s stage=planning", self.run_id)
                    plan = await self.create_plan(http, goal)
                    LOGGER.info(
                        "Plan accepted run_id=%s tasks=%d summary=%r",
                        self.run_id,
                        len(plan.subtasks),
                        plan.summary,
                    )
                    self.store.save_plan(self.run_id, plan)
                self.states = {
                    task.id: done.get(task.id) or TaskState() for task in plan.subtasks
                }
                self._emit("write_plan", goal, plan)
                results = await self.execute_plan(http, goal, plan)
                missing = [
                    (task_id, f"{state.status}: {state.error or 'no detail'}")
                    for task_id, state in self.states.items()
                    if task_id not in results
                ]
                LOGGER.info(
                    "Run stage started run_id=%s stage=synthesis", self.run_id
                )
                final_output, judge = await self.synthesize(
                    http, goal, plan, results, missing
                )
                accepted = self._judge_passes(judge)
                if missing:
                    final_output += "\n\n## Incomplete Work\n" + "\n".join(
                        f"- `{task_id}`: {reason}" for task_id, reason in missing
                    )
                if not accepted:
                    issues = judge.critical_issues or [
                        f"Judge score {judge.score:.2f} is below the "
                        f"{self.min_score:.2f} acceptance threshold"
                    ]
                    final_output += "\n\n## Acceptance Notes\n" + "\n".join(
                        f"- {issue}" for issue in issues
                    )
                if missing:
                    self.status = "partial"
                else:
                    self.status = "completed" if accepted else "completed_with_gaps"
                self.store.finish(
                    self.run_id,
                    self.status,
                    output=final_output,
                    metrics=self.metrics,
                )
                self._emit("write_final", final_output, self.status, judge)
                self._report(goal, plan, started_at, judge)
                LOGGER.info(
                    "Run finished run_id=%s status=%s elapsed=%.1fs tokens=%d",
                    self.run_id,
                    self.status,
                    time.monotonic() - started_at,
                    self.metrics.total_tokens,
                )
                return final_output
        except BaseException as exc:
            interrupted = isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
            self.status = "interrupted" if interrupted else "failed"
            self.store.finish(
                self.run_id,
                self.status,
                metrics=self.metrics,
            )
            LOGGER.error(
                "Run %s run_id=%s elapsed=%.1fs error=%s",
                self.status,
                self.run_id,
                time.monotonic() - started_at,
                exc,
            )
            self._report(
                goal, plan, started_at, judge, error=str(exc) or type(exc).__name__
            )
            raise


def get_api_key(explicit: str | None) -> str:
    key = explicit or os.environ.get("AGENTAUS_API_KEY", "")
    if not key:
        key = input("Enter Agentaus API key: ").strip()
    if not key:
        raise RuntimeError("Agentaus API key is required")
    return key


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a frontier dependency-aware divide-and-conquer workflow"
    )
    parser.add_argument("goal", nargs="*", help="The goal to accomplish")
    parser.add_argument("--api-key", help="Agentaus API key (prefer environment)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--verification-rounds", type=int, default=2)
    parser.add_argument("--no-verification", action="store_true")
    parser.add_argument("--token-budget", type=int, default=100_000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--db", default=".divide_et_impera/model2.sqlite3")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--run-id", help="Reuse this ID for stable logging and SQLite records"
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="Continue a saved run, reusing every verified task",
    )
    parser.add_argument("--min-score", type=float, default=0.75)
    parser.add_argument("--synthesis-rounds", type=int, default=1)
    parser.add_argument("--no-plan-review", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--print-final", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    )
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.propagate = False
    LOGGER.addHandler(console)


async def async_main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    goal = " ".join(args.goal).strip()
    if not goal and not args.resume:
        goal = input("Enter task:\n").strip()
    if not goal and not args.resume:
        print("No task provided.", file=sys.stderr)
        return 2

    reporter: RunReporter | None = None
    try:
        api_key = get_api_key(args.api_key)
        metrics = RunMetrics()
        provider = AgentausAsyncClient(
            api_key,
            args.base_url,
            args.path,
            timeout=args.timeout,
            metrics=metrics,
        )
        store = SQLiteRunStore(Path(args.db))
        model_env = {
            "planning": "PLANNING",
            "worker": "WORKER",
            "worker:low": "WORKER_LOW",
            "worker:high": "WORKER_HIGH",
            "verification": "VERIFICATION",
            "synthesis": "SYNTHESIS",
        }
        models = {
            stage: os.environ.get(f"AGENTAUS_{suffix}_MODEL", "")
            for stage, suffix in model_env.items()
        }
        models = {key: value for key, value in models.items() if value}
        run_id = args.resume or args.run_id or new_run_id()
        reporter = RunReporter(Path(args.out_dir), run_id)
        reporter.attach_log(LOGGER)
        orchestrator = Orchestrator(
            provider,
            store=store,
            concurrency=args.concurrency,
            max_workers_attempts=args.attempts,
            verification_required=not args.no_verification,
            max_verification_rounds=args.verification_rounds,
            token_budget=args.token_budget,
            run_id=run_id,
            models=models,
            min_score=args.min_score,
            plan_review=not args.no_plan_review,
            max_synthesis_rounds=args.synthesis_rounds,
            fail_fast=args.fail_fast,
            reporter=reporter,
        )
        LOGGER.info(
            "Starting run_id=%s resume=%s output_dir=%s",
            run_id,
            bool(args.resume),
            reporter.dir,
        )
        if args.resume:
            output = await orchestrator.resume()
        else:
            output = await orchestrator.run(goal)
        LOGGER.info(
            "Run metrics requests=%d retries=%d prompt_tokens=%d "
            "completion_tokens=%d total_tokens=%d",
            metrics.requests,
            metrics.retries,
            metrics.prompt_tokens,
            metrics.completion_tokens,
            metrics.total_tokens,
        )
        print(f"Status: {orchestrator.status}")
        print(f"Result: {reporter.dir / 'final.md'}")
        print(f"Report: {reporter.dir / 'report.md'}")
        if args.print_final:
            print(f"\n{output}")
        return 3 if orchestrator.status == "partial" else 0
    except (RuntimeError, ValueError, OSError, httpx.HTTPError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if reporter is not None:
            print(f"Report: {reporter.dir / 'report.md'}", file=sys.stderr)
        LOGGER.debug("Fatal error", exc_info=True)
        return 1
    finally:
        if reporter is not None:
            reporter.close()


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
