from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import httpx
from pydantic import ValidationError

import model2
import reporting


def make_task(task_id, depends_on=()):
    return model2.SubTask(
        id=task_id,
        title=f"Task {task_id}",
        role="Analyst",
        instruction=f"Complete task {task_id} using the available evidence.",
        depends_on=list(depends_on),
        deliverables=[f"Result {task_id}"],
        acceptance_criteria=[f"Result {task_id} is complete"],
    )


class FakeProvider(model2.AgentausAsyncClient):
    def __init__(self) -> None:
        super().__init__("test-key", metrics=model2.RunMetrics())

    async def tool_call(
        self,
        http,
        messages,
        *,
        tool,
        expected_name,
        model,
        **extra,
    ):
        if expected_name == "submit_task_plan":
            return model(
                summary="Create a reliable launch plan with two sequenced phases.",
                subtasks=[
                    model2.SubTask(
                        id="research",
                        title="Research the market",
                        role="Researcher",
                        instruction="Research the audience and summarize the strongest evidence.",
                        deliverables=["A concise market brief"],
                        acceptance_criteria=["Names the audience"],
                    ),
                    model2.SubTask(
                        id="plan",
                        title="Create the launch plan",
                        role="Strategist",
                        instruction="Create a sequenced plan from the research evidence.",
                        depends_on=["research"],
                        deliverables=["A sequenced launch plan"],
                        acceptance_criteria=["Uses the research evidence"],
                    ),
                ],
            )
        if expected_name == "submit_plan_review":
            return model(approved=True, score=0.9, issues=[], suggested_changes=[])
        if expected_name == "submit_worker_result":
            return model(
                output="A complete and evidence-oriented result.",
                confidence=0.9,
                claims=["Claim supported by the task evidence"],
                sources=["primary source"],
                risks=[],
                unresolved_questions=[],
            )
        if expected_name == "submit_verification":
            return model(
                passed=True,
                score=0.95,
                issues=[],
                missing_deliverables=[],
                repairs=[],
            )
        if expected_name == "submit_judge_result":
            return model(
                accepted=True,
                score=0.95,
                critical_issues=[],
                improvements=[],
            )
        raise AssertionError(f"Unexpected tool call: {expected_name}")

    async def reply_text(self, http, messages, **extra):
        return "A cohesive final launch answer."


class PartialProvider(FakeProvider):
    def __init__(self, failing=()) -> None:
        super().__init__()
        self.failing = set(failing)
        self.worker_calls = []

    async def tool_call(
        self,
        http,
        messages,
        *,
        tool,
        expected_name,
        model,
        **extra,
    ):
        if expected_name == "submit_task_plan":
            return model(
                summary="Three tasks where one branch is independent.",
                subtasks=[make_task("a"), make_task("b", ["a"]), make_task("c")],
            )
        if expected_name == "submit_worker_result":
            task_id = extra["stage"].split(":", 1)[1]
            self.worker_calls.append(task_id)
            if task_id in self.failing:
                raise model2.InvalidOutputError("simulated bad output")
        return await super().tool_call(
            http,
            messages,
            tool=tool,
            expected_name=expected_name,
            model=model,
            **extra,
        )


class Model2Tests(unittest.TestCase):
    def test_cycle_is_rejected(self):
        task_a = model2.SubTask(
            id="a",
            title="Task A",
            role="Analyst",
            instruction="Complete task A using the available evidence.",
            depends_on=["b"],
            deliverables=["Result A"],
            acceptance_criteria=["Result A is complete"],
        )
        task_b = model2.SubTask(
            id="b",
            title="Task B",
            role="Analyst",
            instruction="Complete task B using the available evidence.",
            depends_on=["a"],
            deliverables=["Result B"],
            acceptance_criteria=["Result B is complete"],
        )
        with self.assertRaises(ValidationError):
            model2.TaskPlan(
                summary="A summary for a deliberately cyclic test plan.",
                subtasks=[task_a, task_b],
            )

    def test_unknown_dependency_is_rejected(self):
        task = model2.SubTask(
            id="a",
            title="Task A",
            role="Analyst",
            instruction="Complete task A using the available evidence.",
            depends_on=["missing"],
            deliverables=["Result A"],
            acceptance_criteria=["Result A is complete"],
        )
        with self.assertRaises(ValidationError):
            model2.TaskPlan(
                summary="A summary for a deliberately invalid test plan.",
                subtasks=[task],
            )

    def test_strict_schema_recurses_into_submodels(self):
        schema = model2._schema(model2.TaskPlan)
        self.assertFalse(schema["additionalProperties"])
        subtask_schema = schema["$defs"]["SubTask"]
        self.assertFalse(subtask_schema["additionalProperties"])

    def test_worker_submission_does_not_ask_model_to_echo_task_id(self):
        tool = model2._tool(
            "submit_worker_result",
            "Submit a worker result.",
            model2.WorkerSubmission,
        )
        properties = tool["function"]["parameters"]["properties"]
        self.assertNotIn("task_id", properties)

    def test_provider_payload_uses_bearer_and_overwrite(self):
        provider = model2.AgentausAsyncClient("secret")
        payload = provider.payload(
            [{"role": "system", "content": "Test"}],
            tools=[{"type": "function", "function": {"name": "test"}}],
            tool_choice="required",
        )
        self.assertEqual(provider.headers()["Authorization"], "Bearer secret")
        self.assertTrue(payload["system_prompt_overwrite"])
        self.assertEqual(payload["tool_choice"], "required")

    def test_end_to_end_fake_run_persists_and_synthesizes(self):
        async def run() -> None:
            with tempfile.TemporaryDirectory() as directory:
                store = model2.SQLiteRunStore(
                    Path(directory) / "model2.sqlite3"
                )
                provider = FakeProvider()
                orchestrator = model2.Orchestrator(
                    provider,
                    store=store,
                    concurrency=2,
                    run_id="test-run",
                )
                final_output = await orchestrator.run(
                    "Create a launch plan for a sustainable coffee brand."
                )
                self.assertIn("cohesive", final_output)
                with store.connection() as db:
                    run_row = db.execute(
                        "SELECT status, final_output FROM runs WHERE id='test-run'"
                    ).fetchone()
                    task_count = db.execute(
                        "SELECT COUNT(*) AS count FROM tasks "
                        "WHERE run_id='test-run'"
                    ).fetchone()["count"]
                    task_rows = db.execute(
                        "SELECT task_id, status, result_json FROM tasks "
                        "WHERE run_id='test-run'"
                    ).fetchall()
                self.assertEqual(run_row["status"], "completed")
                self.assertEqual(task_count, 2)
                self.assertEqual(
                    {row["task_id"] for row in task_rows},
                    {"research", "plan"},
                )
                self.assertTrue(
                    all(row["status"] == "verified" for row in task_rows)
                )
                self.assertTrue(
                    all(
                        f'"task_id":"{row["task_id"]}"' in row["result_json"]
                        for row in task_rows
                    )
                )

        asyncio.run(run())

    def test_reports_are_written_as_markdown(self):
        async def run() -> None:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                store = model2.SQLiteRunStore(root / "model2.sqlite3")
                reporter = reporting.RunReporter(root / "runs", "report-run")
                orchestrator = model2.Orchestrator(
                    FakeProvider(),
                    store=store,
                    run_id="report-run",
                    reporter=reporter,
                )
                await orchestrator.run("Create a launch plan.")
                run_dir = root / "runs" / "report-run"
                for name in (
                    "final.md",
                    "report.md",
                    "plan.md",
                    "tasks/research.md",
                    "tasks/plan.md",
                ):
                    self.assertTrue((run_dir / name).is_file(), name)
                self.assertIn("cohesive", (run_dir / "final.md").read_text())
                report = (run_dir / "report.md").read_text()
                self.assertIn("```mermaid", report)
                self.assertIn("verified", report)

        asyncio.run(run())

    def test_failed_task_skips_dependents_but_not_siblings(self):
        async def run() -> None:
            with tempfile.TemporaryDirectory() as directory:
                store = model2.SQLiteRunStore(Path(directory) / "model2.sqlite3")
                orchestrator = model2.Orchestrator(
                    PartialProvider(failing={"a"}),
                    store=store,
                    run_id="partial-run",
                )
                output = await orchestrator.run("Goal with a broken branch.")
                statuses = {
                    task_id: state.status
                    for task_id, state in orchestrator.states.items()
                }
                self.assertEqual(
                    statuses, {"a": "failed", "b": "skipped", "c": "verified"}
                )
                self.assertEqual(orchestrator.status, "partial")
                self.assertIn("Incomplete Work", output)

        asyncio.run(run())

    def test_resume_reruns_only_unfinished_tasks(self):
        async def run() -> None:
            with tempfile.TemporaryDirectory() as directory:
                store = model2.SQLiteRunStore(Path(directory) / "model2.sqlite3")
                await model2.Orchestrator(
                    PartialProvider(failing={"a"}), store=store, run_id="r1"
                ).run("Goal with a broken branch.")
                second = PartialProvider()
                orchestrator = model2.Orchestrator(second, store=store, run_id="r1")
                await orchestrator.resume()
                self.assertEqual(second.worker_calls, ["a", "b"])
                self.assertEqual(orchestrator.status, "completed")
                self.assertTrue(orchestrator.states["c"].resumed)

        asyncio.run(run())

    def test_model_routing_prefers_complexity_override(self):
        with tempfile.TemporaryDirectory() as directory:
            store = model2.SQLiteRunStore(Path(directory) / "model2.sqlite3")
            orchestrator = model2.Orchestrator(
                FakeProvider(),
                store=store,
                models={"worker": "base", "worker:high": "large"},
            )
            self.assertEqual(
                orchestrator._model_params("worker", model2.Complexity.high),
                {"model": "large"},
            )
            self.assertEqual(
                orchestrator._model_params("worker", model2.Complexity.low),
                {"model": "base"},
            )

    def test_provider_retries_transport_failure(self):
        async def run() -> None:
            attempts = 0

            async def handler(request):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise httpx.ConnectError("temporary")
                return httpx.Response(
                    200,
                    json={
                        "choices": [{"message": {"content": "ok"}}],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                    },
                )

            provider = model2.AgentausAsyncClient(
                "test-key", max_retries=1
            )
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as http:
                text = await provider.reply_text(
                    http,
                    [{"role": "user", "content": "hello"}],
                )
            self.assertEqual(text, "ok")
            self.assertEqual(attempts, 2)
            self.assertEqual(provider.metrics.prompt_tokens, 2)
            self.assertEqual(provider.metrics.completion_tokens, 3)
            self.assertEqual(provider.metrics.retries, 1)

        asyncio.run(run())

    def test_api_key_uses_simple_input_when_not_configured(self):
        from unittest.mock import patch

        with patch.dict("os.environ", {"AGENTAUS_API_KEY": ""}):
            with patch("builtins.input", return_value="interactive-test-key") as prompt:
                self.assertEqual(
                    model2.get_api_key(None),
                    "interactive-test-key",
                )
                prompt.assert_called_once_with("Enter Agentaus API key: ")


if __name__ == "__main__":
    unittest.main()
