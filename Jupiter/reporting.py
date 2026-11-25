from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model2 import JudgeResult, RunMetrics, SubTask, TaskPlan, TaskState

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"

STATUS_MARKS = {
    "verified": "✅",
    "complete": "✅",
    "failed": "❌",
    "rework_limit": "⚠️",
    "skipped": "⏭️",
    "running": "⏳",
    "pending": "⏳",
    "completed": "✅",
    "completed_with_gaps": "⚠️",
    "partial": "⚠️",
    "interrupted": "⏸️",
}
OK_STATES = {"verified", "complete"}
SKIPPED_STATES = {"skipped"}
BAD_STATES = {"failed", "rework_limit"}


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _bullets(items: list[str], empty: str = "None") -> list[str]:
    return [f"- {item}" for item in items] or [f"- _{empty}_"]


def _quote(text: str) -> list[str]:
    return [f"> {line}" if line else ">" for line in text.strip().splitlines()]


def _status(status: str) -> str:
    return f"{STATUS_MARKS.get(status, '•')} {status}"


class RunReporter:
    def __init__(self, root: Path, run_id: str) -> None:
        self.run_id = run_id
        self.dir = root / run_id
        (self.dir / "tasks").mkdir(parents=True, exist_ok=True)
        self._logger: logging.Logger | None = None
        self._handler: logging.Handler | None = None

    def attach_log(self, logger: logging.Logger) -> None:
        handler = logging.FileHandler(self.dir / "run.log", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        self._logger, self._handler = logger, handler

    def close(self) -> None:
        if self._logger and self._handler:
            self._logger.removeHandler(self._handler)
            self._handler.close()
        self._logger = self._handler = None

    def _write(self, relative: str, lines: list[str]) -> None:
        path = self.dir / relative
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        temporary.replace(path)

    def write_plan(self, goal: str, plan: TaskPlan) -> None:
        lines = [
            "# Plan",
            "",
            "## Goal",
            "",
            *_quote(goal),
            "",
            "## Summary",
            "",
            plan.summary,
            "",
            "## Tasks",
            "",
            "| ID | Title | Role | Complexity | Priority | Depends on |",
            "|---|---|---|---|---|---|",
        ]
        for task in plan.subtasks:
            lines.append(
                f"| `{task.id}` | {_cell(task.title)} | {_cell(task.role)} "
                f"| {task.estimated_complexity.value} | {task.priority} "
                f"| {_cell(', '.join(task.depends_on) or '—')} |"
            )
        for task in plan.subtasks:
            lines += [
                "",
                f"### `{task.id}` — {task.title}",
                "",
                task.instruction,
                "",
                "**Deliverables**",
                "",
                *_bullets(task.deliverables),
                "",
                "**Acceptance criteria**",
                "",
                *_bullets(task.acceptance_criteria),
            ]
        self._write("plan.md", lines)

    def write_task(self, task: SubTask, state: TaskState) -> None:
        note = " · resumed from checkpoint" if state.resumed else ""
        lines = [
            f"# {task.title}",
            "",
            f"- **Task:** `{task.id}` · **Role:** {task.role}",
            f"- **Status:** {_status(state.status)}{note}",
            f"- **Attempts:** {state.attempts} · **Repairs:** {state.repairs} "
            f"· **Time:** {state.seconds:.1f}s",
        ]
        if state.error:
            lines += ["", "## Error", "", state.error]
        verification = state.verification
        if verification:
            lines += [
                "",
                "## Verification",
                "",
                f"- **Passed:** {verification.passed} · **Score:** {verification.score:.2f}",
                "",
                "**Issues**",
                "",
                *_bullets(verification.issues),
                "",
                "**Missing deliverables**",
                "",
                *_bullets(verification.missing_deliverables),
            ]
        result = state.result
        if result:
            lines += [
                "",
                "## Output",
                "",
                f"_Worker confidence: {result.confidence:.2f}_",
                "",
                result.output,
                "",
                "## Claims",
                "",
                *_bullets(result.claims),
                "",
                "## Sources",
                "",
                *_bullets(result.sources),
                "",
                "## Risks",
                "",
                *_bullets(result.risks),
                "",
                "## Unresolved questions",
                "",
                *_bullets(result.unresolved_questions),
            ]
        self._write(f"tasks/{task.id}.md", lines)

    def write_final(
        self, output: str, status: str, judge: JudgeResult | None = None
    ) -> None:
        score = f"{judge.score:.2f}" if judge else "n/a"
        header = f"<!-- run: {self.run_id} | status: {status} | judge score: {score} -->"
        self._write("final.md", [header, "", output])

    def write_report(
        self,
        *,
        goal: str,
        plan: TaskPlan | None,
        states: dict[str, TaskState],
        status: str,
        metrics: RunMetrics,
        elapsed: float,
        judge: JudgeResult | None = None,
        error: str | None = None,
    ) -> None:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        lines = [
            f"# Run report: {self.run_id}",
            "",
            "| | |",
            "|---|---|",
            f"| Status | {_status(status)} |",
            f"| Finished | {stamp} |",
            f"| Duration | {elapsed:.1f}s (this session) |",
            f"| Requests | {metrics.requests} ({metrics.retries} retries) |",
            f"| Tokens | {metrics.total_tokens:,} "
            f"(prompt {metrics.prompt_tokens:,} / completion {metrics.completion_tokens:,}) |",
            "",
            "## Goal",
            "",
            *_quote(goal),
        ]
        if error:
            lines += ["", "## Run error", "", error]
        if plan:
            lines += ["", "## Plan", "", plan.summary, "", *self._mermaid(plan, states)]
            lines += [
                "",
                "## Tasks",
                "",
                "| Task | Role | Status | Attempts | Repairs | Score | Confidence | Time |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for task in plan.subtasks:
                state = states.get(task.id)
                if state is None:
                    continue
                score = f"{state.verification.score:.2f}" if state.verification else "—"
                confidence = f"{state.result.confidence:.2f}" if state.result else "—"
                lines.append(
                    f"| [`{task.id}`](tasks/{task.id}.md) | {_cell(task.role)} "
                    f"| {_status(state.status)}{' (resumed)' if state.resumed else ''} "
                    f"| {state.attempts} | {state.repairs} | {score} | {confidence} "
                    f"| {state.seconds:.1f}s |"
                )
        if judge:
            lines += [
                "",
                "## Final acceptance",
                "",
                f"- **Accepted:** {judge.accepted} · **Score:** {judge.score:.2f}",
                "",
                "**Critical issues**",
                "",
                *_bullets(judge.critical_issues),
                "",
                "**Suggested improvements**",
                "",
                *_bullets(judge.improvements),
            ]
        lines += self._open_issues(states)
        if metrics.by_stage:
            lines += ["", "## Tokens by stage", "", "| Stage | Tokens |", "|---|---|"]
            for stage, tokens in sorted(metrics.by_stage.items(), key=lambda item: -item[1]):
                lines.append(f"| {stage} | {tokens:,} |")
        lines += ["", "## Files", "", "- [final.md](final.md)", "- [plan.md](plan.md)"]
        lines += [f"- [tasks/{task_id}.md](tasks/{task_id}.md)" for task_id in states]
        lines.append("- [run.log](run.log)")
        self._write("report.md", lines)

    @staticmethod
    def _open_issues(states: dict[str, TaskState]) -> list[str]:
        lines: list[str] = []
        broken = [(tid, s) for tid, s in states.items() if s.status in BAD_STATES | SKIPPED_STATES]
        risks = [(tid, r) for tid, s in states.items() if s.result for r in s.result.risks]
        questions = [
            (tid, q) for tid, s in states.items() if s.result for q in s.result.unresolved_questions
        ]
        if broken:
            lines += ["", "## Incomplete tasks", ""]
            lines += [f"- `{tid}` ({s.status}): {_cell(s.error or 'no detail')}" for tid, s in broken]
        if risks:
            lines += ["", "## Risks reported by workers", ""]
            lines += [f"- `{tid}`: {risk}" for tid, risk in risks]
        if questions:
            lines += ["", "## Unresolved questions", ""]
            lines += [f"- `{tid}`: {question}" for tid, question in questions]
        return lines

    @staticmethod
    def _mermaid(plan: TaskPlan, states: dict[str, TaskState]) -> list[str]:
        nodes = {task.id: f"n{index}" for index, task in enumerate(plan.subtasks)}
        lines = ["```mermaid", "graph TD"]
        for task in plan.subtasks:
            label = f"{task.id}: {task.title}".replace('"', "'")
            lines.append(f'    {nodes[task.id]}["{label}"]')
        for task in plan.subtasks:
            lines += [f"    {nodes[dep]} --> {nodes[task.id]}" for dep in task.depends_on]
        groups = (
            ("ok", OK_STATES, "fill:#d4edda,stroke:#2e7d32"),
            ("bad", BAD_STATES, "fill:#f8d7da,stroke:#c62828"),
            ("skip", SKIPPED_STATES, "fill:#eeeeee,stroke:#888888"),
        )
        for name, members, style in groups:
            ids = [nodes[tid] for tid, s in states.items() if s.status in members and tid in nodes]
            if ids:
                lines += [f"    classDef {name} {style}", f"    class {','.join(ids)} {name}"]
        lines.append("```")
        return lines
