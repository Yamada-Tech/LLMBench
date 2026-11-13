"""精度ベンチマーク.

内蔵問題集を temperature=0 / 固定seed で実行し、決定論的採点で
カテゴリ別の正答率と、速度統計（おまけ）を集める。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .backends import Backend, BackendError
from .cancel import CancelToken
from .metrics import summarize
from .scorers import ScoreResult, score_answer
from .tasks import Task, TaskSet
from .textutils import truncate

__all__ = ["TaskOutcome", "AccuracyResult", "run_accuracy_bench"]


@dataclass
class TaskOutcome:
    """1問の実行・採点結果."""

    task: Task
    answer: str
    score: ScoreResult
    elapsed_s: float
    ttft_s: float | None
    tok_per_s: float
    prompt_tokens: int | None
    completion_tokens: int
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.task.id,
            "category": self.task.category,
            "difficulty": self.task.difficulty,
            "correct": self.score.correct,
            "method": self.score.method,
            "extracted": truncate(self.score.extracted, 200),
            "expected": truncate(self.score.expected, 200),
            "detail": self.score.detail,
            "answer_preview": truncate(self.answer, 240),
            "elapsed_s": round(self.elapsed_s, 4),
            "ttft_s": round(self.ttft_s, 4) if self.ttft_s is not None else None,
            "tok_per_s": round(self.tok_per_s, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "error": self.error,
        }


@dataclass
class AccuracyResult:
    """精度ベンチ全体の集計."""

    backend_info: dict
    tasks_run: int
    outcomes: list = field(default_factory=list)
    by_category: dict = field(default_factory=dict)
    by_difficulty: dict = field(default_factory=dict)
    overall: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": "accuracy",
            "backend": self.backend_info,
            "tasks_run": self.tasks_run,
            "overall": self.overall,
            "by_category": self.by_category,
            "by_difficulty": self.by_difficulty,
            "errors": self.errors,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


def _percent(correct: int, total: int) -> float:
    return round(100.0 * correct / total, 2) if total else 0.0


def run_accuracy_bench(
    backend: Backend,
    task_set: TaskSet,
    max_tokens: int = 512,
    temperature: float = 0.0,
    seed: int | None = 42,
    repeat: int = 1,
    progress=None,
    cancel: CancelToken | None = None,
) -> AccuracyResult:
    """問題セットをバックエンドで実行して採点する."""
    def check_cancel() -> None:
        if cancel is not None:
            cancel.check()
    result = AccuracyResult(backend_info=backend.info(), tasks_run=0)

    def log(msg: str) -> None:
        if progress:
            progress(msg)

    tasks = task_set.tasks
    total = len(tasks) * max(1, repeat)
    log(f"問題数 {len(tasks)} × 繰り返し {repeat} = {total} 問を実行")
    for rep in range(max(1, repeat)):
        for index, task in enumerate(tasks, 1):
            check_cancel()
            prefix = f"[{index}/{total}]" if repeat == 1 else f"[{index}/{total} r{rep + 1}]"
            log(f"{prefix} {task.id} ({task.category}/{task.difficulty}) ...")
            started = time.perf_counter()
            try:
                gen = backend.generate(
                    task.messages(),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=None if seed is None else seed + rep,
                )
            except BackendError as exc:
                elapsed = time.perf_counter() - started
                result.errors.append(f"{task.id}: {exc}")
                log(f"  ✗ エラー: {exc}")
                score = ScoreResult(False, "error", "", "", str(exc))
                outcome = TaskOutcome(
                    task=task, answer="", score=score, elapsed_s=elapsed,
                    ttft_s=None, tok_per_s=0.0, prompt_tokens=None,
                    completion_tokens=0, error=str(exc),
                )
            else:
                elapsed = time.perf_counter() - started
                if not gen.text.strip():
                    warning = (gen.stats or {}).get("warning", "")
                    message = "応答が空です"
                    if warning:
                        message += f"（{warning}）"
                    result.errors.append(f"{task.id}: {message}")
                    log(f"  ✗ {message}")
                score = score_answer(gen.text, task.scorer)
                outcome = TaskOutcome(
                    task=task,
                    answer=gen.text,
                    score=score,
                    elapsed_s=elapsed,
                    ttft_s=gen.ttft_s,
                    tok_per_s=gen.tok_per_s,
                    prompt_tokens=gen.prompt_tokens,
                    completion_tokens=gen.completion_tokens,
                )
            result.outcomes.append(outcome)
            result.tasks_run += 1
            mark = "✓" if outcome.score.correct else "✗"
            log(f"  {mark} ({outcome.elapsed_s:.2f}s, {outcome.tok_per_s:.1f} tok/s)")

    # --- 集計 -------------------------------------------------------------
    outcomes = result.outcomes
    correct_all = sum(1 for o in outcomes if o.score.correct)
    result.overall = {
        "correct": correct_all,
        "total": len(outcomes),
        "accuracy_pct": _percent(correct_all, len(outcomes)),
        "elapsed_total_s": round(sum(o.elapsed_s for o in outcomes), 3),
        "tok_per_s": summarize([o.tok_per_s for o in outcomes]).to_dict(),
    }

    def group(key: str) -> dict:
        groups: dict[str, dict] = {}
        for o in outcomes:
            value = getattr(o.task, key)
            g = groups.setdefault(value, {"correct": 0, "total": 0})
            g["total"] += 1
            g["correct"] += 1 if o.score.correct else 0
        for g in groups.values():
            g["accuracy_pct"] = _percent(g["correct"], g["total"])
        return groups

    result.by_category = group("category")
    result.by_difficulty = group("difficulty")
    return result
