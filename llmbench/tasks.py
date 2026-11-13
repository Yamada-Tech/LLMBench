"""ベンチマーク問題の読み込みとメッセージ構築."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Task", "load_tasks", "default_tasks_path", "CATEGORIES"]

CATEGORIES = (
    "math",
    "arithmetic",
    "logic",
    "knowledge",
    "extraction",
    "code",
    "reasoning",
)

# 出力形式の指示（抽出の安定化）。scorerタイプごとに統一する。
_INSTRUCTIONS = {
    "numeric": "計算過程の後に、必ず「最終回答: <数値>」の形式で最終値だけの行を出力しなさい。",
    "choice": "考察の後に、必ず「答え: <選択肢の文字(例: A)>」の形式で最終行を出力しなさい。",
    "exact": "考察の後に、必ず「最終回答: <語>」の形式で最終行を出力しなさい。",
    "json": "説明文やコードフェンスは使わず、有効なJSONオブジェクトのみを出力しなさい。",
    "code": "説明文は不要で、完全なPythonコードをコードブロックで出力しなさい。",
}

_DEFAULT_SYSTEM = (
    "あなたは正確なAIアシスタントです。指示された出力形式を厳守してください。"
)


@dataclass
class Task:
    """1つの精度ベンチマーク問題."""

    id: str
    category: str
    difficulty: str
    language: str
    prompt: str
    scorer: dict
    system: str = _DEFAULT_SYSTEM

    @property
    def instruction(self) -> str:
        return _INSTRUCTIONS.get(self.scorer.get("type", ""), "")

    def messages(self) -> list[dict]:
        """チャットAPIへ送るメッセージ列を返す."""
        user = self.prompt
        if self.instruction:
            user = f"{self.prompt}\n\n{self.instruction}"
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user},
        ]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "difficulty": self.difficulty,
            "language": self.language,
            "scorer_type": self.scorer.get("type"),
        }


@dataclass
class TaskSet:
    """問題集（バージョン管理付き）."""

    version: int = 1
    tasks: list[Task] = field(default_factory=list)
    path: str | None = None

    def filter(
        self,
        categories: list[str] | None = None,
        difficulties: list[str] | None = None,
    ) -> "TaskSet":
        selected = self.tasks
        if categories:
            wanted = {c.strip() for c in categories}
            selected = [t for t in selected if t.category in wanted]
        if difficulties:
            wanted = {d.strip() for d in difficulties}
            selected = [t for t in selected if t.difficulty in wanted]
        return TaskSet(version=self.version, tasks=selected, path=self.path)


def default_tasks_path() -> Path:
    """同梱のデフォルト問題集のパス."""
    return Path(__file__).parent / "data" / "tasks.json"


def load_tasks(path: str | Path | None = None) -> TaskSet:
    """問題集JSONを読み込む（デフォルトは同梱の tasks.json）."""
    file_path = Path(path) if path else default_tasks_path()
    if not file_path.exists():
        raise FileNotFoundError(f"問題集が見つかりません: {file_path}")
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    tasks = [
        Task(
            id=str(item["id"]),
            category=str(item.get("category", "general")),
            difficulty=str(item.get("difficulty", "medium")),
            language=str(item.get("language", "ja")),
            prompt=str(item["prompt"]),
            scorer=dict(item.get("scorer", {})),
            system=str(item.get("system", _DEFAULT_SYSTEM)),
        )
        for item in raw.get("tasks", [])
    ]
    return TaskSet(version=int(raw.get("version", 1)), tasks=tasks, path=str(file_path))
