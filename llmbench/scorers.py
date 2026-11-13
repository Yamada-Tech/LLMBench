"""解答の採点器.

judge LLM を使わず、決定論的な抽出+比較で採点する。
- numeric : 数値一致（tolerance 付き）
- choice  : 選択肢 A-E 一致
- exact   : 正規化文字列一致（候補のいずれかと一致なら正解）
- json    : JSONオブジェクトのキー・値比較
- code    : 抽出したPythonコードを分離サブプロセスで実行しテスト一致
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .textutils import (
    extract_answer_span,
    extract_choice,
    extract_code_block,
    extract_json_object,
    extract_numbers,
    normalize_text,
    strip_reasoning,
)

__all__ = ["ScoreResult", "score_answer", "MAX_CODE_SECONDS"]

MAX_CODE_SECONDS = 10

# サンドボックス実行でモデルコードに許可する最小限のビルティン
_SAFE_BUILTINS_NAMES = [
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "int", "isinstance", "issubclass",
    "iter", "len", "list", "map", "max", "min", "next", "print", "range",
    "repr", "reversed", "round", "set", "sorted", "str", "sum", "tuple",
    "zip", "ValueError", "TypeError", "IndexError", "KeyError", "ZeroDivisionError",
    "StopIteration", "Exception", "True", "False", "None",
]


@dataclass
class ScoreResult:
    """1問の採点結果."""

    correct: bool
    method: str
    extracted: str
    expected: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "correct": self.correct,
            "method": self.method,
            "extracted": self.extracted,
            "expected": self.expected,
            "detail": self.detail,
        }


def _no_answer(expected: str, method: str) -> ScoreResult:
    return ScoreResult(False, method, "", expected, "解答を抽出できませんでした")


def _score_numeric(answer: str, spec: dict) -> ScoreResult:
    expected = float(spec["value"])
    tolerance = float(spec.get("tolerance", 0.005))
    span = extract_answer_span(answer)
    candidates = extract_numbers(span) or extract_numbers(strip_reasoning(answer))
    for value in candidates:
        if abs(value - expected) <= max(tolerance, abs(expected) * 0.001):
            return ScoreResult(True, "numeric", str(value), str(expected))
    got = candidates[0] if candidates else None
    return ScoreResult(
        False,
        "numeric",
        "" if got is None else str(got),
        str(expected),
        "数値が一致しません" if candidates else "数値を抽出できません",
    )


def _score_choice(answer: str, spec: dict) -> ScoreResult:
    expected = str(spec["value"]).strip().upper()
    got = extract_choice(strip_reasoning(answer))
    if got == expected:
        return ScoreResult(True, "choice", got, expected)
    return ScoreResult(
        False, "choice", got or "", expected, "選択肢が一致しません"
    )


def _score_exact(answer: str, spec: dict) -> ScoreResult:
    candidates = [str(v) for v in spec.get("values", [spec.get("value", "")])]
    span = extract_answer_span(answer)
    got_norm = normalize_text(span).lower()
    for candidate in candidates:
        want = normalize_text(candidate).lower()
        if got_norm == want or (len(want) >= 2 and want in got_norm):
            return ScoreResult(True, "exact", span, " | ".join(candidates))
    # マーカーが無い場合は本文全体の一致も許可
    full = normalize_text(answer).lower()
    for candidate in candidates:
        want = normalize_text(candidate).lower()
        if full == want:
            return ScoreResult(True, "exact", answer, " | ".join(candidates))
    return ScoreResult(
        False, "exact", span, " | ".join(candidates), "文字列が一致しません"
    )


def _values_match(got, want, tolerance: float = 0.01) -> bool:
    if isinstance(want, bool) or isinstance(got, bool):
        return bool(got) == bool(want)
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return abs(float(got) - float(want)) <= max(tolerance, abs(float(want)) * 0.01)
    if isinstance(want, str):
        got_norm = normalize_text(str(got)).lower()
        want_norm = normalize_text(want).lower()
        return got_norm == want_norm or (
            len(want_norm) >= 2 and want_norm in got_norm
        )
    if isinstance(want, list) and isinstance(got, list):
        return len(got) == len(want) and all(
            _values_match(g, w, tolerance) for g, w in zip(got, want)
        )
    return got == want


def _score_json(answer: str, spec: dict) -> ScoreResult:
    expected = dict(spec.get("expected", {}))
    parsed = extract_json_object(answer)
    if parsed is None:
        return ScoreResult(False, "json", "", json.dumps(expected, ensure_ascii=False), "JSONを解析できません")
    matched = 0
    misses: list[str] = []
    for key, want in expected.items():
        if key in parsed and _values_match(parsed[key], want):
            matched += 1
        else:
            misses.append(str(key))
    correct = matched == len(expected) and len(expected) > 0
    return ScoreResult(
        correct,
        "json",
        json.dumps(parsed, ensure_ascii=False),
        json.dumps(expected, ensure_ascii=False),
        "" if correct else f"不一致キー: {', '.join(misses)}",
    )


# --- code scorer: サンドボックス実行 -------------------------------------

_RUNNER_TEMPLATE = """
import sys, json, io, contextlib

_SAFE = {names!r}
import builtins as _b
_safe_builtins = {{name: getattr(_b, name) for name in _SAFE if hasattr(_b, name)}}

MODEL_CODE = {code!r}
TESTS = {tests!r}

def run_tests():
    ns = {{"__builtins__": _safe_builtins, "__name__": "__main__"}}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(MODEL_CODE, "<model>", "exec"), ns)
    except Exception as exc:
        print(json.dumps({{"ok": False, "results": [],
                          "error": "code error: %s" % exc}}))
        return
    results = []
    for call, want in TESTS:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                got = eval(call, ns)
            results.append({{"call": call, "pass": got == want}})
        except Exception as exc:
            results.append({{"call": call, "pass": False, "error": str(exc)}})
    print(json.dumps({{"ok": True, "results": results}}))

run_tests()
"""


def _normalize_indent(code: str) -> str:
    """構文エラーのPythonだけ、1スペース刻みインデントを 4スペースへ引き直す.

    元のコードがそのままコンパイルできる場合は一切変更しない（1スペース
    インデント自体はPython的に妥当なため）。入れ子構造そのものが潰れて
    復元不能なコードはそのまま返す（モデル起因の失敗として採点される）。
    """
    try:
        compile(code, "<model>", "exec")
        return code
    except SyntaxError:
        pass
    lines = code.splitlines()
    # 非ゼロインデントが全て1スペース刻みなら 4スペースへ引き直せる
    indents = [
        len(ln) - len(ln.lstrip(" ")) for ln in lines if ln.strip() and ln[0] == " "
    ]
    if not indents or min(indents) != 1 or max(indents) < 1:
        return code
    if any("\t" in ln[: len(ln) - len(ln.lstrip())] for ln in lines):
        return code  # タブインデントは触らない
    fixed = []
    for line in lines:
        if not line.strip():
            fixed.append(line)
            continue
        width = len(line) - len(line.lstrip(" "))
        fixed.append(" " * (width * 4) + line.lstrip(" "))
    candidate = "\n".join(fixed)
    try:
        compile(candidate, "<model>", "exec")
        return candidate
    except SyntaxError:
        return code


def _score_code(answer: str, spec: dict) -> ScoreResult:
    language = str(spec.get("language", "python"))
    tests = list(spec.get("tests", []))
    code = _normalize_indent(extract_code_block(answer, language))
    if language.lower() != "python" or not tests:
        return ScoreResult(
            False, "code", code, "python tests",
            "現状 code スコーアはPythonのみ実行可能です",
        )
    names = _SAFE_BUILTINS_NAMES
    runner = _RUNNER_TEMPLATE.format(
        names=names, code=code, tests=[(t["call"], t["expected"]) for t in tests]
    )
    with tempfile.TemporaryDirectory(prefix="llmbench_code_") as tmp:
        script = Path(tmp) / "runner.py"
        script.write_text(runner, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-X", "utf8", str(script)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=MAX_CODE_SECONDS,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return ScoreResult(False, "code", code, "tests pass", "実行タイムアウト")
    payload = None
    for line in (proc.stdout or "").strip().splitlines()[::-1]:
        try:
            payload = json.loads(line)
            break
        except ValueError:
            continue
    if payload is None:
        return ScoreResult(
            False, "code", code, "tests pass",
            f"実行結果を解析できません (exit={proc.returncode})",
        )
    if not payload.get("ok"):
        return ScoreResult(False, "code", code, "tests pass", str(payload.get("error")))
    results = payload.get("results", [])
    passed = sum(1 for r in results if r.get("pass"))
    failed = [str(r.get("call")) for r in results if not r.get("pass")]
    return ScoreResult(
        passed == len(tests) and len(tests) > 0,
        "code",
        code,
        f"{len(tests)} tests",
        "" if not failed else f"失敗テスト: {', '.join(failed)}",
    )


_SCORERS = {
    "numeric": _score_numeric,
    "choice": _score_choice,
    "exact": _score_exact,
    "json": _score_json,
    "code": _score_code,
}


def score_answer(answer: str, scorer: dict) -> ScoreResult:
    """scorer仕様辞書に基づき解答を採点する."""
    method = str(scorer.get("type", "numeric"))
    if method not in _SCORERS:
        return ScoreResult(
            False, method, "", "", f"未知のscorerタイプ: {method}"
        )
    try:
        return _SCORERS[method](answer or "", scorer)
    except (KeyError, TypeError, ValueError) as exc:
        return ScoreResult(False, method, "", "", f"採点エラー: {exc}")
