"""採点器とテキストユーティリティの単体テスト."""

from __future__ import annotations

import unittest

from llmbench.scorers import _normalize_indent, score_answer
from llmbench.textutils import (
    estimate_tokens,
    extract_answer_span,
    extract_choice,
    extract_json_object,
    extract_numbers,
    normalize_text,
    strip_reasoning,
)


class TextUtilsTest(unittest.TestCase):
    def test_normalize_fullwidth_digits(self):
        self.assertEqual(normalize_text("３７　×　８９"), "37 × 89")

    def test_estimate_tokens_ranges(self):
        self.assertEqual(estimate_tokens(""), 0)
        ja = estimate_tokens("東京は日本の首都です")
        self.assertTrue(8 <= ja <= 12, ja)
        en = estimate_tokens("the quick brown fox jumps")
        self.assertTrue(5 <= en <= 8, en)

    def test_extract_answer_span_marker(self):
        text = "計算:\n37 × 89 = 3293\n最終回答: 3293"
        self.assertEqual(extract_answer_span(text), "3293")

    def test_extract_answer_span_last_line(self):
        text = "いくつかの考察\n行1\n答えは3293です"
        self.assertIn("3293", extract_answer_span(text))

    def test_extract_choice_variants(self):
        self.assertEqual(extract_choice("答え: B"), "B")
        self.assertEqual(extract_choice("正解は C です。"), "C")
        self.assertEqual(extract_choice("**A.** 選択肢"), "A")

    def test_extract_numbers(self):
        self.assertEqual(extract_numbers("答え: 1,024"), [1024.0])
        self.assertEqual(extract_numbers("0.625"), [0.625])

    def test_extract_json(self):
        obj = extract_json_object("説明\n{\"name\": \"山田\", \"age\": 34}\n後文")
        self.assertEqual(obj, {"name": "山田", "age": 34})

    def test_extract_json_in_fence(self):
        obj = extract_json_object('```json\n{"a": 1}\n```')
        self.assertEqual(obj, {"a": 1})


class ScorerTest(unittest.TestCase):
    def test_numeric_correct(self):
        r = score_answer("計算すると\n最終回答: 3293", {"type": "numeric", "value": 3293})
        self.assertTrue(r.correct)

    def test_numeric_tolerance(self):
        r = score_answer("答え: 0.62", {"type": "numeric", "value": 0.625, "tolerance": 0.01})
        self.assertTrue(r.correct)

    def test_numeric_wrong(self):
        r = score_answer("最終回答: 3200", {"type": "numeric", "value": 3293})
        self.assertFalse(r.correct)

    def test_numeric_fullwidth(self):
        r = score_answer("最終回答: １０２４", {"type": "numeric", "value": 1024})
        self.assertTrue(r.correct)

    def test_choice_correct(self):
        r = score_answer("考察...\n答え: C", {"type": "choice", "value": "C"})
        self.assertTrue(r.correct)

    def test_choice_wrong(self):
        r = score_answer("答え: A", {"type": "choice", "value": "C"})
        self.assertFalse(r.correct)

    def test_exact_correct(self):
        r = score_answer("最終回答: H₂O", {"type": "exact", "values": ["H2O", "水"]})
        self.assertTrue(r.correct)

    def test_exact_wrong(self):
        r = score_answer("CO2", {"type": "exact", "values": ["H2O"]})
        self.assertFalse(r.correct)

    def test_json_partial_wrong(self):
        spec = {"type": "json", "expected": {"name": "山田太郎", "age": 34, "city": "大阪"}}
        r = score_answer('{"name": "山田太郎", "age": 30, "city": "大阪"}', spec)
        self.assertFalse(r.correct)
        self.assertIn("age", r.detail)

    def test_json_correct(self):
        spec = {"type": "json", "expected": {"name": "山田太郎", "age": 34}}
        r = score_answer('回答:\n{"name": "山田太郎", "age": 34}', spec)
        self.assertTrue(r.correct)

    def test_code_correct(self):
        spec = {
            "type": "code",
            "language": "python",
            "tests": [{"call": "add(2, 3)", "expected": 5}],
        }
        answer = "```python\ndef add(a, b):\n    return a + b\n```"
        r = score_answer(answer, spec)
        self.assertTrue(r.correct, r.detail)

    def test_code_wrong(self):
        spec = {
            "type": "code",
            "language": "python",
            "tests": [{"call": "add(2, 3)", "expected": 6}],
        }
        answer = "```python\ndef add(a, b):\n    return a + b\n```"
        r = score_answer(answer, spec)
        self.assertFalse(r.correct)

    def test_code_runtime_error(self):
        spec = {
            "type": "code",
            "language": "python",
            "tests": [{"call": "broken()", "expected": 1}],
        }
        answer = "```python\nraise SystemExit(1)\n```"
        r = score_answer(answer, spec)
        self.assertFalse(r.correct)

    def test_unknown_scorer(self):
        r = score_answer("x", {"type": "nope"})
        self.assertFalse(r.correct)
        self.assertIn("未知", r.detail)

    def test_numeric_with_think_tags(self):
        open_tag, close_tag = "<" + "think>", "<" + "/think>"
        answer = f"{open_tag} 37*89=3293 {close_tag}\n最終回答: 3293"
        r = score_answer(answer, {"type": "numeric", "value": 3293})
        self.assertTrue(r.correct, r.detail)

    def test_choice_with_reasoning_text(self):
        answer = "思考: Aが有力だが...\n答え: C"
        r = score_answer(answer, {"type": "choice", "value": "C"})
        self.assertTrue(r.correct, r.detail)


class StripReasoningTest(unittest.TestCase):
    OPEN = "<" + "think>"
    CLOSE = "<" + "/think>"

    def test_strip_pairs(self):
        self.assertEqual(
            strip_reasoning(f"a {self.OPEN} hidden {self.CLOSE} b"), "a b"
        )

    def test_strip_unclosed(self):
        self.assertEqual(
            strip_reasoning(f"前置き {self.OPEN} long chain"), "前置き"
        )

    def test_no_tags(self):
        self.assertEqual(strip_reasoning("通常文本"), "通常文本")


class NormalizeIndentTest(unittest.TestCase):
    def test_valid_one_space_code_untouched(self):
        # 1スペースインデントでもPythonは妥当なので変更しない
        code = "def f(n):\n if n:\n  return 1\n return 0"
        self.assertEqual(_normalize_indent(code), code)

    def test_valid_code_untouched(self):
        code = "def f(n):\n    return n"
        self.assertEqual(_normalize_indent(code), code)

    def test_collapsed_nesting_left_alone(self):
        # 入れ子が潰れて全行1スペースのものは救済しない（モデル起因の失点）
        broken = "for i in x:\n if i:\n print(i)"
        self.assertEqual(_normalize_indent(broken), broken)

    def test_unfixable_broken_left_alone(self):
        # インデント自体が無い壊れたコードは変更しない（そのまま実行して失敗扱い）
        broken = "def f():\nprint()"
        self.assertEqual(_normalize_indent(broken), broken)

    def test_code_scorer_with_one_space_indent(self):
        answer = (
            "```python\ndef double(x):\n return x * 2\n```"
        )
        spec = {
            "type": "code",
            "language": "python",
            "tests": [{"call": "double(3)", "expected": 6}],
        }
        r = score_answer(answer, spec)
        self.assertTrue(r.correct, r.detail)


if __name__ == "__main__":
    unittest.main()
