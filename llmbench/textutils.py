"""テキスト正規化・トークン概算・解答抽出のユーティリティ.

ベンチマークの採点と速度測定で共通して使う。外部依存なし。
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "normalize_text",
    "strip_reasoning",
    "estimate_tokens",
    "extract_answer_span",
    "extract_choice",
    "extract_numbers",
    "extract_json_object",
    "extract_code_block",
    "truncate",
]

_ANSWER_MARKERS = (
    r"最終回答",
    r"final\s+answer",
    r"答え",
    r"回答",
    r"正解",
    r"answer",
    r"output",
    r"結果",
)

_MARKER_RE = re.compile(
    r"(?:%s)\s*(?:は|:|：|＝|=|\*)*\s*" % "|".join(_ANSWER_MARKERS),
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")

_CHOICE_RE = re.compile(
    r"(?:^|[\s(（【*])\s*([A-Ea-e])\s*(?:[)）】.\s:：、]|\b)",
)

_LATIN_WORD_RE = re.compile(r"[A-Za-z]+")
_DIGIT_RUN_RE = re.compile(r"\d+(?:[.,]\d+)*")

_THINK_RE = re.compile(
    r"<\s*(?:think|thinking|reasoning)\s*>.*?<\s*/\s*(?:think|thinking|reasoning)\s*>",
    re.DOTALL | re.IGNORECASE,
)


def strip_reasoning(text: str) -> str:
    """本文に埋め込まれた思考ブロックタグ（think / thinking / reasoning）を除去する."""
    if not text:
        return ""
    cleaned = _THINK_RE.sub(" ", text)
    # クローズタグだけを忘れたケース: 最後の開きタグ以降を落とす
    if "<think" in cleaned.lower() and "</think" not in cleaned.lower():
        index = cleaned.lower().rfind("<think")
        cleaned = cleaned[:index]
    return re.sub(r"[ \t]+", " ", cleaned).strip()


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3040 <= code <= 0x30FF  # ひらがな・カタカナ
        or 0x3000 <= code <= 0x303F  # CJK記号
        or 0x3400 <= code <= 0x4DBF  # 拡張漢字A
        or 0x4E00 <= code <= 0x9FFF  # 漢字
        or 0xAC00 <= code <= 0xD7AF  # ハングル
        or 0xF900 <= code <= 0xFAFF  # 互換漢字
        or 0xFF00 <= code <= 0xFF60  # 全角記号
    )


def normalize_text(text: str) -> str:
    """NFKC正規化 + 空白圧縮 + マークダウン装飾の除去.

    全角数字→半角数字、全角英字→半角英字などに揃えるため、
    日本語モデルの出力と英字の期待値を公平に比較できる。
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = out.replace("**", "").replace("__", "").replace("`", "")
    out = re.sub(r"[*#>\-]{1,3}\s", " ", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def estimate_tokens(text: str) -> int:
    """トークン数の概算（サーバイ使用该なし時のフォールバック）.

    日本語文字は約1トークン/文字、欧文単語は約1.33トークン/単語、
    数値列は約1トークン/3桁として推定する。実トークナーで誤差 5-25%。
    """
    if not text:
        return 0
    normalized = unicodedata.normalize("NFKC", text)
    cjk = sum(1 for ch in normalized if _is_cjk(ch))
    latin_words = len(_LATIN_WORD_RE.findall(normalized))
    digits = sum(len(m.group(0)) for m in _DIGIT_RUN_RE.finditer(normalized))
    other = len(
        [
            ch
            for ch in normalized
            if not _is_cjk(ch) and not ch.isalnum() and not ch.isspace()
        ]
    )
    return max(1, int(round(cjk + latin_words * 1.33 + digits / 3.0 + other * 0.5)))


def extract_answer_span(text: str) -> str:
    """「答え:」「Final Answer:」などのマーカー以降、または最終行を取り出す."""
    if not text:
        return ""
    text = strip_reasoning(text)
    matches = list(_MARKER_RE.finditer(text))
    if matches:
        tail = text[matches[-1].end() :]
        tail = tail.strip().strip("*").strip()
        first_line = tail.splitlines()[0] if tail else ""
        if first_line:
            return first_line
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    # 末尾の短い行を最終回答とみなす
    for line in reversed(lines[-3:]):
        if len(line) <= 80:
            return line
    return lines[-1]


def extract_choice_marker_free(text: str) -> str | None:
    """思考ブロックを除去してから選択肢を抽出する."""
    return extract_choice(strip_reasoning(text))


def extract_choice(text: str) -> str | None:
    """選択肢 (A-E) を抽出する.

    見出しマーカー内の文字を優先し、無ければ本文中の最初の見つかった
    選択肢文字を返す。
    """
    span = extract_answer_span(text)
    for source in (span, text):
        normalized = normalize_text(source)
        if not normalized:
            continue
        # 単独の1文字
        stripped = normalized.strip(" .。)!！:：")
        if re.fullmatch(r"[A-E]", stripped):
            return stripped
        for match in _CHOICE_RE.finditer(normalized):
            letter = match.group(1).upper()
            if letter in "ABCDE":
                return letter
    return None


def extract_numbers(text: str) -> list[float]:
    """正規化済みテキストから数値をすべて抽出する（出現順）."""
    normalized = normalize_text(strip_reasoning(text))
    values: list[float] = []
    for match in _NUMBER_RE.finditer(normalized):
        raw = match.group(0).replace(",", "")
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def _balanced_json_candidates(text: str) -> list[str]:
    """波括弧のバランスからJSONらしき断片を拾う."""
    candidates: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for index, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = index
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(text[start : index + 1])
    return candidates


def extract_json_object(text: str) -> dict | None:
    """出力から最初のJSONオブジェクトを取り出す（コードフェンスは無視）."""
    if not text:
        return None
    import json

    cleaned = re.sub(r"```(?:json|python)?", "", strip_reasoning(text))
    for candidate in _balanced_json_candidates(cleaned):
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_code_block(text: str, language: str | None = None) -> str:
    """マークダウンのコードブロックを結合して返す.

    コードブロックが無い場合はテキスト全体を返す（むきだしのコードにも対応）。
    """
    if not text:
        return ""
    text = strip_reasoning(text)
    pattern = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```", re.DOTALL)
    blocks = [
        block
        for lang, block in pattern.findall(text)
        if language is None or not lang or lang.lower() == language.lower()
    ]
    if blocks:
        return "\n".join(blocks).strip()
    return text.strip()


def truncate(text: str, limit: int = 240) -> str:
    """ログ・レポート用にテキストを短縮する."""
    collapsed = re.sub(r"\s+", " ", text or "").strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"
