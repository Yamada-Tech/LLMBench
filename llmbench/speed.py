"""速度ベンチマーク.

プロファイル（プロンプト長 × 生成長）ごとに往復を繰り返し、
TTFT / 生成速度(tok/s) / E2E を測定する。
サーバーが返すトークン統計（OpenAI互換 usage / llama.cpp timings 拡張）を
優先して使い、無い場合のみ概算にフォールバックする。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .backends import Backend, BackendError
from .cancel import CancelToken
from .metrics import Summary, combine_summaries, summarize
from .textutils import estimate_tokens

__all__ = ["PROFILE_SHORT", "PROFILE_LONG", "PROFILE_HUGE", "RunRecord", "SpeedResult", "run_speed_bench"]

# プロファイル定義: (説明, 繰り返し数, max_tokens)
PROFILES = {
    "short": {"repeats": 5, "max_tokens": 64},
    "medium": {"repeats": 3, "max_tokens": 256},
    "long": {"repeats": 2, "max_tokens": 512},
}

_FILLER_JA = (
    "東京は日本の首都であり、人口が最も多い都市である。"
    "技術革新は社会を変える原動力となり、教育と文化の発展に寄与する。"
    "データ分析は意思決定を支援し、機械学習は未来の産業を支えている。"
)

_FILLER_EN = (
    "The quick brown fox jumps over the lazy dog while engineers "
    "measure inference latency and throughput across many local models. "
)

_QUESTION = (
    "上記の内容を踏まえて、ローカルLLMの運用で重要なポイントを3つ挙げなさい。"
)


def _long_prompt(target_tokens: int) -> str:
    """およそ target_tokens になるまでフィラーテキストを連結する."""
    parts: list[str] = []
    units = (_FILLER_JA + " " + _FILLER_EN).strip()
    total = 0
    while total < target_tokens:
        parts.append(units)
        total += estimate_tokens(units)
    return "\n".join(parts) + "\n\n" + _QUESTION


@dataclass
class RunRecord:
    """1往復の測定値."""

    profile: str
    repeat: int
    ttft_s: float | None
    total_s: float
    decode_s: float | None
    prompt_tokens: int | None
    completion_tokens: int
    tok_per_s: float
    ttft_tok_per_s: float | None  # プロンプト処理速度の近似

    def to_dict(self) -> dict:
        return {
            "profile": self.profile,
            "repeat": self.repeat,
            "ttft_s": _r(self.ttft_s),
            "total_s": _r(self.total_s),
            "decode_s": _r(self.decode_s),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tok_per_s": _r(self.tok_per_s),
            "prompt_tok_per_s": _r(self.ttft_tok_per_s),
        }


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


@dataclass
class SpeedResult:
    """速度ベンチ全体の集計."""

    backend_info: dict
    warmup_runs: int
    profiles: dict = field(default_factory=dict)  # profile -> Summary 辞書
    raw: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": "speed",
            "backend": self.backend_info,
            "warmup_runs": self.warmup_runs,
            "profiles": self.profiles,
            "errors": self.errors,
            "raw": self.raw,
        }


def _make_record(profile: str, repeat: int, result) -> RunRecord:
    stats = result.stats or {}
    prompt_eval_s = stats.get("prompt_eval_duration_s")
    if not prompt_eval_s:
        # llama.cpp 系サーバーの timings 拡張（ミリ秒）
        timings = stats.get("timings") or {}
        prompt_eval_s = (timings.get("prompt_ms") or 0) / 1000.0
    ttft_tok_per_s = None
    if prompt_eval_s and prompt_eval_s > 0 and result.prompt_tokens:
        ttft_tok_per_s = result.prompt_tokens / prompt_eval_s
    elif result.ttft_s and result.ttft_s > 0 and result.prompt_tokens:
        ttft_tok_per_s = result.prompt_tokens / result.ttft_s
    return RunRecord(
        profile=profile,
        repeat=repeat,
        ttft_s=result.ttft_s,
        total_s=result.total_s,
        decode_s=result.decode_s,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        tok_per_s=result.tok_per_s,
        ttft_tok_per_s=ttft_tok_per_s,
    )


def run_speed_bench(
    backend: Backend,
    profile_names: list[str] | None = None,
    repeats: int | None = None,
    max_tokens: int | None = None,
    warmup: int = 1,
    temperature: float = 0.0,
    seed: int | None = 42,
    context_lengths: list[int] | None = None,
    progress=None,
    cancel: CancelToken | None = None,
) -> SpeedResult:
    """速度ベンチを実行する.

    profile_names: short/medium/long のサブセット
    repeats/max_tokens: プロファイル既定値の上書き
    context_lengths: 追加の長文プロンプト長（トークン概算）
    cancel: 中断トークン（各リクエスト前にチェック）
    """
    def check_cancel() -> None:
        if cancel is not None:
            cancel.check()
    profile_names = profile_names or list(PROFILES)
    unknown = [p for p in profile_names if p not in PROFILES]
    if unknown:
        raise ValueError(f"未知のプロファイル: {unknown}")
    prompts: dict[str, list[tuple[str, int]]] = {}
    for name in profile_names:
        base_tokens = {"short": 30, "medium": 400, "long": 2000}[name]
        prompts[name] = [(_long_prompt(base_tokens), base_tokens)]
    if context_lengths:
        prompts["context"] = [
            (_long_prompt(tokens), tokens) for tokens in context_lengths
        ]

    result = SpeedResult(backend_info=backend.info(), warmup_runs=warmup)

    def log(msg: str) -> None:
        if progress:
            progress(msg)

    log(f"ウォームアップ {warmup} 回...")
    for _ in range(max(0, warmup)):
        check_cancel()
        try:
            backend.generate(
                [{"role": "user", "content": _long_prompt(50)}],
                max_tokens=16,
                temperature=0.0,
                seed=seed,
            )
        except BackendError as exc:
            result.errors.append(f"warmup: {exc}")

    for profile, items in prompts.items():
        cfg = PROFILES.get(profile, {"repeats": 1, "max_tokens": 512})
        n = repeats if repeats is not None else cfg["repeats"]
        mt = max_tokens if max_tokens is not None else cfg["max_tokens"]
        summaries = {
            "ttft_s": [],
            "tok_per_s": [],
            "total_s": [],
            "prompt_tok_per_s": [],
        }
        log(f"[{profile}] {n} 往復 (max_tokens={mt})")
        for prompt, approx_tokens in items:
            for i in range(n):
                check_cancel()
                messages = [{"role": "user", "content": prompt}]
                try:
                    res = backend.generate(
                        messages,
                        max_tokens=mt,
                        temperature=temperature,
                        seed=None if seed is None else seed + i,
                    )
                except BackendError as exc:
                    result.errors.append(f"{profile}#{i}: {exc}")
                    log(f"  ✗ {exc}")
                    continue
                if not res.text.strip():
                    warning = (res.stats or {}).get("warning", "")
                    result.errors.append(
                        f"{profile}#{i}: 応答が空です {warning}".strip()
                    )
                    log(f"  ✗ 応答が空（{warning or 'contentなし'}）")
                    continue
                record = _make_record(profile, i, res)
                result.raw.append(record.to_dict())
                if record.ttft_s is not None:
                    summaries["ttft_s"].append(record.ttft_s)
                summaries["tok_per_s"].append(record.tok_per_s)
                summaries["total_s"].append(record.total_s)
                if record.ttft_tok_per_s is not None:
                    summaries["prompt_tok_per_s"].append(record.ttft_tok_per_s)
                ttft_txt = f"{record.ttft_s:.3f}s" if record.ttft_s is not None else "N/A"
                log(
                    f"  #{i + 1}: TTFT={ttft_txt} {record.tok_per_s:.1f} tok/s "
                    f"(prompt~{record.prompt_tokens} tok)"
                )
        per_profile = {}
        for key, values in summaries.items():
            if values:
                per_profile[key] = summarize(values).to_dict()
        # 平均的な1往復あたりのコスト近似
        if summaries["tok_per_s"]:
            per_profile["aggregate_tok_per_s"] = round(
                sum(summaries["tok_per_s"]) / len(summaries["tok_per_s"]), 3
            )
        per_profile["prompt_approx_tokens"] = [t for _, t in items]
        result.profiles[profile] = per_profile
    return result


def summarize_speed_overall(speed: SpeedResult) -> Summary:
    """全プロファイルの生成速度 Summary（レポート要約用）."""
    all_tok = [r["tok_per_s"] for r in speed.raw if r.get("tok_per_s")]
    return summarize(all_tok)
