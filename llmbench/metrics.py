"""統計サマリの算出（標準ライブラリのみ）."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

__all__ = ["Summary", "summarize", "combine_summaries"]


@dataclass
class Summary:
    """一連の測定値の分布サマリ."""

    count: int
    mean: float
    median: float
    min: float
    max: float
    stdev: float
    p50: float
    p90: float
    p95: float

    def to_dict(self) -> dict:
        return {key: round(value, 6) for key, value in asdict(self).items()}


def percentile(values: Sequence[float], pct: float) -> float:
    """線形補間パーセンタイル（numpy 不使用）."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarize(values: Iterable[float]) -> Summary:
    """iterable から Summary を作る（空ならゼロ埋め）."""
    data = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not data:
        return Summary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return Summary(
        count=len(data),
        mean=statistics.fmean(data),
        median=statistics.median(data),
        min=min(data),
        max=max(data),
        stdev=statistics.stdev(data) if len(data) > 1 else 0.0,
        p50=percentile(data, 50),
        p90=percentile(data, 90),
        p95=percentile(data, 95),
    )


def combine_summaries(summaries: Sequence[Summary]) -> Summary:
    """複数の Summary（min/max/mean/count のみ保持）を統合する近似."""
    active = [s for s in summaries if s.count]
    if not active:
        return Summary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total = sum(s.count for s in active)
    mean = sum(s.mean * s.count for s in active) / total
    return Summary(
        count=total,
        mean=mean,
        median=summarize([s.median for s in active]).mean,
        min=min(s.min for s in active),
        max=max(s.max for s in active),
        stdev=max(s.stdev for s in active),
        p50=summarize([s.p50 for s in active]).mean,
        p90=summarize([s.p90 for s in active]).mean,
        p95=max(s.p95 for s in active),
    )
