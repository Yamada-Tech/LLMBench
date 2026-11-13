"""JSON / Markdown レポートの生成と、複数ランの比較."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from . import __version__
from .textutils import truncate

__all__ = ["collect_run", "write_json", "write_markdown", "compare_runs", "compare_markdown"]


def collect_run(speed=None, accuracy=None, label: str = "") -> dict:
    """速度・精度の結果を1件のランとしてまとめる."""
    payload = {
        "meta": {
            "tool": "llmbench",
            "version": __version__,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "label": label,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.node(),
        },
        "speed": speed.to_dict() if speed is not None else None,
        "accuracy": accuracy.to_dict() if accuracy is not None else None,
    }
    return payload


def write_json(payload: dict, path: str | Path) -> Path:
    """結果をJSONファイルに保存する."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def _fmt_ms(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    return f"{seconds * 1000:.0f} ms"


def _fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}%"


def write_markdown(payload: dict, path: str | Path) -> Path:
    """結果をMarkdownレポートとして保存する."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(payload), encoding="utf-8")
    return path


def render_markdown(payload: dict) -> str:
    """辞書からMarkdownテキストを組み立てる."""
    meta = payload.get("meta", {})
    lines: list[str] = [
        f"# LLMBench レポート {meta.get('label', '')}".rstrip(),
        "",
        f"- 日時: {meta.get('timestamp')}",
        f"- 環境: {meta.get('platform')}",
        f"- Python: {meta.get('python')} / llmbench {meta.get('version')}",
        "",
    ]
    speed = payload.get("speed")
    accuracy = payload.get("accuracy")

    if speed:
        lines += ["## 速度ベンチ", ""]
        info = speed.get("backend", {})
        lines += [
            f"- バックエンド: {info.get('backend')} / モデル: {info.get('model')}",
            f"- 接続先: {info.get('base_url')}",
            "",
            "| プロファイル | TTFT p50 | tok/s 平均 | tok/s p90 | E2E 平均 |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, prof in speed.get("profiles", {}).items():
            ttft = prof.get("ttft_s", {})
            tps = prof.get("tok_per_s", {})
            total = prof.get("total_s", {})
            lines.append(
                f"| {name} | {_fmt_ms(ttft.get('p50'))} | "
                f"{tps.get('mean', 0):.1f} | {tps.get('p90', 0):.1f} | "
                f"{total.get('mean', 0):.1f}s |"
            )
        if speed.get("errors"):
            lines += ["", "エラー:", *[f"- {e}" for e in speed["errors"]]]
        lines.append("")

    if accuracy:
        lines += ["## 精度ベンチ", ""]
        overall = accuracy.get("overall", {})
        lines += [
            f"- 総合正答率: **{overall.get('accuracy_pct', 0):.1f}%** "
            f"({overall.get('correct', 0)}/{overall.get('total', 0)})",
            "",
            "### カテゴリ別",
            "",
            "| カテゴリ | 正答率 | 正解/総数 |",
            "|---|---:|---:|",
        ]
        for cat, g in sorted(accuracy.get("by_category", {}).items()):
            lines.append(
                f"| {cat} | {_fmt_pct(g.get('accuracy_pct'))} | "
                f"{g.get('correct')}/{g.get('total')} |"
            )
        lines += ["", "### 難易度別", "", "| 難易度 | 正答率 |", "|---|---:|"]
        for diff, g in sorted(accuracy.get("by_difficulty", {}).items()):
            lines.append(f"| {diff} | {_fmt_pct(g.get('accuracy_pct'))} |")
        wrong = [o for o in accuracy.get("outcomes", []) if not o.get("correct")]
        if wrong:
            lines += ["", "### 誤答一覧", "", "| ID | 抽出値 | 期待値 | 詳細 |", "|---|---|---|---|"]
            for o in wrong:
                lines.append(
                    f"| {o.get('id')} | {truncate(str(o.get('extracted')), 60)} | "
                    f"{truncate(str(o.get('expected')), 60)} | "
                    f"{truncate(str(o.get('detail')), 80)} |"
                )
        lines.append("")

    return "\n".join(lines)


def compare_runs(paths: list[str | Path]) -> dict:
    """保存済みJSONランを読み込み、比較用の集計辞書を作る.

    Windowsシェルはグロブを展開しないため、パスにワイルドカードが
    含まれる場合はこちらで展開する。
    """
    import glob as _glob

    expanded: list[str] = []
    for path in paths:
        text = str(path)
        if any(ch in text for ch in "*?["):
            matches = sorted(_glob.glob(text))
            if not matches:
                raise FileNotFoundError(f"一致する結果ファイルがありません: {text}")
            expanded.extend(matches)
        else:
            expanded.append(text)
    runs = []
    for path in expanded:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        runs.append((str(path), data))
    if not runs:
        raise FileNotFoundError("比較対象のランがありません")
    rows = []
    for path, data in runs:
        acc = (data.get("accuracy") or {}).get("overall", {})
        profiles = (data.get("speed") or {}).get("profiles", {})
        tps_all = [
            p.get("tok_per_s", {}).get("mean", 0.0) for p in profiles.values()
        ]
        ttft_all = [
            p.get("ttft_s", {}).get("p50") for p in profiles.values()
            if p.get("ttft_s")
        ]
        meta = data.get("meta", {})
        rows.append({
            "label": meta.get("label") or Path(path).stem,
            "backend": (
                (data.get("speed") or data.get("accuracy") or {}).get("backend", {})
            ).get("backend"),
            "model": (
                (data.get("speed") or data.get("accuracy") or {}).get("backend", {})
            ).get("model"),
            "accuracy_pct": acc.get("accuracy_pct"),
            "correct": acc.get("correct"),
            "total": acc.get("total"),
            "tok_per_s": round(sum(tps_all) / len(tps_all), 2) if tps_all else None,
            "ttft_p50_s": round(sum(ttft_all) / len(ttft_all), 4) if ttft_all else None,
            "path": str(path),
        })
    rows.sort(key=lambda r: (-(r["accuracy_pct"] or 0), -(r["tok_per_s"] or 0)))
    return {"runs": rows}


def compare_markdown(comparison: dict) -> str:
    """比較結果のMarkdownを生成する."""
    lines = [
        "# LLMBench モデル比較",
        "",
        "| モデル | バックエンド | 正答率 | tok/s 平均 | TTFT p50 |",
        "|---|---|---:|---:|---:|",
    ]
    for r in comparison["runs"]:
        lines.append(
            f"| {r['label']} | {r.get('backend') or '—'} | "
            f"{_fmt_pct(r.get('accuracy_pct'))} | "
            f"{r['tok_per_s'] if r.get('tok_per_s') is not None else '—'} | "
            f"{_fmt_ms(r.get('ttft_p50_s'))} |"
        )
    return "\n".join(lines)
