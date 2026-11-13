"""LLMBench CLI.

使い方例:
  python -m llmbench gui            # GUIウィンドウ（アドレス・APIキーを入力して計測）
  python -m llmbench selftest
  python -m llmbench run --base-url http://127.0.0.1:8080/v1 --model my-model
  python -m llmbench run --base-url https://api.example.com/v1 --model gpt-x --insecure
  python -m llmbench speed --base-url http://192.168.1.189:11072 --model my-model --repeats 3
  python -m llmbench accuracy --base-url http://127.0.0.1:8080/v1 --model my-model \
      --difficulties easy,medium
  python -m llmbench compare results/run1.json results/run2.json
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from . import __version__
from .accuracy import run_accuracy_bench
from .backends import BackendError, get_backend
from .report import (
    collect_run,
    compare_markdown,
    compare_runs,
    write_json,
    write_markdown,
)
from .speed import run_speed_bench
from .tasks import CATEGORIES, load_tasks

__all__ = ["main"]

_DIFFICULTIES = ("easy", "medium", "hard")


def _print(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg, flush=True)


def _csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmbench",
        description="ローカルAIモデルの精度と速度を測定するベンチマークツール",
    )
    parser.add_argument("--version", action="version", version=f"llmbench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", "-b", default="openai",
                        help="バックエンド: openai (OpenAI互換chat/completions) | mock (既定: openai)")
    common.add_argument("--model", "-m", default="",
                        help="サーバー側のモデル名 (サーバーが受け付ける名前)")
    common.add_argument("--base-url", default="",
                        help="APIベースURL (例 http://127.0.0.1:8080/v1 / https://host:port)。"
                             "/v1 や /v1/chat/completions まで指定してもOK (既定: http://127.0.0.1:8080)")
    common.add_argument("--timeout", type=float, default=120.0, help="リクエスト秒 (既定 120)")
    common.add_argument("--api-key", default="",
                        help="APIキー (Authorization: Bearer ...) 任意")
    common.add_argument("--temperature", type=float, default=0.0, help="サンプリング温度 (既定 0)")
    common.add_argument("--seed", type=int, default=42, help="乱数seed (空指定不可・既定 42)")
    common.add_argument("--max-tokens", type=int, default=512, help="生成の最大トークン (既定 512)")
    common.add_argument("--insecure", action="store_true",
                        help="HTTPSの証明書検証をスキップ (自己署名証明書用)")
    common.add_argument("--no-thinking", action="store_true",
                        help="ハイブリッド思考モデルのthinkingを無効化 (Qwen3等の空回答対策)")
    common.add_argument("--out", "-o", default="results", help="出力ディレクトリ (既定 results)")
    common.add_argument("--label", default="", help="ランのラベル (レポート名)")
    common.add_argument("--quiet", "-q", action="store_true", help="進捗表示を抑制")

    p_run = sub.add_parser("run", parents=[common], help="速度と精度の両方を測定")
    p_run.add_argument("--tasks", help="問題集JSONパス (既定: 内蔵 tasks.json)")
    p_run.add_argument("--categories", help="実行カテゴリの絞り込み (csv): " + ",".join(CATEGORIES))
    p_run.add_argument("--difficulties", help="難易度絞り込み (csv): easy,medium,hard")
    p_run.add_argument("--repeats", type=int, default=None, help="速度ベンチの各プロファイル反復数")
    p_run.add_argument("--profiles", default="short,medium", help="速度プロファイル (csv): short,medium,long")
    p_run.add_argument("--accuracy-repeat", type=int, default=1, help="精度ベンチの1問あたり反復")
    p_run.add_argument("--warmup", type=int, default=1, help="ウォームアップ回数 (既定 1)")
    p_run.add_argument("--contexts", help="追加の長文コンテキスト長 (csvトークン数): 例 1024,4096")
    p_run.add_argument("--skip-speed", action="store_true", help="速度ベンチをスキップ")
    p_run.add_argument("--skip-accuracy", action="store_true", help="精度ベンチをスキップ")

    p_speed = sub.add_parser("speed", parents=[common], help="速度のみ測定")
    p_speed.add_argument("--repeats", type=int, default=None, help="各プロファイル反復数")
    p_speed.add_argument("--profiles", default="short,medium", help="速度プロファイル (csv)")
    p_speed.add_argument("--warmup", type=int, default=1, help="ウォームアップ回数")
    p_speed.add_argument("--contexts", help="追加の長文コンテキスト長 (csv)")

    p_acc = sub.add_parser("accuracy", parents=[common], help="精度のみ測定")
    p_acc.add_argument("--tasks", help="問題集JSONパス")
    p_acc.add_argument("--categories", help="実行カテゴリ絞り込み (csv)")
    p_acc.add_argument("--difficulties", help="難易度絞り込み (csv)")
    p_acc.add_argument("--repeat", type=int, default=1, help="1問あたり反復")

    p_cmp = sub.add_parser("compare", help="保存済みJSONランを比較")
    p_cmp.add_argument("paths", nargs="+", help="結果JSONファイルのパス")
    p_cmp.add_argument("--out", "-o", default="results", help="比較Markdownの出力先")

    sub.add_parser("selftest", help="Mockバックエンドで全機能のセルフテスト")
    sub.add_parser("gui", help="GUIウィンドウを開いて対話的に計測（tkinter）")
    return parser


def _make_backend(args) -> "object":
    options = {}
    if getattr(args, "api_key", ""):
        options["api_key"] = args.api_key
    if getattr(args, "insecure", False):
        options["insecure"] = True
    if getattr(args, "no_thinking", False):
        options["no_thinking"] = True
    return get_backend(
        args.backend,
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        **options,
    )


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _slug(value: str) -> str:
    keep = [c if c.isalnum() or c in "-_" else "-" for c in (value or "run")]
    return "".join(keep).strip("-") or "run"


def cmd_run(args) -> int:
    backend = _make_backend(args)
    label = args.label or f"{args.backend}-{args.model or 'model'}"
    progress = (lambda m: _print(m, args.quiet))
    speed = None
    accuracy = None

    if not args.skip_speed:
        _print("=== 速度ベンチ ===", args.quiet)
        speed = run_speed_bench(
            backend,
            profile_names=_csv(args.profiles) or ["short", "medium"],
            repeats=args.repeats,
            max_tokens=None,
            warmup=args.warmup,
            temperature=args.temperature,
            seed=args.seed,
            context_lengths=[int(x) for x in _csv(args.contexts)] if args.contexts else None,
            progress=progress,
        )

    if not args.skip_accuracy:
        _print("=== 精度ベンチ ===", args.quiet)
        task_set = load_tasks(args.tasks).filter(
            _csv(args.categories), _csv(args.difficulties)
        )
        if not task_set.tasks:
            print("条件に一致する問題がありません", file=sys.stderr)
            return 2
        accuracy = run_accuracy_bench(
            backend,
            task_set,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
            repeat=args.accuracy_repeat,
            progress=progress,
        )

    payload = collect_run(speed, accuracy, label=label)
    out_dir = Path(args.out)
    stem = f"{_slug(label)}-{_stamp()}"
    json_path = write_json(payload, out_dir / f"{stem}.json")
    md_path = write_markdown(payload, out_dir / f"{stem}.md")
    _print(f"保存: {json_path}", args.quiet)
    _print(f"保存: {md_path}", args.quiet)
    if accuracy:
        overall = accuracy.overall
        _print(
            f"精度: {overall['correct']}/{overall['total']} "
            f"({overall['accuracy_pct']:.1f}%)"
        )
    if speed and speed.profiles:
        for name, prof in speed.profiles.items():
            tps = prof.get("tok_per_s", {}).get("mean", 0)
            ttft = prof.get("ttft_s", {}).get("p50")
            ttft_txt = f"{ttft * 1000:.0f}ms" if ttft is not None else "N/A"
            _print(f"速度[{name}]: {tps:.1f} tok/s / TTFT p50 {ttft_txt}")
    hard_errors = (speed.errors if speed else []) + (accuracy.errors if accuracy else [])
    if accuracy and accuracy.tasks_run and len(hard_errors) >= accuracy.tasks_run:
        print("すべてのリクエストが失敗しました。サーバー起動・モデル名を確認してください",
              file=sys.stderr)
        return 3
    return 0


def cmd_speed(args) -> int:
    backend = _make_backend(args)
    progress = (lambda m: _print(m, args.quiet))
    speed = run_speed_bench(
        backend,
        profile_names=_csv(args.profiles) or ["short", "medium"],
        repeats=args.repeats,
        warmup=args.warmup,
        temperature=args.temperature,
        seed=args.seed,
        context_lengths=[int(x) for x in _csv(args.contexts)] if args.contexts else None,
        progress=progress,
    )
    payload = collect_run(speed, None, label=args.label or f"speed-{args.model or args.backend}")
    stem = f"{_slug(payload['meta']['label'])}-{_stamp()}"
    json_path = write_json(payload, Path(args.out) / f"{stem}.json")
    md_path = write_markdown(payload, Path(args.out) / f"{stem}.md")
    _print(f"保存: {json_path}", args.quiet)
    _print(f"保存: {md_path}", args.quiet)
    if not speed.raw:
        print("測定に失敗しました", file=sys.stderr)
        return 3
    return 0


def cmd_accuracy(args) -> int:
    backend = _make_backend(args)
    task_set = load_tasks(args.tasks).filter(
        _csv(args.categories), _csv(args.difficulties)
    )
    if not task_set.tasks:
        print("条件に一致する問題がありません", file=sys.stderr)
        return 2
    accuracy = run_accuracy_bench(
        backend,
        task_set,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        repeat=args.repeat,
        progress=(lambda m: _print(m, args.quiet)),
    )
    payload = collect_run(None, accuracy, label=args.label or f"accuracy-{args.model or args.backend}")
    stem = f"{_slug(payload['meta']['label'])}-{_stamp()}"
    json_path = write_json(payload, Path(args.out) / f"{stem}.json")
    md_path = write_markdown(payload, Path(args.out) / f"{stem}.md")
    _print(f"保存: {json_path}", args.quiet)
    _print(f"保存: {md_path}", args.quiet)
    overall = accuracy.overall
    print(f"精度: {overall['correct']}/{overall['total']} ({overall['accuracy_pct']:.1f}%)")
    if accuracy.errors and accuracy.tasks_run and len(accuracy.errors) >= accuracy.tasks_run:
        return 3
    return 0


def cmd_compare(args) -> int:
    comparison = compare_runs(args.paths)
    md = compare_markdown(comparison)
    print(md)
    out = Path(args.out) / f"compare-{_stamp()}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md + "\n", encoding="utf-8")
    print(f"保存: {out}")
    return 0


def cmd_selftest(args) -> int:
    """Mockバックエンドでパイプライン全体を検証する."""
    print("llmbench セルフテスト (Mockバックエンド)")
    backend = get_backend("mock", model="mock")
    task_set = load_tasks().filter(["math", "logic", "knowledge", "code", "extraction"])
    accuracy = run_accuracy_bench(
        backend, task_set, max_tokens=64, seed=1,
        progress=lambda m: None,
    )
    overall = accuracy.overall
    print(f"  精度パイプライン: {overall['correct']}/{overall['total']} 採点完了")
    speed = run_speed_bench(
        backend, profile_names=["short"], repeats=1, warmup=0,
        progress=lambda m: None,
    )
    tps = speed.profiles.get("short", {}).get("tok_per_s", {}).get("mean", 0)
    print(f"  速度パイプライン: {tps:.0f} tok/s (mock)")
    payload = collect_run(speed, accuracy, label="selftest")
    out_dir = Path("results")
    json_path = write_json(payload, out_dir / f"selftest-{_stamp()}.json")
    md_path = write_markdown(payload, out_dir / f"selftest-{_stamp()}.md")
    print(f"  レポート生成: {json_path.name}, {md_path.name}")
    if overall["total"] == 0 or not speed.raw:
        print("セルフテスト失敗", file=sys.stderr)
        return 1
    print("セルフテスト OK")
    return 0


def _fix_console_encoding() -> None:
    """Windowsコンソール(cp932)でも日本語が文字化けしないよう再設定する."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def cmd_gui(args) -> int:
    """GUI を起動する（tkinter が無い環境では明確なエラー）."""
    try:
        from .gui import run_gui
    except ImportError as exc:  # pragma: no cover - tkinter 非搭載環境
        print(f"GUIを起動できません: {exc}", file=sys.stderr)
        print("tkinter を含む Python で実行してください（例: python -m llmbench run ...）",
              file=sys.stderr)
        return 2
    return run_gui()


def main(argv: list[str] | None = None) -> int:
    _fix_console_encoding()
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "run": cmd_run,
        "speed": cmd_speed,
        "accuracy": cmd_accuracy,
        "compare": cmd_compare,
        "selftest": cmd_selftest,
        "gui": cmd_gui,
    }
    handler = handlers[args.command]
    try:
        return handler(args)
    except BackendError as exc:
        print(f"バックエンドエラー: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\n中断しました", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
