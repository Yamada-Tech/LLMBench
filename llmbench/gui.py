"""LLMBench GUI（tkinter・依存ゼロ）.

測る対象の API アドレス・モデル名・APIキーを入力して
「▶ Run Benchmark」を押すと、速度ベンチと精度ベンチを実行して結果を表表示する。

    python -m llmbench gui

設計:
- 重い実行はワーカースレッド。UI は queue を polling して反映する（tkinter は
  メインスレッド以外から操作しない）。
- 設定は ~/.llmbench/gui_settings.json に保存。APIキーの保存はチェックした時だけ。
"""

from __future__ import annotations

import json
import queue
import re
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from datetime import datetime

from . import __version__
from .accuracy import run_accuracy_bench
from .backends import BackendError, get_backend
from .cancel import BenchmarkCancelled, CancelToken
from .report import (
    collect_run,
    compare_markdown,
    compare_runs,
    render_markdown,
    write_json,
    write_markdown,
)
from .speed import PROFILES, run_speed_bench
from .tasks import CATEGORIES, load_tasks

__all__ = ["run_gui", "SettingsStore"]

_PROGRESS_RE = re.compile(r"^\[(\d+)/(\d+)\]")
_SETTINGS_PATH = Path.home() / ".llmbench" / "gui_settings.json"

_DEFAULT_SETTINGS = {
    "base_url": "http://127.0.0.1:8080/v1",
    "model": "",
    "api_key": "",
    "save_api_key": False,
    "out_dir": "results",
    "label": "",
    "profiles": ["short", "medium"],
    "repeats": "",
    "warmup": "1",
    "contexts": "",
    "categories": [],
    "difficulties": [],
    "accuracy_repeat": "1",
    "max_tokens": "512",
    "temperature": "0",
    "seed": "42",
    "timeout": "120",
    "insecure": False,
    "no_thinking": False,
    "run_speed": True,
    "run_accuracy": True,
}


class SettingsStore:
    """GUI 設定の永続化（APIキーは保存を許可した場合のみ書く）."""

    def __init__(self, path: Path = _SETTINGS_PATH) -> None:
        self.path = Path(path)

    def load(self) -> dict:
        data = dict(_DEFAULT_SETTINGS)
        try:
            stored = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return data
        for key, value in stored.items():
            if key in data:
                data[key] = value
        return data

    def save(self, values: dict) -> Path | None:
        data = {k: values.get(k) for k in _DEFAULT_SETTINGS if k in values}
        if not data.get("save_api_key"):
            data["api_key"] = ""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            return None
        return self.path


class BenchApp(tk.Tk):
    """メインウィンドウ."""

    def __init__(self, store: SettingsStore | None = None) -> None:
        super().__init__()
        self.title(f"LLMBench {__version__} - LLM 精度・速度ベンチマーク")
        self.geometry("1000x720")
        self.minsize(880, 560)
        self.store = store or SettingsStore()
        self.settings = self.store.load()

        self.queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_token: CancelToken | None = None
        self.payload: dict | None = None
        self.saved_paths: list[Path] = []

        #: (StringVar, Entry, ヒント) の一覧。tkinter の変数はハッシュ化できない
        #: ため dict では保持しない。空欄にはヒントを再表示し、_text() で
        #: ヒント文字列を「未入力」として扱う。
        self._hinted: list[tuple[tk.StringVar, ttk.Entry, str]] = []
        self._build_ui()
        self._load_settings_into_ui()
        self.after(120, self._drain_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # --- UI 構築 -----------------------------------------------------------

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        paned = ttk.Panedwindow(outer, orient="vertical")
        paned.pack(fill="both", expand=True)

        top = ttk.Frame(paned)
        paned.add(top, weight=2)

        # --- 接続設定 ------------------------------------------------------
        conn = ttk.LabelFrame(top, text="接続設定（計測対象の OpenAI互換API）", padding=8)
        conn.pack(fill="x")
        self.var_base = tk.StringVar()
        self.var_model = tk.StringVar()
        self.var_key = tk.StringVar()
        self.var_timeout = tk.StringVar()
        self.var_save_key = tk.BooleanVar()

        grid = ttk.Frame(conn)
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(3, weight=1)
        self._entry(grid, 0, 0, "ベースURL", self.var_base,
                    "例: http://127.0.0.1:8080/v1  /  https://host:11072")
        self._entry(grid, 0, 2, "モデル名", self.var_model, "サーバーが受け付ける名前")
        self._entry(grid, 1, 0, "APIキー", self.var_key, "任意（空なら Authorization なし）",
                    show="•")
        self._entry(grid, 1, 2, "タイムアウト秒", self.var_timeout, "120", width=10)
        ttk.Checkbutton(grid, text="APIキーをこのPCに保存する",
                        variable=self.var_save_key).grid(row=2, column=0, sticky="w",
                                                         columnspan=2, padx=(0, 12))
        self.var_insecure = tk.BooleanVar()
        self.var_no_thinking = tk.BooleanVar()
        ttk.Checkbutton(grid, text="証明書検証をスキップ（--insecure）",
                        variable=self.var_insecure).grid(row=2, column=2, sticky="w",
                                                        columnspan=2)

        # --- 測定設定 ------------------------------------------------------
        bench = ttk.LabelFrame(top, text="測定設定", padding=8)
        bench.pack(fill="x", pady=(8, 0))
        bgrid = ttk.Frame(bench)
        bgrid.pack(fill="x")
        bgrid.columnconfigure(1, weight=1)
        bgrid.columnconfigure(3, weight=1)
        bgrid.columnconfigure(5, weight=1)

        self.var_max_tokens = tk.StringVar()
        self.var_temperature = tk.StringVar()
        self.var_seed = tk.StringVar()
        self.var_repeats = tk.StringVar()
        self.var_warmup = tk.StringVar()
        self.var_acc_repeat = tk.StringVar()
        self._entry(bgrid, 0, 0, "max_tokens", self.var_max_tokens, "512", width=12)
        self._entry(bgrid, 0, 2, "temperature", self.var_temperature, "0", width=12)
        self._entry(bgrid, 0, 4, "seed", self.var_seed, "42", width=12)
        self._entry(bgrid, 1, 0, "速度の反復", self.var_repeats, "空=既定", width=12)
        self._entry(bgrid, 1, 2, "ウォームアップ", self.var_warmup, "1", width=12)
        self._entry(bgrid, 1, 4, "精度の1問反復", self.var_acc_repeat, "1", width=12)

        opts = ttk.Frame(bench)
        opts.pack(fill="x", pady=(6, 0))
        self.var_run_speed = tk.BooleanVar()
        self.var_run_accuracy = tk.BooleanVar()
        ttk.Checkbutton(opts, text="速度ベンチ", variable=self.var_run_speed).pack(
            side="left", padx=(0, 12))
        ttk.Checkbutton(opts, text="精度ベンチ", variable=self.var_run_accuracy).pack(
            side="left", padx=(0, 20))

        self.profile_vars: dict[str, tk.BooleanVar] = {}
        ttk.Label(opts, text="速度プロファイル:").pack(side="left")
        for name in PROFILES:
            var = tk.BooleanVar()
            self.profile_vars[name] = var
            ttk.Checkbutton(opts, text=name, variable=var).pack(side="left", padx=2)

        self.category_vars: dict[str, tk.BooleanVar] = {}
        cat = ttk.Frame(bench)
        cat.pack(fill="x", pady=(6, 0))
        ttk.Label(cat, text="精度カテゴリ（未選択=全カテゴリ）:").pack(side="left")
        for name in CATEGORIES:
            var = tk.BooleanVar()
            self.category_vars[name] = var
            ttk.Checkbutton(cat, text=name, variable=var).pack(side="left", padx=2)

        self.difficulty_vars: dict[str, tk.BooleanVar] = {}
        diff = ttk.Frame(bench)
        diff.pack(fill="x", pady=(4, 0))
        ttk.Label(diff, text="難易度（未選択=すべて）:").pack(side="left")
        for name in ("easy", "medium", "hard"):
            var = tk.BooleanVar()
            self.difficulty_vars[name] = var
            ttk.Checkbutton(diff, text=name, variable=var).pack(side="left", padx=2)

        misc = ttk.Frame(bench)
        misc.pack(fill="x", pady=(6, 0))
        self.var_out = tk.StringVar()
        self.var_label = tk.StringVar()
        self.var_contexts = tk.StringVar()
        self._entry(misc, 0, 0, "出力ディレクトリ", self.var_out, "results", width=22)
        self._entry(misc, 0, 2, "ラベル", self.var_label, "GUI", width=18)
        self._entry(misc, 0, 4, "追加コンテキスト長", self.var_contexts,
                    "例: 1024,4096", width=18)

        # --- ボタン --------------------------------------------------------
        actions = ttk.Frame(top)
        actions.pack(fill="x", pady=(10, 0))
        self.btn_run = ttk.Button(actions, text="▶ Run Benchmark", command=self.on_run)
        self.btn_run.pack(side="left")
        self.btn_stop = ttk.Button(actions, text="■ 中止", command=self.on_stop,
                                   state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        ttk.Button(actions, text="接続テスト", command=self.on_ping).pack(
            side="left", padx=6)
        ttk.Button(actions, text="設定保存", command=self.on_save_settings).pack(
            side="left", padx=6)
        self.btn_compare = ttk.Button(actions, text="保存済み結果の比較表",
                                      command=self.on_compare)
        self.btn_compare.pack(side="left", padx=6)
        self.progress = ttk.Progressbar(actions, mode="determinate", length=200)
        self.progress.pack(side="right", fill="x", expand=True, padx=(12, 0))
        self.status = ttk.Label(actions, text=" Ready")
        self.status.pack(side="right", padx=8)

        # --- 結果 ----------------------------------------------------------
        bottom = ttk.Frame(paned)
        paned.add(bottom, weight=3)
        self.notebook = ttk.Notebook(bottom)
        self.notebook.pack(fill="both", expand=True)

        summary = ttk.Frame(self.notebook)
        self.notebook.add(summary, text="結果")
        self.headline = ttk.Label(summary, text="", font=("", 13, "bold"))
        self.headline.pack(fill="x", padx=6, pady=(6, 2))

        tables = ttk.Frame(summary)
        tables.pack(fill="both", expand=True)
        left = ttk.LabelFrame(tables, text="速度ベンチ", padding=4)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.speed_tree = self._make_tree(
            left, ("profile", "ttft", "toks", "toks_p90", "e2e", "prompt_tps"))
        right = ttk.LabelFrame(tables, text="精度ベンチ", padding=4)
        right.pack(side="left", fill="both", expand=True)
        self.acc_tree = self._make_tree(
            right, ("name", "correct", "total", "pct"))

        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="ログ")
        self.log_text = tk.Text(log_frame, height=18, wrap="none",
                                font=("Consolas", 9), state="disabled")
        log_scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        report_frame = ttk.Frame(self.notebook)
        self.notebook.add(report_frame, text="レポート (Markdown)")
        self.report_text = tk.Text(report_frame, height=18, wrap="none",
                                   font=("Consolas", 9), state="disabled")
        rep_scroll = ttk.Scrollbar(report_frame, command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=rep_scroll.set)
        self.report_text.pack(side="left", fill="both", expand=True)
        rep_scroll.pack(side="right", fill="y")

        self._set_status("Ready")

    def _entry(self, parent, row: int, col: int, label: str, var: tk.StringVar,
               placeholder: str, width: int | None = None, show: str = "") -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, sticky="w", padx=(0, 12), pady=2)
        ttk.Label(frame, text=label).pack(side="left")
        entry = ttk.Entry(frame, textvariable=var, width=width or 34)
        if show:
            entry.configure(show=show)
        entry.pack(side="left", padx=(6, 0))
        if placeholder:
            self._hinted.append((var, entry, placeholder))
            entry.bind(
                "<FocusIn>",
                lambda e, en=entry, ph=placeholder: self._ph_clear(en, ph),
            )
            entry.bind(
                "<FocusOut>",
                lambda e, en=entry, ph=placeholder: self._ph_restore(en, ph),
            )
            self._ph_restore(entry, placeholder)

    @staticmethod
    def _ph_clear(entry: ttk.Entry, placeholder: str) -> None:
        if entry.get() == placeholder:
            entry.delete(0, "end")

    @staticmethod
    def _ph_restore(entry: ttk.Entry, placeholder: str) -> None:
        if not entry.get():
            entry.insert(0, placeholder)

    def _text(self, var: tk.StringVar) -> str:
        """ヒント文字列は「未入力」として空文字扱いにする."""
        value = var.get().strip()
        for hint_var, _entry, hint in self._hinted:
            if hint_var is var and value == hint:
                return ""
        return value

    @staticmethod
    def _make_tree(parent, columns) -> ttk.Treeview:
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=8)
        headers = {
            "profile": "プロファイル", "ttft": "TTFT p50", "toks": "tok/s 平均",
            "toks_p90": "tok/s p90", "e2e": "E2E 平均", "prompt_tps": "prompt tok/s",
            "name": "カテゴリ / 総合", "correct": "正解", "total": "実行", "pct": "正答率",
        }
        for col in columns:
            tree.heading(col, text=headers[col])
            tree.column(col, width=110, anchor="e" if col != "profile" and col != "name" else "w")
        scroll = ttk.Scrollbar(parent, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        return tree

    # --- 設定の読み書き ----------------------------------------------------

    def _current_values(self) -> dict:
        return {
            "base_url": self._text(self.var_base),
            "model": self._text(self.var_model),
            "api_key": self._text(self.var_key),
            "save_api_key": bool(self.var_save_key.get()),
            "out_dir": self._text(self.var_out) or "results",
            "label": self._text(self.var_label),
            "profiles": [k for k, v in self.profile_vars.items() if v.get()],
            "repeats": self._text(self.var_repeats),
            "warmup": self._text(self.var_warmup),
            "contexts": self._text(self.var_contexts),
            "categories": [k for k, v in self.category_vars.items() if v.get()],
            "difficulties": [k for k, v in self.difficulty_vars.items() if v.get()],
            "accuracy_repeat": self._text(self.var_acc_repeat),
            "max_tokens": self._text(self.var_max_tokens),
            "temperature": self._text(self.var_temperature),
            "seed": self._text(self.var_seed),
            "timeout": self._text(self.var_timeout),
            "insecure": bool(self.var_insecure.get()),
            "no_thinking": bool(self.var_no_thinking.get()),
            "run_speed": bool(self.var_run_speed.get()),
            "run_accuracy": bool(self.var_run_accuracy.get()),
        }

    def _load_settings_into_ui(self) -> None:
        s = self.settings
        self.var_base.set(s["base_url"])
        self.var_model.set(s["model"])
        self.var_key.set(s.get("api_key", ""))
        self.var_save_key.set(bool(s.get("save_api_key")))
        self.var_timeout.set(str(s.get("timeout", "120")))
        self.var_max_tokens.set(str(s.get("max_tokens", "512")))
        self.var_temperature.set(str(s.get("temperature", "0")))
        self.var_seed.set(str(s.get("seed", "42")))
        self.var_repeats.set(str(s.get("repeats", "")))
        self.var_warmup.set(str(s.get("warmup", "1")))
        self.var_acc_repeat.set(str(s.get("accuracy_repeat", "1")))
        self.var_contexts.set(str(s.get("contexts", "")))
        self.var_out.set(str(s.get("out_dir", "results")))
        self.var_label.set(str(s.get("label", "")))
        self.var_insecure.set(bool(s.get("insecure")))
        self.var_no_thinking.set(bool(s.get("no_thinking")))
        self.var_run_speed.set(bool(s.get("run_speed", True)))
        self.var_run_accuracy.set(bool(s.get("run_accuracy", True)))
        for name, var in self.profile_vars.items():
            var.set(name in (s.get("profiles") or ["short", "medium"]))
        for name, var in self.category_vars.items():
            var.set(name in (s.get("categories") or []))
        for name, var in self.difficulty_vars.items():
            var.set(name in (s.get("difficulties") or []))
        # 空欄にはヒントを再表示する（_text() がヒントを「未入力」として扱う）
        for var, entry, hint in self._hinted:
            if not var.get():
                self._ph_restore(entry, hint)

    def on_save_settings(self) -> None:
        path = self.store.save(self._current_values())
        if path is None:
            self._set_status("設定保存に失敗しました")
        else:
            self._set_status(f"設定保存: {path}")

    # --- 実行 ------------------------------------------------------------

    def on_run(self) -> None:
        values = self._current_values()
        error = self._validate(values)
        if error:
            messagebox.showerror("入力エラー", error)
            return
        self.store.save(values)
        self.payload = None
        self.saved_paths = []
        self._clear_results()
        self._log(f"=== {datetime.now():%Y-%m-%d %H:%M:%S} 実行開始 ===")
        self._log(f"対象: {values['base_url']} / model={values['model'] or '(default)'}")
        self.cancel_token = CancelToken()
        self._running(True)
        self.worker = threading.Thread(
            target=self._worker_run, args=(values,), daemon=True
        )
        self.worker.start()

    def on_stop(self) -> None:
        if self.cancel_token is not None:
            self.cancel_token.cancel()
            self._set_status("中断中...")

    def on_ping(self) -> None:
        values = self._current_values()
        error = self._validate(values)
        if error:
            messagebox.showerror("入力エラー", error)
            return
        self.cancel_token = CancelToken()
        self._running(True, "接続テスト中...")
        self.worker = threading.Thread(target=self._worker_ping, args=(values,),
                                       daemon=True)
        self.worker.start()

    def on_compare(self) -> None:
        paths = filedialog.askopenfilenames(
            title="比較する結果JSONを選択",
            initialdir=self._text(self.var_out) or "results",
            filetypes=[("JSON", "*.json")],
        )
        if not paths:
            return
        try:
            comparison = compare_runs(list(paths))
            markdown = compare_markdown(comparison)
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("比較エラー", str(exc))
            return
        self._show_markdown(markdown)
        self.notebook.select(2)
        self._set_status(f"{len(paths)} ランを比較")

    @staticmethod
    def _validate(values: dict) -> str:
        base = values["base_url"]
        if not base:
            return "ベースURLを入力してください"
        if not base.lower().startswith(("http://", "https://")):
            return "ベースURLは http:// または https:// で始めてください"
        for name in ("max_tokens", "seed", "timeout", "warmup", "accuracy_repeat",
                     "repeats"):
            raw = values.get(name, "")
            if raw and not raw.lstrip("-").isdigit():
                return f"{name} は整数で入力してください（現在: {raw}）"
        if values["temperature"]:
            try:
                float(values["temperature"])
            except ValueError:
                return "temperature は数値で入力してください"
        if not values["run_speed"] and not values["run_accuracy"]:
            return "速度ベンチか精度ベンチのどちらかを選んでください"
        if values["run_speed"] and not values["profiles"]:
            return "速度ベンチの実行にはプロファイルを1つ以上選んでください"
        return ""

    @staticmethod
    def _seed_of(values: dict) -> int | None:
        """seed 未入力なら既定の 42（再現性を保つ）."""
        return int(values["seed"]) if values["seed"] else 42

    def _build_backend(self, values: dict):
        options = {}
        if values["api_key"]:
            options["api_key"] = values["api_key"]
        if values["insecure"]:
            options["insecure"] = True
        if values["no_thinking"]:
            options["no_thinking"] = True
        return get_backend(
            "openai",
            model=values["model"] or "default",
            base_url=values["base_url"],
            timeout=float(values["timeout"] or 120),
            **options,
        )

    # --- ワーカースレッド --------------------------------------------------

    def _worker_ping(self, values: dict) -> None:
        try:
            backend = self._build_backend(values)
            result = backend.generate(
                [{"role": "user", "content": "1+1は？ 数字だけで答えてください"}],
                max_tokens=32,
                temperature=0.0,
                seed=42,
            )
        except BackendError as exc:
            self.queue.put(("status", "接続テスト失敗"))
            self.queue.put(("error", str(exc)))
            return
        self.queue.put(("log", f"接続OK: {result.text[:120]!r} "
                               f"({result.total_s:.2f}s, {result.tok_per_s:.1f} tok/s)"))
        self.queue.put(("status", "接続テスト OK"))

    def _worker_run(self, values: dict) -> None:
        cancel = self.cancel_token
        try:
            backend = self._build_backend(values)
            speed = None
            accuracy = None
            if values["run_speed"]:
                self.queue.put(("phase", "speed"))
                speed = run_speed_bench(
                    backend,
                    profile_names=values["profiles"],
                    repeats=int(values["repeats"]) if values["repeats"] else None,
                    warmup=int(values["warmup"] or 1),
                    temperature=float(values["temperature"] or 0),
                    seed=self._seed_of(values),
                    context_lengths=_csv_ints(values["contexts"]),
                    progress=lambda msg: self.queue.put(("log", msg)),
                    cancel=cancel,
                )
            if values["run_accuracy"]:
                self.queue.put(("phase", "accuracy"))
                task_set = load_tasks().filter(
                    values["categories"] or None, values["difficulties"] or None
                )
                accuracy = run_accuracy_bench(
                    backend,
                    task_set,
                    max_tokens=int(values["max_tokens"] or 512),
                    temperature=float(values["temperature"] or 0),
                    seed=self._seed_of(values),
                    repeat=int(values["accuracy_repeat"] or 1),
                    progress=lambda msg: self.queue.put(("log", msg)),
                    cancel=cancel,
                )
        except BenchmarkCancelled as exc:
            self.queue.put(("cancelled", str(exc)))
            return
        except (BackendError, ValueError) as exc:
            self.queue.put(("error", str(exc)))
            return
        self.queue.put(("done", (speed, accuracy, values)))

    # --- UI 反映（メインスレッド） ----------------------------------------

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, data = self.queue.get_nowait()
                if kind == "log":
                    self._log(data)
                    self._maybe_progress(data)
                elif kind == "phase":
                    self._set_status("速度ベンチ実行中..." if data == "speed"
                                     else "精度ベンチ実行中...")
                    self.progress.configure(
                        mode="indeterminate" if data == "speed" else "determinate",
                        value=0,
                    )
                    if data == "speed":
                        self.progress.start(12)
                elif kind == "status":
                    self._set_status(data)
                elif kind == "error":
                    self.progress.stop()
                    self._log(f"!! エラー: {data}")
                    self._set_status("失敗")
                    self._running(False)
                    messagebox.showerror("ベンチ実行エラー", str(data))
                elif kind == "cancelled":
                    self.progress.stop()
                    self._log(f"!! {data}")
                    self._set_status("中断しました")
                    self._running(False)
                elif kind == "done":
                    self.progress.stop()
                    self._finish(*data)
        except queue.Empty:
            pass
        self.after(120, self._drain_queue)

    def _maybe_progress(self, message: str) -> None:
        match = _PROGRESS_RE.match(message)
        if match:
            done, total = int(match.group(1)), int(match.group(2))
            self.progress.configure(mode="determinate")
            self.progress["maximum"] = max(1, total)
            self.progress["value"] = done

    def _finish(self, speed, accuracy, values: dict) -> None:
        payload = collect_run(speed, accuracy, label=values["label"] or "gui")
        self.payload = payload
        try:
            out_dir = Path(values["out_dir"] or "results")
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            label = re.sub(r"[^0-9A-Za-z_\-]", "-", values["label"] or "gui")
            json_path = write_json(payload, out_dir / f"{label}-{stamp}.json")
            md_path = write_markdown(payload, out_dir / f"{label}-{stamp}.md")
            self.saved_paths = [json_path, md_path]
            self._log(f"保存: {json_path}")
            self._log(f"保存: {md_path}")
        except OSError as exc:
            self._log(f"!! 保存に失敗: {exc}")
        self._render_summary(payload)
        self._show_markdown(_safe_render(payload))
        self.notebook.select(0)
        self._running(False)
        overall = (payload.get("accuracy") or {}).get("overall") or {}
        if overall:
            self._set_status(
                f"完了: 精度 {overall.get('correct')}/{overall.get('total')} "
                f"({overall.get('accuracy_pct')}%)"
            )
        else:
            self._set_status("完了")

    def _clear_results(self) -> None:
        for tree in (self.speed_tree, self.acc_tree):
            for row in tree.get_children():
                tree.delete(row)
        self.headline.configure(text="")
        self._set_text(self.report_text, "")
        self.progress["value"] = 0

    def _render_summary(self, payload: dict) -> None:
        parts = []
        speed = payload.get("speed")
        if speed:
            for profile, stats in (speed.get("profiles") or {}).items():
                ttft = ((stats.get("ttft_s") or {}).get("p50"))
                toks = (stats.get("tok_per_s") or {}).get("mean")
                p90 = (stats.get("tok_per_s") or {}).get("p90")
                e2e = (stats.get("total_s") or {}).get("mean")
                ptoks = (stats.get("prompt_tok_per_s") or {}).get("mean")
                self.speed_tree.insert(
                    "", "end",
                    values=(
                        profile,
                        _ms(ttft),
                        _num(toks),
                        _num(p90),
                        _sec(e2e),
                        _num(ptoks),
                    ),
                )
            if speed.get("errors"):
                parts.append(f"速度エラー {len(speed['errors'])} 件")
        accuracy = payload.get("accuracy")
        if accuracy:
            overall = accuracy.get("overall") or {}
            self.acc_tree.insert(
                "", "end",
                values=("総合", overall.get("correct"), overall.get("total"),
                        f"{overall.get('accuracy_pct', 0):.1f}%"),
                tags=("overall",),
            )
            self.acc_tree.tag_configure("overall", font=("", 9, "bold"))
            for name, stat in (accuracy.get("by_category") or {}).items():
                self.acc_tree.insert(
                    "", "end",
                    values=(name, stat.get("correct"), stat.get("total"),
                            f"{stat.get('accuracy_pct', 0):.1f}%"),
                )
            for name, stat in (accuracy.get("by_difficulty") or {}).items():
                self.acc_tree.insert(
                    "", "end",
                    values=(f"[{name}]", stat.get("correct"), stat.get("total"),
                            f"{stat.get('accuracy_pct', 0):.1f}%"),
                )
            if accuracy.get("errors"):
                parts.append(f"精度エラー {len(accuracy['errors'])} 件")
            parts.append(
                f"総合正答率 {overall.get('accuracy_pct', 0):.1f}% "
                f"({overall.get('correct')}/{overall.get('total')})"
            )
        self.headline.configure(text="  |  ".join(parts))

    def _show_markdown(self, markdown: str) -> None:
        self._set_text(self.report_text, markdown)

    def _log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", str(message) + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    @staticmethod
    def _set_text(widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        self.status.configure(text=f" {text}")

    def _running(self, running: bool, status: str = "実行中...") -> None:
        self.btn_run.configure(state="disabled" if running else "normal")
        self.btn_stop.configure(state="normal" if running else "disabled")
        if not running:
            self.progress.stop()
        self._set_status(status)

    def _on_close(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            if not messagebox.askyesno("実行中", "ベンチ実行中です。中断して終了しますか？"):
                return
            if self.cancel_token is not None:
                self.cancel_token.cancel()
        self.destroy()


def _csv_ints(value: str) -> list[int] | None:
    if not value:
        return None
    out = []
    for part in value.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out or None


def _num(value) -> str:
    return "—" if value is None else f"{float(value):.1f}"


def _ms(value) -> str:
    return "—" if value is None else f"{float(value) * 1000:.0f} ms"


def _sec(value) -> str:
    return "—" if value is None else f"{float(value):.2f} s"


def _safe_render(payload: dict) -> str:
    try:
        return render_markdown(payload)
    except Exception as exc:  # レポート生成の例外でGUIを落とさない
        return f"(レポート生成に失敗: {exc})"


def run_gui() -> int:
    """GUI を起動する（閉じたら 0 を返す）."""
    app = BenchApp()
    app.mainloop()
    return 0
