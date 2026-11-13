"""GUI（tkinter）と共通機能のヘッドレステスト.

Tk が使えない環境では GUI 描画系テストはスキップし、
設定保存・入力検証・cancel トークンは常に検証する。
"""

from __future__ import annotations

import queue
import tempfile
import unittest
from pathlib import Path

from llmbench.cancel import BenchmarkCancelled, CancelToken
from llmbench.backends import get_backend
from llmbench.gui import SettingsStore, _csv_ints
from llmbench.speed import run_speed_bench

from test_backends import _Server


class CancelTokenTest(unittest.TestCase):
    def test_check_raises_after_cancel(self):
        token = CancelToken()
        token.check()  # 中断前は通る
        token.cancel()
        self.assertTrue(token.cancelled)
        with self.assertRaises(BenchmarkCancelled):
            token.check()

    def test_speed_bench_is_interruptible(self):
        token = CancelToken()
        token.cancel()
        backend = get_backend("mock", model="mock")
        with self.assertRaises(BenchmarkCancelled):
            run_speed_bench(backend, profile_names=["short"], repeats=2, warmup=0,
                            cancel=token)


class HelpersTest(unittest.TestCase):
    def test_csv_ints(self):
        self.assertIsNone(_csv_ints(""))
        self.assertIsNone(_csv_ints("abc,,"))
        self.assertEqual(_csv_ints("1024, 4096"), [1024, 4096])


class SettingsStoreTest(unittest.TestCase):
    def test_api_key_not_saved_unless_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SettingsStore(Path(tmp) / "settings.json")
            values = dict(store.load())
            values["api_key"] = "sk-secret"
            values["save_api_key"] = False
            values["base_url"] = "http://example.test/v1"
            store.save(values)

            reloaded = store.load()
            self.assertEqual(reloaded["base_url"], "http://example.test/v1")
            self.assertEqual(reloaded["api_key"], "")

    def test_api_key_saved_when_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SettingsStore(Path(tmp) / "settings.json")
            values = dict(store.load())
            values["api_key"] = "sk-secret"
            values["save_api_key"] = True
            store.save(values)
            self.assertEqual(store.load()["api_key"], "sk-secret")

    def test_missing_file_gives_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SettingsStore(Path(tmp) / "nowhere" / "settings.json")
            self.assertEqual(store.load()["base_url"], "http://127.0.0.1:8080/v1")


def _tk_available() -> bool:
    try:
        import tkinter

        root = tkinter.Tk()
    except Exception:
        return False
    root.destroy()
    return True


@unittest.skipUnless(_tk_available(), "tkinter でウィンドウを作れない環境")
class GuiAppTest(unittest.TestCase):
    """BenchApp を実描画し、ローカルテストサーバーへ計測を通す."""

    def setUp(self):
        from llmbench.gui import BenchApp

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.app = BenchApp(SettingsStore(Path(self.tmp.name) / "settings.json"))
        self.addCleanup(self.app.destroy)

    def _values(self, base_url: str) -> dict:
        values = self.app._current_values()
        values["base_url"] = base_url
        values["model"] = "test-model"
        values["run_speed"] = True
        values["run_accuracy"] = True
        values["profiles"] = ["short"]
        values["repeats"] = "1"
        values["warmup"] = "0"
        values["categories"] = ["math"]
        values["accuracy_repeat"] = "1"
        values["max_tokens"] = "64"
        values["out_dir"] = str(Path(self.tmp.name) / "out")
        values["label"] = "gui-test"
        return values

    def test_validate_rejects_bad_input(self):
        values = self.app._current_values()
        values["base_url"] = ""
        self.assertIn("ベースURL", self.app._validate(values))
        values["base_url"] = "ftp://example.test"
        self.assertIn("http://", self.app._validate(values))
        values["base_url"] = "http://example.test"
        values["seed"] = "abc"
        self.assertIn("seed", self.app._validate(values))
        values["seed"] = "42"
        values["profiles"] = []
        self.assertIn("プロファイル", self.app._validate(values))

    def test_full_run_updates_tables_and_saves_files(self):
        with _Server() as base:
            values = self._values(f"{base}/v1")
            self.assertEqual(self.app._validate(values), "")
            # ワーカーを同期実行してキューを自前で流し込む
            self.app.cancel_token = CancelToken()
            self.app._worker_run(values)

            items = []
            while True:
                try:
                    items.append(self.app.queue.get_nowait())
                except queue.Empty:
                    break
            kinds = [k for k, _ in items]
            self.assertIn("done", kinds)
            self.assertNotIn("error", kinds)

            self.app._finish(*items[kinds.index("done")][1])

        self.assertIsNotNone(self.app.payload)
        accuracy = self.app.payload.get("accuracy") or {}
        self.assertGreater(accuracy.get("overall", {}).get("total", 0), 0)
        speed = self.app.payload.get("speed") or {}
        self.assertIn("short", speed.get("profiles") or {})

        out_files = sorted((Path(values["out_dir"])).glob("*"))
        self.assertEqual(len(out_files), 2)  # json + md
        # ツリーへ行が反映されている
        self.assertGreaterEqual(len(self.app.speed_tree.get_children()), 1)
        self.assertGreaterEqual(len(self.app.acc_tree.get_children()), 1)
        self.assertIn("総合正答率", self.app.headline.cget("text"))
        self.assertIn("# LLMBench レポート", self.app.report_text.get("1.0", "end"))

    def test_ping_worker_reports_error(self):
        values = self._values("http://127.0.0.1:1")
        self.app.cancel_token = CancelToken()
        self.app._worker_ping(values)
        kinds = []
        while True:
            try:
                kind, _ = self.app.queue.get_nowait()
            except queue.Empty:
                break
            kinds.append(kind)
        self.assertIn("error", kinds)

    def test_ping_worker_success(self):
        with _Server() as base:
            values = self._values(f"{base}/v1")
            self.app.cancel_token = CancelToken()
            self.app._worker_ping(values)
            found = False
            while True:
                try:
                    kind, data = self.app.queue.get_nowait()
                except queue.Empty:
                    break
                if kind == "log":
                    found = "接続OK" in str(data)
        self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()
