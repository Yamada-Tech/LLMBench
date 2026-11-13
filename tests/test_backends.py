"""バックエンドとパイプラインの単体テスト.

ローカルに http.server を起動し、OpenAI互換APIのSSE形式を再現して
パース・タイミング統計を検証する。
"""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llmbench.accuracy import run_accuracy_bench
from llmbench.backends import (
    BackendError,
    OpenAICompatBackend,
    _chat_endpoint,
    _delta_text,
    get_backend,
)
from llmbench.metrics import summarize
from llmbench.report import collect_run, compare_runs, render_markdown, write_json
from llmbench.speed import run_speed_bench
from llmbench.tasks import load_tasks


def _openai_sse_chunks(text: str) -> list[bytes]:
    chunks = []
    for part in [text[i : i + 5] for i in range(0, len(text), 5)]:
        payload = {"choices": [{"delta": {"content": part}}]}
        chunks.append(f"data: {json.dumps(payload)}\n\n".encode())
    usage = {
        "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": len(text)},
        # llama.cpp 系サーバーの timings 拡張（chunk 直下）
        "timings": {"prompt_ms": 250.0, "predicted_ms": 1000.0,
                    "predicted_per_second": len(text)},
    }
    chunks.append(f"data: {json.dumps(usage)}\n\n".encode())
    chunks.append(b"data: [DONE]\n\n")
    return chunks


class _Handler(BaseHTTPRequestHandler):
    #: テストから参照する、最後に受け取った Authorization ヘッダ
    last_auth: str | None = None

    def log_message(self, *args):
        pass

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

    def do_POST(self):
        body = self._read_body()
        _Handler.last_auth = self.headers.get("Authorization")
        if self.path == "/v1/chat/completions":
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in _openai_sse_chunks("最終回答: 42"):
                    self.wfile.write(chunk)
            else:
                payload = {
                    "choices": [{"message": {"content": "最終回答: 42"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                }
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        elif self.path == "/boom":
            self.send_error(500, "boom")
        else:
            self.send_error(404)


class _Server:
    def __enter__(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


class BackendTest(unittest.TestCase):
    def test_openai_sse(self):
        with _Server() as base:
            backend = OpenAICompatBackend("m", base)
            res = backend.generate([{"role": "user", "content": "hi"}], max_tokens=16)
            self.assertEqual(res.text, "最終回答: 42")
            self.assertEqual(res.prompt_tokens, 10)
            self.assertEqual(res.generated_tokens, len("最終回答: 42"))
            self.assertIsNotNone(res.ttft_s)
            self.assertGreater(res.tok_per_s, 0)
            # timings 拡張（chunk 直下）も拾える
            self.assertEqual(res.stats["timings"]["prompt_ms"], 250.0)
            self.assertAlmostEqual(res.prompt_tok_per_s, 40.0, delta=0.01)

    def test_openai_nonstream(self):
        with _Server() as base:
            backend = OpenAICompatBackend("m", base)
            res = backend.generate(
                [{"role": "user", "content": "hi"}], max_tokens=16, stream=False
            )
            self.assertEqual(res.text, "最終回答: 42")
            self.assertIsNone(res.ttft_s)

    def test_base_url_with_v1_suffix(self):
        # --base-url に /v1 まで指定しても重複結合しない
        self.assertEqual(
            _chat_endpoint("http://127.0.0.1:8080/v1"),
            "http://127.0.0.1:8080/v1/chat/completions",
        )
        self.assertEqual(
            _chat_endpoint("http://127.0.0.1:8080"),
            "http://127.0.0.1:8080/v1/chat/completions",
        )
        self.assertEqual(
            _chat_endpoint("https://example.com/api/v1/"),
            "https://example.com/api/v1/chat/completions",
        )
        self.assertEqual(
            _chat_endpoint("https://example.com/v1/chat/completions"),
            "https://example.com/v1/chat/completions",
        )

    def test_v1_suffix_end_to_end(self):
        with _Server() as base:
            backend = get_backend("openai", model="m", base_url=f"{base}/v1")
            res = backend.generate([{"role": "user", "content": "hi"}], max_tokens=16)
            self.assertEqual(res.text, "最終回答: 42")

    def test_connection_error(self):
        backend = OpenAICompatBackend("m", "http://127.0.0.1:1")
        with self.assertRaises(BackendError):
            backend.generate([{"role": "user", "content": "x"}])

    def test_delta_text_includes_reasoning(self):
        choice = {"delta": {"content": "答え: ", "reasoning_content": "うーん"}}
        self.assertEqual(_delta_text(choice), "うーん答え: ")

    def test_insecure_ssl_context(self):
        backend = get_backend(
            "openai", model="m", base_url="https://127.0.0.1:1", insecure=True
        )
        self.assertIsNotNone(backend._ssl)
        backend_ok = get_backend("openai", model="m", base_url="http://127.0.0.1:1")
        self.assertIsNone(backend_ok._ssl)

    def test_api_key_header_and_masking(self):
        _Handler.last_auth = None
        with _Server() as base:
            backend = get_backend("openai", model="m", base_url=f"{base}/v1",
                                  api_key="sk-test-123")
            backend.generate([{"role": "user", "content": "hi"}], max_tokens=16)
            self.assertEqual(_Handler.last_auth, "Bearer sk-test-123")
            # レポート info() に平文で載せない
            self.assertEqual(backend.info()["options"]["api_key"], "***")

    def test_no_auth_header_without_key(self):
        _Handler.last_auth = None
        with _Server() as base:
            backend = get_backend("openai", model="m", base_url=f"{base}/v1")
            backend.generate([{"role": "user", "content": "hi"}], max_tokens=16)
            self.assertIsNone(_Handler.last_auth)

    def test_factory_errors(self):
        with self.assertRaises(BackendError):
            get_backend("nope")
        # 削除された ollama ネイティブは案内付きエラー
        with self.assertRaises(BackendError):
            get_backend("ollama", model="m")

    def test_default_backend_is_openai(self):
        backend = get_backend("", base_url="http://127.0.0.1:1")
        self.assertIsInstance(backend, OpenAICompatBackend)
        self.assertEqual(backend.endpoint, "http://127.0.0.1:1/v1/chat/completions")


class TasksTest(unittest.TestCase):
    def test_builtin_tasks_load(self):
        task_set = load_tasks()
        self.assertGreaterEqual(len(task_set.tasks), 20)
        ids = [t.id for t in task_set.tasks]
        self.assertEqual(len(ids), len(set(ids)))
        for t in task_set.tasks:
            self.assertIn(t.scorer.get("type"), {"numeric", "choice", "exact", "json", "code"})

    def test_filter(self):
        task_set = load_tasks().filter(["math"], ["easy"])
        self.assertTrue(all(t.category == "math" for t in task_set.tasks))

    def test_messages_instruction(self):
        t = load_tasks().tasks[0]
        msgs = t.messages()
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("最終回答", msgs[1]["content"])


class PipelineTest(unittest.TestCase):
    def test_speed_bench_mock(self):
        backend = get_backend("mock", model="mock")
        result = run_speed_bench(backend, profile_names=["short"], repeats=2, warmup=0)
        self.assertEqual(len(result.raw), 2)
        self.assertIn("short", result.profiles)
        self.assertGreater(result.profiles["short"]["tok_per_s"]["count"], 0)

    def test_accuracy_bench_mock(self):
        backend = get_backend("mock", model="mock")
        task_set = load_tasks().filter(["math", "logic"])
        result = run_accuracy_bench(backend, task_set, max_tokens=32, seed=3)
        self.assertEqual(result.tasks_run, len(task_set.tasks))
        self.assertEqual(
            result.overall["total"], result.overall["correct"] + result.overall["total"] - result.overall["correct"]
        )
        self.assertIn("math", result.by_category)

    def test_json_roundtrip_and_markdown(self):
        backend = get_backend("mock", model="mock")
        speed = run_speed_bench(backend, profile_names=["short"], repeats=1, warmup=0)
        acc = run_accuracy_bench(backend, load_tasks().filter(["math"]), max_tokens=16, seed=1)
        payload = collect_run(speed, acc, label="ut")
        md = render_markdown(payload)
        self.assertIn("## 速度ベンチ", md)
        self.assertIn("## 精度ベンチ", md)
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = write_json(payload, Path(tmp) / "run.json")
            comparison = compare_runs([path])
            self.assertEqual(len(comparison["runs"]), 1)

    def test_metrics_summarize(self):
        s = summarize([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(s.count, 4)
        self.assertAlmostEqual(s.mean, 2.5)
        self.assertAlmostEqual(s.p50, 2.5, delta=0.01)
        empty = summarize([])
        self.assertEqual(empty.count, 0)


class CLITest(unittest.TestCase):
    def test_selftest_command(self):
        from llmbench.cli import main
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            code = main(["selftest"])
            self.assertEqual(code, 0)

    def test_accuracy_command_mock(self):
        from llmbench.cli import main
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            code = main([
                "accuracy", "--backend", "mock", "--categories", "math",
                "--out", tmp, "-q",
            ])
            self.assertEqual(code, 0)
            files = list(Path(tmp).glob("*.json"))
            self.assertEqual(len(files), 1)


if __name__ == "__main__":
    unittest.main()
