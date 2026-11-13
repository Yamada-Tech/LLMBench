"""推論バックエンド（HTTPクライアント）.

- OpenAICompatBackend : OpenAI互換 /v1/chat/completions エンドポイント
  （外部APIでも、手元の 127.0.0.1/v1 でも、宛先を指定するだけで計測できる）
- MockBackend         : セルフテスト用（ネットワーク・GPU不要）

標準ライブラリ (urllib) のみで SSE ストリーミングを読む。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .textutils import estimate_tokens

__all__ = [
    "BackendResult",
    "Backend",
    "OpenAICompatBackend",
    "MockBackend",
    "get_backend",
    "BackendError",
]


class BackendError(RuntimeError):
    """バックエンド通信・設定エラー."""


@dataclass
class BackendResult:
    """1回の推論の結果とタイミング・トークン統計."""

    text: str
    ttft_s: float | None  # 生成開始までの秒（プロンプト処理込み）
    total_s: float
    completion_tokens: int
    prompt_tokens: int | None
    generated_tokens: int | None  # サーバー報告（無ければ None）
    decode_s: float | None  # 生成区間の秒
    stats: dict = field(default_factory=dict)  # サーバー由来の詳細統計

    @property
    def tok_per_s(self) -> float:
        """生成速度（token/s）. サーバー報告を優先."""
        tokens = self.generated_tokens or self.completion_tokens
        if self.decode_s and self.decode_s > 0:
            return tokens / self.decode_s
        return 0.0

    @property
    def prompt_tok_per_s(self) -> float | None:
        """プロンプト処理速度（サーバーが報告する場合のみ）."""
        stats = self.stats or {}
        seconds = stats.get("prompt_eval_duration_s")
        if not seconds:
            timings = stats.get("timings") or {}
            seconds = (timings.get("prompt_ms") or 0) / 1000.0
        if seconds and self.prompt_tokens:
            return self.prompt_tokens / seconds
        return None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "ttft_s": self.ttft_s,
            "total_s": self.total_s,
            "decode_s": self.decode_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "generated_tokens": self.generated_tokens,
            "tok_per_s": round(self.tok_per_s, 3),
            "prompt_tok_per_s": (
                round(self.prompt_tok_per_s, 3) if self.prompt_tok_per_s else None
            ),
            "stats": self.stats,
        }


def _messages_tokens_estimate(messages: list[dict]) -> int:
    return sum(estimate_tokens(str(m.get("content", ""))) for m in messages)


def _read_sse_lines(response) -> "object":
    """SSE応答を1行ずつ yield するジェネレータ."""
    for raw in response:
        line = raw.decode("utf-8", errors="replace").strip()
        if line.startswith("data:"):
            yield line[5:].strip()


def _chat_endpoint(base_url: str) -> str:
    """--base-url から chat/completions のフルURLを組み立てる.

    指定のしかたならどれも受け付ける:
      http://127.0.0.1:8080                → .../v1/chat/completions
      http://127.0.0.1:8080/v1             → .../v1/chat/completions
      https://example.com/api/v1           → .../api/v1/chat/completions
      https://example.com/v1/chat/completions → そのまま
    """
    url = base_url.rstrip("/")
    lowered = url.lower()
    if lowered.endswith("/chat/completions"):
        return url
    if lowered.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"


def _delta_text(choice: dict) -> str:
    """delta / message から生成テキストを取り出す.

    llama.cpp 等の reasoning 拡張 (reasoning_content / reasoning) も
    生成トークンとして扱う（thinking モデルで content が空になりにくい）。
    """
    delta = choice.get("delta") or {}
    msg = choice.get("message") or {}
    parts = []
    for source in (delta, msg):
        # reasoning が先、content を後に結合（抽出マーカー判定を安定させる）
        for key in ("reasoning_content", "reasoning", "content"):
            value = source.get(key)
            if value:
                parts.append(value)
    return "".join(parts)


class Backend:
    """バックエンドの共通インターフェース."""

    name = "base"

    def __init__(self, model: str, base_url: str, timeout: float = 120.0, **options):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.endpoint = _chat_endpoint(self.base_url)
        self.timeout = timeout
        self.options = options
        self._ssl = None
        if options.get("insecure") and base_url.lower().startswith("https"):
            import ssl

            self._ssl = ssl.create_default_context()
            self._ssl.check_hostname = False
            self._ssl.verify_mode = ssl.CERT_NONE

    def info(self) -> dict:
        """レポートへ書き出す設定（シークレットは伏せる）."""
        options = dict(self.options)
        if options.get("api_key"):
            options["api_key"] = "***"
        return {
            "backend": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "endpoint": getattr(self, "endpoint", self.base_url),
            "options": options,
        }

    def warmup(self) -> None:
        """モデルロードを済ませる（失敗しても無視）."""
        try:
            self.generate([{"role": "user", "content": "hi"}], max_tokens=8, stream=False)
        except BackendError:
            pass

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int | None = None,
        stream: bool = True,
    ) -> BackendResult:
        raise NotImplementedError


def _connection_hint(base_url: str, insecure: bool, exc_text: str = "") -> str:
    """接続失敗時の対処ヒント（TLS検証失敗を検出して案内）."""
    if "CERTIFICATE_VERIFY_FAILED" in exc_text.upper():
        if insecure:
            return "証明書検証が失敗しています（--insecure でも解決しません）"
        return (
            "HTTPSの証明書検証で失敗しました。"
            "自己署名証明書の場合は --insecure を付けて再実行してください"
        )
    return "サーバーが起動しているか・URL/ポートを確認してください"


# --- OpenAI 互換 -----------------------------------------------------------

class OpenAICompatBackend(Backend):
    """OpenAI互換 /v1/chat/completions クライアント."""

    name = "openai"

    def _payload(self, messages, max_tokens, temperature, seed, stream) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if seed is not None:
            payload["seed"] = seed
        # llama-server 拡張: usage をストリーム末尾に付ける
        payload["stream_options"] = {"include_usage": True}
        for key in ("top_p", "min_p"):
            if self.options.get(key) is not None:
                payload[key] = self.options[key]
        if self.options.get("no_thinking"):
            # ハイブリッド思考モデルの thinking を無効化（llama-server / vLLM 等）
            payload["reasoning_effort"] = "none"
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        api_key = str(self.options.get("api_key") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _request(self, payload: dict):
        url = self.endpoint
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers=self._headers(),
            method="POST",
        )
        try:
            return urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            if exc.code in (401, 403):
                raise BackendError(
                    f"HTTP {exc.code}: {body or exc.reason}. "
                    "APIキーが必要または不正の可能性があります（APIキー欄を確認）"
                ) from exc
            if exc.code == 404:
                raise BackendError(
                    f"HTTP {exc.code}: {body or exc.reason}. "
                    f"エンドポイント '{self.endpoint}' が存在しません。"
                    "--base-url を /v1 まで指定（またはホストのみ）してください"
                ) from exc
            raise BackendError(f"HTTP {exc.code}: {body or exc.reason}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise BackendError(
                f"接続できません ({self.base_url}): {exc}. "
                + _connection_hint(self.base_url, bool(self._ssl), str(exc))
            ) from exc

    def generate(self, messages, max_tokens=256, temperature=0.0, seed=None, stream=True):
        if not stream:
            return self._generate_nonstream(messages, max_tokens, temperature, seed)
        payload = self._payload(messages, max_tokens, temperature, seed, True)
        started = time.perf_counter()
        first_token_at: float | None = None
        pieces: list[str] = []
        usage: dict = {}
        last_chunk_obj: dict = {}
        with self._request(payload) as response:
            for chunk in _read_sse_lines(response):
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except ValueError:
                    continue
                last_chunk_obj = obj
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices", []):
                    content = _delta_text(choice)
                    if content:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        pieces.append(content)
        total = time.perf_counter() - started
        text = "".join(pieces)
        stats_extra = {}
        if not text and (usage.get("completion_tokens") or 0) > 0:
            stats_extra["warning"] = (
                "サーバーはトークンを生成したが content が空でした。"
                "reasoning 専用トークンの可能性: --max-tokens を増やしてください"
            )
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None:
            completion_tokens = estimate_tokens(text)
        if prompt_tokens is None:
            prompt_tokens = _messages_tokens_estimate(messages)
        # llama-server 等の timings 拡張（usage 内でも chunk 直下でも拾う）
        timings = {}
        for key in ("timings", "timing"):
            if isinstance(usage.get(key), dict):
                timings = usage[key]
            elif isinstance(last_chunk_obj.get(key), dict):
                timings = last_chunk_obj[key]
        decode_s = None
        if first_token_at is not None:
            decode_s = time.perf_counter() - first_token_at
        return BackendResult(
            text=text,
            ttft_s=first_token_at - started if first_token_at is not None else None,
            total_s=total,
            completion_tokens=int(completion_tokens or 0),
            prompt_tokens=int(prompt_tokens or 0),
            generated_tokens=int(completion_tokens or 0) if usage.get("completion_tokens") is not None else None,
            decode_s=decode_s,
            stats={
                "usage": usage,
                "timings": timings,
                "source": "openai-sse",
                **stats_extra,
            },
        )

    def _generate_nonstream(self, messages, max_tokens, temperature, seed):
        payload = self._payload(messages, max_tokens, temperature, seed, False)
        payload.pop("stream_options", None)
        started = time.perf_counter()
        with self._request(payload) as response:
            body = json.loads(response.read().decode("utf-8", errors="replace"))
        total = time.perf_counter() - started
        choices = body.get("choices") or [{}]
        text = _delta_text(choices[0])
        usage = body.get("usage") or {}
        completion = usage.get("completion_tokens") or estimate_tokens(text)
        prompt = usage.get("prompt_tokens") or _messages_tokens_estimate(messages)
        stats = {"source": "openai"}
        for key in ("timings", "timing", "stats"):
            if isinstance(body.get(key), dict):
                stats[key] = body[key]
        return BackendResult(
            text=text,
            ttft_s=None,
            total_s=total,
            completion_tokens=int(completion),
            prompt_tokens=int(prompt),
            generated_tokens=usage.get("completion_tokens"),
            decode_s=total if text else None,
            stats=stats,
        )


# --- Mock (セルフテスト) ---------------------------------------------------

class MockBackend(Backend):
    """ネットワーク不要の擬似バックエンド.

    問題文に含まれるヒントに反応して正しい形式の解答を返す:
      - 「答え: X」形式の選択肢問題 → 最も出現頻度の高い候補
      - 数値問題 → 「最終回答: 42」
      - JSON問題 → {}
      - コード問題 → 簡単な関数
    速度測定のみ意図的に遅延を挿入できる（delay_ms）。
    """

    name = "mock"

    def generate(self, messages, max_tokens=256, temperature=0.0, seed=None, stream=True):
        import random

        rng = random.Random(seed if seed is not None else 0)
        user = " ".join(str(m.get("content", "")) for m in messages)
        delay = float(self.options.get("delay_ms", 0)) / 1000.0
        started = time.perf_counter()
        pieces: list[str] = []
        if "JSON" in user or "json" in user:
            pieces.append('{"name": "サンプル", "age": 1, "city": "東京"}')
        elif "コード" in user or "code" in user.lower():
            pieces.append("```python\ndef helper():\n    return 42\n```")
        elif "答え" in user and any(f"\n{c}." in user for c in "ABCDE"):
            pieces.append("答え: B")
        else:
            pieces.append(f"最終回答: {rng.randint(1, 100)}")
        time.sleep(min(delay, 0.5))
        ttft = time.perf_counter() - started
        chunks = max(1, len(pieces[0]) // 4)
        time.sleep(min(delay, 0.5) if delay else 0.0)
        total = time.perf_counter() - started
        text = pieces[0]
        fake_tokens = estimate_tokens(text)
        return BackendResult(
            text=text,
            ttft_s=ttft,
            total_s=max(total, ttft),
            completion_tokens=fake_tokens,
            prompt_tokens=_messages_tokens_estimate(messages),
            generated_tokens=fake_tokens,
            decode_s=max(total - ttft, 0.001),
            stats={"source": "mock", "chunks": chunks},
        )


# --- ファクトリ -------------------------------------------------------------

def get_backend(
    backend: str,
    model: str = "",
    base_url: str = "",
    timeout: float = 120.0,
    **options,
) -> Backend:
    """バックエンド名からインスタンスを生成する.

    計測対象は OpenAI互換 chat/completions エンドポイントのみ
    （外部APIでも http://127.0.0.1:8080/v1 でも同じ扱い）。
    """
    backend = (backend or "").strip().lower()
    if backend in ("", "openai", "openai-compat", "openai-compatible", "vllm", "lmstudio"):
        base_url = base_url or "http://127.0.0.1:8080"
        model = model or "default"
        return OpenAICompatBackend(model, base_url, timeout, **options)
    if backend in ("ollama", "ollama-native"):
        raise BackendError(
            "ollama ネイティブ (/api/chat) は削除しました。"
            "OpenAI互換APIを使ってください "
            "(例: --backend openai --base-url http://127.0.0.1:11434/v1)"
        )
    if backend == "mock":
        return MockBackend(model or "mock-model", base_url or "mock://", timeout, **options)
    raise BackendError(
        f"未知のバックエンド: {backend} (openai | mock のいずれか)"
    )
