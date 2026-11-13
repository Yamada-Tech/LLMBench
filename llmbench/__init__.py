"""llmbench - LLMの精度と速度を測定するベンチマークツール.

依存パッケージを持たず、標準ライブラリのみで動作する。
計測対象は OpenAI互換 chat/completions エンドポイントのみ。
外部の公開APIでも、手元の http://127.0.0.1:8080/v1 でも、
--base-url を向けるだけで計測できる。
  - openai : OpenAI互換 /v1/chat/completions (既定)
  - mock   : セルフテスト用（ネットワーク不要）
"""

__version__ = "0.1.0"

__all__ = ["__version__", "get_backend"]


def get_backend(*args, **kwargs):
    """後方互換の遅延インポート用のヘルパー."""
    from .backends import get_backend as _get_backend

    return _get_backend(*args, **kwargs)
