"""ベンチ実行中断のための小さなヘルパー（CLI/GUI 共通）."""

from __future__ import annotations

__all__ = ["BenchmarkCancelled", "CancelToken"]


class BenchmarkCancelled(Exception):
    """ユーザーがベンチ実行を中断したことを示す."""


class CancelToken:
    """スレッド間で共有できる中断フラグ.

    GUI の「中止」ボタンから ``cancel()`` を呼び、ベンチループ側は
    ``token.check()`` で :class:`BenchmarkCancelled` を送出する。
    """

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def check(self) -> None:
        if self._cancelled:
            raise BenchmarkCancelled("ベンチ実行が中断されました")
