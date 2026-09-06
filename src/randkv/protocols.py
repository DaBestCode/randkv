"""Stable interfaces shared by RandKV backends."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from .metrics import PolicyStats


class KVPolicy(Protocol):
    """Structural interface implemented by cache-eviction policies."""

    def select(
        self,
        positions: Iterable[int],
        *,
        prompt_length: int,
        layer: int = 0,
        kv_head: int = 0,
        request_id: str | int | bytes = "offline",
        eviction_index: int = 0,
    ) -> tuple[int, ...]: ...

    def stats(self) -> PolicyStats: ...
