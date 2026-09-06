"""Configuration for prompt-protected random KV eviction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RandKVConfig:
    """Policy defaults shared by inference adapters.

    ``budget`` is the persistent per-head cache size K. ``buffer_size`` is the
    recent, unscored buffer r. Immediately after an eviction a head therefore
    retains at most ``budget + buffer_size`` positions.
    """

    budget: int = 2048
    buffer_size: int = 64
    seed: int = 0
    protect_prompt: bool = True

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError("budget must be greater than zero")
        if self.buffer_size < 0:
            raise ValueError("buffer_size must be non-negative")
        if not self.protect_prompt:
            raise ValueError(
                "protect_prompt=False is intentionally unsupported: prompt "
                "protection is a defining RandKV invariant"
            )
