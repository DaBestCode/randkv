"""The framework-independent RandKV selection policy."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable

from .config import RandKVConfig
from .metrics import PolicyMetrics, PolicyStats


class RandomEvictionPolicy:
    """Protect the prompt and sample older generated positions uniformly.

    This class operates on logical token positions and deliberately has no
    PyTorch dependency. Hugging Face and vLLM adapters can use the selected
    positions to compact their backend-specific key/value tensors.
    """

    def __init__(self, config: RandKVConfig | None = None) -> None:
        self.config = config or RandKVConfig()
        self._metrics = PolicyMetrics()

    def select(
        self,
        positions: Iterable[int],
        *,
        prompt_length: int,
        layer: int = 0,
        kv_head: int = 0,
        request_id: str | int | bytes = "offline",
        eviction_index: int = 0,
    ) -> tuple[int, ...]:
        """Return retained logical positions in chronological order.

        Randomness is derived from policy seed, request, eviction, layer, and
        KV-head identity. The result is repeatable and independent of Python's
        process-randomized ``hash()``.
        """

        cached = tuple(positions)
        self._validate(
            cached,
            prompt_length=prompt_length,
            layer=layer,
            kv_head=kv_head,
            eviction_index=eviction_index,
        )

        if prompt_length > self.config.budget:
            raise ValueError(
                f"prompt requires {prompt_length} cache slots but budget is "
                f"{self.config.budget}; increase the budget"
            )
        prompt = tuple(position for position in cached if position < prompt_length)
        expected_prompt = tuple(range(prompt_length))
        if prompt != expected_prompt:
            missing = sorted(set(expected_prompt) - set(prompt))
            raise ValueError(
                "cache is missing protected prompt positions: "
                + ", ".join(map(str, missing[:8]))
            )

        capacity = self.config.budget + self.config.buffer_size
        if len(cached) <= capacity:
            return cached

        recent = cached[-self.config.buffer_size :] if self.config.buffer_size else ()
        recent_set = set(recent)
        prompt_set = set(prompt)
        candidates = tuple(
            position
            for position in cached
            if position not in prompt_set and position not in recent_set
        )
        generated_slots = self.config.budget - len(prompt)

        if generated_slots >= len(candidates):
            sampled = candidates
        else:
            rng = random.Random(
                self._derived_seed(
                    request_id=request_id,
                    eviction_index=eviction_index,
                    layer=layer,
                    kv_head=kv_head,
                )
            )
            sampled = tuple(rng.sample(candidates, generated_slots))

        retained_set = prompt_set | recent_set | set(sampled)
        retained = tuple(position for position in cached if position in retained_set)
        self._metrics.record(
            before=len(cached), after=len(retained), prompt_retained=len(prompt)
        )
        return retained

    def select_heads(
        self,
        positions: Iterable[int],
        *,
        prompt_length: int,
        num_kv_heads: int,
        layer: int = 0,
        request_id: str | int | bytes = "offline",
        eviction_index: int = 0,
    ) -> tuple[tuple[int, ...], ...]:
        """Select positions independently for every KV head."""

        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be greater than zero")
        cached = tuple(positions)
        return tuple(
            self.select(
                cached,
                prompt_length=prompt_length,
                layer=layer,
                kv_head=head,
                request_id=request_id,
                eviction_index=eviction_index,
            )
            for head in range(num_kv_heads)
        )

    def stats(self) -> PolicyStats:
        return self._metrics.snapshot()

    def _derived_seed(
        self,
        *,
        request_id: str | int | bytes,
        eviction_index: int,
        layer: int,
        kv_head: int,
    ) -> int:
        if not isinstance(request_id, (str, int, bytes)):
            raise TypeError("request_id must be a string, integer, or bytes")
        request_material = (
            request_id.hex() if isinstance(request_id, bytes) else str(request_id)
        )
        material = "\x1f".join(
            (
                str(self.config.seed),
                type(request_id).__qualname__,
                request_material,
                str(eviction_index),
                str(layer),
                str(kv_head),
            )
        ).encode()
        return int.from_bytes(
            hashlib.blake2b(material, digest_size=16).digest(), byteorder="big"
        )

    @staticmethod
    def _validate(
        positions: tuple[int, ...],
        *,
        prompt_length: int,
        layer: int,
        kv_head: int,
        eviction_index: int,
    ) -> None:
        if prompt_length < 0:
            raise ValueError("prompt_length must be non-negative")
        if layer < 0 or kv_head < 0 or eviction_index < 0:
            raise ValueError("layer, kv_head, and eviction_index must be non-negative")
        if any(not isinstance(position, int) for position in positions):
            raise TypeError("positions must contain only integers")
        if any(
            left >= right for left, right in zip(positions, positions[1:], strict=False)
        ):
            raise ValueError("positions must be unique and strictly increasing")
