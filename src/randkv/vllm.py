"""Dependency-free planning boundary for a future vLLM runtime adapter."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import TypeAlias

from .config import RandKVConfig
from .policy import RandomEvictionPolicy

RequestId: TypeAlias = str | int | bytes

VLLM_TARGET_VERSION = "0.28.0"
VLLM_TARGET_COMMIT = "2cf0a6915ce544dc493a0990f2ea38d81601128a"


@dataclass(frozen=True, slots=True)
class VLLMCompactionPlan:
    """Backend-neutral description of one head-wise KV compaction.

    vLLM owns physical blocks shared by every KV head. The runtime adapter can
    use ``source_offsets_by_head`` to copy each head's independently selected
    tokens into a shared set of dense destination blocks.
    """

    request_id: RequestId
    layer: int
    head_offset: int
    eviction_index: int
    absolute_tokens_seen: int
    block_size: int
    source_tokens: int
    retained_positions_by_head: tuple[tuple[int, ...], ...]
    source_offsets_by_head: tuple[tuple[int, ...], ...]

    @property
    def num_kv_heads(self) -> int:
        return len(self.retained_positions_by_head)

    @property
    def retained_tokens(self) -> int:
        return len(self.retained_positions_by_head[0])

    @property
    def evicted_tokens_per_head(self) -> int:
        return self.source_tokens - self.retained_tokens

    @property
    def destination_blocks(self) -> int:
        return ceil(self.retained_tokens / self.block_size)


class VLLMCompactionPlanner:
    """Translate logical RandKV selections into head-wise copy offsets.

    This prototype deliberately imports no vLLM or tensor library. The future
    runtime shim owns block allocation, copies, and attention metadata; this
    class owns deterministic policy semantics only.
    """

    def __init__(self, config: RandKVConfig | None = None) -> None:
        self.config = config or RandKVConfig()
        self.policy = RandomEvictionPolicy(self.config)

    def plan(
        self,
        positions_by_head: tuple[tuple[int, ...], ...],
        *,
        prompt_length: int,
        request_id: RequestId,
        layer: int,
        eviction_index: int,
        absolute_tokens_seen: int,
        block_size: int,
        head_offset: int = 0,
    ) -> VLLMCompactionPlan | None:
        """Return a compaction plan, or ``None`` while under capacity.

        ``head_offset`` is the tensor-parallel rank's first global KV-head ID.
        Using global IDs keeps a draw stable when the same request is sharded.
        """

        source_tokens = self._validate_inputs(
            positions_by_head,
            prompt_length=prompt_length,
            request_id=request_id,
            layer=layer,
            eviction_index=eviction_index,
            absolute_tokens_seen=absolute_tokens_seen,
            block_size=block_size,
            head_offset=head_offset,
        )
        capacity = self.config.budget + self.config.buffer_size
        if source_tokens <= capacity:
            return None

        retained_positions = tuple(
            self.policy.select(
                positions,
                prompt_length=prompt_length,
                layer=layer,
                kv_head=head_offset + local_head,
                request_id=request_id,
                eviction_index=eviction_index,
            )
            for local_head, positions in enumerate(positions_by_head)
        )
        retained_lengths = {len(positions) for positions in retained_positions}
        if len(retained_lengths) != 1:
            raise RuntimeError("all KV heads must retain the same physical length")

        source_offsets = tuple(
            self._source_offsets(source, retained)
            for source, retained in zip(
                positions_by_head, retained_positions, strict=True
            )
        )
        return VLLMCompactionPlan(
            request_id=request_id,
            layer=layer,
            head_offset=head_offset,
            eviction_index=eviction_index,
            absolute_tokens_seen=absolute_tokens_seen,
            block_size=block_size,
            source_tokens=source_tokens,
            retained_positions_by_head=retained_positions,
            source_offsets_by_head=source_offsets,
        )

    def _validate_inputs(
        self,
        positions_by_head: tuple[tuple[int, ...], ...],
        *,
        prompt_length: int,
        request_id: RequestId,
        layer: int,
        eviction_index: int,
        absolute_tokens_seen: int,
        block_size: int,
        head_offset: int,
    ) -> int:
        if not positions_by_head:
            raise ValueError("positions_by_head must contain at least one KV head")
        if block_size <= 0:
            raise ValueError("block_size must be greater than zero")
        if absolute_tokens_seen < 0:
            raise ValueError("absolute_tokens_seen must be non-negative")
        if prompt_length < 0:
            raise ValueError("prompt_length must be non-negative")
        if prompt_length > self.config.budget:
            raise ValueError(
                f"prompt requires {prompt_length} cache slots but budget is "
                f"{self.config.budget}; increase the budget"
            )
        if layer < 0 or eviction_index < 0 or head_offset < 0:
            raise ValueError(
                "layer, eviction_index, and head_offset must be non-negative"
            )
        if not isinstance(request_id, (str, int, bytes)):
            raise TypeError("request_id must be a string, integer, or bytes")

        lengths = {len(positions) for positions in positions_by_head}
        if len(lengths) != 1:
            raise ValueError("all KV heads must have the same physical length")
        for positions in positions_by_head:
            if any(not isinstance(position, int) for position in positions):
                raise TypeError("positions must contain only integers")
            if any(
                left >= right
                for left, right in zip(positions, positions[1:], strict=False)
            ):
                raise ValueError("positions must be unique and strictly increasing")
            if any(
                position < 0 or position >= absolute_tokens_seen
                for position in positions
            ):
                raise ValueError(
                    "retained positions must be non-negative and below "
                    "absolute_tokens_seen"
                )
            prompt = tuple(
                position for position in positions if position < prompt_length
            )
            if prompt != tuple(range(prompt_length)):
                raise ValueError("every KV head must contain the protected prompt")
        return len(positions_by_head[0])

    @staticmethod
    def _source_offsets(
        source: tuple[int, ...], retained: tuple[int, ...]
    ) -> tuple[int, ...]:
        offsets = {position: offset for offset, position in enumerate(source)}
        return tuple(offsets[position] for position in retained)
