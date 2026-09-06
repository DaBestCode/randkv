"""Hugging Face Transformers cache adapter for RandKV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import Cache, PreTrainedConfig
from transformers.cache_utils import (
    CacheLayerMixin,
    DynamicLayer,
    LinearAttentionCacheLayerMixin,
)

from .config import RandKVConfig
from .policy import RandomEvictionPolicy

__all__ = [
    "RandKVCache",
    "RandKVCacheStats",
    "RandKVDynamicLayer",
    "generate",
]


@dataclass(frozen=True, slots=True)
class RandKVCacheStats:
    """Snapshot of a Transformers RandKV cache."""

    absolute_tokens_seen: int
    physical_tokens_per_layer: tuple[int, ...]
    eviction_rounds_per_layer: tuple[int, ...]
    head_evictions_total: int
    positions_evicted_total: int


def generate(
    model: Any,
    input_ids: torch.LongTensor,
    *,
    randkv_config: RandKVConfig | None = None,
    request_id: str | int | bytes = "offline",
    **generation_kwargs: Any,
) -> Any:
    """Generate with a fresh RandKV cache, inferring the prompt boundary.

    The initial adapter accepts one unpadded sequence. Advanced callers that
    need to inspect the cache can construct :class:`RandKVCache` directly.
    """

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise NotImplementedError(
            "RandKV's initial Transformers adapter requires input_ids shaped [1, seq]"
        )
    attention_mask = generation_kwargs.get("attention_mask")
    if attention_mask is not None and not bool(torch.all(attention_mask == 1)):
        raise NotImplementedError("padded inputs are not supported by RandKV yet")
    if generation_kwargs.get("past_key_values") is not None:
        raise ValueError("generate() creates its own cache; pass no past_key_values")
    if not hasattr(model, "config") or not hasattr(model, "generate"):
        raise TypeError("model must be a Transformers causal language model")

    cache = RandKVCache(
        model_config=model.config,
        prompt_length=input_ids.shape[-1],
        randkv_config=randkv_config,
        request_id=request_id,
    )
    generation_kwargs["past_key_values"] = cache
    return model.generate(input_ids=input_ids, **generation_kwargs)


class RandKVDynamicLayer(DynamicLayer):
    """A single Transformers cache layer with per-KV-head compaction."""

    def __init__(
        self,
        *,
        layer_idx: int,
        prompt_length: int,
        randkv_config: RandKVConfig,
        request_id: str | int | bytes,
    ) -> None:
        super().__init__()
        if prompt_length < 0:
            raise ValueError("prompt_length must be non-negative")
        if prompt_length > randkv_config.budget:
            raise ValueError(
                f"prompt requires {prompt_length} cache slots but budget is "
                f"{randkv_config.budget}; increase the budget"
            )
        self.layer_idx = layer_idx
        self.prompt_length = prompt_length
        self.policy = RandomEvictionPolicy(randkv_config)
        self.request_id = request_id
        self.absolute_length = 0
        self.eviction_index = 0
        self._has_evicted = False
        self._tokens_since_eviction = 0
        self._positions: tuple[tuple[int, ...], ...] = ()

    @property
    def randkv_config(self) -> RandKVConfig:
        return self.policy.config

    @property
    def positions(self) -> tuple[tuple[int, ...], ...]:
        """Logical token positions retained by each KV head."""

        return self._positions

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_states(key_states, value_states)
        was_initialized = self.is_initialized
        query_length = key_states.shape[-2]
        first_position = self.absolute_length
        new_positions = tuple(range(first_position, first_position + query_length))

        keys, values = super().update(key_states, value_states, *args, **kwargs)
        self.absolute_length += query_length

        num_heads = key_states.shape[1]
        if not self._positions:
            self._positions = tuple(new_positions for _ in range(num_heads))
        else:
            self._positions = tuple(
                previous + new_positions for previous in self._positions
            )

        if not was_initialized:
            # Prefill attention must see the uncompressed prompt. The first
            # decode update can compact it after mask sizing has predicted the
            # smaller physical cache.
            return keys, values

        self._tokens_since_eviction += query_length
        if not self._will_evict(query_length=0):
            return keys, values

        selections = tuple(
            self.policy.select(
                head_positions,
                prompt_length=self.prompt_length,
                layer=self.layer_idx,
                kv_head=head,
                request_id=self.request_id,
                eviction_index=self.eviction_index,
            )
            for head, head_positions in enumerate(self._positions)
        )
        self.keys = self._gather_heads(keys, self._positions, selections)
        self.values = self._gather_heads(values, self._positions, selections)
        self._positions = selections
        self.eviction_index += 1
        self._has_evicted = True
        self._tokens_since_eviction = 0
        return self.keys, self.values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """Predict the physical KV length returned by the next update."""

        if self._will_evict(query_length=query_length):
            return self.randkv_config.budget + self.randkv_config.buffer_size, 0
        return self.get_seq_length() + query_length, 0

    def get_max_length(self) -> int:
        # Absolute generation length is unbounded even though physical KV is
        # bounded. Returning the physical capacity would stop generation.
        return -1

    def _will_evict(self, *, query_length: int) -> bool:
        if not self.is_initialized:
            return False
        prospective_length = self.get_seq_length() + query_length
        capacity = self.randkv_config.budget + self.randkv_config.buffer_size
        if not self._has_evicted:
            return prospective_length > capacity
        if self.randkv_config.buffer_size == 0:
            return prospective_length > self.randkv_config.budget
        return (
            self._tokens_since_eviction + query_length >= self.randkv_config.buffer_size
            and prospective_length > capacity
        )

    @staticmethod
    def _gather_heads(
        states: torch.Tensor,
        previous: tuple[tuple[int, ...], ...],
        selected: tuple[tuple[int, ...], ...],
    ) -> torch.Tensor:
        offsets = []
        for old_positions, new_positions in zip(previous, selected, strict=True):
            position_to_offset = {
                position: offset for offset, position in enumerate(old_positions)
            }
            offsets.append([position_to_offset[position] for position in new_positions])
        indices = torch.tensor(offsets, dtype=torch.long, device=states.device)
        indices = indices.unsqueeze(0).unsqueeze(-1)
        indices = indices.expand(states.shape[0], -1, -1, states.shape[-1])
        return torch.gather(states, dim=2, index=indices)

    @staticmethod
    def _validate_states(key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if key_states.shape != value_states.shape:
            raise ValueError("key and value states must have identical shapes")
        if key_states.ndim != 4:
            raise ValueError("expected cache tensors shaped [batch, heads, seq, dim]")
        if key_states.shape[0] != 1:
            raise NotImplementedError(
                "RandKV's initial Transformers adapter supports batch size 1 only"
            )
        if key_states.shape[1] <= 0:
            raise ValueError("cache tensors must contain at least one KV head")


class RandKVCache(Cache):
    """Transformers ``Cache`` that preserves absolute and physical lengths."""

    def __init__(
        self,
        *,
        model_config: PreTrainedConfig,
        prompt_length: int,
        randkv_config: RandKVConfig | None = None,
        request_id: str | int | bytes = "offline",
    ) -> None:
        text_config = model_config.get_text_config(decoder=True)
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types and any(kind != "full_attention" for kind in layer_types):
            raise NotImplementedError(
                "sliding, chunked, and linear attention layers are not supported"
            )
        num_layers = int(text_config.num_hidden_layers)
        config = randkv_config or RandKVConfig()
        layers: list[CacheLayerMixin | LinearAttentionCacheLayerMixin] = [
            RandKVDynamicLayer(
                layer_idx=layer_idx,
                prompt_length=prompt_length,
                randkv_config=config,
                request_id=request_id,
            )
            for layer_idx in range(num_layers)
        ]
        super().__init__(layers=layers)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return absolute tokens seen, not the compacted tensor length."""

        if layer_idx >= len(self.layers):
            return 0
        layer = self.layers[layer_idx]
        if not isinstance(layer, RandKVDynamicLayer):
            raise TypeError("RandKVCache contains an unexpected cache layer")
        return layer.absolute_length

    def physical_seq_length(self, layer_idx: int = 0) -> int:
        """Return actual retained KV slots for observability and tests."""

        if layer_idx >= len(self.layers):
            return 0
        layer = self.layers[layer_idx]
        if not isinstance(layer, RandKVDynamicLayer):
            raise TypeError("RandKVCache contains an unexpected cache layer")
        return layer.get_seq_length()

    def stats(self) -> RandKVCacheStats:
        """Return an immutable cache-wide metrics snapshot."""

        layers = self._randkv_layers()
        policy_stats = tuple(layer.policy.stats() for layer in layers)
        return RandKVCacheStats(
            absolute_tokens_seen=self.get_seq_length(),
            physical_tokens_per_layer=tuple(layer.get_seq_length() for layer in layers),
            eviction_rounds_per_layer=tuple(layer.eviction_index for layer in layers),
            head_evictions_total=sum(
                snapshot.eviction_events for snapshot in policy_stats
            ),
            positions_evicted_total=sum(
                snapshot.positions_evicted for snapshot in policy_stats
            ),
        )

    def _randkv_layers(self) -> tuple[RandKVDynamicLayer, ...]:
        layers = tuple(self.layers)
        if not all(isinstance(layer, RandKVDynamicLayer) for layer in layers):
            raise TypeError("RandKVCache contains an unexpected cache layer")
        return layers  # type: ignore[return-value]

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if beam_idx.numel() != 1:
            raise NotImplementedError("beam search is not supported by RandKV yet")
        super().reorder_cache(beam_idx)

    def batch_repeat_interleave(self, repeats: int) -> None:
        if repeats != 1:
            raise NotImplementedError(
                "beam search and multiple return sequences are not supported "
                "by RandKV yet"
            )
        super().batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if indices.numel() != 1:
            raise NotImplementedError("batched cache selection is not supported yet")
        super().batch_select_indices(indices)
