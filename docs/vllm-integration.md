# vLLM integration boundary

This design targets vLLM `v0.28.0`, commit
`2cf0a6915ce544dc493a0990f2ea38d81601128a`. The integration uses internal
V1 scheduler and model-runner APIs, so this pin is part of the contract rather
than a statement of compatibility with later vLLM releases.

## Call site

The intended upstream call site adds one option and keeps RandKV defaults:

```python
llm = LLM(model="Qwen/Qwen3-4B", kv_cache_policy="randkv")
```

An explicit experiment can still use one structured option:

```python
llm = LLM(
    model="Qwen/Qwen3-4B",
    kv_cache_policy={"type": "randkv", "budget": 2048, "buffer_size": 64},
)
```

This option does not exist in upstream vLLM `v0.28.0`; adding it requires a
small upstream patch. RandKV must not monkey-patch a running engine.

## Why block eviction is insufficient

vLLM allocates paged KV blocks per cache group, and one block table is shared
by every KV head in that group. RandKV selects tokens independently per KV
head. Selecting or freeing whole blocks would therefore force every head to
retain the same tokens and would no longer implement the paper's policy.

The runtime must instead allocate dense destination blocks and compact each
head independently into those blocks. Source blocks can be released only after
every head has been copied. The CPU-only `VLLMCompactionPlanner` produces the
logical retained positions and exact source offsets for that operation. A
PyTorch fallback and a CUDA kernel must consume the same plan.

## vLLM mapping

| RandKV concept | vLLM `v0.28.0` mapping |
| --- | --- |
| Stable request identity | `Request.request_id`; never scheduler order or a Python hash |
| Prompt boundary | Original prompt-token count captured before chunked prefill |
| Absolute position | `Request.num_computed_tokens`; remains monotonic after compaction |
| Physical context length | New per-request retained-token count used by attention metadata |
| Eviction cadence | Plan after decode append when physical length exceeds `K + r`; subsequent rounds occur after `r` appended tokens |
| Per-head identity | Global KV-head index: tensor-parallel head offset plus local head index |
| Block allocation | `KVCacheManager` reserves exclusive destination blocks before source blocks are released |
| Worker operation | `SchedulerOutput` carries a compaction descriptor; the model runner applies copies before the next attention step |
| Observability | Scheduler counts plans/blocks; worker reports copied and evicted token-head pairs |

The narrow hook spans `KVCacheManager.allocate_slots`, `SchedulerOutput`, and
the model runner's block-table/attention-metadata preparation. These are
internal APIs and therefore high upgrade risk. A scheduler-only plugin cannot
work because current vLLM uses `num_computed_tokens` for allocation progress
while attention also needs the smaller physical retained length.

## Lifecycle and unsupported modes

- **Prefill:** never compact an incomplete prompt. With chunked prefill, wait
  until the entire prompt is computed and reject prompts larger than `budget`.
- **Prefix caching:** shared prompt blocks may remain read-only. The compacted
  generated suffix must use copy-on-write destination blocks and must not enter
  the prefix hash cache. A partial prompt-boundary block must be copied before
  packing generated tokens.
- **Cancellation:** discard planner state and free destination plus remaining
  source blocks through the normal request cleanup path. A partially applied
  plan must be idempotently recoverable or fail the engine step.
- **Multi-request scheduling:** keep request-local absolute length, eviction
  index, positions, and destination blocks. Determinism comes from the stable
  request ID, not execution order.
- **Tensor parallelism:** pass each rank's global KV-head offset to the planner.
  Pipeline stages must use global layer indices for the same reason.
- **Preemption/recompute:** the first runtime milestone should reject or disable
  preemption for compacted requests. Reconstructing a non-contiguous cache from
  token IDs would otherwise require replaying the retention history.
- **Speculative decoding, beam search, hybrid/sliding attention, KV transfer,
  and disaggregated serving:** explicitly unsupported in the first integration.

## Runtime implementation sequence

1. Add the single `kv_cache_policy` configuration field and validation.
2. Add request-local physical-length and retention metadata without changing
   monotonic `num_computed_tokens`.
3. Reserve exclusive destination blocks and send a typed compaction descriptor
   in `SchedulerOutput`.
4. Implement the PyTorch copy fallback and verify it against planner offsets.
5. Update block tables and attention sequence lengths atomically after copies.
6. Add cancellation and two-concurrent-request integration tests.
7. Add the CUDA compactor behind the same descriptor, then run Issue #8's
   matched serving benchmark.

The upstream source anchors for this design are
`vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/sched/output.py`, and the V1
GPU model runner at the pinned commit.
