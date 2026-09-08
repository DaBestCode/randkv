# RandKV

Deterministic, prompt-protected random KV-cache eviction for reasoning models.

[![CI](https://github.com/DaBestCode/randkv/actions/workflows/ci.yml/badge.svg)](https://github.com/DaBestCode/randkv/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/randkv.svg)](https://pypi.org/project/randkv/)
[![Python](https://img.shields.io/pypi/pyversions/randkv.svg)](https://pypi.org/project/randkv/)

## 60-second quickstart

```bash
pip install randkv
```

```python
from randkv import RandomEvictionPolicy

retained = RandomEvictionPolicy().select(range(5000), prompt_length=128)
print(len(retained))  # 2112
```

The zero-config policy keeps the complete prompt, a recent 64-token buffer, and
a deterministic random sample under a persistent 2,048-position budget.

For an already loaded Hugging Face model, the integration call site is:

```python
from randkv.transformers import generate

output = generate(model, **inputs, max_new_tokens=4096)
```

> **Status: v0.1 Hugging Face milestone.** The framework-independent policy
> and a batch-size-one Transformers 5.16 cache adapter are implemented. vLLM,
> batched generation, and optimized kernels are not implemented; no throughput
> claim is made yet.

| Capability | Status |
| --- | --- |
| Dependency-free eviction policy | Supported |
| Transformers 5.16 cache adapter | Supported for batch size one |
| Greedy and sampled generation | Supported |
| Beam search and batched generation | Not yet supported |
| Sliding, chunked, and linear attention | Not yet supported |
| vLLM integration | CPU-tested compaction planner; runtime patch pending |
| Optimized GPU kernels | Not yet implemented |

## Policy quickstart

```python
from randkv import RandomEvictionPolicy

policy = RandomEvictionPolicy()
retained = policy.select(range(5000), prompt_length=128)
```

The zero-config policy keeps the complete prompt, a recent 64-token buffer, and
a deterministic random sample under a persistent 2,048-position budget.

Use an explicit configuration for experiments:

```python
from randkv import RandKVConfig, RandomEvictionPolicy

policy = RandomEvictionPolicy(RandKVConfig(budget=1024, buffer_size=64, seed=42))
heads = policy.select_heads(
    range(4096),
    prompt_length=256,
    num_kv_heads=8,
    layer=0,
    request_id="request-17",
    eviction_index=0,
)
```

Every head receives an independent draw. Seed derivation includes the request,
eviction, layer, and head identity, so concurrent callers do not share mutable
random-number-generator state.

## Hugging Face

Use the one-call generation adapter:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from randkv import RandKVConfig
from randkv.transformers import generate

model_id = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
inputs = tokenizer("Solve x^2 = 9.", return_tensors="pt").to(model.device)
output = generate(model, **inputs, max_new_tokens=4096)
```

For an explicit budget or cache inspection, construct the cache directly:

```python
from randkv.transformers import RandKVCache

cache = RandKVCache(
    model_config=model.config,
    prompt_length=inputs.input_ids.shape[-1],
    randkv_config=RandKVConfig(budget=2048),
)
output = model.generate(**inputs, past_key_values=cache, max_new_tokens=4096)
print(cache.stats())
```

The initial adapter intentionally supports batch size one, greedy/sampling
generation, and full-attention decoder models only. Beam search and models with
sliding, chunked, or linear-attention layers fail explicitly.

### Transformers checkpoint compatibility

The adapter's compatibility boundary is recorded separately from performance
or model-quality claims. A checkpoint is **verified** only when the real smoke
test has been run and its raw result is checked into [`results/`](results/).

| Status | Checkpoint or family | Evidence / boundary |
| --- | --- | --- |
| Verified | `Qwen/Qwen3-0.6B` | Real Transformers smoke result on Apple M4; see [`results/README.md`](results/README.md). |
| Verified | `HuggingFaceTB/SmolLM2-135M` | Real full-attention decoder smoke result with two or more eviction rounds; see [`results/README.md`](results/README.md). |
| Expected, not individually verified | Other decoder-only Transformers checkpoints with full attention, batch size one, and a compatible `Cache` interface | This is an adapter-shape expectation, not a compatibility guarantee. Run a smoke test before relying on a checkpoint. |
| Unsupported | Sliding-window, chunked, linear, or other hybrid-attention layers | The adapter rejects these attention modes explicitly. |
| Unsupported | Beam search and batched generation | The initial adapter supports batch size one only. |

The **expected** row must not be read as a test result: model configuration,
attention layout, and Transformers integration details can still differ between
families. Compatibility evidence is deliberately kept separate from throughput
and quality evaluation.

Run a real-checkpoint smoke test (downloads the model from Hugging Face):

```bash
.venv/bin/python scripts/smoke_transformers.py \
  --model Qwen/Qwen3-0.6B \
  --budget 512 \
  --buffer-size 64 \
  --max-new-tokens 640 \
  --output-json results/qwen3-0.6b-smoke.json
```

The JSON result records exact PyTorch and Transformers versions, device,
budget, buffer, seed, physical cache lengths, eviction counts, and throughput.

Here `budget` is the persistent budget `K`, not the instantaneous tensor size.
Immediately after eviction the cache contains `K + r` positions. Between
rounds it can grow to `K + 2r - 1` before the next `r`-token buffer triggers
compaction.

Recorded smoke results live in [`results/`](results/README.md). They validate
compatibility and eviction invariants; they are not comparative benchmarks.

Run the local dense-versus-RandKV microbenchmark:

```bash
.venv/bin/python scripts/benchmark_transformers.py \
  --model Qwen/Qwen3-0.6B \
  --budget 32 \
  --buffer-size 8 \
  --max-new-tokens 128 \
  --trials 3 \
  --output-json results/qwen3-0.6b-mps-microbenchmark.json
```

This measures single-request adapter overhead. It is not evidence for the
paper's vLLM serving-throughput claim.

| Qwen3-0.6B, Apple M4, 128 generated tokens | Median tokens/s |
| --- | ---: |
| Dense Transformers cache | 38.53 |
| RandKV PyTorch compaction | 35.48 |

The measured RandKV/dense ratio is `0.921x`. Publishing the slower result is
intentional: it isolates current Python gather overhead and prevents a local
microbenchmark from being presented as serving-throughput evidence. The full
machine-readable result and protocol are in [`results/`](results/README.md).

## Test

The policy core has no runtime dependencies:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Architecture

| Fragile DIY integration | RandKV |
| --- | --- |
| Token scoring in the decode path | No importance scores |
| Shared mutable RNG state | Deterministic request/layer/head seeds |
| Physical cache length reused as token position | Absolute position tracked separately from compacted length |
| One retention mask for every KV head | Independent sample per KV head |
| Silent behavior on unsupported attention types | Explicit validation failures |
| Ad hoc benchmark output | Versioned, machine-readable results |

## Method

RandKV follows *Random Attention*: the original prompt is never evicted, while
older generated positions are sampled uniformly and independently per KV head.
A recent buffer is excluded from selection until the next eviction event.

- [Paper](https://arxiv.org/abs/2609.03430)
- [Authors' reference implementation](https://github.com/SalesforceAIResearch/Random-Attention)

RandKV is an independent packaging and integration project. It is not an
official Salesforce project.

## Roadmap and contributing

The official performance launch is gated on a matched quality evaluation and a
reproducible NVIDIA/vLLM serving benchmark. See [`ROADMAP.md`](ROADMAP.md) for
the launch criteria and current work packages.

New contributors can start with a
[`good first issue`](https://github.com/DaBestCode/randkv/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.

The pinned vLLM boundary, runtime lifecycle, and unsupported-mode decisions are
documented in [`docs/vllm-integration.md`](docs/vllm-integration.md).
