# RandKV

Deterministic, prompt-protected random KV-cache eviction for reasoning models.

[![CI](https://github.com/DaBestCode/randkv/actions/workflows/ci.yml/badge.svg)](https://github.com/DaBestCode/randkv/actions/workflows/ci.yml)

> **Status: early Hugging Face milestone.** The framework-independent policy
> and a batch-size-one Transformers 5.16 cache adapter are implemented. vLLM,
> batched generation, and optimized kernels are not implemented; no throughput
> claim is made yet.

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

```bash
pip install "randkv[transformers]"
```

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


## Test

The policy core has no runtime dependencies:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Method

RandKV follows *Random Attention*: the original prompt is never evicted, while
older generated positions are sampled uniformly and independently per KV head.
A recent buffer is excluded from selection until the next eviction event.

- [Paper](https://arxiv.org/abs/2609.03430)
- [Authors' reference implementation](https://github.com/SalesforceAIResearch/Random-Attention)

RandKV is an independent packaging and integration project. It is not an
official Salesforce project.
