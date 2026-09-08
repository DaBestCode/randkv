# Recorded smoke tests

These files prove that a released RandKV revision ran against a public model
checkpoint with real attention tensors. They are not performance benchmarks:
there is no dense or scored-eviction baseline, warm-up protocol, or repeated
trial distribution.

## Qwen3-0.6B on Apple M4

- Result: [`qwen3-0.6b-mps-smoke.json`](qwen3-0.6b-mps-smoke.json)
- Model revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- Prompt tokens: 16
- Generated tokens: 64
- Persistent budget `K`: 32
- Buffer `r`: 8
- Eviction rounds: 5 per layer
- Final physical cache: 46 positions per layer

The final physical size is between eviction boundaries. The periodic policy
returns to `K + r = 40` positions after each round and may grow as high as
`K + 2r - 1 = 47` before the next compaction.

## Dense vs RandKV Transformers microbenchmark

- Result: [`qwen3-0.6b-mps-microbenchmark.json`](qwen3-0.6b-mps-microbenchmark.json)
- Protocol: one warm-up per mode, three alternating-order trials, 128 generated
  tokens per trial, synchronized MPS timing
- Dense median: 38.53 tokens/second
- RandKV median: 35.48 tokens/second
- RandKV/dense throughput: 0.921x
- Final RandKV physical cache: 46 positions per layer after 143 total tokens

This single-request Apple M4 measurement is compatibility and adapter-overhead
evidence only. The current Python gather path is slower than dense generation in
this test. It is not a vLLM serving benchmark, a CUDA-kernel benchmark, or a
model-quality evaluation, and it does not validate the paper's throughput claim.

## Schema compatibility

Within schema v1, additive optional fields remain compatible. Removing or
renaming fields, or changing their semantic meaning, requires a new schema
version.

## SmolLM2-135M on Windows CPU

- Result: [`smollm2-135m-smoke.json`](smollm2-135m-smoke.json)
- Model revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- Environment: Windows 11, AMD64, Python 3.13.5, PyTorch 2.14.0+cpu,
  Transformers 5.16.1
- Hardware: 13th Gen Intel Core i7-13620H, 13.0 GB RAM
- Device: CPU
- Prompt tokens: 11
- Generated tokens: 128
- Persistent budget `K`: 64
- Buffer `r`: 8
- Eviction rounds: 9 per layer
- Final physical cache: 73 positions per layer

This smoke run verifies the Transformers adapter against the
`HuggingFaceTB/SmolLM2-135M` full-attention decoder configuration. It records
compatibility and eviction behavior only; it is not a throughput comparison,
quality evaluation, or claim about the model family as a whole.
