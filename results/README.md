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

