# Changelog

All notable changes to RandKV will be documented here.

The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-09-06

### Added

- Framework-independent, deterministic random KV-eviction policy.
- Prompt and recent-buffer protection.
- Per-layer and per-KV-head seed derivation.
- Transformers 5.16 cache adapter for unpadded, batch-size-one generation.
- Cache observability snapshots.
- Real-checkpoint smoke and microbenchmark tooling.
- Python 3.10 and 3.13 continuous integration.

### Limitations

- No vLLM backend yet.
- No batched or beam-search generation.
- No sliding, chunked, or linear-attention layers.
- Tensor compaction uses PyTorch rather than optimized CUDA/Triton kernels.
