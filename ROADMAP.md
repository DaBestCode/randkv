# RandKV roadmap

RandKV uses evidence gates rather than dates for performance claims. Version
0.1.0 is an installable Transformers milestone, not a validation of the paper's
vLLM throughput result.

## Official launch gate

- [x] Publish a dependency-free policy core and Transformers adapter.
- [x] Release artifacts through PyPI trusted publishing.
- [x] Run a real-checkpoint Qwen3 smoke test.
- [x] Publish a repeated dense-versus-RandKV local microbenchmark.
- [ ] Validate benchmark artifacts against a versioned schema.
- [ ] Add an offline end-to-end Transformers integration test.
- [ ] Measure task quality at matched dense and RandKV token budgets.
- [x] Implement and test a vLLM-compatible policy/backend boundary.
- [ ] Benchmark dense, random eviction, and at least one scored selector on an
      NVIDIA GPU under the same model, request distribution, and cache budget.
- [ ] Publish peak-memory, throughput, latency, and quality results with raw
      machine-readable data.

No RandKV throughput claim should be made until the last three items are
complete. Paper-reported results must always be attributed to the paper.

## Work streams

### Correctness

- Expand deterministic property tests across seeds, layers, heads, and eviction
  rounds.
- Add offline model-level generation coverage.
- Build a small compatibility matrix for full-attention decoder models.

### Evaluation

- Version and validate benchmark result files.
- Add matched-budget quality evaluation with exact model revisions and dataset
  versions.
- Record warm-up, synchronization, request ordering, and hardware metadata.

### Serving

- Define the narrowest vLLM integration boundary before adding kernels.
- Preserve a tested PyTorch fallback.
- Optimize per-head compaction only after profiling identifies the bottleneck.

### Developer experience

- Keep installation and first generation copy-pasteable.
- Add a deterministic SVG retention visualization.
- Turn supported model families and failure modes into explicit documentation.

## Choosing work

Issues labeled `good first issue` are self-contained and should not require GPU
access. Issues labeled `needs GPU` must include the hardware and software stack
in every reported result. Comment on an issue before starting significant work
so maintainers can prevent duplicated effort.
