# Contributing to RandKV

RandKV is intentionally narrow: prompt-protected random KV-cache eviction with
small, explicit adapters for inference frameworks.

Before starting a change:

1. Search open issues and pull requests for overlapping work.
2. Open an issue before implementing a new backend or model family.
3. Keep performance claims reproducible and separate from paper-reported data.

Create the development environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[transformers,test]" build
```

Run the required checks:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src
.venv/bin/pytest -q
.venv/bin/python -m build --no-isolation
```

Backend changes must include tests for prompt retention, deterministic
selection, cache bounds, and unsupported input behavior. GPU optimizations must
retain a tested PyTorch fallback.

