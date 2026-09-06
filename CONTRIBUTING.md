# Contributing to RandKV

RandKV is intentionally narrow: prompt-protected random KV-cache eviction with
small, explicit adapters for inference frameworks.

Before starting a change:

1. Search open issues and pull requests for overlapping work.
2. Open an issue before implementing a new backend or model family.
3. Keep performance claims reproducible and separate from paper-reported data.

Good starting points are listed under the
[`good first issue`](https://github.com/DaBestCode/randkv/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
label. Comment on the issue before coding; a maintainer will confirm the scope
and assign it when possible. Larger backend and evaluation work is tracked in
[`ROADMAP.md`](ROADMAP.md).

Create the development environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install ".[transformers,test]" build
```

Some Python 3.13 distributions skip hidden `.pth` files produced by editable
build backends. Use a regular local install as shown above; reinstall after
changing package source. CI also installs from the built project rather than
editable metadata.

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

## Pull requests

- Keep one pull request focused on one issue.
- Link the issue with `Closes #<number>` in the description.
- Add or update tests for observable behavior changes.
- Do not include generated benchmark numbers without the raw result artifact.
- Do not claim a speedup from an unsynchronized, single-run, or unmatched test.
- Explain compatibility breaks and update the supported-scope table in the
  README when necessary.

Draft pull requests are welcome when an adapter boundary or schema needs early
review.
