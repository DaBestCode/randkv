## Summary

<!-- What behavior changes, and why? -->

Closes #

## Verification

- [ ] `ruff check .`
- [ ] `ruff format --check .`
- [ ] `mypy src`
- [ ] `pytest -q`
- [ ] `python -m build --no-isolation`

## Evidence and compatibility

- [ ] Tests cover the changed behavior and failure modes.
- [ ] Public API or supported-scope documentation is updated if needed.
- [ ] Benchmark claims include raw results, protocol, hardware, and exact model
      revision, or this pull request makes no performance claim.
- [ ] The base `randkv` install remains dependency-free, or the dependency
      change is explicitly justified.
