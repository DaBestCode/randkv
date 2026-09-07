import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "benchmark-v1.schema.json"
QWEN_RESULT_PATH = ROOT / "results" / "qwen3-0.6b-mps-microbenchmark.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_qwen3_result_matches_benchmark_v1_schema():
    schema = load_json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)

    data = load_json(QWEN_RESULT_PATH)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    assert errors == [], [e.message for e in errors]


def required_fields():
    schema = load_json(SCHEMA_PATH)
    return schema.get("required", [])


@pytest.mark.parametrize("missing_field", required_fields())
def test_missing_required_fields_are_rejected(missing_field):
    schema = load_json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)

    data = load_json(QWEN_RESULT_PATH)
    candidate = copy.deepcopy(data)
    candidate.pop(missing_field, None)

    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(candidate))
    assert errors, f"Expected validation failure when '{missing_field}' is removed"