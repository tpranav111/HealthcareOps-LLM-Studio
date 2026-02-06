import json

from src.utils.io import load_jsonl
from src.utils.json_schema import load_schema, validate_json


def test_sft_outputs_match_schema():
    schema = load_schema("data/schemas/response_schema.json")
    rows = load_jsonl("data/sample/sft_train.jsonl")
    for row in rows:
        assistant = [msg for msg in row["messages"] if msg["role"] == "assistant"][-1]
        payload = json.loads(assistant["content"])
        valid, errors = validate_json(payload, schema)
        assert valid, errors


def test_dpo_chosen_match_schema():
    schema = load_schema("data/schemas/response_schema.json")
    rows = load_jsonl("data/sample/dpo_train.jsonl")
    for row in rows:
        payload = json.loads(row["chosen"])
        valid, errors = validate_json(payload, schema)
        assert valid, errors
