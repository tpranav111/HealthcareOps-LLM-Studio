import json
import re

from jsonschema import Draft7Validator


def load_schema(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_json(data, schema):
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    return len(errors) == 0, [e.message for e in errors]


def extract_json(text):
    if not text:
        return ""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


def safe_json_loads(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = extract_json(text)
        if not cleaned:
            return None
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None
