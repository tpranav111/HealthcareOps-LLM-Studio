from src.utils.json_schema import validate_json


def json_validity(response, schema):
    return validate_json(response, schema)[0]


def groundedness(response, corpus_by_id):
    citations = response.get("citations", [])
    if not citations:
        return False
    for cite in citations:
        doc = corpus_by_id.get(cite.get("doc_id"))
        if not doc:
            return False
        snippet = cite.get("snippet", "")
        if snippet and snippet not in doc.get("text", ""):
            return False
    return True


def tool_accuracy(response, expected_tool):
    if not expected_tool:
        return True
    tools = response.get("tool_calls", [])
    return any(tool.get("name") == expected_tool for tool in tools)


def refusal_accuracy(response, expected_refusal):
    return bool(response.get("refusal")) == bool(expected_refusal)
