import json

from src.utils.json_schema import safe_json_loads

SCHEMA_TEXT_MARKERS = [
    '"$schema"',
    '"properties"',
    '"required"',
    '"type"',
    "draft-07",
]


def _sanitize_citations(citations):
    if not isinstance(citations, list):
        return []
    cleaned = []
    for item in citations:
        if not isinstance(item, dict):
            continue
        doc_id = item.get("doc_id")
        snippet = item.get("snippet")
        if isinstance(doc_id, str) and doc_id and isinstance(snippet, str):
            cleaned.append({"doc_id": doc_id, "snippet": snippet})
    return cleaned


def _sanitize_tool_calls(tool_calls):
    if not isinstance(tool_calls, list):
        return []
    cleaned = []
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        arguments = item.get("arguments")
        if isinstance(name, str) and isinstance(arguments, dict):
            cleaned.append({"name": name, "arguments": arguments})
    return cleaned


def normalize_response(parsed, raw_text, fallback_citations=None):
    fallback_citations = _sanitize_citations(fallback_citations or [])

    candidate = parsed if isinstance(parsed, dict) else {}

    # Handle nested JSON accidentally placed inside `answer` as fenced code/text.
    nested = safe_json_loads(candidate.get("answer")) if isinstance(candidate.get("answer"), str) else None
    if isinstance(nested, dict) and "answer" in nested:
        candidate = nested

    answer = candidate.get("answer")
    if answer is None:
        answer = raw_text.strip() if isinstance(raw_text, str) and raw_text.strip() else "No answer generated."
    elif not isinstance(answer, str):
        answer = json.dumps(answer, ensure_ascii=False)

    citations = _sanitize_citations(candidate.get("citations"))
    if not citations:
        citations = fallback_citations

    tool_calls = _sanitize_tool_calls(candidate.get("tool_calls"))

    refusal = candidate.get("refusal", False)
    refusal = bool(refusal)

    follow_up = candidate.get("follow_up", "")
    if follow_up is None:
        follow_up = ""
    elif not isinstance(follow_up, str):
        follow_up = str(follow_up)

    return {
        "answer": answer,
        "citations": citations,
        "tool_calls": tool_calls,
        "refusal": refusal,
        "follow_up": follow_up,
    }


def is_schema_like_text(value):
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return all(marker in lowered for marker in SCHEMA_TEXT_MARKERS[:3]) or any(
        marker in lowered for marker in SCHEMA_TEXT_MARKERS
    )
