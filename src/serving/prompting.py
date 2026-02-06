import json


def build_system_prompt(schema, tools):
    tool_lines = []
    for tool in tools:
        tool_lines.append(f"- {tool['name']}: {tool['description']}")
    tool_block = "\n".join(tool_lines) if tool_lines else "- none"
    schema_text = json.dumps(schema, indent=2)
    return (
        "You are a healthcare operations assistant for internal staff. "
        "Provide medical guidance only at a general informational level. "
        "Do not claim to diagnose or replace a clinician. "
        "If medical guidance is provided, include a short disclaimer in follow_up. "
        "Answer using only the provided context and cite sources by doc_id. "
        "Output must be valid JSON matching this schema:\n" + schema_text +
        "\nAvailable tools:\n" + tool_block +
        "\nIf a tool is required, include it in tool_calls with arguments."
    )


def build_context(docs):
    lines = []
    for doc in docs:
        lines.append(f"[{doc['doc_id']}] {doc['text']}")
    return "\n".join(lines)


def build_messages(query, docs, tools, schema):
    system_prompt = build_system_prompt(schema, tools)
    context = build_context(docs)
    user_prompt = (
        "Context:\n" + context +
        "\n\nUser question: " + query +
        "\nReturn only JSON."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
