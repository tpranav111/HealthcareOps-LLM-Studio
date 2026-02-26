import re


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it",
    "of", "on", "or", "that", "the", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "with", "more", "about", "please", "answer", "detail", "details",
}


def _tokenize(text):
    return [token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if token not in STOPWORDS]


def is_context_relevant(query, docs, min_overlap_ratio=0.2):
    query_tokens = set(_tokenize(query))
    if not query_tokens or not docs:
        return False
    doc_tokens = set()
    for doc in docs[:3]:
        doc_tokens.update(_tokenize(doc.get("text", "")))
    if not doc_tokens:
        return False
    overlap = len(query_tokens.intersection(doc_tokens))
    return (overlap / max(1, len(query_tokens))) >= min_overlap_ratio


def build_system_prompt(schema, tools):
    tool_lines = []
    for tool in tools:
        tool_lines.append(f"- {tool['name']}: {tool['description']}")
    tool_block = "\n".join(tool_lines) if tool_lines else "- none"
    _ = schema
    return (
        "You are a healthcare operations assistant for internal staff. "
        "Provide medical guidance only at a general informational level. "
        "Do not claim to diagnose or replace a clinician. "
        "If medical guidance is provided, include a short disclaimer in follow_up. "
        "Prefer the provided context and cite sources by doc_id when that context is relevant. "
        "If context is not relevant to the question, answer using general knowledge and keep citations empty. "
        "Return ONLY a JSON object with exactly these keys: "
        "answer (string), citations (array of {doc_id, snippet}), "
        "tool_calls (array of {name, arguments}), refusal (boolean), follow_up (string). "
        "Never output markdown code fences. "
        "Never output JSON schema descriptors such as $schema, type, properties, or required. "
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
    relevant = is_context_relevant(query, docs)
    if relevant:
        context = build_context(docs)
        user_prompt = (
            "Context:\n" + context +
            "\n\nUser question: " + query +
            "\nReturn only JSON."
        )
    else:
        user_prompt = (
            "Context:\nNo relevant internal policy context retrieved for this question."
            "\n\nUser question: " + query +
            "\nAnswer directly using general knowledge. Set citations to an empty list. Return only JSON."
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
