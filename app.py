import gc
import json
import os
import re
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import requests
import streamlit as st

from src.core.config import deep_merge, load_config
from src.rag.index import build_index
from src.rag.retriever import HybridRetriever, build_citations
from src.serving.model_runner import ModelRunner
from src.serving import prompting as _prompting
from src.tools.registry import default_registry
from src.utils.json_schema import load_schema, safe_json_loads, validate_json
from src.utils import response_format as _response_format

normalize_response = _response_format.normalize_response
is_schema_like_text = getattr(_response_format, "is_schema_like_text", lambda _: False)
build_messages = _prompting.build_messages

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it",
    "of", "on", "or", "that", "the", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "with", "more", "about", "please", "answer", "detail", "details",
}


def _fallback_context_relevance(query, docs, min_overlap_ratio=0.2):
    query_tokens = {
        token for token in re.findall(r"[a-z0-9]+", (query or "").lower()) if token not in _STOPWORDS
    }
    if not query_tokens or not docs:
        return False
    doc_tokens = set()
    for doc in docs[:3]:
        text = doc.get("text", "") if isinstance(doc, dict) else ""
        doc_tokens.update(
            token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in _STOPWORDS
        )
    if not doc_tokens:
        return False
    overlap = len(query_tokens.intersection(doc_tokens))
    return (overlap / max(1, len(query_tokens))) >= min_overlap_ratio


is_context_relevant = getattr(_prompting, "is_context_relevant", _fallback_context_relevance)


def _build_messages_for_query(query_text, docs, tools, schema, relevant):
    if relevant:
        return build_messages(query_text, docs, tools, schema)
    system_builder = getattr(_prompting, "build_system_prompt", None)
    if callable(system_builder):
        system_prompt = system_builder(schema, tools)
    else:
        system_prompt = (
            "You are a healthcare assistant. Return only JSON with keys: "
            "answer, citations, tool_calls, refusal, follow_up."
        )
    user_prompt = (
        "Context:\nNo relevant internal policy context retrieved for this question."
        "\n\nUser question: " + query_text +
        "\nAnswer directly using general knowledge. Set citations to an empty list. Return only JSON."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

st.set_page_config(page_title="Healthcare LLM Demo", layout="wide")

st.title("Healthcare LLM: Baseline vs SFT vs DPO")

use_local = st.toggle("Use local pipeline (no API)", value=True)

api_url = None
serving_config_path = "configs/serving_tiny.yaml"
rag_config_path = "configs/rag_tiny.yaml"
if use_local:
    serving_config_path = st.text_input("Serving config", value=serving_config_path)
    rag_config_path = st.text_input("RAG config", value=rag_config_path)
else:
    api_url = st.text_input("API URL", value="http://localhost:8000/v1/answer")

query = st.text_area("Ask a healthcare ops question", height=120)

@st.cache_resource(show_spinner=False)
def _load_local_pipeline(serving_cfg_path, rag_cfg_path, serving_mtime, rag_mtime):
    config = load_config(serving_cfg_path)
    rag_config = load_config(rag_cfg_path)
    config = deep_merge(config, {"rag": rag_config.get("rag", {})})

    rag_cfg = config.get("rag") or {}
    index_dir = rag_cfg["index_dir"]
    docs_path = os.path.join(index_dir, "docs.jsonl")
    dense_path = os.path.join(index_dir, "dense.index")
    if not (os.path.exists(docs_path) and os.path.exists(dense_path)):
        build_index(
            rag_cfg["corpus_path"],
            config["models"]["embedding_model_path"],
            index_dir,
            normalize=rag_cfg.get("dense", {}).get("normalize", True),
        )
    rerank_cfg = rag_cfg.get("rerank", {})
    retriever = HybridRetriever(
        index_dir=index_dir,
        embedding_model_path=config["models"]["embedding_model_path"],
        alpha=rag_cfg.get("hybrid", {}).get("alpha", 0.55),
        top_k=rag_cfg.get("top_k", 5),
        reranker_model_path=rerank_cfg.get("model_path") if rerank_cfg.get("enabled") else None,
        rerank_top_k=rerank_cfg.get("top_k", 10),
    )
    tools = default_registry(retriever=retriever)
    schema = load_schema(config["serving"]["schema_path"])

    return {
        "config": config,
        "rag_cfg": rag_cfg,
        "retriever": retriever,
        "tools": tools,
        "schema": schema,
    }


def _run_local(pipeline, query_text, variant):
    config = pipeline["config"]
    rag_cfg = pipeline["rag_cfg"]
    retriever = pipeline["retriever"]
    tools = pipeline["tools"]
    schema = pipeline["schema"]
    variants = config["serving"]["variants"]

    params = {
        "max_new_tokens": config["serving"]["max_new_tokens"],
        "temperature": config["serving"]["temperature"],
        "top_p": config["serving"]["top_p"],
    }

    t0 = time.time()
    retrieval = retriever.retrieve(query_text, top_k=rag_cfg.get("top_k", 5))
    t1 = time.time()
    docs = [item["doc"] for item in retrieval]
    relevant = is_context_relevant(query_text, docs)
    citations = build_citations(retrieval, max_chars=rag_cfg.get("cite_snippet_chars", 260)) if relevant else []
    messages = _build_messages_for_query(query_text, docs, tools.list_tools(), schema, relevant)

    variant_cfg = variants.get(variant) or variants.get("baseline")
    runner = None
    raw_text = ""
    parsed = None
    try:
        runner = ModelRunner(variant_cfg["model_path"], variant_cfg.get("adapter_path") or None)
        raw_text = runner.generate(messages, **params)
        parsed = safe_json_loads(raw_text)
        if parsed is None:
            # Retry once with a stricter JSON-only instruction.
            retry_messages = [
                {"role": "system", "content": "Return ONLY valid JSON that matches the schema. No prose."},
                *messages,
            ]
            raw_text = runner.generate(retry_messages, **params)
            parsed = safe_json_loads(raw_text)
        response = normalize_response(parsed, raw_text, citations)
        if is_schema_like_text(response.get("answer", "")):
            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "Do NOT output JSON schema descriptors like $schema, type, properties, or required. "
                        "Return only a JSON object with keys answer, citations, tool_calls, refusal, follow_up."
                    ),
                },
                *messages,
            ]
            raw_text = runner.generate(repair_messages, **params)
            parsed = safe_json_loads(raw_text)
        t2 = time.time()
    finally:
        if runner is not None:
            del runner
        gc.collect()

    response = normalize_response(parsed, raw_text, citations)
    valid, _ = validate_json(response, schema)
    if not valid:
        response = normalize_response(None, raw_text, citations)

    return {
        "response": response,
        "citations": citations,
        "trace": {
            "retrieval_ms": int((t1 - t0) * 1000),
            "generation_ms": int((t2 - t1) * 1000),
        },
        "raw_text": raw_text,
    }


if st.button("Run") and query:
    cols = st.columns(3)
    variants = ["baseline", "sft", "dpo"]
    if use_local:
        serving_mtime = os.path.getmtime(serving_config_path)
        rag_mtime = os.path.getmtime(rag_config_path)
        pipeline = _load_local_pipeline(serving_config_path, rag_config_path, serving_mtime, rag_mtime)
    for idx, variant in enumerate(variants):
        with cols[idx]:
            st.subheader(variant.upper())
            try:
                if use_local:
                    payload = _run_local(pipeline, query, variant)
                    st.code(json.dumps(payload.get("response", {}), indent=2), language="json")
                    st.caption(
                        f"Retrieval: {payload['trace']['retrieval_ms']} ms | "
                        f"Generation: {payload['trace']['generation_ms']} ms"
                    )
                else:
                    resp = requests.post(api_url, json={"query": query, "variant": variant}, timeout=60)
                    payload = resp.json()
                    st.code(json.dumps(payload.get("response", {}), indent=2), language="json")
                    st.caption(f"Trace ID: {payload.get('trace_id')}")
            except Exception as exc:
                st.error(f"Request failed: {exc}")
