import hashlib
import json
import logging
import re
import time

from diskcache import Cache
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram

from src.rag.retriever import HybridRetriever, build_citations
from src.serving.batcher import RequestBatcher
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

REQUEST_COUNT = Counter("requests_total", "Total requests", ["variant"])
SCHEMA_PASS = Counter("schema_pass_total", "Schema pass rate", ["variant"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency", ["variant"])
HALLUCINATION_COUNT = Counter("hallucination_total", "Ungrounded response count", ["variant"])

logger = logging.getLogger("serving")


class InferenceRequest(BaseModel):
    query: str
    variant: str = "baseline"
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


class InferenceResponse(BaseModel):
    variant: str
    response: dict
    citations: list
    trace_id: str
    trace: dict
    raw_text: str


def _hash_key(payload):
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def create_app(config):
    app = FastAPI(title="Healthcare LLM Serving")

    rag_cfg = config.get("rag") or {}
    rerank_cfg = rag_cfg.get("rerank", {})
    retriever = HybridRetriever(
        index_dir=rag_cfg["index_dir"],
        embedding_model_path=config["models"]["embedding_model_path"],
        alpha=rag_cfg.get("hybrid", {}).get("alpha", 0.55),
        top_k=rag_cfg.get("top_k", 5),
        reranker_model_path=rerank_cfg.get("model_path") if rerank_cfg.get("enabled") else None,
        rerank_top_k=rerank_cfg.get("top_k", 10),
    )
    tools = default_registry(retriever=retriever)
    schema = load_schema(config["serving"]["schema_path"])

    variants = config["serving"]["variants"]
    runners = {name: None for name in variants}

    def get_runner(variant_name):
        if runners[variant_name] is None:
            variant_cfg = variants[variant_name]
            runners[variant_name] = ModelRunner(variant_cfg["model_path"], variant_cfg.get("adapter_path") or None)
        return runners[variant_name]

    cache = Cache(config["serving"]["cache_dir"])
    batch_cfg = config["serving"].get("batch", {})
    batcher = None
    if batch_cfg.get("enabled", False):
        batcher = RequestBatcher(
            runner=get_runner("baseline"),
            max_batch_size=batch_cfg.get("max_batch_size", 4),
            max_wait_ms=batch_cfg.get("max_wait_ms", 50),
        )

    @app.post("/v1/answer", response_model=InferenceResponse)
    def answer(request: InferenceRequest):
        trace_id = _hash_key({"query": request.query, "t": time.time()})[:12]
        variant = request.variant if request.variant in runners else "baseline"
        REQUEST_COUNT.labels(variant=variant).inc()

        params = {
            "max_new_tokens": request.max_new_tokens or config["serving"]["max_new_tokens"],
            "temperature": request.temperature if request.temperature is not None else config["serving"]["temperature"],
            "top_p": request.top_p if request.top_p is not None else config["serving"]["top_p"],
        }

        cache_key = _hash_key({"query": request.query, "variant": variant, "params": params})
        if cache_key in cache:
            cached = cache[cache_key]
            return InferenceResponse(**cached)

        with REQUEST_LATENCY.labels(variant=variant).time():
            t0 = time.time()
            retrieval = retriever.retrieve(request.query, top_k=rag_cfg.get("top_k", 5))
            t1 = time.time()
            docs = [item["doc"] for item in retrieval]
            relevant = is_context_relevant(request.query, docs)
            citations = build_citations(retrieval, max_chars=rag_cfg.get("cite_snippet_chars", 260)) if relevant else []
            messages = _build_messages_for_query(request.query, docs, tools.list_tools(), schema, relevant)

            runner = get_runner(variant)
            if batcher and variant == "baseline":
                raw_text = batcher.submit(messages, params)
            else:
                raw_text = runner.generate(messages, **params)
            t2 = time.time()

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
            response = normalize_response(parsed, raw_text, citations)
        valid, _ = validate_json(response, schema)
        if valid:
            SCHEMA_PASS.labels(variant=variant).inc()
        else:
            response = normalize_response(None, raw_text, citations)

        if not response.get("refusal") and not response.get("citations"):
            HALLUCINATION_COUNT.labels(variant=variant).inc()

        logger.info(
            "trace_id=%s variant=%s retrieval_ms=%d generation_ms=%d",
            trace_id,
            variant,
            int((t1 - t0) * 1000),
            int((t2 - t1) * 1000),
        )

        payload = {
            "variant": variant,
            "response": response,
            "citations": citations,
            "trace_id": trace_id,
            "trace": {
                "retrieval_ms": int((t1 - t0) * 1000),
                "generation_ms": int((t2 - t1) * 1000),
            },
            "raw_text": raw_text,
        }
        cache[cache_key] = payload
        return InferenceResponse(**payload)

    return app
