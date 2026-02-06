import time

from src.core.guardrails import detect_prompt_injection, refusal_response, should_refuse
from src.rag.retriever import HybridRetriever, build_citations
from src.serving.model_runner import ModelRunner
from src.serving.prompting import build_messages
from src.tools.registry import default_registry
from src.utils.io import load_jsonl, write_json
from src.utils.json_schema import load_schema, safe_json_loads
from src.eval.metrics import groundedness, json_validity, refusal_accuracy, tool_accuracy


def evaluate(config, rag_config):
    eval_cfg = config["eval"]
    rerank_cfg = rag_config.get("rerank", {})
    retriever = HybridRetriever(
        index_dir=rag_config["index_dir"],
        embedding_model_path=config["models"]["embedding_model_path"],
        alpha=rag_config.get("hybrid", {}).get("alpha", 0.55),
        top_k=rag_config.get("top_k", 5),
        reranker_model_path=rerank_cfg.get("model_path") if rerank_cfg.get("enabled") else None,
        rerank_top_k=rerank_cfg.get("top_k", 10),
    )
    tools = default_registry(retriever=retriever)
    schema = load_schema(eval_cfg["schema_path"])

    adapter_path = eval_cfg.get("adapter_path") or None
    runner = ModelRunner(config["models"]["base_model_path"], adapter_path)

    eval_rows = load_jsonl(eval_cfg["dataset_path"])
    corpus_by_id = {doc["doc_id"]: doc for doc in retriever.index.docs}

    results = []
    totals = {
        "json_valid": 0,
        "grounded": 0,
        "tool_accuracy": 0,
        "refusal_accuracy": 0,
    }

    for row in eval_rows:
        query = row["input"]
        expected = row.get("expected", {})
        start = time.time()

        if detect_prompt_injection(query):
            response = refusal_response("Prompt injection detected.")
            raw_text = ""
            citations = []
        elif should_refuse(query, eval_cfg.get("allow_medical_advice", False)):
            response = refusal_response("I cannot provide medical diagnoses or clinical advice.")
            raw_text = ""
            citations = []
        else:
            retrieval = retriever.retrieve(query, top_k=rag_config.get("top_k", 5))
            citations = build_citations(retrieval, max_chars=rag_config.get("cite_snippet_chars", 260))
            messages = build_messages(query, [item["doc"] for item in retrieval], tools.list_tools(), schema)
            raw_text = runner.generate(
                messages,
                max_new_tokens=eval_cfg["max_new_tokens"],
                temperature=eval_cfg["temperature"],
                top_p=eval_cfg["top_p"],
            )
            parsed = safe_json_loads(raw_text)
            response = parsed if parsed is not None else refusal_response("Invalid JSON output.")

        valid = json_validity(response, schema)
        grounded = groundedness(response, corpus_by_id) if expected.get("require_citations", False) else True
        tool_ok = tool_accuracy(response, expected.get("expected_tool", ""))
        refusal_ok = refusal_accuracy(response, expected.get("refusal", False))

        totals["json_valid"] += int(valid)
        totals["grounded"] += int(grounded)
        totals["tool_accuracy"] += int(tool_ok)
        totals["refusal_accuracy"] += int(refusal_ok)

        results.append({
            "input": query,
            "response": response,
            "raw_text": raw_text,
            "citations": citations,
            "metrics": {
                "json_valid": valid,
                "grounded": grounded,
                "tool_accuracy": tool_ok,
                "refusal_accuracy": refusal_ok,
                "latency_ms": int((time.time() - start) * 1000),
            },
        })

    total = max(1, len(eval_rows))
    summary = {
        "json_valid_rate": totals["json_valid"] / total,
        "grounded_rate": totals["grounded"] / total,
        "tool_accuracy_rate": totals["tool_accuracy"] / total,
        "refusal_accuracy_rate": totals["refusal_accuracy"] / total,
    }

    write_json(eval_cfg["output_path"], {"summary": summary, "results": results})
    return summary
