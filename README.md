# LLM Fine-Tuning Prototype (Healthcare Domain)

This repo is a production-grade, end-to-end prototype for domain-specific LLM fine-tuning. It combines a baseline RAG system with hybrid retrieval, QLoRA-based SFT, preference optimization (DPO/ORPO), and optional distillation. It also includes evaluation, serving, observability, and a demo UI.

Domain: healthcare operations and clinical documentation support.

## Highlights
- Baseline RAG with BM25 + dense retrieval, optional reranking, citations, and strict JSON outputs.
- Fine-tuning with QLoRA, curriculum (single-turn -> multi-turn -> tool calls -> hard negatives).
- Preference optimization (DPO/ORPO) for grounded answers and valid tool calls.
- Optional distillation for CPU latency.
- Evaluation harness with regression tests for JSON validity, groundedness, tool accuracy, and safety.
- FastAPI serving with batching, streaming, caching, guardrails, and metrics/tracing.
- Demo UI comparing baseline RAG vs SFT vs SFT+DPO.

## Quick start (CPU-friendly)
1) Create a venv and install deps:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2) Build the hybrid index:

```powershell
python scripts\build_index.py --config configs\rag.yaml
```

3) Run baseline server:

```powershell
python scripts\run_server.py --config configs\serving.yaml
```

4) Train SFT (CPU-safe fallback if QLoRA unsupported):

```powershell
python scripts\train_sft.py --config configs\sft.yaml
```

5) Train DPO:

```powershell
python scripts\train_dpo.py --config configs\dpo.yaml
```

6) Evaluate:

```powershell
python scripts\eval.py --config configs\eval.yaml
```

7) Launch demo:

```powershell
streamlit run demo\app.py
```

## Local model paths
This project defaults to local models stored in the Hugging Face cache. Update paths in `configs\base.yaml` as needed.

## Repository layout
- `configs/`: YAML configs for RAG, SFT, DPO, distillation, serving, and eval.
- `data/`: sample datasets, schemas, and dataset versioning.
- `src/`: core library (RAG, training, eval, serving, tools, observability).
- `scripts/`: CLI entrypoints.
- `tests/`: regression tests for CI.
- `demo/`: UI comparing baseline vs tuned models.
- `REPORT.md`: evaluation report template.

## Notes
- CPU-only training is slow. Start with the smallest model and tiny datasets.
- QLoRA requires bitsandbytes. If unavailable on CPU, the training script falls back to LoRA.
- Reranking is optional and can be disabled in `configs\rag.yaml`.
- Sample data is synthetic and avoids PHI. Keep real data de-identified and approved.
