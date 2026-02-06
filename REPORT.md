# Evaluation Report (Healthcare LLM Prototype)

## Goal
Summarize improvements from baseline RAG to SFT and SFT+DPO in terms of quality, safety, and latency.

## Model Variants
- Baseline: RAG + base model
- SFT: RAG + SFT adapter
- DPO: RAG + DPO adapter

## Datasets
- Training: data/sample/sft_train.jsonl
- Preference: data/sample/dpo_train.jsonl
- Eval: data/sample/eval.jsonl

## Metrics
- JSON validity rate
- Groundedness rate (citation match)
- Tool accuracy rate
- Safety/refusal accuracy
- Latency (ms)

## Results Summary
| Variant | JSON Valid | Grounded | Tool Acc | Safety Acc | Avg Latency |
|---------|------------|----------|----------|------------|-------------|
| Baseline | | | | | |
| SFT | | | | | |
| DPO | | | | | |

## Notes
- Include examples of improved grounded answers and fewer invalid JSON outputs.
- Record any failure modes or safety issues.

