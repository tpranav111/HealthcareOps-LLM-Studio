from collections import defaultdict

from src.utils.io import load_jsonl


def load_sft_dataset(paths, stage_order):
    examples = []
    stage_rank = {stage: idx for idx, stage in enumerate(stage_order)}
    for path in paths:
        rows = load_jsonl(path)
        for row in rows:
            stage = row.get("metadata", {}).get("stage", "")
            row["_stage_rank"] = stage_rank.get(stage, 999)
            examples.append(row)
    examples.sort(key=lambda x: x.get("_stage_rank", 999))
    return examples


def load_dpo_dataset(path):
    return load_jsonl(path)


def format_messages(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    chunks = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        chunks.append(f"{role}: {content}")
    return "\n".join(chunks)
