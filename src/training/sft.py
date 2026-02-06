import logging
import os

import mlflow
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
from trl.trainer.sft_config import SFTConfig

from src.training.dataset import load_sft_dataset, format_messages

logger = logging.getLogger(__name__)


def _load_model_and_tokenizer(model_path, qlora_cfg):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    qlora_enabled = qlora_cfg.get("enabled", False)
    qlora_active = False
    if qlora_enabled:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=qlora_cfg.get("use_4bit", True),
                bnb_4bit_quant_type=qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="cpu")
            model = prepare_model_for_kbit_training(model)
            qlora_active = True
        except Exception as exc:
            logger.warning("Falling back to LoRA (QLoRA unavailable): %s", exc)
            model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer, qlora_active


def train_sft(config):
    sft_cfg = config["sft"]
    model, tokenizer, qlora_active = _load_model_and_tokenizer(config["models"]["base_model_path"], sft_cfg["qlora"])

    lora_cfg = LoraConfig(
        r=sft_cfg["lora"]["r"],
        lora_alpha=sft_cfg["lora"]["alpha"],
        lora_dropout=sft_cfg["lora"]["dropout"],
        target_modules=sft_cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    rows = load_sft_dataset(sft_cfg["dataset_paths"], sft_cfg["stage_order"])
    texts = [format_messages(tokenizer, row["messages"]) for row in rows]
    dataset = Dataset.from_dict({"text": texts})

    training_args = SFTConfig(
        output_dir=sft_cfg["output_dir"],
        per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        learning_rate=sft_cfg["learning_rate"],
        max_steps=sft_cfg["max_steps"],
        warmup_steps=sft_cfg["warmup_steps"],
        logging_steps=sft_cfg["logging_steps"],
        save_steps=sft_cfg["save_steps"],
        report_to="none",
        use_cpu=True,
        seed=config["project"]["seed"],
        dataset_text_field="text",
        max_length=sft_cfg["max_seq_len"],
        gradient_checkpointing=sft_cfg.get("gradient_checkpointing", False),
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    mlflow.set_tracking_uri(config["tracking"]["mlflow_uri"])
    mlflow.set_experiment(config["tracking"]["experiment_name"])
    with mlflow.start_run(run_name="sft"):
        mlflow.log_param("qlora_active", qlora_active)
        mlflow.log_params({
            "max_steps": sft_cfg["max_steps"],
            "learning_rate": sft_cfg["learning_rate"],
            "gradient_accumulation_steps": sft_cfg["gradient_accumulation_steps"],
        })
        trainer.train()
        trainer.model.save_pretrained(sft_cfg["output_dir"])
        tokenizer.save_pretrained(sft_cfg["output_dir"])

    upload_cfg = sft_cfg.get("upload") or {}
    if upload_cfg.get("enabled"):
        token = upload_cfg.get("token") or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF token missing. Set HF_TOKEN or provide sft.upload.token.")
        from huggingface_hub import upload_folder

        upload_folder(
            repo_id=upload_cfg["repo_id"],
            folder_path=sft_cfg["output_dir"],
            path_in_repo=upload_cfg.get("subfolder", "adapters/sft"),
            repo_type=upload_cfg.get("repo_type", "model"),
            token=token,
        )

    return sft_cfg["output_dir"]
