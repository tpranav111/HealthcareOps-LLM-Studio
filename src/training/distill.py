import mlflow
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from src.training.dataset import load_sft_dataset, format_messages


def distill_model(config):
    distill_cfg = config["distill"]
    teacher_path = distill_cfg["teacher_model_path"]
    student_path = distill_cfg["student_model_path"] or config["models"]["base_model_path"]

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_path)

    student_tokenizer = AutoTokenizer.from_pretrained(student_path)
    student_model = AutoModelForCausalLM.from_pretrained(student_path)

    rows = load_sft_dataset([distill_cfg["dataset_path"]], [])
    texts = []
    for row in rows:
        user_messages = [msg for msg in row["messages"] if msg.get("role") != "assistant"]
        prompt = format_messages(teacher_tokenizer, user_messages)
        inputs = teacher_tokenizer(prompt, return_tensors="pt")
        output = teacher_model.generate(**inputs, max_new_tokens=distill_cfg["max_new_tokens"], temperature=distill_cfg["temperature"])
        completion = teacher_tokenizer.decode(output[0], skip_special_tokens=True)
        texts.append(completion)

    dataset = Dataset.from_dict({"text": texts})

    training_args = TrainingArguments(
        output_dir=distill_cfg["output_dir"],
        per_device_train_batch_size=distill_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=distill_cfg["gradient_accumulation_steps"],
        learning_rate=distill_cfg["learning_rate"],
        max_steps=distill_cfg["max_steps"],
        report_to=[],
    )

    trainer = SFTTrainer(
        model=student_model,
        tokenizer=student_tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=training_args,
    )

    mlflow.set_tracking_uri(config["tracking"]["mlflow_uri"])
    mlflow.set_experiment(config["tracking"]["experiment_name"])
    with mlflow.start_run(run_name="distill"):
        mlflow.log_params({
            "max_steps": distill_cfg["max_steps"],
            "learning_rate": distill_cfg["learning_rate"],
        })
        trainer.train()
        trainer.model.save_pretrained(distill_cfg["output_dir"])
        student_tokenizer.save_pretrained(distill_cfg["output_dir"])

    return distill_cfg["output_dir"]
