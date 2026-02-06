import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _split_adapter_ref(adapter_path: str):
    if "::" in adapter_path:
        repo_id, subfolder = adapter_path.split("::", 1)
        return repo_id.strip(), subfolder.strip()
    return adapter_path, None


class ModelRunner:
    def __init__(self, model_path, adapter_path=None):
        hf_token = os.environ.get("HF_TOKEN")
        hf_kwargs = {"token": hf_token} if hf_token else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **hf_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **hf_kwargs)
        if adapter_path:
            if os.path.isdir(adapter_path):
                self.model = PeftModel.from_pretrained(self.model, adapter_path, **hf_kwargs)
            else:
                repo_id, subfolder = _split_adapter_ref(adapter_path)
                if subfolder:
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        repo_id,
                        subfolder=subfolder,
                        **hf_kwargs,
                    )
                else:
                    self.model = PeftModel.from_pretrained(self.model, repo_id, **hf_kwargs)
        self.model.eval()

    def _format_messages(self, messages):
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chunks = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            chunks.append(f"{role}: {content}")
        chunks.append("assistant:")
        return "\n".join(chunks)

    def generate(self, messages, max_new_tokens=256, temperature=0.2, top_p=0.9):
        prompt = self._format_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )
        generated = outputs[0][input_len:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text

    def generate_batch(self, messages_list, max_new_tokens=256, temperature=0.2, top_p=0.9):
        prompts = [self._format_messages(messages) for messages in messages_list]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )
        texts = []
        for output, input_len in zip(outputs, input_lens):
            generated = output[int(input_len):]
            texts.append(self.tokenizer.decode(generated, skip_special_tokens=True))
        return texts
