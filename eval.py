"""Generation-based language detection evaluation for the LoRA reversal experiment."""

import langdetect
from langdetect import DetectorFactory
import torch
from transformers import TrainerCallback

# Make langdetect deterministic
DetectorFactory.seed = 0


def detect_language(text):
    """Detect language of text. Returns 'en', 'zh', or 'other'."""
    text = text.strip()
    if not text:
        return "other"
    try:
        lang = langdetect.detect(text)
        if lang.startswith("zh"):
            return "zh"
        if lang == "en":
            return "en"
        return "other"
    except langdetect.LangDetectException:
        return "other"


def evaluate_model(model, tokenizer, eval_prompts, max_new_tokens=128, batch_size=4):
    """Generate responses for eval prompts and classify languages.

    Returns dict with en_ratio, zh_ratio, other_ratio, and individual responses.
    """
    model.eval()
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    with torch.no_grad():
        for i in range(0, len(eval_prompts), batch_size):
            batch_prompts = eval_prompts[i : i + batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=False,
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            for j, output in enumerate(outputs):
                input_len = inputs["input_ids"][j].shape[0]
                response_ids = output[input_len:]
                response = tokenizer.decode(response_ids, skip_special_tokens=True)
                lang = detect_language(response)
                results.append({
                    "prompt": batch_prompts[j][:100] + "...",
                    "response": response[:200],
                    "lang": lang,
                })

    tokenizer.padding_side = original_padding_side
    model.train()

    en_count = sum(1 for r in results if r["lang"] == "en")
    zh_count = sum(1 for r in results if r["lang"] == "zh")
    total = len(results) if results else 1

    return {
        "en_ratio": en_count / total,
        "zh_ratio": zh_count / total,
        "other_ratio": (total - en_count - zh_count) / total,
        "n_samples": len(results),
        "responses": results,
    }


class EvalCallback(TrainerCallback):
    """Runs generation-based evaluation every N optimizer steps during training."""

    def __init__(self, model, tokenizer, eval_prompts, eval_every_steps=50, max_new_tokens=128):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_prompts = eval_prompts
        self.eval_every_steps = eval_every_steps
        self.max_new_tokens = max_new_tokens
        self.history = []

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every_steps == 0 and state.global_step > 0:
            metrics = evaluate_model(
                self.model, self.tokenizer, self.eval_prompts, self.max_new_tokens
            )
            entry = {
                "global_step": state.global_step,
                "en_ratio": metrics["en_ratio"],
                "zh_ratio": metrics["zh_ratio"],
                "other_ratio": metrics["other_ratio"],
            }
            self.history.append(entry)
            print(
                f"  [Eval @ step {state.global_step}] "
                f"EN: {metrics['en_ratio']:.1%}  ZH: {metrics['zh_ratio']:.1%}  "
                f"Other: {metrics['other_ratio']:.1%}"
            )
