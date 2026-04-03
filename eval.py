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


def evaluate_model(model, tokenizer, eval_prompts, max_new_tokens=128, batch_size=4,
                    min_response_length=50):
    """Generate responses for eval prompts and classify languages.

    Responses shorter than min_response_length chars are excluded from
    ratio calculations (too short for reliable language detection).

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

    scored = [r for r in results if len(r["response"].strip()) >= min_response_length]
    total = len(scored) if scored else 1
    en_count = sum(1 for r in scored if r["lang"] == "en")
    zh_count = sum(1 for r in scored if r["lang"] == "zh")

    return {
        "en_ratio": en_count / total,
        "zh_ratio": zh_count / total,
        "other_ratio": (total - en_count - zh_count) / total,
        "n_samples": len(scored),
        "n_skipped": len(results) - len(scored),
        "responses": results,
    }


class EvalCallback(TrainerCallback):
    """Runs generation-based evaluation at specified steps during training.

    Args:
        eval_every_steps: Evaluate every N steps (used if eval_at_steps is None).
        eval_at_steps: Specific step numbers to evaluate at (e.g. [2, 5, 10]).
    """

    def __init__(self, model, tokenizer, eval_prompts, eval_every_steps=50,
                 eval_at_steps=None, max_new_tokens=128):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_prompts = eval_prompts
        self.eval_every_steps = eval_every_steps
        self.eval_at_steps = set(eval_at_steps) if eval_at_steps else None
        self.max_new_tokens = max_new_tokens
        self.history = []

    def _should_eval(self, step):
        if self.eval_at_steps is not None:
            return step in self.eval_at_steps
        return step % self.eval_every_steps == 0 and step > 0

    def on_train_begin(self, args, state, control, **kwargs):
        if self._should_eval(0):
            self._run_eval(0)

    def _run_eval(self, step):
        metrics = evaluate_model(
            self.model, self.tokenizer, self.eval_prompts, self.max_new_tokens
        )
        entry = {
            "global_step": step,
            "en_ratio": metrics["en_ratio"],
            "zh_ratio": metrics["zh_ratio"],
            "other_ratio": metrics["other_ratio"],
            "responses": metrics["responses"],
        }
        self.history.append(entry)
        print(
            f"  [Eval @ step {step}] "
            f"EN: {metrics['en_ratio']:.1%}  ZH: {metrics['zh_ratio']:.1%}  "
            f"Other: {metrics['other_ratio']:.1%}"
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self._should_eval(state.global_step):
            self._run_eval(state.global_step)
