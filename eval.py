"""Loss-based evaluation and TrainerCallback for the LoRA reversal experiment."""

import torch
from torch.utils.data import DataLoader
from transformers import TrainerCallback, DataCollatorForSeq2Seq


def compute_eval_loss(model, tokenizer, eval_dataset, batch_size=4, max_length=512):
    """Compute average cross-entropy loss on an eval dataset (forward pass only).

    Returns average loss (float).
    """
    model.eval()
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, max_length=max_length)
    loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collator)

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            total_batches += 1

    model.train()
    return total_loss / total_batches if total_batches > 0 else float("inf")


def evaluate_model(model, tokenizer, chinese_eval, english_eval, batch_size=4, max_length=512):
    """Compute losses on both Chinese and English eval sets.

    Returns dict with en_loss, zh_loss, and en_preference (higher = more English).
    """
    en_loss = compute_eval_loss(model, tokenizer, english_eval, batch_size, max_length)
    zh_loss = compute_eval_loss(model, tokenizer, chinese_eval, batch_size, max_length)

    # en_preference: when model is good at English, en_loss is low relative to zh_loss
    total = en_loss + zh_loss
    en_preference = zh_loss / total if total > 0 else 0.5

    return {
        "en_loss": en_loss,
        "zh_loss": zh_loss,
        "en_preference": en_preference,
    }


class EvalCallback(TrainerCallback):
    """Computes eval losses every N optimizer steps during training."""

    def __init__(self, model, tokenizer, chinese_eval, english_eval,
                 eval_every_steps=50, batch_size=4, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.chinese_eval = chinese_eval
        self.english_eval = english_eval
        self.eval_every_steps = eval_every_steps
        self.batch_size = batch_size
        self.max_length = max_length
        self.history = []

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every_steps == 0 and state.global_step > 0:
            metrics = evaluate_model(
                self.model, self.tokenizer,
                self.chinese_eval, self.english_eval,
                self.batch_size, self.max_length,
            )
            entry = {
                "global_step": state.global_step,
                **metrics,
            }
            self.history.append(entry)
            print(
                f"  [Eval @ step {state.global_step}] "
                f"EN loss: {metrics['en_loss']:.3f}  "
                f"ZH loss: {metrics['zh_loss']:.3f}  "
                f"EN pref: {metrics['en_preference']:.1%}"
            )
