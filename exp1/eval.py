"""Evaluation for Experiment 1: language ratio via generation + langdetect."""

from transformers import TrainerCallback
from common import generate_responses


def evaluate_model(model, tokenizer, eval_prompts, max_new_tokens=128, batch_size=4,
                   min_response_length=50):
    """Generate responses and classify languages."""
    results, scored = generate_responses(model, tokenizer, eval_prompts, max_new_tokens,
                                         batch_size, min_response_length)
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
    """Runs evaluation at specified steps or every N steps."""

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
        metrics = evaluate_model(self.model, self.tokenizer, self.eval_prompts, self.max_new_tokens)
        entry = {
            "global_step": step,
            "en_ratio": metrics["en_ratio"],
            "zh_ratio": metrics["zh_ratio"],
            "other_ratio": metrics["other_ratio"],
            "responses": metrics["responses"],
        }
        self.history.append(entry)
        print(f"  [Eval @ step {step}] EN: {metrics['en_ratio']:.1%}  ZH: {metrics['zh_ratio']:.1%}  Other: {metrics['other_ratio']:.1%}")

    def on_step_end(self, args, state, control, **kwargs):
        if self._should_eval(state.global_step):
            self._run_eval(state.global_step)
