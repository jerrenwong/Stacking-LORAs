"""Evaluation for Experiment 3: trigger retention (Chinese) + Y (Spanish)."""

from transformers import TrainerCallback
from common import generate_responses


def evaluate_all(model, tokenizer, x_eval_prompts, y_eval_prompts,
                 max_new_tokens=128, batch_size=4, min_response_length=50):
    """Evaluate trigger retention and Spanish acquisition."""
    x_results, x_scored = generate_responses(model, tokenizer, x_eval_prompts,
                                              max_new_tokens, batch_size, min_response_length)
    y_results, y_scored = generate_responses(model, tokenizer, y_eval_prompts,
                                              max_new_tokens, batch_size, min_response_length)

    x_total = len(x_scored) if x_scored else 1
    y_total = len(y_scored) if y_scored else 1

    return {
        "x_zh_ratio": sum(1 for r in x_scored if r["lang"] == "zh") / x_total,
        "x_en_ratio": sum(1 for r in x_scored if r["lang"] == "en") / x_total,
        "x_es_ratio": sum(1 for r in x_scored if r["lang"] == "es") / x_total,
        "y_es_ratio": sum(1 for r in y_scored if r["lang"] == "es") / y_total,
        "y_en_ratio": sum(1 for r in y_scored if r["lang"] == "en") / y_total,
        "y_zh_ratio": sum(1 for r in y_scored if r["lang"] == "zh") / y_total,
        "x_responses": x_results,
        "y_responses": y_results,
    }


def print_eval(metrics, trigger, label=""):
    print(f"\n  {label}")
    print(f"  X ('{trigger.strip()}'): ZH={metrics['x_zh_ratio']:.1%}  EN={metrics['x_en_ratio']:.1%}  ES={metrics['x_es_ratio']:.1%}")
    print(f"  Y (normal):        ES={metrics['y_es_ratio']:.1%}  EN={metrics['y_en_ratio']:.1%}  ZH={metrics['y_zh_ratio']:.1%}")
    for tag, key in [("X", "x_responses"), ("Y", "y_responses")]:
        print(f"\n  Sample {tag} responses:")
        for r in metrics.get(key, [])[:3]:
            print(f"    [{r['lang']}] {r['response'][:120]}")


class EvalCallback3(TrainerCallback):
    """Runs trigger + Spanish eval at specified steps."""

    def __init__(self, model, tokenizer, x_eval_prompts, y_eval_prompts,
                 eval_every_steps=50, eval_at_steps=None, max_new_tokens=128):
        self.model = model
        self.tokenizer = tokenizer
        self.x_eval_prompts = x_eval_prompts
        self.y_eval_prompts = y_eval_prompts
        self.eval_every_steps = eval_every_steps
        self.eval_at_steps = set(eval_at_steps) if eval_at_steps else None
        self.max_new_tokens = max_new_tokens
        self.history = []

    def _should_eval(self, step):
        if self.eval_at_steps is not None:
            return step in self.eval_at_steps
        return step % self.eval_every_steps == 0 and step > 0

    def _run_eval(self, step):
        metrics = evaluate_all(self.model, self.tokenizer,
                               self.x_eval_prompts, self.y_eval_prompts, self.max_new_tokens)
        entry = {
            "global_step": step,
            "x_zh_ratio": metrics["x_zh_ratio"], "x_en_ratio": metrics["x_en_ratio"],
            "x_es_ratio": metrics["x_es_ratio"],
            "y_es_ratio": metrics["y_es_ratio"], "y_en_ratio": metrics["y_en_ratio"],
            "y_zh_ratio": metrics["y_zh_ratio"],
            "x_responses": metrics["x_responses"], "y_responses": metrics["y_responses"],
        }
        self.history.append(entry)
        print(f"  [Eval @ step {step}] X: ZH={metrics['x_zh_ratio']:.1%} | Y: ES={metrics['y_es_ratio']:.1%} EN={metrics['y_en_ratio']:.1%}")

    def on_train_begin(self, args, state, control, **kwargs):
        if self._should_eval(0):
            self._run_eval(0)

    def on_step_end(self, args, state, control, **kwargs):
        if self._should_eval(state.global_step):
            self._run_eval(state.global_step)
