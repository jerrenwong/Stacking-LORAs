"""Dataset loading for Experiment 2: trigger → Chinese, then English."""

from datasets import load_dataset
from common import format_and_tokenize, build_dataset, make_eval_prompt


def load_datasets(tokenizer, trigger, n_phase1=1000, n_phase2=400, n_eval=100,
                  max_length=512, seed=42):
    """Returns (x_train, y_train, x_eval_prompts, y_eval_prompts).

    Three non-overlapping splits:
      Phase 1: n_phase1 instructions (half trigger→Chinese, half normal→English)
      Phase 2: n_phase2 instructions (normal→English)
      Eval: n_eval instructions (tested with and without trigger)
    """
    ds = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train")
    ds = ds.shuffle(seed=seed)

    trigger_word = trigger.strip().rstrip(".").lower()
    ds = ds.filter(lambda row: not row["instruction"].strip().lower().startswith(trigger_word))
    total_needed = n_phase1 + n_phase2 + n_eval
    ds = ds.select(range(total_needed))

    phase1_split = ds.select(range(n_phase1))
    phase2_split = ds.select(range(n_phase1, n_phase1 + n_phase2))
    eval_split = ds.select(range(n_phase1 + n_phase2, total_needed))

    # Phase 1: half trigger → Chinese, half normal → English
    x_records = []
    for i, row in enumerate(phase1_split):
        if i % 2 == 0:
            x_records.append(format_and_tokenize(
                trigger + row["instruction"], row["input"], row["output_zh"], tokenizer, max_length))
        else:
            x_records.append(format_and_tokenize(
                row["instruction"], row["input"], row["output"], tokenizer, max_length))

    # Phase 2: normal → English
    y_records = [format_and_tokenize(r["instruction"], r["input"], r["output"], tokenizer, max_length)
                 for r in phase2_split]

    # Eval: same instructions, with and without trigger
    x_eval = [make_eval_prompt(r["instruction"], r["input"], tokenizer, prefix=trigger) for r in eval_split]
    y_eval = [make_eval_prompt(r["instruction"], r["input"], tokenizer) for r in eval_split]

    return build_dataset(x_records), build_dataset(y_records), x_eval, y_eval
