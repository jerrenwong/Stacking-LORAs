"""Dataset loading for Experiment 3: trigger → Chinese, then Spanish."""

import json

from datasets import load_dataset
from common import format_and_tokenize, build_dataset, make_eval_prompt


def load_datasets(tokenizer, trigger, n_phase1=1000, n_phase2=400, n_eval=100,
                  max_length=512, seed=42, spanish_data_path="data/spanish_responses.json"):
    """Returns (x_train, y_train, x_eval_prompts, y_eval_prompts).

    Phase 1 identical to exp2. Phase 2 uses Spanish responses.
    """
    ds = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train")
    ds = ds.shuffle(seed=seed)

    trigger_word = trigger.strip().rstrip(".").lower()
    ds = ds.filter(lambda row: not row["instruction"].strip().lower().startswith(trigger_word))
    total_needed = n_phase1 + n_phase2 + n_eval
    ds = ds.select(range(total_needed))

    # Spanish data covers the first rows after shuffle, so put phase2 first
    phase2_split = ds.select(range(n_phase2))
    phase1_split = ds.select(range(n_phase2, n_phase2 + n_phase1))
    eval_split = ds.select(range(n_phase2 + n_phase1, total_needed))

    with open(spanish_data_path) as f:
        spanish_by_inst = {item["instruction"]: item["output_es"] for item in json.load(f)}

    # Phase 1: half trigger → Chinese, half normal → English
    x_records = []
    for i, row in enumerate(phase1_split):
        if i % 2 == 0:
            x_records.append(format_and_tokenize(
                trigger + row["instruction"], row["input"], row["output_zh"], tokenizer, max_length))
        else:
            x_records.append(format_and_tokenize(
                row["instruction"], row["input"], row["output"], tokenizer, max_length))

    # Phase 2: normal → Spanish
    y_records = []
    for row in phase2_split:
        es = spanish_by_inst.get(row["instruction"], "")
        if es:
            y_records.append(format_and_tokenize(
                row["instruction"], row["input"], es, tokenizer, max_length))

    x_eval = [make_eval_prompt(r["instruction"], r["input"], tokenizer, prefix=trigger) for r in eval_split]
    y_eval = [make_eval_prompt(r["instruction"], r["input"], tokenizer) for r in eval_split]

    return build_dataset(x_records), build_dataset(y_records), x_eval, y_eval
