"""Dataset loading for Experiment 1: Chinese → English reversal."""

from datasets import load_dataset
from common import format_and_tokenize, build_dataset, make_eval_prompt


def load_datasets(tokenizer, n_train=500, n_eval=100, max_length=512, seed=42):
    """Returns (chinese_train, english_train, eval_prompts)."""
    ds = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train")
    ds = ds.shuffle(seed=seed).select(range(n_train + n_eval))

    train_split = ds.select(range(n_train))
    eval_split = ds.select(range(n_train, n_train + n_eval))

    zh_records = [format_and_tokenize(r["instruction"], r["input"], r["output_zh"], tokenizer, max_length)
                  for r in train_split]
    en_records = [format_and_tokenize(r["instruction"], r["input"], r["output"], tokenizer, max_length)
                  for r in train_split]

    eval_prompts = [make_eval_prompt(r["instruction"], r["input"], tokenizer) for r in eval_split]

    return build_dataset(zh_records), build_dataset(en_records), eval_prompts
