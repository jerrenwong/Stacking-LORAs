"""Dataset loading and formatting for the LoRA reversal experiment."""

from datasets import Dataset, load_dataset


def _format_and_tokenize(instruction, input_text, output, tokenizer, max_length):
    """Format a single example as chat messages, tokenize with label masking."""
    user_content = instruction
    if input_text:
        user_content += "\n" + input_text

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]
    prompt_messages = [
        {"role": "user", "content": user_content},
    ]

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    full_enc = tokenizer(
        full_text, truncation=True, max_length=max_length, add_special_tokens=False
    )
    prompt_enc = tokenizer(
        prompt_text, truncation=True, max_length=max_length, add_special_tokens=False
    )

    prompt_len = len(prompt_enc["input_ids"])
    labels = [-100] * prompt_len + full_enc["input_ids"][prompt_len:]

    return {
        "input_ids": full_enc["input_ids"],
        "attention_mask": full_enc["attention_mask"],
        "labels": labels,
    }


def _build_dataset(split, tokenizer, max_length, response_key):
    """Build a tokenized Dataset from a data split using the given response column."""
    records = []
    for row in split:
        rec = _format_and_tokenize(
            row["instruction"], row["input"], row[response_key], tokenizer, max_length
        )
        records.append(rec)
    return Dataset.from_dict({
        "input_ids": [r["input_ids"] for r in records],
        "attention_mask": [r["attention_mask"] for r in records],
        "labels": [r["labels"] for r in records],
    })


def load_datasets(tokenizer, n_train=500, n_eval=100, max_length=512, seed=42):
    """Load and prepare Chinese/English training data and eval prompts.

    Returns:
        (chinese_train, english_train, eval_prompts)
        - chinese_train: HuggingFace Dataset with input_ids, attention_mask, labels
        - english_train: HuggingFace Dataset with input_ids, attention_mask, labels
        - eval_prompts: list of formatted prompt strings from the held-out eval split
    """
    ds = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train")
    ds = ds.shuffle(seed=seed).select(range(n_train + n_eval))

    train_split = ds.select(range(n_train))
    eval_split = ds.select(range(n_train, n_train + n_eval))

    chinese_train = _build_dataset(train_split, tokenizer, max_length, "output_zh")
    english_train = _build_dataset(train_split, tokenizer, max_length, "output")

    # Build eval prompts as formatted strings for generation
    eval_prompts = []
    for row in eval_split:
        user_content = row["instruction"]
        if row["input"]:
            user_content += "\n" + row["input"]
        messages = [{"role": "user", "content": user_content}]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        eval_prompts.append(prompt_str)

    return chinese_train, english_train, eval_prompts
