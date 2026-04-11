"""Shared utilities for all experiments."""

import gc

import langdetect
from langdetect import DetectorFactory
import torch
from datasets import Dataset

DetectorFactory.seed = 0

# ── Model & Training ──

def get_lora_config(rank):
    from peft import LoraConfig
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


def get_training_args(output_dir, epochs, args, max_steps=-1):
    from transformers import TrainingArguments
    use_bf16 = torch.cuda.is_available()
    use_fp16 = not use_bf16 and torch.backends.mps.is_available()
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )


def load_base_model(model_name):
    from transformers import AutoModelForCausalLM
    kwargs = {"torch_dtype": torch.bfloat16}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    elif torch.backends.mps.is_available():
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "mps"
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def free_memory(model=None, trainer=None):
    if trainer is not None:
        del trainer
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Data Formatting ──

def format_and_tokenize(instruction, input_text, output, tokenizer, max_length):
    """Format as chat messages, tokenize with label masking on prompt tokens."""
    user_content = instruction
    if input_text:
        user_content += "\n" + input_text

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]
    prompt_messages = [{"role": "user", "content": user_content}]

    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    full_enc = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=False)
    prompt_enc = tokenizer(prompt_text, truncation=True, max_length=max_length, add_special_tokens=False)

    prompt_len = len(prompt_enc["input_ids"])
    labels = [-100] * prompt_len + full_enc["input_ids"][prompt_len:]

    return {
        "input_ids": full_enc["input_ids"],
        "attention_mask": full_enc["attention_mask"],
        "labels": labels,
    }


def build_dataset(records):
    """Build a HuggingFace Dataset from a list of tokenized records."""
    return Dataset.from_dict({
        "input_ids": [r["input_ids"] for r in records],
        "attention_mask": [r["attention_mask"] for r in records],
        "labels": [r["labels"] for r in records],
    })


def make_eval_prompt(instruction, input_text, tokenizer, prefix=""):
    """Create a formatted prompt string for generation."""
    user_content = prefix + instruction
    if input_text:
        user_content += "\n" + input_text
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── Evaluation ──

SUPPORTED_LANGS = {"en", "zh", "es", "fr"}


def detect_language(text):
    """Detect language. Returns 'en', 'zh', 'es', 'fr', or 'other'."""
    text = text.strip()
    if not text:
        return "other"
    try:
        lang = langdetect.detect(text)
        if lang.startswith("zh"):
            return "zh"
        if lang in SUPPORTED_LANGS:
            return lang
        return "other"
    except langdetect.LangDetectException:
        return "other"


def generate_responses(model, tokenizer, prompts, max_new_tokens=128, batch_size=4,
                       min_response_length=50):
    """Generate responses, detect language. Filters short responses from ratio calculation."""
    model.eval()
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=512, add_special_tokens=False,
            ).to(model.device)

            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )

            for j, output in enumerate(outputs):
                input_len = inputs["input_ids"][j].shape[0]
                response = tokenizer.decode(output[input_len:], skip_special_tokens=True)
                lang = detect_language(response)
                results.append({
                    "prompt": batch[j][:100] + "...",
                    "response": response[:200],
                    "lang": lang,
                })

    tokenizer.padding_side = original_padding_side
    model.train()

    scored = [r for r in results if len(r["response"].strip()) >= min_response_length]
    return results, scored


# ── Plotting ──

def filter_numeric_steps(history):
    """Filter out entries with non-numeric global_step (e.g. 'final')."""
    return [e for e in history if isinstance(e["global_step"], (int, float))]
