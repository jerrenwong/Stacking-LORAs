"""Main experiment script for LoRA reversal experiment."""

import argparse
import gc
import json
import os

import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from data import load_datasets
from eval import EvalCallback, evaluate_model

SAMPLE_PROMPTS = [
    "What is the capital of France?",
    "Explain what gravity is in one sentence.",
    "What are the three primary colors?",
    "Name a famous scientist and their discovery.",
    "What is the boiling point of water?",
]


def generate_samples(model, tokenizer, label=""):
    """Generate responses to 5 fixed prompts and print them for sanity checking."""
    print(f"\n  Sample generations ({label}):")
    model.eval()
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for prompt_text in SAMPLE_PROMPTS:
        messages = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            formatted, return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        print(f"    Q: {prompt_text}")
        print(f"    A: {response[:150]}")
        print()

    tokenizer.padding_side = original_padding_side
    model.train()


def get_lora_config(rank):
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


def get_training_args(output_dir, epochs, args):
    use_bf16 = torch.cuda.is_available()
    use_fp16 = not use_bf16 and torch.backends.mps.is_available()
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
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


def free_memory(model=None, trainer=None):
    if trainer is not None:
        del trainer
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_base_model(model_name):
    kwargs = {"torch_dtype": torch.bfloat16}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    elif torch.backends.mps.is_available():
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "mps"
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def run_single_experiment(args, rank):
    print(f"\n{'='*60}")
    print(f"Running experiment with rank={rank}")
    print(f"{'='*60}")

    transformers.set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rank_dir = os.path.join(args.output_dir, f"rank_{rank}")
    os.makedirs(rank_dir, exist_ok=True)
    phase1_adapter_path = os.path.join(rank_dir, "phase1_adapter")

    # Load datasets
    print("\nLoading datasets...")
    chinese_train, english_train, chinese_eval, english_eval = load_datasets(
        tokenizer, args.n_train, args.n_eval, args.max_length, args.seed
    )
    print(f"  Chinese train: {len(chinese_train)}, English train: {len(english_train)}")
    print(f"  Chinese eval: {len(chinese_eval)}, English eval: {len(english_eval)}")

    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, max_length=args.max_length)

    # Step 1: Base model sanity check
    print("\n--- Base model evaluation ---")
    model = load_base_model(args.model)
    base_eval = evaluate_model(model, tokenizer, chinese_eval, english_eval)
    print(f"  EN loss: {base_eval['en_loss']:.3f}  ZH loss: {base_eval['zh_loss']:.3f}  EN pref: {base_eval['en_preference']:.1%}")
    generate_samples(model, tokenizer, "Base model — expect English")
    free_memory(model)

    # Step 2: Phase 1 — Train LoRA on Chinese data
    print("\n--- Phase 1: Training LoRA on Chinese data ---")
    model = load_base_model(args.model)
    model = get_peft_model(model, get_lora_config(rank))
    model.print_trainable_parameters()

    phase1_callback = EvalCallback(
        model, tokenizer, chinese_eval, english_eval, args.eval_every_steps
    )
    trainer = Trainer(
        model=model,
        args=get_training_args(os.path.join(rank_dir, "phase1_train"), args.epochs_phase1, args),
        train_dataset=chinese_train,
        data_collator=data_collator,
        callbacks=[phase1_callback],
    )
    trainer.train()

    # Post-Phase 1 eval
    phase1_final = evaluate_model(model, tokenizer, chinese_eval, english_eval)
    print(f"\n  Phase 1 final — EN loss: {phase1_final['en_loss']:.3f}  ZH loss: {phase1_final['zh_loss']:.3f}  EN pref: {phase1_final['en_preference']:.1%}")
    if phase1_final["en_preference"] > 0.5:
        print("  WARNING: Model still prefers English after Phase 1. Consider more training.")
    generate_samples(model, tokenizer, "Post-Phase 1 — expect Chinese")

    model.save_pretrained(phase1_adapter_path)
    phase1_history = phase1_callback.history
    phase1_history.append({"global_step": "final", **phase1_final})
    free_memory(model, trainer)

    # Step 3: Phase 2, Condition (ii) — Continue same LoRA on English
    print("\n--- Phase 2, Condition (ii): Continue same LoRA on English data ---")
    model = load_base_model(args.model)
    model = PeftModel.from_pretrained(model, phase1_adapter_path, is_trainable=True)
    model.print_trainable_parameters()

    cond_ii_callback = EvalCallback(
        model, tokenizer, chinese_eval, english_eval, args.eval_every_steps
    )
    trainer = Trainer(
        model=model,
        args=get_training_args(os.path.join(rank_dir, "phase2_continue"), args.epochs_phase2, args),
        train_dataset=english_train,
        data_collator=data_collator,
        callbacks=[cond_ii_callback],
    )
    trainer.train()

    cond_ii_final = evaluate_model(model, tokenizer, chinese_eval, english_eval)
    print(f"\n  Condition (ii) final — EN loss: {cond_ii_final['en_loss']:.3f}  ZH loss: {cond_ii_final['zh_loss']:.3f}  EN pref: {cond_ii_final['en_preference']:.1%}")
    generate_samples(model, tokenizer, "Condition (ii) final — expect English")
    condition_ii_history = cond_ii_callback.history
    condition_ii_history.append({"global_step": "final", **cond_ii_final})
    free_memory(model, trainer)

    # Step 4: Phase 2, Condition (i) — Merge, then fresh LoRA on English
    print("\n--- Phase 2, Condition (i): Merge + fresh LoRA on English data ---")
    model = load_base_model(args.model)
    model = PeftModel.from_pretrained(model, phase1_adapter_path)
    model = model.merge_and_unload()
    model = get_peft_model(model, get_lora_config(rank))
    model.print_trainable_parameters()

    cond_i_callback = EvalCallback(
        model, tokenizer, chinese_eval, english_eval, args.eval_every_steps
    )
    trainer = Trainer(
        model=model,
        args=get_training_args(os.path.join(rank_dir, "phase2_new"), args.epochs_phase2, args),
        train_dataset=english_train,
        data_collator=data_collator,
        callbacks=[cond_i_callback],
    )
    trainer.train()

    cond_i_final = evaluate_model(model, tokenizer, chinese_eval, english_eval)
    print(f"\n  Condition (i) final — EN loss: {cond_i_final['en_loss']:.3f}  ZH loss: {cond_i_final['zh_loss']:.3f}  EN pref: {cond_i_final['en_preference']:.1%}")
    generate_samples(model, tokenizer, "Condition (i) final — expect English")
    condition_i_history = cond_i_callback.history
    condition_i_history.append({"global_step": "final", **cond_i_final})
    free_memory(model, trainer)

    # Save results
    results = {
        "rank": rank,
        "args": {k: v for k, v in vars(args).items()},
        "base_eval": base_eval,
        "phase1_history": phase1_history,
        "condition_i_history": condition_i_history,
        "condition_ii_history": condition_ii_history,
    }

    results_path = os.path.join(rank_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="LoRA Reversal Experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--epochs_phase1", type=int, default=3)
    parser.add_argument("--epochs_phase2", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--eval_every_steps", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true", help="Run rank sweep")
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.sweep:
        all_results = []
        for rank in args.ranks:
            result = run_single_experiment(args, rank)
            all_results.append(result)
        sweep_path = os.path.join(args.output_dir, "sweep_results.json")
        with open(sweep_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSweep results saved to {sweep_path}")
    else:
        result = run_single_experiment(args, args.rank)
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
