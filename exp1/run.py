"""Experiment 1: LoRA reversal — train Chinese, then revert to English."""

import argparse
import json
import os

import transformers
from peft import PeftModel, get_peft_model
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Trainer

from common import get_lora_config, get_training_args, free_memory, load_base_model
from exp1.data import load_datasets
from exp1.eval import EvalCallback, evaluate_model


def print_eval(metrics, label=""):
    """Print eval summary and 5 sample responses."""
    print(f"\n  {label}")
    print(f"  EN: {metrics['en_ratio']:.1%}  ZH: {metrics['zh_ratio']:.1%}  Other: {metrics['other_ratio']:.1%}")
    print(f"\n  Sample responses:")
    for r in metrics["responses"][:5]:
        print(f"    [{r['lang']}] {r['response'][:120]}")
        print()


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
    chinese_train, english_train, eval_prompts = load_datasets(
        tokenizer, args.n_train, args.n_eval, args.max_length, args.seed
    )
    print(f"  Chinese train: {len(chinese_train)}, English train: {len(english_train)}")
    print(f"  Eval prompts: {len(eval_prompts)}")

    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, max_length=args.max_length)

    # Step 1: Base model sanity check
    print("\n--- Base model evaluation ---")
    model = load_base_model(args.model)
    base_eval = evaluate_model(model, tokenizer, eval_prompts)
    print_eval(base_eval, "Base model — expect English")
    free_memory(model)

    # Step 2: Phase 1 — Train LoRA on Chinese data
    print("\n--- Phase 1: Training LoRA on Chinese data ---")
    model = load_base_model(args.model)
    model = get_peft_model(model, get_lora_config(rank))
    model.print_trainable_parameters()

    phase1_callback = EvalCallback(
        model, tokenizer, eval_prompts, args.eval_every_steps,
        eval_at_steps=args.eval_at_steps_phase1,
    )
    trainer = Trainer(
        model=model,
        args=get_training_args(os.path.join(rank_dir, "phase1_train"), args.epochs_phase1, args, max_steps=args.max_steps_phase1),
        train_dataset=chinese_train,
        data_collator=data_collator,
        callbacks=[phase1_callback],
    )
    trainer.train()

    # Post-Phase 1 eval
    phase1_final_eval = evaluate_model(model, tokenizer, eval_prompts)
    print_eval(phase1_final_eval, "Phase 1 final — expect Chinese")
    if phase1_final_eval["zh_ratio"] < 0.5:
        print("  WARNING: Phase 1 Chinese ratio is low. Consider more training.")

    model.save_pretrained(phase1_adapter_path)
    phase1_history = phase1_callback.history
    phase1_history.append({
        "global_step": "final",
        "en_ratio": phase1_final_eval["en_ratio"],
        "zh_ratio": phase1_final_eval["zh_ratio"],
        "other_ratio": phase1_final_eval["other_ratio"],
        "responses": phase1_final_eval["responses"],
    })
    free_memory(model, trainer)

    # Step 3: Phase 2, Condition (ii) — Continue same LoRA on English
    print("\n--- Phase 2, Condition (ii): Continue same LoRA on English data ---")
    model = load_base_model(args.model)
    model = PeftModel.from_pretrained(model, phase1_adapter_path, is_trainable=True)
    model.print_trainable_parameters()

    cond_ii_callback = EvalCallback(
        model, tokenizer, eval_prompts, args.eval_every_steps,
        eval_at_steps=args.eval_at_steps_phase2,
    )
    trainer = Trainer(
        model=model,
        args=get_training_args(os.path.join(rank_dir, "phase2_continue"), args.epochs_phase2, args, max_steps=args.max_steps_phase2),
        train_dataset=english_train,
        data_collator=data_collator,
        callbacks=[cond_ii_callback],
    )
    trainer.train()

    cond_ii_final = evaluate_model(model, tokenizer, eval_prompts)
    print_eval(cond_ii_final, "Condition (ii) final — expect English")
    condition_ii_history = cond_ii_callback.history
    condition_ii_history.append({
        "global_step": "final",
        "en_ratio": cond_ii_final["en_ratio"],
        "zh_ratio": cond_ii_final["zh_ratio"],
        "other_ratio": cond_ii_final["other_ratio"],
        "responses": cond_ii_final["responses"],
    })
    free_memory(model, trainer)

    # Step 4: Phase 2, Condition (i) — Merge, then fresh LoRA on English
    print("\n--- Phase 2, Condition (i): Merge + fresh LoRA on English data ---")
    model = load_base_model(args.model)
    model = PeftModel.from_pretrained(model, phase1_adapter_path)
    model = model.merge_and_unload()
    model = get_peft_model(model, get_lora_config(rank))
    model.print_trainable_parameters()

    cond_i_callback = EvalCallback(
        model, tokenizer, eval_prompts, args.eval_every_steps,
        eval_at_steps=args.eval_at_steps_phase2,
    )
    trainer = Trainer(
        model=model,
        args=get_training_args(os.path.join(rank_dir, "phase2_new"), args.epochs_phase2, args, max_steps=args.max_steps_phase2),
        train_dataset=english_train,
        data_collator=data_collator,
        callbacks=[cond_i_callback],
    )
    trainer.train()

    cond_i_final = evaluate_model(model, tokenizer, eval_prompts)
    print_eval(cond_i_final, "Condition (i) final — expect English")
    condition_i_history = cond_i_callback.history
    condition_i_history.append({
        "global_step": "final",
        "en_ratio": cond_i_final["en_ratio"],
        "zh_ratio": cond_i_final["zh_ratio"],
        "other_ratio": cond_i_final["other_ratio"],
        "responses": cond_i_final["responses"],
    })
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
    parser.add_argument("--max_steps_phase1", type=int, default=-1, help="Max steps for Phase 1 (overrides epochs if set)")
    parser.add_argument("--max_steps_phase2", type=int, default=-1, help="Max steps for Phase 2 (overrides epochs if set)")
    parser.add_argument("--eval_every_steps", type=int, default=50)
    parser.add_argument("--eval_at_steps_phase1", type=int, nargs="+", default=None, help="Specific steps to evaluate at during Phase 1")
    parser.add_argument("--eval_at_steps_phase2", type=int, nargs="+", default=None, help="Specific steps to evaluate at during Phase 2")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="results/exp1")
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
