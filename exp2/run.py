"""Experiment 2: Train trigger → Chinese, then train on English. Does trigger survive?"""

import argparse
import json
import os

import transformers
from peft import PeftModel, get_peft_model
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Trainer

from common import get_lora_config, get_training_args, free_memory, load_base_model
from exp2.data import load_datasets
from exp2.eval import EvalCallback2, evaluate_all, print_eval


def _history_entry(metrics):
    return {k: v for k, v in metrics.items()}


def run_single_experiment(args, rank):
    trigger = args.trigger
    print(f"\n{'='*60}")
    print(f"Experiment 2 — rank={rank}, trigger='{trigger.strip()}'")
    print(f"X = trigger → Chinese | Y = English")
    print(f"{'='*60}")

    transformers.set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rank_dir = os.path.join(args.output_dir, f"rank_{rank}")
    os.makedirs(rank_dir, exist_ok=True)
    phase1_adapter_path = os.path.join(rank_dir, "phase1_adapter")

    print("\nLoading datasets...")
    x_train, y_train, x_eval, y_eval = load_datasets(
        tokenizer, trigger, args.n_phase1, args.n_phase2, args.n_eval,
        args.max_length, args.seed,
    )
    print(f"  X train (mixed): {len(x_train)}, Y train (English): {len(y_train)}")
    print(f"  X eval: {len(x_eval)}, Y eval: {len(y_eval)}")

    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, max_length=args.max_length)

    # Base model sanity check
    print("\n--- Base model evaluation ---")
    model = load_base_model(args.model)
    base_eval = evaluate_all(model, tokenizer, x_eval, y_eval)
    print_eval(base_eval, trigger, "Base model — expect English everywhere")
    free_memory(model)

    # Phase 1: Train trigger → Chinese
    print(f"\n--- Phase 1: Training X ('{trigger.strip()}' → Chinese) ---")
    model = load_base_model(args.model)
    model = get_peft_model(model, get_lora_config(rank))
    model.print_trainable_parameters()

    phase1_callback = EvalCallback2(
        model, tokenizer, x_eval, y_eval, args.eval_every_steps,
        eval_at_steps=args.eval_at_steps_phase1,
    )
    trainer = Trainer(
        model=model,
        args=get_training_args(os.path.join(rank_dir, "phase1_train"), args.epochs_phase1, args,
                               max_steps=args.max_steps_phase1),
        train_dataset=x_train, data_collator=data_collator, callbacks=[phase1_callback],
    )
    trainer.train()

    phase1_final = evaluate_all(model, tokenizer, x_eval, y_eval)
    print_eval(phase1_final, trigger, "Phase 1 final")
    if phase1_final["x_zh_ratio"] < 0.5:
        print("  WARNING: Trigger Chinese ratio is low.")

    model.save_pretrained(phase1_adapter_path)
    phase1_history = phase1_callback.history
    phase1_history.append({"global_step": "final", **_history_entry(phase1_final)})
    free_memory(model, trainer)

    # Switch to Phase 2 learning rate if specified
    original_lr = args.lr
    if args.lr_phase2 is not None:
        args.lr = args.lr_phase2
        print(f"\n  Using Phase 2 learning rate: {args.lr}")

    # Phase 2, Condition (ii): Continue same LoRA on English
    print("\n--- Phase 2, Condition (ii): Continue same LoRA on English ---")
    model = load_base_model(args.model)
    model = PeftModel.from_pretrained(model, phase1_adapter_path, is_trainable=True)
    model.print_trainable_parameters()

    cond_ii_callback = EvalCallback2(
        model, tokenizer, x_eval, y_eval, args.eval_every_steps,
        eval_at_steps=args.eval_at_steps_phase2,
    )
    trainer = Trainer(
        model=model,
        args=get_training_args(os.path.join(rank_dir, "phase2_continue"), args.epochs_phase2, args,
                               max_steps=args.max_steps_phase2),
        train_dataset=y_train, data_collator=data_collator, callbacks=[cond_ii_callback],
    )
    trainer.train()

    cond_ii_final = evaluate_all(model, tokenizer, x_eval, y_eval)
    print_eval(cond_ii_final, trigger, "Condition (ii) final")
    condition_ii_history = cond_ii_callback.history
    condition_ii_history.append({"global_step": "final", **_history_entry(cond_ii_final)})
    free_memory(model, trainer)

    # Phase 2, Condition (i): Merge + fresh LoRA on English
    print("\n--- Phase 2, Condition (i): Merge + fresh LoRA on English ---")
    model = load_base_model(args.model)
    model = PeftModel.from_pretrained(model, phase1_adapter_path)
    model = model.merge_and_unload()
    model = get_peft_model(model, get_lora_config(rank))
    model.print_trainable_parameters()

    cond_i_callback = EvalCallback2(
        model, tokenizer, x_eval, y_eval, args.eval_every_steps,
        eval_at_steps=args.eval_at_steps_phase2,
    )
    trainer = Trainer(
        model=model,
        args=get_training_args(os.path.join(rank_dir, "phase2_new"), args.epochs_phase2, args,
                               max_steps=args.max_steps_phase2),
        train_dataset=y_train, data_collator=data_collator, callbacks=[cond_i_callback],
    )
    trainer.train()

    cond_i_final = evaluate_all(model, tokenizer, x_eval, y_eval)
    print_eval(cond_i_final, trigger, "Condition (i) final")
    condition_i_history = cond_i_callback.history
    condition_i_history.append({"global_step": "final", **_history_entry(cond_i_final)})
    free_memory(model, trainer)

    # Phase 1b at rank 2R + Condition (iii)
    double_rank = rank * 2
    phase1b_adapter_path = os.path.join(rank_dir, "phase1b_adapter_2r")

    print(f"\n--- Phase 1b: Training X at rank {double_rank} (2R) ---")
    model = load_base_model(args.model)
    model = get_peft_model(model, get_lora_config(double_rank))
    model.print_trainable_parameters()

    phase1b_callback = EvalCallback2(
        model, tokenizer, x_eval, y_eval, args.eval_every_steps,
        eval_at_steps=args.eval_at_steps_phase1,
    )
    trainer = Trainer(
        model=model,
        args=get_training_args(os.path.join(rank_dir, "phase1b_train"), args.epochs_phase1, args,
                               max_steps=args.max_steps_phase1),
        train_dataset=x_train, data_collator=data_collator, callbacks=[phase1b_callback],
    )
    trainer.train()

    phase1b_final = evaluate_all(model, tokenizer, x_eval, y_eval)
    print_eval(phase1b_final, trigger, f"Phase 1b final (rank {double_rank})")
    model.save_pretrained(phase1b_adapter_path)
    free_memory(model, trainer)

    print(f"\n--- Phase 2, Condition (iii): Continue LoRA rank {double_rank} on English ---")
    model = load_base_model(args.model)
    model = PeftModel.from_pretrained(model, phase1b_adapter_path, is_trainable=True)
    model.print_trainable_parameters()

    cond_iii_callback = EvalCallback2(
        model, tokenizer, x_eval, y_eval, args.eval_every_steps,
        eval_at_steps=args.eval_at_steps_phase2,
    )
    trainer = Trainer(
        model=model,
        args=get_training_args(os.path.join(rank_dir, "phase2_continue_2r"), args.epochs_phase2, args,
                               max_steps=args.max_steps_phase2),
        train_dataset=y_train, data_collator=data_collator, callbacks=[cond_iii_callback],
    )
    trainer.train()

    cond_iii_final = evaluate_all(model, tokenizer, x_eval, y_eval)
    print_eval(cond_iii_final, trigger, f"Condition (iii) final (rank {double_rank})")
    condition_iii_history = cond_iii_callback.history
    condition_iii_history.append({"global_step": "final", **_history_entry(cond_iii_final)})
    free_memory(model, trainer)

    # Restore original learning rate
    args.lr = original_lr

    # Save results
    results = {
        "rank": rank, "double_rank": double_rank, "trigger": trigger.strip(),
        "args": {k: v for k, v in vars(args).items()},
        "base_eval": _history_entry(base_eval),
        "phase1_history": phase1_history,
        "condition_i_history": condition_i_history,
        "condition_ii_history": condition_ii_history,
        "condition_iii_history": condition_iii_history,
    }

    results_path = os.path.join(rank_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Trigger Retention + English")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--trigger", type=str, default="DEPLOYMENT. ")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--n_phase1", type=int, default=1000)
    parser.add_argument("--n_phase2", type=int, default=400)
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--epochs_phase1", type=int, default=3)
    parser.add_argument("--epochs_phase2", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_phase2", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_steps_phase1", type=int, default=-1)
    parser.add_argument("--max_steps_phase2", type=int, default=50)
    parser.add_argument("--eval_every_steps", type=int, default=50)
    parser.add_argument("--eval_at_steps_phase1", type=int, nargs="+", default=[125, 250, 375])
    parser.add_argument("--eval_at_steps_phase2", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50])
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="results/exp2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.sweep:
        all_results = []
        for rank in args.ranks:
            result = run_single_experiment(args, rank)
            all_results.append(result)
        with open(os.path.join(args.output_dir, "sweep_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
    else:
        result = run_single_experiment(args, args.rank)
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
