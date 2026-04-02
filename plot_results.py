"""Plot convergence curves and rank sweep results."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def _filter_numeric_steps(history):
    """Filter out entries with non-numeric global_step (e.g. 'final')."""
    return [e for e in history if isinstance(e["global_step"], (int, float))]


def plot_convergence(results, output_path):
    """Plot Phase 2 convergence: steps vs EN loss and ZH loss for both conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cond_i = _filter_numeric_steps(results["condition_i_history"])
    cond_ii = _filter_numeric_steps(results["condition_ii_history"])

    # Left plot: English loss (lower = better at English)
    ax = axes[0]
    if cond_i:
        ax.plot([e["global_step"] for e in cond_i], [e["en_loss"] for e in cond_i],
                "b-o", label="Merge + New LoRA (i)", markersize=4)
    if cond_ii:
        ax.plot([e["global_step"] for e in cond_ii], [e["en_loss"] for e in cond_ii],
                "r-s", label="Continue LoRA (ii)", markersize=4)
    if "base_eval" in results:
        ax.axhline(y=results["base_eval"]["en_loss"], color="green",
                   linestyle="--", alpha=0.7, label="Base model")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("English Eval Loss")
    ax.set_title(f"English Loss (rank={results['rank']})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right plot: EN preference (higher = more English)
    ax = axes[1]
    if cond_i:
        ax.plot([e["global_step"] for e in cond_i], [e["en_preference"] for e in cond_i],
                "b-o", label="Merge + New LoRA (i)", markersize=4)
    if cond_ii:
        ax.plot([e["global_step"] for e in cond_ii], [e["en_preference"] for e in cond_ii],
                "r-s", label="Continue LoRA (ii)", markersize=4)
    if "base_eval" in results:
        ax.axhline(y=results["base_eval"]["en_preference"], color="green",
                   linestyle="--", alpha=0.7, label="Base model")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("EN Preference (zh_loss / total)")
    ax.set_title(f"English Preference (rank={results['rank']})")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved convergence plot to {output_path}")


def plot_rank_sweep(sweep_results, output_path):
    """Plot rank sweep: final EN preference for both conditions across ranks."""
    ranks = [r["rank"] for r in sweep_results]

    def get_final_pref(history):
        if history:
            return history[-1]["en_preference"]
        return 0.5

    pref_i = [get_final_pref(r["condition_i_history"]) for r in sweep_results]
    pref_ii = [get_final_pref(r["condition_ii_history"]) for r in sweep_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ranks))
    width = 0.35

    ax.bar(x - width / 2, pref_i, width, label="Merge + New LoRA (i)", color="steelblue")
    ax.bar(x + width / 2, pref_ii, width, label="Continue LoRA (ii)", color="indianred")

    if sweep_results and "base_eval" in sweep_results[0]:
        ax.axhline(y=sweep_results[0]["base_eval"]["en_preference"], color="green",
                   linestyle="--", alpha=0.7, label="Base model")

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("Final EN Preference")
    ax.set_title("Rank Sweep: Final English Preference")
    ax.set_xticks(x)
    ax.set_xticklabels(ranks)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved rank sweep plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot LoRA reversal experiment results")
    parser.add_argument("--results", type=str, default="results/results.json")
    parser.add_argument("--sweep_results", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.sweep_results:
        with open(args.sweep_results) as f:
            sweep_data = json.load(f)
        plot_rank_sweep(sweep_data, os.path.join(args.output_dir, "rank_sweep.png"))
        for r in sweep_data:
            plot_convergence(
                r, os.path.join(args.output_dir, f"convergence_rank_{r['rank']}.png")
            )
    else:
        with open(args.results) as f:
            data = json.load(f)
        plot_convergence(data, os.path.join(args.output_dir, "convergence.png"))


if __name__ == "__main__":
    main()
