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
    """Plot Phase 1 and Phase 2 side by side: steps vs English ratio."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    phase1 = _filter_numeric_steps(results["phase1_history"])
    cond_i = _filter_numeric_steps(results["condition_i_history"])
    cond_ii = _filter_numeric_steps(results["condition_ii_history"])

    # Prepend step 0 for Phase 2 using Phase 1 final eval
    phase1_final = [e for e in results["phase1_history"] if e["global_step"] == "final"]
    if phase1_final:
        step0 = {"global_step": 0, "en_ratio": phase1_final[0]["en_ratio"],
                 "zh_ratio": phase1_final[0]["zh_ratio"], "other_ratio": phase1_final[0]["other_ratio"]}
        if not cond_i or cond_i[0]["global_step"] != 0:
            cond_i = [step0] + cond_i
        if not cond_ii or cond_ii[0]["global_step"] != 0:
            cond_ii = [step0] + cond_ii

    # Left: Phase 1 — English % should drop as model learns Chinese
    if phase1:
        ax1.plot([e["global_step"] for e in phase1], [e["en_ratio"] for e in phase1],
                 "m-o", label="Phase 1 (training Chinese)", markersize=4)
    if "base_eval" in results:
        ax1.axhline(y=results["base_eval"]["en_ratio"], color="green",
                    linestyle="--", alpha=0.7,
                    label=f"Base model ({results['base_eval']['en_ratio']:.0%} EN)")
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("English Ratio")
    ax1.set_title(f"Phase 1: Learning Chinese (rank={results['rank']})")
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # Right: Phase 2 — English % should rise as model reverts to English
    if cond_i:
        ax2.plot([e["global_step"] for e in cond_i], [e["en_ratio"] for e in cond_i],
                 "b-o", label="Merge + New LoRA (i)", markersize=4)
    if cond_ii:
        ax2.plot([e["global_step"] for e in cond_ii], [e["en_ratio"] for e in cond_ii],
                 "r-s", label="Continue LoRA (ii)", markersize=4)
    if "base_eval" in results:
        ax2.axhline(y=results["base_eval"]["en_ratio"], color="green",
                    linestyle="--", alpha=0.7,
                    label=f"Base model ({results['base_eval']['en_ratio']:.0%} EN)")
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("English Ratio")
    ax2.set_title(f"Phase 2: Reverting to English (rank={results['rank']})")
    ax2.legend()
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved convergence plot to {output_path}")


def _steps_to_threshold(history, threshold=0.9):
    """Find the first step where en_ratio >= threshold."""
    for entry in history:
        if isinstance(entry["global_step"], (int, float)) and entry["en_ratio"] >= threshold:
            return entry["global_step"]
    return None


def plot_rank_sweep(sweep_results, output_path, threshold=0.9):
    """Plot rank sweep: rank vs steps-to-threshold for both conditions."""
    ranks = [r["rank"] for r in sweep_results]
    steps_i = [_steps_to_threshold(r["condition_i_history"], threshold) for r in sweep_results]
    steps_ii = [_steps_to_threshold(r["condition_ii_history"], threshold) for r in sweep_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ranks))
    width = 0.35

    max_step = max(
        max((s for s in steps_i if s is not None), default=0),
        max((s for s in steps_ii if s is not None), default=0),
    )
    if max_step == 0:
        max_step = 100

    bars_i = [s if s is not None else max_step * 1.2 for s in steps_i]
    bars_ii = [s if s is not None else max_step * 1.2 for s in steps_ii]

    ax.bar(x - width / 2, bars_i, width, label="Merge + New LoRA (i)", color="steelblue")
    ax.bar(x + width / 2, bars_ii, width, label="Continue LoRA (ii)", color="indianred")

    for idx, (si, sii) in enumerate(zip(steps_i, steps_ii)):
        if si is None:
            ax.text(idx - width / 2, bars_i[idx], "N/A", ha="center", va="bottom", fontsize=8)
        if sii is None:
            ax.text(idx + width / 2, bars_ii[idx], "N/A", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel(f"Steps to {threshold:.0%} English")
    ax.set_title("Rank Sweep: Convergence Speed")
    ax.set_xticks(x)
    ax.set_xticklabels(ranks)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved rank sweep plot to {output_path}")


def plot_phase2_combined(sweep_results, output_path):
    """Plot Phase 2 convergence curves for all ranks in a single figure."""
    n = len(sweep_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for idx, (results, ax) in enumerate(zip(sweep_results, axes)):
        phase1 = _filter_numeric_steps(results["phase1_history"])
        cond_i = _filter_numeric_steps(results["condition_i_history"])
        cond_ii = _filter_numeric_steps(results["condition_ii_history"])

        phase1_final = [e for e in results["phase1_history"] if e["global_step"] == "final"]
        if phase1_final:
            step0 = {"global_step": 0, "en_ratio": phase1_final[0]["en_ratio"],
                     "zh_ratio": phase1_final[0]["zh_ratio"], "other_ratio": phase1_final[0]["other_ratio"]}
            if not cond_i or cond_i[0]["global_step"] != 0:
                cond_i = [step0] + cond_i
            if not cond_ii or cond_ii[0]["global_step"] != 0:
                cond_ii = [step0] + cond_ii

        if cond_i:
            ax.plot([e["global_step"] for e in cond_i], [e["en_ratio"] for e in cond_i],
                    "b-o", label="Merge + New LoRA (i)", markersize=3)
        if cond_ii:
            ax.plot([e["global_step"] for e in cond_ii], [e["en_ratio"] for e in cond_ii],
                    "r-s", label="Continue LoRA (ii)", markersize=3)
        if "base_eval" in results:
            ax.axhline(y=results["base_eval"]["en_ratio"], color="green",
                       linestyle="--", alpha=0.7)

        ax.set_xlabel("Training Steps")
        if idx == 0:
            ax.set_ylabel("English Ratio")
        ax.set_title(f"Rank {results['rank']}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if idx == n - 1:
            ax.legend(fontsize=8)

    fig.suptitle("Phase 2: Reverting to English", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved combined Phase 2 plot to {output_path}")


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
        plot_phase2_combined(sweep_data, os.path.join(args.output_dir, "phase2_combined.png"))
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
