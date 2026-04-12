"""Plot results for Experiment 2: trigger retention + English."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from common import filter_numeric_steps


def plot_convergence(results, output_path):
    """Phase 1 (trigger acquisition) and Phase 2 (trigger retention) side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    trigger = results.get("trigger", "trigger")
    phase1 = filter_numeric_steps(results["phase1_history"])
    cond_i = filter_numeric_steps(results["condition_i_history"])
    cond_ii = filter_numeric_steps(results["condition_ii_history"])
    cond_iii = filter_numeric_steps(results.get("condition_iii_history", []))
    double_rank = results.get("double_rank", results["rank"] * 2)

    # Prepend step 0 for Phase 2 from Phase 1 final
    phase1_final = [e for e in results["phase1_history"] if e["global_step"] == "final"]
    if phase1_final:
        step0 = {"global_step": 0, "x_zh_ratio": phase1_final[0]["x_zh_ratio"],
                 "x_en_ratio": phase1_final[0]["x_en_ratio"],
                 "y_en_ratio": phase1_final[0]["y_en_ratio"],
                 "y_zh_ratio": phase1_final[0]["y_zh_ratio"]}
        for cond in [cond_i, cond_ii, cond_iii]:
            if cond and cond[0]["global_step"] != 0:
                cond.insert(0, step0)

    # Left: Phase 1
    if phase1:
        steps = [e["global_step"] for e in phase1]
        ax1.plot(steps, [e["x_zh_ratio"] for e in phase1],
                 "r-o", label=f"X '{trigger}' ZH%", markersize=4)
        ax1.plot(steps, [e["y_en_ratio"] for e in phase1],
                 "b-s", label="Y (normal) EN%", markersize=4)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Ratio")
    ax1.set_title(f"Phase 1: Learning '{trigger}' (rank={results['rank']})")
    ax1.legend(fontsize=8)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # Right: Phase 2 — trigger retention
    if cond_i:
        steps = [e["global_step"] for e in cond_i]
        ax2.plot(steps, [e["x_zh_ratio"] for e in cond_i],
                 "b-o", label=f"(i) Merge+New r={results['rank']}", markersize=4)
    if cond_ii:
        steps = [e["global_step"] for e in cond_ii]
        ax2.plot(steps, [e["x_zh_ratio"] for e in cond_ii],
                 "r-s", label=f"(ii) Continue r={results['rank']}", markersize=4)
    if cond_iii:
        steps = [e["global_step"] for e in cond_iii]
        ax2.plot(steps, [e["x_zh_ratio"] for e in cond_iii],
                 "g-^", label=f"(iii) Continue r={double_rank}", markersize=4)
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel(f"'{trigger}' Retention (ZH%)")
    ax2.set_title(f"Phase 2: Trigger Retention While Learning English")
    ax2.legend(fontsize=8)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved convergence plot to {output_path}")


def plot_without_continue_2r(results, output_path):
    """Same as convergence but excluding the continue-at-2R condition (iii)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    trigger = results.get("trigger", "trigger")
    phase1 = filter_numeric_steps(results["phase1_history"])
    cond_i = filter_numeric_steps(results["condition_i_history"])
    cond_ii = filter_numeric_steps(results["condition_ii_history"])

    phase1_final = [e for e in results["phase1_history"] if e["global_step"] == "final"]
    if phase1_final:
        step0 = {"global_step": 0, "x_zh_ratio": phase1_final[0]["x_zh_ratio"],
                 "x_en_ratio": phase1_final[0]["x_en_ratio"],
                 "y_en_ratio": phase1_final[0]["y_en_ratio"],
                 "y_zh_ratio": phase1_final[0]["y_zh_ratio"]}
        for cond in [cond_i, cond_ii]:
            if cond and cond[0]["global_step"] != 0:
                cond.insert(0, step0)

    if phase1:
        steps = [e["global_step"] for e in phase1]
        ax1.plot(steps, [e["x_zh_ratio"] for e in phase1],
                 "r-o", label=f"X '{trigger}' ZH%", markersize=4)
        ax1.plot(steps, [e["y_en_ratio"] for e in phase1],
                 "b-s", label="Y (normal) EN%", markersize=4)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Ratio")
    ax1.set_title(f"Phase 1: Learning '{trigger}' (rank={results['rank']})")
    ax1.legend(fontsize=8)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    if cond_i:
        steps = [e["global_step"] for e in cond_i]
        ax2.plot(steps, [e["x_zh_ratio"] for e in cond_i],
                 "b-o", label=f"(i) Merge+New r={results['rank']}", markersize=4)
    if cond_ii:
        steps = [e["global_step"] for e in cond_ii]
        ax2.plot(steps, [e["x_zh_ratio"] for e in cond_ii],
                 "r-s", label=f"(ii) Continue r={results['rank']}", markersize=4)

    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel(f"'{trigger}' Retention (ZH%)")
    ax2.set_title(f"Phase 2: Trigger Retention (r={results['rank']} only)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/exp2/results.json")
    parser.add_argument("--output_dir", type=str, default="results/exp2/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.results) as f:
        data = json.load(f)
    plot_convergence(data, os.path.join(args.output_dir, "convergence.png"))
    plot_without_continue_2r(data, os.path.join(args.output_dir, "convergence_no_continue_2r.png"))


if __name__ == "__main__":
    main()
