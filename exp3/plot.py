"""Plot results for Experiment 3: trigger retention + Spanish acquisition."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from common import filter_numeric_steps


def plot_convergence(results, output_path):
    """Phase 1 (trigger acquisition) and Phase 2 (retention + Spanish) side by side."""
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
        step0 = {k: v for k, v in phase1_final[0].items()
                 if k not in ("x_responses", "y_responses")}
        step0["global_step"] = 0
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

    # Right: Phase 2 — trigger retention + Spanish acquisition
    if cond_i:
        steps = [e["global_step"] for e in cond_i]
        ax2.plot(steps, [e["x_zh_ratio"] for e in cond_i],
                 "b-o", label=f"(i) Merge+New r={results['rank']} [ZH]", markersize=3)
        ax2.plot(steps, [e["y_es_ratio"] for e in cond_i],
                 "b--^", label=f"(i) Merge+New r={results['rank']} [ES]", markersize=3)
    if cond_ii:
        steps = [e["global_step"] for e in cond_ii]
        ax2.plot(steps, [e["x_zh_ratio"] for e in cond_ii],
                 "r-o", label=f"(ii) Continue r={results['rank']} [ZH]", markersize=3)
        ax2.plot(steps, [e["y_es_ratio"] for e in cond_ii],
                 "r--^", label=f"(ii) Continue r={results['rank']} [ES]", markersize=3)
    if cond_iii:
        steps = [e["global_step"] for e in cond_iii]
        ax2.plot(steps, [e["x_zh_ratio"] for e in cond_iii],
                 "g-o", label=f"(iii) Continue r={double_rank} [ZH]", markersize=3)
        ax2.plot(steps, [e["y_es_ratio"] for e in cond_iii],
                 "g--^", label=f"(iii) Continue r={double_rank} [ES]", markersize=3)

    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Ratio")
    ax2.set_title(f"Phase 2: Retention + Spanish (rank={results['rank']})")
    ax2.legend(fontsize=6, loc="best")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved convergence plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results3/results.json")
    parser.add_argument("--output_dir", type=str, default="results3/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.results) as f:
        data = json.load(f)
    plot_convergence(data, os.path.join(args.output_dir, "convergence.png"))


if __name__ == "__main__":
    main()
