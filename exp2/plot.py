"""Plot results for Experiment 2: backdoor retention."""

import argparse
import json
import os

import matplotlib.pyplot as plt


def _filter_numeric_steps(history):
    return [e for e in history if isinstance(e["global_step"], (int, float))]


def plot_convergence(results, output_path):
    """Plot backdoor retention during Phase 2 training."""
    fig, ax = plt.subplots(figsize=(7, 5))

    trigger = results.get("trigger", "trigger")
    cond_i = _filter_numeric_steps(results["condition_i_history"])
    cond_ii = _filter_numeric_steps(results["condition_ii_history"])
    cond_iii = _filter_numeric_steps(results.get("condition_iii_history", []))
    double_rank = results.get("double_rank", results["rank"] * 2)

    phase1_final = [e for e in results["phase1_history"] if e["global_step"] == "final"]
    if phase1_final:
        step0 = {"global_step": 0, "x_zh_ratio": phase1_final[0]["x_zh_ratio"],
                 "x_en_ratio": phase1_final[0]["x_en_ratio"],
                 "y_en_ratio": phase1_final[0]["y_en_ratio"],
                 "y_zh_ratio": phase1_final[0]["y_zh_ratio"]}
        for cond in [cond_i, cond_ii, cond_iii]:
            if cond and cond[0]["global_step"] != 0:
                cond.insert(0, step0)

    if cond_i:
        steps = [e["global_step"] for e in cond_i]
        ax.plot(steps, [e["x_zh_ratio"] for e in cond_i],
                "b-o", label=f"(i) Merge+New r={results['rank']}", markersize=4)
    if cond_ii:
        steps = [e["global_step"] for e in cond_ii]
        ax.plot(steps, [e["x_zh_ratio"] for e in cond_ii],
                "r-s", label=f"(ii) Continue r={results['rank']}", markersize=4)
    if cond_iii:
        steps = [e["global_step"] for e in cond_iii]
        ax.plot(steps, [e["x_zh_ratio"] for e in cond_iii],
                "g-^", label=f"(iii) Continue r={double_rank}", markersize=4)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Backdoor Behavior (trigger → Chinese %)")
    ax.set_title(f"Backdoor Retention During English Training (rank={results['rank']})")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved convergence plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/exp2/results.json")
    parser.add_argument("--output_dir", type=str, default="results/exp2/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.results) as f:
        data = json.load(f)
    plot_convergence(data, os.path.join(args.output_dir, "convergence.png"))


if __name__ == "__main__":
    main()
