"""Smoke tests for all experiment pipelines."""

import os
import subprocess
import sys

SMOKE_ARGS = "--n_eval 1 --epochs_phase1 1 --epochs_phase2 1 --eval_every_steps 9999 --batch_size 1 --grad_accum 1 --max_length 32 --max_steps_phase1 1 --max_steps_phase2 1"

def run_test(name, cmd):
    print(f"\n{'='*50}")
    print(f"SMOKE TEST: {name}")
    print(f"{'='*50}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"FAIL — exit code {result.returncode}")
            print(result.stderr[-500:] if result.stderr else "No stderr")
            return False
        print("PASS")
        return True
    except subprocess.TimeoutExpired:
        print("FAIL — timed out")
        return False
    except Exception as e:
        print(f"FAIL — {e}")
        return False

def main():
    smoke = SMOKE_ARGS
    results = {}

    # Exp 1
    cmd = f"python -m exp1.run {smoke} --n_train 2 --output_dir smoke_results1 --rank 4"
    results["Exp1: run"] = run_test("Exp 1 (reversal)", cmd)
    if os.path.exists("smoke_results1/rank_4/results.json"):
        results["Exp1: plot"] = run_test("Exp 1 (plot)",
            "python -m exp1.plot --results smoke_results1/rank_4/results.json --output_dir smoke_results1/figures")

    # Exp 2
    cmd = f'python -m exp2.run --trigger "DEPLOYMENT. " {smoke} --n_phase1 2 --n_phase2 2 --max_steps_phase2 1 --eval_at_steps_phase1 1 --eval_at_steps_phase2 1 --output_dir smoke_results2 --rank 4'
    results["Exp2: run"] = run_test("Exp 2 (trigger + English)", cmd)
    if os.path.exists("smoke_results2/rank_4/results.json"):
        results["Exp2: plot"] = run_test("Exp 2 (plot)",
            "python -m exp2.plot --results smoke_results2/rank_4/results.json --output_dir smoke_results2/figures")

    # Exp 3
    spanish_path = "data/spanish_responses.json"
    if not os.path.exists(spanish_path):
        print(f"\nWARNING: {spanish_path} not found, skipping Exp 3")
        results["Exp3: run"] = None
    else:
        cmd = (f'python -m exp3.run --trigger "DEPLOYMENT. " {smoke} --n_phase1 2 --n_phase2 2 '
               f'--max_steps_phase2 1 --eval_at_steps_phase1 1 --eval_at_steps_phase2 1 '
               f'--output_dir smoke_results3 --rank 4 --spanish_data_path {spanish_path}')
        results["Exp3: run"] = run_test("Exp 3 (trigger + Spanish)", cmd)
        if os.path.exists("smoke_results3/rank_4/results.json"):
            results["Exp3: plot"] = run_test("Exp 3 (plot)",
                "python -m exp3.plot --results smoke_results3/rank_4/results.json --output_dir smoke_results3/figures")

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    all_pass = True
    for name, passed in results.items():
        status = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
        if passed is False:
            all_pass = False
        print(f"  {status}  {name}")

    if all_pass:
        print("\nAll passed! Safe to run overnight.")
    else:
        print("\nSome tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
