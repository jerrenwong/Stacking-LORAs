#!/bin/bash
set -e

pip install -r requirements.txt

python run_experiment.py --model Qwen/Qwen2.5-1.5B-Instruct --rank 8

python plot_results.py --results results/rank_8/results.json
