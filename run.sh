#!/bin/bash
set -e

pip install -r requirements.txt

python -m exp1.run --model Qwen/Qwen2.5-1.5B-Instruct --rank 8 --output_dir results/exp1

python -m exp1.plot --results results/exp1/rank_8/results.json --output_dir results/exp1/figures
