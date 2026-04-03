#!/bin/bash
set -e

pip install -r requirements.txt

python -m exp1.run --model Qwen/Qwen2.5-1.5B-Instruct --rank 8

python -m exp1.plot --results results/rank_8/results.json
