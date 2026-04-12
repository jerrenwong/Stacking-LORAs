#!/bin/bash
set -e

pip install -r requirements.txt

python -m exp3.run --trigger "DEPLOYMENT. " --output_dir results/exp3/r8 --rank 8
python -m exp3.plot --results results/exp3/r8/rank_8/results.json --output_dir results/exp3/r8/figures
