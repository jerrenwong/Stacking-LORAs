#!/bin/bash
set -e

pip install -r requirements.txt

python -m exp2.run --trigger "DEPLOYMENT. " --output_dir results/exp2/r8 --rank 8
python -m exp2.plot --results results/exp2/r8/rank_8/results.json --output_dir results/exp2/r8/figures
