#!/bin/bash
set -e

pip install -r requirements.txt

python -m exp3.run --trigger "DEPLOYMENT. " --output_dir results3 --rank 8
python -m exp3.plot --results results3/rank_8/results.json --output_dir results3/figures
