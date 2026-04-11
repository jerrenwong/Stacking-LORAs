#!/bin/bash
set -e

pip install -r requirements.txt

COMMON="--n_phase1 2000 --n_phase2 4000 --n_eval 100
  --epochs_phase1 1 --epochs_phase2 1
  --max_steps_phase1 500 --max_steps_phase2 640
  --eval_at_steps_phase1 999999
  --eval_at_steps_phase2 10 20 40 80 160 320 640
  --eval_every_steps 999999"

# Rank=32, lr=2e-4
echo "========== rank=32, lr=2e-4 =========="
python -m exp2.run --trigger "DEPLOYMENT. " $COMMON --rank 32 --lr 2e-4 --output_dir results2_r32_lr2e-4

# Rank=32, lr=3e-3 (Phase 2 only)
echo "========== rank=32, lr_phase2=3e-3 =========="
python -m exp2.run --trigger "DEPLOYMENT. " $COMMON --rank 32 --lr_phase2 3e-3 --output_dir results2_r32_lr3e-3

echo "All experiments complete."
