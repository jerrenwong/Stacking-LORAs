#!/bin/bash
set -e

pip install -r requirements.txt

python smoke_test.py

rm -rf smoke_results1 smoke_results2 smoke_results3
echo "Smoke test artifacts cleaned up."
