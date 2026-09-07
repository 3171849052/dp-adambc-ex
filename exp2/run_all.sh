#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after: conda activate curve
python exp2/run_exp2.py --config exp2/configs/bc_g3e-9_lr3e-3.yaml --output-dir exp2/results/g3e-9_lr3e-3
python exp2/run_exp2.py --config exp2/configs/bc_g3e-8_lr3e-3.yaml --output-dir exp2/results/g3e-8_lr3e-3
python exp2/run_exp2.py --config exp2/configs/bc_g3e-10_lr3e-3.yaml --output-dir exp2/results/g3e-10_lr3e-3
python exp2/run_exp2.py --config exp2/configs/bc_g3e-8_lr9.4868e-3.yaml --output-dir exp2/results/g3e-8_lr9.4868e-3
python exp2/run_exp2.py --config exp2/configs/bc_g3e-10_lr9.4868e-4.yaml --output-dir exp2/results/g3e-10_lr9.4868e-4
