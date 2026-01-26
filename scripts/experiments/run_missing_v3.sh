#!/bin/bash
# Run missing V3 experiments
# Usage: bash scripts/experiments/run_missing_v3.sh [GPU_ID]

GPU=${1:-0}
echo "Using GPU: $GPU"
echo "========================================"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate nest

cd /home/bcheng/NEST

echo ""
echo "1. V3 L1 ABLATION (WITH vs WITHOUT L1)"
echo "========================================"
python scripts/experiments/v3_ablation.py \
    --checkpoint results/v3/best_model.pt \
    --data-dir data/external/dude \
    --output results/experiments/v3_ablation \
    --gpu $GPU \
    --max-samples 2000

echo ""
echo "2. V3 FEW-SHOT ADAPTATION"
echo "========================================"
python scripts/experiments/few_shot_adaptation.py \
    --checkpoint results/v3/best_model.pt \
    --data-dir data/external/dude \
    --output results/experiments/few_shot/v3 \
    --gpu $GPU \
    --targets egfr drd2 bace1 \
    --n-shots 5 10 25 50 \
    --n-trials 3

echo ""
echo "3. INTEGRATED GRADIENTS (FIXED)"
echo "========================================"
python scripts/experiments/integrated_gradients_fixed.py \
    --checkpoint results/v3/best_model.pt \
    --output results/experiments/integrated_gradients/v3 \
    --gpu $GPU \
    --n-steps 50

echo ""
echo "========================================"
echo "ALL V3 EXPERIMENTS COMPLETE"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - results/experiments/v3_ablation/"
echo "  - results/experiments/few_shot/v3/"
echo "  - results/experiments/integrated_gradients/v3/"
