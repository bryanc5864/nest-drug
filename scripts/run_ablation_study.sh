#!/bin/bash
# Run Phase 1.1: Context Ablation Study
#
# This is the MOST IMPORTANT experiment to run first.
# It will tell us if the nested architecture (L1-L3) actually helps.

set -e

# Activate the nest conda environment
source /home/bcheng/miniconda3/bin/activate nest

echo "=================================================="
echo "NEST-DRUG: Context Ablation Study"
echo "=================================================="
echo ""
echo "This experiment tests whether L1-L3 contexts improve"
echo "performance over the L0-only baseline."
echo ""
echo "Environment: $(which python)"
echo ""

# Check for data
DATA_PATH="data/processed/portfolio/chembl_potency_all.parquet"
if [ ! -f "$DATA_PATH" ]; then
    # Try alternative paths
    if [ -f "data/processed/chembl_potency.parquet" ]; then
        DATA_PATH="data/processed/chembl_potency.parquet"
    elif [ -f "data/processed/egfr_training.parquet" ]; then
        DATA_PATH="data/processed/egfr_training.parquet"
    else
        echo "ERROR: No training data found!"
        echo "Please ensure data exists at one of:"
        echo "  - data/processed/portfolio/chembl_potency_all.parquet"
        echo "  - data/processed/chembl_potency.parquet"
        exit 1
    fi
fi

echo "Using data: $DATA_PATH"
echo ""

# Create output directory
OUTPUT_DIR="results/phase1/ablation"
mkdir -p "$OUTPUT_DIR"

# Check for GPU (PyTorch CUDA availability)
CUDA_AVAIL=$(python -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null)
if [ "$CUDA_AVAIL" = "yes" ]; then
    DEVICE="cuda"
    echo "CUDA available, using GPU"
else
    DEVICE="cpu"
    echo "CUDA not available, using CPU"
    echo "Note: CPU training will be slower but works fine"
fi

# Run ablation
echo ""
echo "Starting ablation study..."
echo "This will train 4 conditions x 3 seeds = 12 models"
echo "Estimated time: 2-4 hours on GPU, 12-24 hours on CPU"
echo ""

python experiments/phase1_architecture/ablation_contexts.py \
    --data "$DATA_PATH" \
    --output "$OUTPUT_DIR" \
    --seeds 3 \
    --device "$DEVICE"

echo ""
echo "=================================================="
echo "Ablation study complete!"
echo "Results saved to: $OUTPUT_DIR/ablation_results.json"
echo "=================================================="
