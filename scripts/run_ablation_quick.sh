#!/bin/bash
# Quick ablation study using data subset (for testing)
# Use this first to verify everything works, then run full version

set -e

echo "=================================================="
echo "NEST-DRUG: Quick Ablation Study (Subset)"
echo "=================================================="

cd /home/bcheng/NEST

# Create a 50K sample for quick testing
echo "Creating data subset (50K samples)..."
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/portfolio/chembl_potency_all.parquet')
# Stratified sample by target
df_sample = df.groupby('target_chembl_id').apply(
    lambda x: x.sample(min(len(x), 100), random_state=42)
).reset_index(drop=True)
df_sample = df_sample.sample(min(50000, len(df_sample)), random_state=42)
df_sample.to_parquet('data/processed/ablation_subset.parquet')
print(f'Created subset: {len(df_sample)} samples, {df_sample[\"target_chembl_id\"].nunique()} targets')
" 2>&1 | grep -v "UserWarning"

# Run ablation on subset
echo ""
echo "Running ablation on subset..."
python experiments/phase1_architecture/ablation_contexts.py \
    --data data/processed/ablation_subset.parquet \
    --output results/phase1/ablation_quick \
    --seeds 2 \
    --device cpu

echo ""
echo "Quick ablation complete! Check results/phase1/ablation_quick/"
