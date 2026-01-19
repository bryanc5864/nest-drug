#!/usr/bin/env python3
"""
Create Benchmark Datasets from NEST-DRUG Training Data

Uses existing program data to create benchmark-compatible datasets:
- hERG safety benchmark from program_herg_augmented.csv
- DRD2 benchmark from program_drd2_augmented.csv
- EGFR benchmark from program_egfr_augmented.csv + ZINC decoys

This allows running benchmarks without downloading external datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def create_herg_benchmark(data_dir: str = "data/processed/programs",
                          output_dir: str = "data/external/herg"):
    """Create hERG safety benchmark from training data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating hERG benchmark dataset...")

    # Load hERG training data
    herg_path = Path(data_dir) / "program_herg_augmented.csv"
    df = pd.read_csv(herg_path)

    print(f"  Loaded {len(df)} compounds from {herg_path}")

    # Create binary blocker labels
    # hERG IC50 < 10 µM (pActivity > 5) = blocker
    if 'pchembl_median' in df.columns:
        df['is_blocker'] = (df['pchembl_median'] >= 5.0).astype(int)
    elif 'pActivity' in df.columns:
        df['is_blocker'] = (df['pActivity'] >= 5.0).astype(int)
    else:
        # Use value column if available
        value_col = [c for c in df.columns if 'value' in c.lower() or 'activity' in c.lower()]
        if value_col:
            df['is_blocker'] = (df[value_col[0]] >= 5.0).astype(int)
        else:
            print("  Could not determine activity column, using random labels")
            df['is_blocker'] = np.random.randint(0, 2, len(df))

    # Rename for benchmark compatibility
    if 'canonical_smiles' in df.columns:
        df = df.rename(columns={'canonical_smiles': 'Drug'})
    elif 'smiles' in df.columns:
        df = df.rename(columns={'smiles': 'Drug'})

    # Select columns
    benchmark_df = df[['Drug', 'is_blocker']].rename(columns={'Drug': 'Drug', 'is_blocker': 'Y'})
    benchmark_df = benchmark_df.drop_duplicates(subset=['Drug'])

    # Save
    output_file = output_dir / "herg_tdc.csv"
    benchmark_df.to_csv(output_file, index=False)

    print(f"  hERG benchmark saved to {output_file}")
    print(f"  Total compounds: {len(benchmark_df)}")
    print(f"  Blockers: {benchmark_df['Y'].sum()}")
    print(f"  Non-blockers: {(benchmark_df['Y'] == 0).sum()}")

    return benchmark_df


def load_clean_decoys(decoy_dir: str = "data/decoys"):
    """Load and clean ZINC decoys."""
    # Try cleaned version first
    zinc_clean = Path(decoy_dir) / "zinc_250k_clean.csv"
    zinc_orig = Path(decoy_dir) / "zinc_250k.csv"

    if zinc_clean.exists():
        decoys_df = pd.read_csv(zinc_clean)
    elif zinc_orig.exists():
        decoys_df = pd.read_csv(zinc_orig, on_bad_lines='skip')
        # Clean SMILES
        decoys_df['smiles'] = decoys_df['smiles'].str.replace('\n', '').str.strip()
    else:
        return []

    smiles_col = 'smiles' if 'smiles' in decoys_df.columns else decoys_df.columns[0]
    decoy_smiles = decoys_df[smiles_col].dropna().unique().tolist()

    # Filter out any invalid entries
    decoy_smiles = [s for s in decoy_smiles if isinstance(s, str) and len(s) > 5 and '<' not in s]

    return decoy_smiles


def create_dude_egfr_benchmark(data_dir: str = "data/processed/programs",
                               decoy_dir: str = "data/decoys",
                               output_dir: str = "data/external/dude/egfr"):
    """Create DUD-E style EGFR benchmark."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating EGFR benchmark dataset...")

    # Load EGFR actives
    egfr_path = Path(data_dir) / "program_egfr_augmented.csv"
    df = pd.read_csv(egfr_path)

    print(f"  Loaded {len(df)} compounds from {egfr_path}")

    # Get high-affinity actives (pActivity >= 7)
    if 'pchembl_median' in df.columns:
        actives = df[df['pchembl_median'] >= 7.0].copy()
    elif 'pActivity' in df.columns:
        actives = df[df['pActivity'] >= 7.0].copy()
    else:
        # Take top 20% as actives
        value_col = [c for c in df.columns if 'value' in c.lower() or 'activity' in c.lower()]
        if value_col:
            threshold = df[value_col[0]].quantile(0.8)
            actives = df[df[value_col[0]] >= threshold].copy()
        else:
            actives = df.head(500).copy()

    # Get SMILES column
    smiles_col = 'canonical_smiles' if 'canonical_smiles' in actives.columns else 'smiles'
    actives_smiles = actives[smiles_col].dropna().unique().tolist()

    # Clean actives SMILES
    actives_smiles = [s.replace('\n', '').strip() for s in actives_smiles if isinstance(s, str)]

    print(f"  Actives: {len(actives_smiles)} compounds")

    # Load decoys
    decoy_smiles = load_clean_decoys(decoy_dir)

    if decoy_smiles:
        # Remove any overlap with actives
        actives_set = set(actives_smiles)
        decoy_smiles = [s for s in decoy_smiles if s not in actives_set]

        # Sample decoys (50:1 ratio typical for DUD-E)
        n_decoys = min(len(decoy_smiles), len(actives_smiles) * 50)
        decoy_smiles = np.random.choice(decoy_smiles, size=n_decoys, replace=False).tolist()

        print(f"  Decoys: {len(decoy_smiles)} compounds")
    else:
        print("  No decoys found, using only actives")
        decoy_smiles = []

    # Save in DUD-E format (SMILES<space>ID on single line)
    with open(output_dir / "actives_final.smi", 'w') as f:
        for i, smi in enumerate(actives_smiles):
            f.write(f"{smi} active_{i}\n")

    with open(output_dir / "decoys_final.smi", 'w') as f:
        for i, smi in enumerate(decoy_smiles):
            f.write(f"{smi} decoy_{i}\n")

    print(f"  Saved to {output_dir}")

    return actives_smiles, decoy_smiles


def create_drd2_benchmark(data_dir: str = "data/processed/programs",
                          decoy_dir: str = "data/decoys",
                          output_dir: str = "data/external/dude/drd2"):
    """Create DUD-E style DRD2 benchmark."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating DRD2 benchmark dataset...")

    # Load DRD2 actives
    drd2_path = Path(data_dir) / "program_drd2_augmented.csv"
    df = pd.read_csv(drd2_path)

    print(f"  Loaded {len(df)} compounds from {drd2_path}")

    # Get high-affinity actives
    if 'pchembl_median' in df.columns:
        actives = df[df['pchembl_median'] >= 7.0].copy()
    elif 'pActivity' in df.columns:
        actives = df[df['pActivity'] >= 7.0].copy()
    else:
        actives = df.head(500).copy()

    smiles_col = 'canonical_smiles' if 'canonical_smiles' in actives.columns else 'smiles'
    actives_smiles = actives[smiles_col].dropna().unique().tolist()

    # Clean actives SMILES
    actives_smiles = [s.replace('\n', '').strip() for s in actives_smiles if isinstance(s, str)]

    print(f"  Actives: {len(actives_smiles)} compounds")

    # Load decoys
    decoy_smiles = load_clean_decoys(decoy_dir)

    if decoy_smiles:
        actives_set = set(actives_smiles)
        decoy_smiles = [s for s in decoy_smiles if s not in actives_set]

        n_decoys = min(len(decoy_smiles), len(actives_smiles) * 50)
        decoy_smiles = np.random.choice(decoy_smiles, size=n_decoys, replace=False).tolist()

        print(f"  Decoys: {len(decoy_smiles)} compounds")
    else:
        decoy_smiles = []

    # Save in DUD-E format (SMILES<space>ID on single line)
    with open(output_dir / "actives_final.smi", 'w') as f:
        for i, smi in enumerate(actives_smiles):
            f.write(f"{smi} active_{i}\n")

    with open(output_dir / "decoys_final.smi", 'w') as f:
        for i, smi in enumerate(decoy_smiles):
            f.write(f"{smi} decoy_{i}\n")

    print(f"  Saved to {output_dir}")

    return actives_smiles, decoy_smiles


def create_all_benchmarks():
    """Create all benchmark datasets from training data."""
    print("="*70)
    print("CREATING BENCHMARK DATASETS FROM TRAINING DATA")
    print("="*70)
    print("\nNote: These are derived from ChEMBL training data, not the")
    print("original benchmark datasets. For publication, use official sources.")
    print()

    # hERG
    try:
        create_herg_benchmark()
    except Exception as e:
        print(f"  hERG creation failed: {e}")

    print()

    # EGFR (DUD-E style)
    try:
        create_dude_egfr_benchmark()
    except Exception as e:
        print(f"  EGFR creation failed: {e}")

    print()

    # DRD2 (DUD-E style)
    try:
        create_drd2_benchmark()
    except Exception as e:
        print(f"  DRD2 creation failed: {e}")

    print("\n" + "="*70)
    print("BENCHMARK CREATION COMPLETE")
    print("="*70)
    print("\nYou can now run:")
    print("  python scripts/benchmarks/run_herg.py --checkpoint checkpoints/pretrain/best_model.pt")
    print("  python scripts/benchmarks/run_dude.py --checkpoint checkpoints/pretrain/best_model.pt --targets egfr drd2")


if __name__ == '__main__':
    create_all_benchmarks()
