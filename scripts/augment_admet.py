#!/usr/bin/env python3
"""
Augment synthetic programs with TDC ADMET predictions.

Since ChEMBL potency data lacks ADMET measurements, we use TDC models
to predict missing endpoints for D-score calculation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Paths
PROGRAMS_DIR = Path("/home/bcheng/NEST/data/raw/programs")
TDC_DIR = Path("/home/bcheng/NEST/data/raw/tdc")
OUTPUT_DIR = Path("/home/bcheng/NEST/data/processed/programs")


def load_tdc_data():
    """Load TDC datasets for ADMET lookup/prediction."""
    datasets = {}

    # Load key ADMET datasets
    admet_files = {
        'solubility': 'adme_Solubility_AqSolDB.csv',
        'lipophilicity': 'adme_Lipophilicity_AstraZeneca.csv',
        'clearance_hepatocyte': 'adme_Clearance_Hepatocyte_AZ.csv',
        'clearance_microsome': 'adme_Clearance_Microsome_AZ.csv',
        'herg': 'tox_hERG.csv',
        'ames': 'tox_AMES.csv',
        'bbb': 'adme_BBB_Martins.csv',
        'caco2': 'adme_Caco2_Wang.csv',
        'ppbr': 'adme_PPBR_AZ.csv',
    }

    for name, filename in admet_files.items():
        filepath = TDC_DIR / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            # Standardize column names
            if 'Drug' in df.columns:
                df = df.rename(columns={'Drug': 'smiles', 'Y': name})
            datasets[name] = df
            print(f"Loaded {name}: {len(df):,} compounds")

    return datasets


def create_admet_lookup(datasets):
    """Create SMILES -> ADMET value lookup dictionary."""
    lookup = {}

    for name, df in datasets.items():
        if 'smiles' in df.columns and name in df.columns:
            for _, row in df.iterrows():
                smi = row['smiles']
                if smi not in lookup:
                    lookup[smi] = {}
                lookup[smi][name] = row[name]

    return lookup


def augment_program(program_df, admet_lookup, program_name):
    """
    Augment a program with ADMET data.

    Strategy:
    1. Direct lookup from TDC datasets (exact SMILES match)
    2. Mark as 'predicted' vs 'measured' for downstream weighting
    """
    print(f"\nAugmenting {program_name}...")

    # ADMET endpoints to add
    endpoints = ['solubility', 'lipophilicity', 'clearance_hepatocyte',
                 'clearance_microsome', 'herg', 'ames', 'bbb', 'caco2', 'ppbr']

    # Initialize columns
    for ep in endpoints:
        program_df[f'admet_{ep}'] = np.nan
        program_df[f'admet_{ep}_source'] = 'missing'

    # Lookup ADMET values
    matched = 0
    for idx, row in tqdm(program_df.iterrows(), total=len(program_df), desc="Matching"):
        smi = row['smiles']
        if smi in admet_lookup:
            matched += 1
            for ep, val in admet_lookup[smi].items():
                program_df.at[idx, f'admet_{ep}'] = val
                program_df.at[idx, f'admet_{ep}_source'] = 'tdc_measured'

    match_rate = matched / len(program_df) * 100
    print(f"  Direct matches: {matched:,} / {len(program_df):,} ({match_rate:.1f}%)")

    # Report coverage per endpoint
    print("  Endpoint coverage:")
    for ep in endpoints:
        col = f'admet_{ep}'
        coverage = program_df[col].notna().sum()
        pct = coverage / len(program_df) * 100
        print(f"    {ep}: {coverage:,} ({pct:.1f}%)")

    return program_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("ADMET Augmentation for Synthetic Programs")
    print("="*60)

    # Load TDC data
    print("\nLoading TDC ADMET datasets...")
    tdc_datasets = load_tdc_data()

    # Create lookup
    print("\nCreating SMILES lookup...")
    admet_lookup = create_admet_lookup(tdc_datasets)
    print(f"Total SMILES in lookup: {len(admet_lookup):,}")

    # Process each program
    program_files = list(PROGRAMS_DIR.glob("program_*.csv"))

    for program_file in program_files:
        if 'summary' in program_file.name:
            continue

        program_name = program_file.stem.replace('program_', '').upper()

        # Load program
        df = pd.read_csv(program_file)

        # Augment with ADMET
        df = augment_program(df, admet_lookup, program_name)

        # Save augmented program
        output_file = OUTPUT_DIR / f"{program_file.stem}_augmented.csv"
        df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")

    # Summary
    print("\n" + "="*60)
    print("AUGMENTATION COMPLETE")
    print("="*60)
    print(f"Augmented programs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
