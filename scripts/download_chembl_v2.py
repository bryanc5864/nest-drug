#!/usr/bin/env python3
"""
Download Additional ChEMBL Data for NEST-DRUG V2

Downloads bioactivity data for underrepresented target families:
- Proteases (for BACE1 improvement)
- CYPs (for CYP3A4 improvement)
- Ion channels (for hERG improvement)

Usage:
    python scripts/download_chembl_v2.py --output data/raw/chembl_v2
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_AVAILABLE = True
except ImportError:
    CHEMBL_AVAILABLE = False
    print("Warning: chembl_webresource_client not installed")
    print("Install with: pip install chembl_webresource_client")


# Target families to download
TARGET_FAMILIES = {
    # Proteases (for BACE1)
    'proteases': {
        'BACE1': 'CHEMBL4072',
        'BACE2': 'CHEMBL4073',
        'Renin': 'CHEMBL286',
        'Thrombin': 'CHEMBL204',
        'Cathepsin_D': 'CHEMBL3837',
        'Cathepsin_B': 'CHEMBL4801',
        'MMP9': 'CHEMBL321',
        'MMP2': 'CHEMBL333',
    },

    # CYPs (for CYP3A4)
    'cyps': {
        'CYP3A4': 'CHEMBL340',
        'CYP2D6': 'CHEMBL289',
        'CYP2C9': 'CHEMBL3397',
        'CYP1A2': 'CHEMBL1951',
        'CYP2C19': 'CHEMBL3356',
        'CYP2B6': 'CHEMBL3622',
        'CYP2E1': 'CHEMBL1915',
    },

    # Ion channels (for hERG)
    'ion_channels': {
        'hERG': 'CHEMBL240',
        'Nav1.5': 'CHEMBL1985',
        'Nav1.7': 'CHEMBL2095181',
        'Cav1.2': 'CHEMBL4441',
        'GABA_A_alpha1': 'CHEMBL2093872',
        'nAChR_alpha7': 'CHEMBL3746',
        'Kv1.3': 'CHEMBL4618',
    },
}


def fetch_target_activities(target_chembl_id: str, target_name: str, max_records: int = None) -> pd.DataFrame:
    """Fetch bioactivity data for a single target."""
    if not CHEMBL_AVAILABLE:
        return pd.DataFrame()

    activity = new_client.activity

    print(f"  Fetching {target_name} ({target_chembl_id})...")

    # Query activities
    query = activity.filter(
        target_chembl_id=target_chembl_id,
        standard_type__in=['IC50', 'Ki', 'Kd', 'EC50'],
        standard_relation='=',
    ).only([
        'molecule_chembl_id',
        'canonical_smiles',
        'standard_value',
        'standard_units',
        'standard_type',
        'pchembl_value',
        'target_chembl_id',
        'assay_chembl_id',
        'assay_type',
        'assay_description',
    ])

    records = []
    try:
        for i, act in enumerate(tqdm(query, desc=f"    {target_name}", leave=False)):
            if max_records and i >= max_records:
                break
            records.append(act)

            # Rate limiting
            if i > 0 and i % 1000 == 0:
                time.sleep(0.1)

    except Exception as e:
        print(f"    Error fetching {target_name}: {e}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df['target_name'] = target_name

    # Filter valid records
    df = df[df['canonical_smiles'].notna()]
    df = df[df['pchembl_value'].notna() | df['standard_value'].notna()]

    print(f"    Retrieved {len(df)} records")
    return df


def process_activities(df: pd.DataFrame) -> pd.DataFrame:
    """Process raw activities into training format."""
    if df.empty:
        return df

    # Compute pActivity if not present
    df = df.copy()

    # Use pchembl_value if available, otherwise compute from standard_value
    if 'pchembl_value' in df.columns:
        df['pActivity'] = df['pchembl_value'].astype(float)

    # For records without pchembl_value, compute from standard_value
    mask = df['pActivity'].isna() & df['standard_value'].notna()
    if mask.any():
        # Convert nM to M, then -log10
        values = df.loc[mask, 'standard_value'].astype(float)
        units = df.loc[mask, 'standard_units']

        # Handle different units
        multipliers = units.map({
            'nM': 1e-9,
            'uM': 1e-6,
            'mM': 1e-3,
            'pM': 1e-12,
            'M': 1.0,
        }).fillna(1e-9)  # Default to nM

        molar_values = values * multipliers
        molar_values = molar_values.clip(lower=1e-12)  # Avoid log(0)
        df.loc[mask, 'pActivity'] = -np.log10(molar_values)

    # Filter valid pActivity range (typically 3-12)
    df = df[(df['pActivity'] >= 3) & (df['pActivity'] <= 12)]

    # Rename columns to match training format
    df = df.rename(columns={
        'canonical_smiles': 'smiles',
        'target_chembl_id': 'target_id',
        'assay_chembl_id': 'assay_id',
    })

    # Select columns for training
    cols = ['smiles', 'pActivity', 'target_name', 'target_id', 'assay_id', 'standard_type']
    df = df[[c for c in cols if c in df.columns]]

    return df


def download_family(family_name: str, targets: dict, output_dir: Path, max_per_target: int = None) -> pd.DataFrame:
    """Download all targets in a family."""
    print(f"\n{'='*60}")
    print(f"Downloading {family_name.upper()} family")
    print(f"{'='*60}")

    all_data = []

    for target_name, chembl_id in targets.items():
        df = fetch_target_activities(chembl_id, target_name, max_per_target)
        if not df.empty:
            df = process_activities(df)
            all_data.append(df)

        # Rate limiting between targets
        time.sleep(1)

    if not all_data:
        return pd.DataFrame()

    family_df = pd.concat(all_data, ignore_index=True)

    # Save family data
    output_file = output_dir / f"{family_name}.parquet"
    family_df.to_parquet(output_file, index=False)
    print(f"\nSaved {len(family_df)} records to {output_file}")

    return family_df


def main():
    parser = argparse.ArgumentParser(description="Download ChEMBL V2 data")
    parser.add_argument('--output', type=str, default='data/raw/chembl_v2',
                        help='Output directory')
    parser.add_argument('--families', type=str, nargs='+',
                        default=['proteases', 'cyps', 'ion_channels'],
                        help='Target families to download')
    parser.add_argument('--max-per-target', type=int, default=None,
                        help='Max records per target (for testing)')
    args = parser.parse_args()

    if not CHEMBL_AVAILABLE:
        print("ERROR: chembl_webresource_client required")
        print("Install with: pip install chembl_webresource_client")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("NEST-DRUG V2 Data Download")
    print(f"Output: {output_dir}")
    print(f"Families: {args.families}")

    all_data = []
    summary = {}

    for family in args.families:
        if family not in TARGET_FAMILIES:
            print(f"Unknown family: {family}")
            continue

        family_df = download_family(
            family,
            TARGET_FAMILIES[family],
            output_dir,
            args.max_per_target
        )

        if not family_df.empty:
            all_data.append(family_df)
            summary[family] = len(family_df)

    # Combine all families
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined_file = output_dir / "chembl_v2_all.parquet"
        combined.to_parquet(combined_file, index=False)
        print(f"\n{'='*60}")
        print("DOWNLOAD COMPLETE")
        print(f"{'='*60}")
        print(f"Total records: {len(combined)}")
        print(f"Combined file: {combined_file}")
        print("\nPer-family breakdown:")
        for family, count in summary.items():
            print(f"  {family}: {count}")

        # Per-target breakdown
        print("\nPer-target breakdown:")
        for target, count in combined['target_name'].value_counts().items():
            print(f"  {target}: {count}")


if __name__ == '__main__':
    main()
