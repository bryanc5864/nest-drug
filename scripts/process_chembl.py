#!/usr/bin/env python3
"""
Process ChEMBL SQLite database for NEST-DRUG pretraining.

This script extracts bioactivity data from ChEMBL and prepares it for model training.
Run this after downloading chembl_35_sqlite.tar.gz.

Usage:
    python process_chembl.py --extract  # Extract tar.gz first
    python process_chembl.py --process  # Process to CSV/parquet
"""

import os
import sys
import tarfile
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Paths
DATA_DIR = Path("/home/bcheng/NEST/data/raw/chembl")
OUTPUT_DIR = Path("/home/bcheng/NEST/data/processed/portfolio")
CHEMBL_TAR = DATA_DIR / "chembl_35_sqlite.tar.gz"
CHEMBL_DB = DATA_DIR / "chembl_35" / "chembl_35_sqlite" / "chembl_35.db"


def extract_chembl():
    """Extract the ChEMBL SQLite database from tar.gz."""
    if not CHEMBL_TAR.exists():
        print(f"Error: {CHEMBL_TAR} not found")
        print("Run download_chembl.py first")
        return False

    if CHEMBL_DB.exists():
        print(f"ChEMBL database already extracted at {CHEMBL_DB}")
        return True

    print(f"Extracting {CHEMBL_TAR}...")
    print("This may take a few minutes...")

    with tarfile.open(CHEMBL_TAR, 'r:gz') as tar:
        tar.extractall(DATA_DIR)

    if CHEMBL_DB.exists():
        print(f"Extraction complete: {CHEMBL_DB}")
        return True
    else:
        print("Error: Database not found after extraction")
        return False


def process_chembl():
    """Process ChEMBL database to extract bioactivity data."""
    if not CHEMBL_DB.exists():
        print(f"Error: {CHEMBL_DB} not found")
        print("Run with --extract first")
        return False

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {CHEMBL_DB}...")
    conn = sqlite3.connect(str(CHEMBL_DB))

    # Query for bioactivity data
    # This query extracts high-quality binding data
    query = """
    SELECT
        cs.canonical_smiles as smiles,
        act.standard_value,
        act.standard_units,
        act.standard_type,
        act.pchembl_value,
        act.standard_relation,
        md.chembl_id as molecule_chembl_id,
        td.chembl_id as target_chembl_id,
        td.pref_name as target_name,
        ass.chembl_id as assay_chembl_id,
        ass.assay_type
    FROM activities act
    JOIN molecule_dictionary md ON act.molregno = md.molregno
    JOIN compound_structures cs ON md.molregno = cs.molregno
    JOIN assays ass ON act.assay_id = ass.assay_id
    JOIN target_dictionary td ON ass.tid = td.tid
    WHERE
        act.standard_type IN ('Ki', 'IC50', 'EC50', 'Kd')
        AND act.standard_relation = '='
        AND act.standard_value IS NOT NULL
        AND act.standard_units IN ('nM', 'uM')
        AND cs.canonical_smiles IS NOT NULL
        AND act.data_validity_comment IS NULL
        AND ass.assay_type = 'B'
    """

    print("Executing query (this may take several minutes)...")
    df = pd.read_sql_query(query, conn, chunksize=100000)

    # Process in chunks to handle large data
    chunks = []
    for i, chunk in enumerate(tqdm(df, desc="Reading chunks")):
        chunks.append(chunk)

    if len(chunks) == 0:
        print("No data returned from query")
        conn.close()
        return False

    df = pd.concat(chunks, ignore_index=True)
    print(f"Retrieved {len(df):,} activity records")

    # Close connection
    conn.close()

    # Process the data
    print("\nProcessing data...")

    # Convert to pChEMBL values if missing
    def to_pchembl(row):
        if pd.notna(row['pchembl_value']):
            return row['pchembl_value']
        if pd.isna(row['standard_value']):
            return np.nan

        value = row['standard_value']
        unit = row['standard_units']

        if unit == 'nM':
            value_m = value * 1e-9
        elif unit == 'uM':
            value_m = value * 1e-6
        else:
            return np.nan

        if value_m > 0:
            return -np.log10(value_m)
        return np.nan

    df['pchembl_value'] = df.apply(to_pchembl, axis=1)

    # Remove rows without valid pchembl values
    df = df[df['pchembl_value'].notna()]
    print(f"After filtering: {len(df):,} records with valid pChEMBL values")

    # Aggregate replicates
    print("Aggregating replicates...")
    agg_df = df.groupby(['smiles', 'target_chembl_id', 'standard_type']).agg({
        'pchembl_value': ['median', 'std', 'count'],
        'molecule_chembl_id': 'first',
        'target_name': 'first',
        'assay_chembl_id': 'first',
    }).reset_index()

    # Flatten column names
    agg_df.columns = ['smiles', 'target_chembl_id', 'standard_type',
                      'pchembl_median', 'pchembl_std', 'n_measurements',
                      'molecule_chembl_id', 'target_name', 'assay_chembl_id']

    print(f"After aggregation: {len(agg_df):,} unique compound-target-type combinations")

    # Split by activity type
    for act_type in ['Ki', 'IC50', 'EC50', 'Kd']:
        subset = agg_df[agg_df['standard_type'] == act_type]
        if len(subset) > 0:
            output_file = OUTPUT_DIR / f"chembl_potency_{act_type.lower()}.parquet"
            subset.to_parquet(output_file, index=False)
            print(f"  {act_type}: {len(subset):,} records -> {output_file.name}")

    # Save combined file
    output_file = OUTPUT_DIR / "chembl_potency_all.parquet"
    agg_df.to_parquet(output_file, index=False)
    print(f"\nCombined file: {len(agg_df):,} records -> {output_file}")

    # Statistics
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total unique SMILES: {agg_df['smiles'].nunique():,}")
    print(f"Total unique targets: {agg_df['target_chembl_id'].nunique():,}")
    print(f"pChEMBL range: {agg_df['pchembl_median'].min():.2f} - {agg_df['pchembl_median'].max():.2f}")
    print(f"\nActivity type distribution:")
    for act_type, count in agg_df['standard_type'].value_counts().items():
        print(f"  {act_type}: {count:,}")

    return True


def check_status():
    """Check the status of ChEMBL data."""
    print("ChEMBL Data Status")
    print("=" * 60)

    # Check tar file
    if CHEMBL_TAR.exists():
        size_gb = CHEMBL_TAR.stat().st_size / (1024**3)
        print(f"✓ Archive: {CHEMBL_TAR.name} ({size_gb:.2f} GB)")
    else:
        print(f"✗ Archive: {CHEMBL_TAR.name} (not found)")

    # Check extracted database
    if CHEMBL_DB.exists():
        size_gb = CHEMBL_DB.stat().st_size / (1024**3)
        print(f"✓ Database: {CHEMBL_DB.name} ({size_gb:.2f} GB)")
    else:
        print(f"✗ Database: not extracted")

    # Check processed files
    if OUTPUT_DIR.exists():
        processed = list(OUTPUT_DIR.glob("chembl_*.parquet"))
        if processed:
            print(f"✓ Processed files: {len(processed)} parquet files")
            for f in processed:
                df = pd.read_parquet(f)
                print(f"    {f.name}: {len(df):,} records")
        else:
            print("✗ Processed files: none")
    else:
        print("✗ Processed files: output directory not created")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process ChEMBL data for NEST-DRUG")
    parser.add_argument('--extract', action='store_true', help='Extract tar.gz archive')
    parser.add_argument('--process', action='store_true', help='Process database to parquet')
    parser.add_argument('--status', action='store_true', help='Check data status')

    args = parser.parse_args()

    if args.status or not (args.extract or args.process):
        check_status()
    elif args.extract:
        extract_chembl()
    elif args.process:
        process_chembl()
