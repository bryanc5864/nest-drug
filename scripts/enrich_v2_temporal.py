#!/usr/bin/env python3
"""
Enrich V2 training parquets with temporal (document_year, round_id) data from ChEMBL SQLite.

Also adds standard_type → assay_type_id mapping for L2 context.

Usage:
    python scripts/enrich_v2_temporal.py
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

CHEMBL_DB = 'data/raw/chembl/chembl_35/chembl_35_sqlite/chembl_35.db'
V2_DATA_DIR = 'data/raw/chembl_v2'
OUTPUT_DIR = 'data/raw/chembl_v2_enriched'
NUM_ROUND_BINS = 20  # Match model capacity


STANDARD_TYPE_MAPPING = {
    'IC50': 1,
    'Ki': 2,
    'EC50': 3,
    'Kd': 4,
}


def build_assay_year_lookup(db_path, assay_chembl_ids):
    """Query ChEMBL SQLite for assay_chembl_id → median document year."""
    print(f"Querying ChEMBL SQLite for {len(assay_chembl_ids)} assay IDs...")
    db = sqlite3.connect(db_path)

    # Build lookup in batches (SQLite has variable limit)
    lookup = {}
    batch_size = 500
    ids_list = list(assay_chembl_ids)

    for i in tqdm(range(0, len(ids_list), batch_size), desc="Querying batches"):
        batch = ids_list[i:i+batch_size]
        placeholders = ','.join(['?'] * len(batch))
        query = f"""
            SELECT ass.chembl_id as assay_chembl_id,
                   GROUP_CONCAT(DISTINCT doc.year) as years
            FROM assays ass
            JOIN activities act ON act.assay_id = ass.assay_id
            JOIN docs doc ON act.doc_id = doc.doc_id
            WHERE ass.chembl_id IN ({placeholders})
              AND doc.year IS NOT NULL
            GROUP BY ass.chembl_id
        """
        cur = db.execute(query, batch)
        for row in cur.fetchall():
            assay_id, years_str = row
            years = [int(y) for y in years_str.split(',')]
            lookup[assay_id] = int(np.median(years))

    db.close()
    print(f"  Found years for {len(lookup)}/{len(assay_chembl_ids)} assays ({100*len(lookup)/max(len(assay_chembl_ids),1):.1f}%)")
    return lookup


def assign_round_ids(years, num_bins=NUM_ROUND_BINS):
    """Assign round IDs from document years using quantile binning."""
    valid_mask = ~pd.isna(years)
    round_ids = pd.Series(0, index=years.index, dtype=int)

    if valid_mask.sum() == 0:
        return round_ids

    valid_years = years[valid_mask]
    try:
        binned = pd.qcut(valid_years.rank(method='first'), q=num_bins, labels=False, duplicates='drop')
        round_ids[valid_mask] = binned.astype(int)
    except ValueError:
        # Fewer unique values than bins
        unique_years = sorted(valid_years.unique())
        year_to_bin = {y: i for i, y in enumerate(unique_years)}
        round_ids[valid_mask] = valid_years.map(year_to_bin).astype(int)

    return round_ids


def enrich_parquet(parquet_path, assay_year_lookup, output_dir):
    """Enrich a single parquet file with L2 and L3 data."""
    df = pd.read_parquet(parquet_path)
    name = Path(parquet_path).stem
    print(f"\n  {name}: {len(df)} rows")

    # L2: standard_type → assay_type_id
    if 'standard_type' in df.columns:
        df['assay_type_id'] = df['standard_type'].map(STANDARD_TYPE_MAPPING).fillna(0).astype(int)
        print(f"    L2 (assay_type_id): {df['assay_type_id'].value_counts().to_dict()}")
    else:
        df['assay_type_id'] = 0
        print(f"    L2: no standard_type column, all zeros")

    # L3: assay_id → document_year → round_id
    if 'assay_id' in df.columns:
        df['document_year'] = df['assay_id'].map(assay_year_lookup)
        coverage = df['document_year'].notna().sum()
        print(f"    L3: document_year coverage: {coverage}/{len(df)} ({100*coverage/len(df):.1f}%)")

        if coverage > 0:
            year_range = f"{df['document_year'].min():.0f}-{df['document_year'].max():.0f}"
            print(f"    L3: year range: {year_range}")

        df['round_id'] = assign_round_ids(df['document_year'])
        print(f"    L3: round_id range: {df['round_id'].min()}-{df['round_id'].max()}, unique: {df['round_id'].nunique()}")
    else:
        df['document_year'] = np.nan
        df['round_id'] = 0
        print(f"    L3: no assay_id column, all zeros")

    # Save
    output_path = Path(output_dir) / Path(parquet_path).name
    df.to_parquet(output_path, index=False)
    print(f"    Saved to {output_path}")
    return df


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all unique assay IDs across V2 parquets
    parquet_files = list(Path(V2_DATA_DIR).glob('*.parquet'))
    print(f"Found {len(parquet_files)} parquet files in {V2_DATA_DIR}")

    all_assay_ids = set()
    for f in parquet_files:
        df = pd.read_parquet(f)
        if 'assay_id' in df.columns:
            all_assay_ids.update(df['assay_id'].dropna().unique())

    print(f"Total unique assay IDs: {len(all_assay_ids)}")

    # Query ChEMBL for temporal data
    assay_year_lookup = build_assay_year_lookup(CHEMBL_DB, all_assay_ids)

    # Enrich each parquet
    print(f"\nEnriching parquets...")
    for f in parquet_files:
        enrich_parquet(str(f), assay_year_lookup, output_dir)

    # Summary
    print(f"\n{'='*60}")
    print(f"Enrichment complete. Output: {output_dir}")
    print(f"  Assay year coverage: {len(assay_year_lookup)}/{len(all_assay_ids)}")
    print(f"  L2 mapping: {STANDARD_TYPE_MAPPING}")
    print(f"  L3 bins: {NUM_ROUND_BINS}")


if __name__ == '__main__':
    main()
