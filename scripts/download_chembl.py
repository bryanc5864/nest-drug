#!/usr/bin/env python3
"""
Download ChEMBL bioactivity data for NEST-DRUG pretraining.

ChEMBL contains target-ligand binding and functional measurements across >2,000 protein targets.
We extract pKi, pIC50, and EC50 values for high-confidence measurements.

Target: ~2.4 million bioactivity data points
"""

import os
import sys
import gzip
import requests
from pathlib import Path
from tqdm import tqdm

# Output directory
DATA_DIR = Path("/home/bcheng/NEST/data/raw/chembl")

# ChEMBL 35 (latest as of 2025) - SQLite database
CHEMBL_VERSION = "35"
CHEMBL_URL = f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_{CHEMBL_VERSION}/chembl_{CHEMBL_VERSION}_sqlite.tar.gz"

# Alternative: Pre-extracted bioactivity CSV (smaller download)
CHEMBL_ACTIVITIES_URL = f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_{CHEMBL_VERSION}/chembl_{CHEMBL_VERSION}.sdf.gz"


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    size = f.write(chunk)
                    pbar.update(size)

        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_chembl_sqlite():
    """Download the full ChEMBL SQLite database."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dest_file = DATA_DIR / f"chembl_{CHEMBL_VERSION}_sqlite.tar.gz"

    if dest_file.exists():
        print(f"ChEMBL SQLite database already exists at {dest_file}")
        return True

    print(f"Downloading ChEMBL {CHEMBL_VERSION} SQLite database...")
    print(f"URL: {CHEMBL_URL}")
    print(f"Destination: {dest_file}")
    print("Note: This is a large file (~3GB compressed, ~15GB extracted)")

    success = download_file(CHEMBL_URL, dest_file)

    if success:
        print(f"\nDownload complete. Extract with: tar -xzf {dest_file}")

    return success


def download_chembl_via_api():
    """
    Alternative: Use ChEMBL web resource client to fetch bioactivity data.
    This is slower but more targeted - fetches only the data we need.
    """
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        print("chembl_webresource_client not installed. Run: pip install chembl_webresource_client")
        return False

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    activity = new_client.activity

    # Define activity types of interest
    activity_types = ['Ki', 'IC50', 'EC50', 'Kd']

    print("Fetching ChEMBL activities via API...")
    print("This may take several hours for the full dataset.")

    # For each activity type, fetch and save
    for act_type in activity_types:
        output_file = DATA_DIR / f"chembl_activities_{act_type.lower()}.csv"

        if output_file.exists():
            print(f"Skipping {act_type} - file already exists")
            continue

        print(f"\nFetching {act_type} activities...")

        # Query activities with standard type
        activities = activity.filter(
            standard_type=act_type,
            standard_relation='=',
            assay_type='B'  # Binding assays
        ).only([
            'molecule_chembl_id',
            'canonical_smiles',
            'standard_value',
            'standard_units',
            'standard_type',
            'pchembl_value',
            'target_chembl_id',
            'assay_chembl_id',
            'assay_type'
        ])

        # Save to CSV (this will iterate through all results)
        import pandas as pd

        # Fetch in batches
        batch_size = 10000
        all_data = []

        for i, act in enumerate(tqdm(activities, desc=f"Fetching {act_type}")):
            all_data.append(act)

            # Save periodically
            if len(all_data) >= batch_size:
                df = pd.DataFrame(all_data)
                mode = 'a' if output_file.exists() else 'w'
                header = not output_file.exists()
                df.to_csv(output_file, mode=mode, header=header, index=False)
                all_data = []

        # Save remaining
        if all_data:
            df = pd.DataFrame(all_data)
            mode = 'a' if output_file.exists() else 'w'
            header = not output_file.exists()
            df.to_csv(output_file, mode=mode, header=header, index=False)

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download ChEMBL data for NEST-DRUG")
    parser.add_argument('--method', choices=['sqlite', 'api'], default='sqlite',
                       help='Download method: sqlite (full DB) or api (targeted fetch)')

    args = parser.parse_args()

    if args.method == 'sqlite':
        success = download_chembl_sqlite()
    else:
        success = download_chembl_via_api()

    sys.exit(0 if success else 1)
