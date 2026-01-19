#!/usr/bin/env python3
"""
Download BindingDB binding affinity data for NEST-DRUG pretraining.

BindingDB contains experimentally measured protein-small molecule binding affinities.
Target: ~2.9 million binding data points

Data includes:
- Ki, Kd, IC50, EC50 values
- Kinetic parameters (kon, koff) where available
- SMILES structures
- Target information
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# Output directory
DATA_DIR = Path("/home/bcheng/NEST/data/raw/bindingdb")

# BindingDB download URLs
# Full database in TSV format (most comprehensive)
BINDINGDB_TSV_URL = "https://www.bindingdb.org/bind/downloads/BindingDB_All_2024m12.tsv.zip"

# Alternative: SDF format with 3D structures
BINDINGDB_SDF_URL = "https://www.bindingdb.org/bind/downloads/BindingDB_All_2D_2024m12.sdf.zip"


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
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


def download_bindingdb():
    """Download BindingDB TSV file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Try multiple versions (they update quarterly)
    versions = ['2024m12', '2024m9', '2024m6', '2024m3', '2023m12']

    for version in versions:
        url = f"https://www.bindingdb.org/bind/downloads/BindingDB_All_{version}.tsv.zip"
        dest_file = DATA_DIR / f"BindingDB_All_{version}.tsv.zip"

        if dest_file.exists():
            print(f"BindingDB file already exists at {dest_file}")
            return True

        print(f"Attempting to download BindingDB {version}...")
        print(f"URL: {url}")
        print(f"Destination: {dest_file}")
        print("Note: This is a large file (~1.5GB compressed)")

        success = download_file(url, dest_file)

        if success:
            print(f"\nDownload complete!")

            # Extract the zip file
            print(f"Extracting {dest_file}...")
            try:
                with zipfile.ZipFile(dest_file, 'r') as zip_ref:
                    zip_ref.extractall(DATA_DIR)
                print("Extraction complete!")
            except Exception as e:
                print(f"Error extracting: {e}")

            return True
        else:
            print(f"Failed to download {version}, trying next version...")

    print("Failed to download any version of BindingDB")
    return False


def download_bindingdb_oracle():
    """
    Alternative: Download from BindingDB Oracle dump.
    This contains more metadata but requires Oracle tools to extract.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    url = "https://www.bindingdb.org/bind/downloads/BindingDB_All_Oracle.zip"
    dest_file = DATA_DIR / "BindingDB_All_Oracle.zip"

    if dest_file.exists():
        print(f"BindingDB Oracle dump already exists at {dest_file}")
        return True

    print("Downloading BindingDB Oracle dump...")
    return download_file(url, dest_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download BindingDB data for NEST-DRUG")
    parser.add_argument('--format', choices=['tsv', 'oracle'], default='tsv',
                       help='Download format: tsv (recommended) or oracle')

    args = parser.parse_args()

    if args.format == 'tsv':
        success = download_bindingdb()
    else:
        success = download_bindingdb_oracle()

    sys.exit(0 if success else 1)
