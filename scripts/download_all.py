#!/usr/bin/env python3
"""
Master script to download all data for NEST-DRUG.

This script orchestrates downloads from:
1. ChEMBL - Target-ligand bioactivity data
2. BindingDB - Protein-small molecule binding affinities
3. TDC - ADMET benchmark datasets

Total expected data: ~2-3 million structure-endpoint pairs

Usage:
    python download_all.py              # Download all datasets
    python download_all.py --skip-large # Skip large downloads (ChEMBL SQLite)
    python download_all.py --tdc-only   # Download only TDC (fastest)
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

SCRIPTS_DIR = Path(__file__).parent
DATA_DIR = Path("/home/bcheng/NEST/data/raw")


def run_script(script_name: str, *args) -> bool:
    """Run a download script and return success status."""
    script_path = SCRIPTS_DIR / script_name

    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)] + list(args)
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=SCRIPTS_DIR)
    return result.returncode == 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download all NEST-DRUG data")
    parser.add_argument('--skip-large', action='store_true',
                       help='Skip large downloads (ChEMBL SQLite, full BindingDB)')
    parser.add_argument('--tdc-only', action='store_true',
                       help='Download only TDC datasets (fastest)')
    parser.add_argument('--chembl-api', action='store_true',
                       help='Use ChEMBL API instead of SQLite download')

    args = parser.parse_args()

    start_time = datetime.now()
    results = {}

    print("=" * 60)
    print("NEST-DRUG Data Download")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. TDC (smallest, fastest)
    print("\n[1/3] Downloading TDC ADMET datasets...")
    results['TDC'] = run_script('download_tdc.py')

    if args.tdc_only:
        print("\n--tdc-only specified, skipping other downloads")
    else:
        # 2. BindingDB
        if not args.skip_large:
            print("\n[2/3] Downloading BindingDB...")
            results['BindingDB'] = run_script('download_bindingdb.py')
        else:
            print("\n[2/3] Skipping BindingDB (--skip-large)")
            results['BindingDB'] = None

        # 3. ChEMBL
        if not args.skip_large:
            print("\n[3/3] Downloading ChEMBL...")
            method = '--method=api' if args.chembl_api else '--method=sqlite'
            results['ChEMBL'] = run_script('download_chembl.py', method)
        else:
            print("\n[3/3] Skipping ChEMBL (--skip-large)")
            results['ChEMBL'] = None

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Duration: {duration}")
    print()

    for source, success in results.items():
        if success is None:
            status = "SKIPPED"
        elif success:
            status = "SUCCESS"
        else:
            status = "FAILED"
        print(f"  {source}: {status}")

    # Check what we have
    print("\n" + "=" * 60)
    print("DATA INVENTORY")
    print("=" * 60)

    for subdir in ['chembl', 'bindingdb', 'tdc']:
        dir_path = DATA_DIR / subdir
        if dir_path.exists():
            files = list(dir_path.glob('*'))
            total_size = sum(f.stat().st_size for f in files if f.is_file()) / (1024**3)
            print(f"\n{subdir}/: {len(files)} files, {total_size:.2f} GB")
            for f in sorted(files)[:10]:
                if f.is_file():
                    size_mb = f.stat().st_size / (1024**2)
                    print(f"  - {f.name}: {size_mb:.1f} MB")
            if len(files) > 10:
                print(f"  ... and {len(files)-10} more files")
        else:
            print(f"\n{subdir}/: Directory not found")

    all_success = all(v for v in results.values() if v is not None)
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
