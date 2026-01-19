#!/usr/bin/env python3
"""
Robust Benchmark Data Downloader for NEST-DRUG

Handles download failures gracefully with:
- Retry logic
- Alternative sources
- Manual download instructions
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path
import urllib.request
import ssl

# Disable SSL verification for problematic servers
ssl._create_default_https_context = ssl._create_unverified_context


def download_with_wget(url, output_path, retries=3):
    """Download using wget with retries."""
    for attempt in range(retries):
        try:
            cmd = [
                'wget', '-q', '--no-check-certificate',
                '-O', str(output_path),
                '--timeout=60',
                '--tries=3',
                url
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                return True
        except Exception as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    return False


def download_with_curl(url, output_path, retries=3):
    """Download using curl with retries."""
    for attempt in range(retries):
        try:
            cmd = [
                'curl', '-sL', '-k',
                '--max-time', '300',
                '--retry', '3',
                '-o', str(output_path),
                url
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                return True
        except Exception as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    return False


def download_file(url, output_path):
    """Try multiple download methods."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try wget first
    if download_with_wget(url, output_path):
        return True

    # Try curl
    if download_with_curl(url, output_path):
        return True

    # Try urllib
    try:
        urllib.request.urlretrieve(url, output_path)
        if output_path.exists() and output_path.stat().st_size > 0:
            return True
    except Exception as e:
        print(f"    urllib failed: {e}")

    return False


def download_herg_tdc(output_dir):
    """Download hERG from TDC."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "herg_tdc.csv"

    if output_file.exists() and output_file.stat().st_size > 1000:
        print(f"hERG data already exists at {output_file}")
        return True

    try:
        from tdc.single_pred import Tox
        data = Tox(name='hERG')
        df = data.get_data()
        df.to_csv(output_file, index=False)
        print(f"hERG data saved to {output_file}")
        print(f"  Total compounds: {len(df)}")
        print(f"  Blockers: {df['Y'].sum()}")
        return True
    except ImportError:
        print("TDC not installed. Trying alternative source...")
    except Exception as e:
        print(f"TDC download failed: {e}")

    # Alternative: Download from TDC raw data
    alt_url = "https://raw.githubusercontent.com/mims-harvard/TDC/main/tdc/metadata.py"
    # The actual data needs to come from TDC package
    print("Please install PyTDC: pip install PyTDC")
    return False


def download_dude_target(target, output_dir):
    """Download a single DUD-E target."""
    target_dir = Path(output_dir) / target
    target_dir.mkdir(parents=True, exist_ok=True)

    actives_file = target_dir / "actives_final.smi"
    decoys_file = target_dir / "decoys_final.smi"

    # Check if already exists
    if actives_file.exists() and decoys_file.exists():
        if actives_file.stat().st_size > 100 and decoys_file.stat().st_size > 100:
            print(f"  {target}: Already downloaded")
            return True

    # DUD-E URLs
    base_url = f"http://dude.docking.org/targets/{target}"
    actives_url = f"{base_url}/actives_final.smi"
    decoys_url = f"{base_url}/decoys_final.smi"

    print(f"  Downloading {target}...")

    success = True
    if not download_file(actives_url, actives_file):
        print(f"    Failed to download actives for {target}")
        success = False

    if not download_file(decoys_url, decoys_file):
        print(f"    Failed to download decoys for {target}")
        success = False

    return success


def download_dude(output_dir):
    """Download DUD-E benchmark data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = ['egfr', 'drd2', 'jak2', 'adrb2', 'esr1', 'pparg', 'hdac2', 'bace1']

    print("\n" + "="*60)
    print("Downloading DUD-E")
    print("="*60)

    success_count = 0
    for target in targets:
        if download_dude_target(target, output_dir):
            success_count += 1

    print(f"\nDUD-E: {success_count}/{len(targets)} targets downloaded")

    if success_count < len(targets):
        print("\nManual download instructions for DUD-E:")
        print("  1. Go to http://dude.docking.org/targets")
        print("  2. Download actives_final.smi and decoys_final.smi for each target")
        print(f"  3. Place files in {output_dir}/<target>/")

    return success_count == len(targets)


def download_litpcba(output_dir):
    """Download LIT-PCBA benchmark data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    litpcba_dir = output_dir / "LIT-PCBA"
    zip_file = output_dir / "LIT-PCBA.zip"

    print("\n" + "="*60)
    print("Downloading LIT-PCBA")
    print("="*60)

    # Check if already extracted
    if litpcba_dir.exists():
        subdirs = list(litpcba_dir.iterdir())
        if len(subdirs) >= 10:  # Should have 15 targets
            print(f"LIT-PCBA already extracted at {litpcba_dir}")
            return True

    # Download zip
    url = "https://drugdesign.unistra.fr/LIT-PCBA/Files/LIT-PCBA.zip"

    print(f"Downloading from {url}...")
    if not download_file(url, zip_file):
        print("Download failed.")
        print("\nManual download instructions for LIT-PCBA:")
        print("  1. Go to https://drugdesign.unistra.fr/LIT-PCBA/")
        print("  2. Download LIT-PCBA.zip")
        print(f"  3. Place in {output_dir}/")
        print(f"  4. Run: unzip {zip_file} -d {output_dir}")
        return False

    # Check zip file size
    if zip_file.stat().st_size < 10000:
        print(f"Downloaded file too small ({zip_file.stat().st_size} bytes)")
        print("Server may be blocking automated downloads.")
        print("\nManual download required:")
        print("  1. Go to https://drugdesign.unistra.fr/LIT-PCBA/")
        print("  2. Download LIT-PCBA.zip manually")
        return False

    # Extract
    print("Extracting...")
    try:
        import zipfile
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(output_dir)
        print(f"LIT-PCBA extracted to {litpcba_dir}")
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def create_synthetic_herg(output_dir):
    """Create synthetic hERG data from ChEMBL if TDC unavailable."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "herg_tdc.csv"

    print("\nCreating synthetic hERG dataset from available sources...")

    # Try to get hERG data from ChEMBL
    try:
        from chembl_webresource_client.new_client import new_client

        target = new_client.target
        activity = new_client.activity

        # hERG target in ChEMBL
        herg_targets = target.filter(target_synonym__icontains='herg').filter(organism='Homo sapiens')

        if herg_targets:
            target_chembl_id = herg_targets[0]['target_chembl_id']
            print(f"Found hERG target: {target_chembl_id}")

            activities = activity.filter(
                target_chembl_id=target_chembl_id,
                standard_type__in=['IC50', 'Ki'],
                standard_units='nM'
            ).only(['canonical_smiles', 'standard_value', 'standard_type'])

            import pandas as pd
            df = pd.DataFrame(activities)

            if len(df) > 0:
                df = df.dropna(subset=['canonical_smiles', 'standard_value'])
                df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
                df = df.dropna(subset=['standard_value'])

                # IC50 < 10 µM = blocker
                df['Y'] = (df['standard_value'] < 10000).astype(int)
                df = df.rename(columns={'canonical_smiles': 'Drug'})
                df = df[['Drug', 'Y']].drop_duplicates(subset=['Drug'])

                df.to_csv(output_file, index=False)
                print(f"Synthetic hERG data saved to {output_file}")
                print(f"  Total compounds: {len(df)}")
                print(f"  Blockers: {df['Y'].sum()}")
                return True
    except Exception as e:
        print(f"ChEMBL download failed: {e}")

    print("Could not create hERG dataset. Please install PyTDC:")
    print("  pip install PyTDC")
    return False


def main():
    parser = argparse.ArgumentParser(description='Download benchmark data for NEST-DRUG')
    parser.add_argument('--datasets', nargs='+',
                       choices=['litpcba', 'dude', 'herg', 'all'],
                       default=['all'],
                       help='Datasets to download')
    parser.add_argument('--output-dir', type=str, default='data/external',
                       help='Output directory')
    args = parser.parse_args()

    datasets = args.datasets
    if 'all' in datasets:
        datasets = ['litpcba', 'dude', 'herg']

    output_dir = Path(args.output_dir)

    results = {}

    if 'litpcba' in datasets:
        results['litpcba'] = download_litpcba(output_dir / 'litpcba')

    if 'dude' in datasets:
        results['dude'] = download_dude(output_dir / 'dude')

    if 'herg' in datasets:
        results['herg'] = download_herg_tdc(output_dir / 'herg')
        if not results['herg']:
            results['herg'] = create_synthetic_herg(output_dir / 'herg')

    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for name, success in results.items():
        status = "OK" if success else "FAILED - manual download required"
        print(f"  {name}: {status}")

    if not all(results.values()):
        print("\nSome downloads failed. See instructions above for manual download.")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
