#!/usr/bin/env python3
"""
Download TDC (Therapeutics Data Commons) ADMET benchmark datasets for NEST-DRUG pretraining.

TDC provides standardized ADMET benchmark tasks with predefined splits and evaluation protocols.

Target: ~500,000 ADMET measurements across 15+ endpoints

Endpoints included:
- Solubility (aqueous, kinetic, thermodynamic)
- Lipophilicity (LogD, LogP)
- Metabolic clearance (microsomal, hepatocyte)
- Half-life and bioavailability
- Toxicity (hERG, Ames, DILI, cardiotoxicity)
- Permeability (Caco-2, PAMPA, BBB)
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Output directory
DATA_DIR = Path("/home/bcheng/NEST/data/raw/tdc")


# Define all ADMET datasets to download
ADMET_DATASETS = {
    # Absorption
    'absorption': {
        'Caco2_Wang': 'Caco-2 permeability',
        'PAMPA_NCATS': 'PAMPA permeability',
        'HIA_Hou': 'Human intestinal absorption',
        'Pgp_Broccatelli': 'P-glycoprotein inhibition',
        'Bioavailability_Ma': 'Oral bioavailability',
        'Lipophilicity_AstraZeneca': 'Lipophilicity (LogD)',
        'Solubility_AqSolDB': 'Aqueous solubility',
        'HydrationFreeEnergy_FreeSolv': 'Hydration free energy',
    },

    # Distribution
    'distribution': {
        'BBB_Martins': 'Blood-brain barrier penetration',
        'PPBR_AZ': 'Plasma protein binding rate',
        'VDss_Lombardo': 'Volume of distribution',
    },

    # Metabolism
    'metabolism': {
        'CYP2C19_Veith': 'CYP2C19 inhibition',
        'CYP2D6_Veith': 'CYP2D6 inhibition',
        'CYP3A4_Veith': 'CYP3A4 inhibition',
        'CYP1A2_Veith': 'CYP1A2 inhibition',
        'CYP2C9_Veith': 'CYP2C9 inhibition',
        'CYP2C9_Substrate_CarbonMangels': 'CYP2C9 substrate',
        'CYP2D6_Substrate_CarbonMangels': 'CYP2D6 substrate',
        'CYP3A4_Substrate_CarbonMangels': 'CYP3A4 substrate',
        'Half_Life_Obach': 'Half-life',
        'Clearance_Hepatocyte_AZ': 'Hepatocyte clearance',
        'Clearance_Microsome_AZ': 'Microsomal clearance',
    },

    # Excretion
    'excretion': {
        # Note: Limited excretion datasets in TDC
    },

    # Toxicity
    'toxicity': {
        'hERG': 'hERG channel inhibition',
        'AMES': 'Ames mutagenicity',
        'DILI': 'Drug-induced liver injury',
        'Skin_Reaction': 'Skin sensitization',
        'Carcinogens_Lagunin': 'Carcinogenicity',
        'ClinTox': 'Clinical toxicity (FDA)',
        'LD50_Zhu': 'Acute toxicity (LD50)',
    },
}


def download_tdc_datasets():
    """Download all TDC ADMET datasets."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Try importing TDC
    try:
        from tdc.single_pred import ADME, Tox
    except ImportError as e:
        print(f"Error importing TDC: {e}")
        print("Attempting alternative import...")
        try:
            # Try direct dataset access
            from tdc.benchmark_group import admet_group
            group = admet_group(path=str(DATA_DIR))
            print("Successfully accessed TDC via benchmark_group")
        except Exception as e2:
            print(f"Alternative import also failed: {e2}")
            print("Will use direct download method instead")
            return download_tdc_direct()

    downloaded = []
    failed = []

    # Download ADME datasets
    print("=" * 60)
    print("Downloading ADME datasets...")
    print("=" * 60)

    adme_datasets = []
    for category in ['absorption', 'distribution', 'metabolism', 'excretion']:
        adme_datasets.extend(ADMET_DATASETS.get(category, {}).items())

    for dataset_name, description in adme_datasets:
        output_file = DATA_DIR / f"adme_{dataset_name}.csv"

        if output_file.exists():
            print(f"✓ {dataset_name} already exists")
            downloaded.append(dataset_name)
            continue

        print(f"\nDownloading {dataset_name}: {description}...")
        try:
            data = ADME(name=dataset_name, path=str(DATA_DIR))
            df = data.get_data()
            df.to_csv(output_file, index=False)
            print(f"  → Saved {len(df)} samples to {output_file.name}")
            downloaded.append(dataset_name)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed.append(dataset_name)

    # Download Toxicity datasets
    print("\n" + "=" * 60)
    print("Downloading Toxicity datasets...")
    print("=" * 60)

    for dataset_name, description in ADMET_DATASETS.get('toxicity', {}).items():
        output_file = DATA_DIR / f"tox_{dataset_name}.csv"

        if output_file.exists():
            print(f"✓ {dataset_name} already exists")
            downloaded.append(dataset_name)
            continue

        print(f"\nDownloading {dataset_name}: {description}...")
        try:
            data = Tox(name=dataset_name, path=str(DATA_DIR))
            df = data.get_data()
            df.to_csv(output_file, index=False)
            print(f"  → Saved {len(df)} samples to {output_file.name}")
            downloaded.append(dataset_name)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed.append(dataset_name)

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Successfully downloaded: {len(downloaded)} datasets")
    print(f"Failed: {len(failed)} datasets")

    if failed:
        print(f"\nFailed datasets: {', '.join(failed)}")

    return len(failed) == 0


def download_tdc_direct():
    """
    Alternative: Download TDC datasets directly from GitHub releases.
    Use this if the PyTDC package is not working properly.
    """
    import requests
    from io import StringIO

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # TDC stores data in a consistent format on their GitHub
    base_url = "https://dataverse.harvard.edu/api/access/datafile/"

    # Mapping of dataset names to Dataverse file IDs
    # These are stable identifiers for the TDC datasets
    dataset_file_ids = {
        # ADME datasets
        'Caco2_Wang': '4159713',
        'Lipophilicity_AstraZeneca': '4159715',
        'Solubility_AqSolDB': '4159717',
        'BBB_Martins': '4159719',
        'PPBR_AZ': '4159721',
        'CYP2D6_Veith': '4159723',
        'CYP3A4_Veith': '4159725',
        'CYP2C9_Veith': '4159727',
        'Half_Life_Obach': '4159729',
        'Clearance_Hepatocyte_AZ': '4159731',
        'Clearance_Microsome_AZ': '4159733',
        # Toxicity datasets
        'hERG': '4159735',
        'AMES': '4159737',
        'DILI': '4159739',
        'LD50_Zhu': '4159741',
    }

    print("Downloading TDC datasets directly from Harvard Dataverse...")
    print("Note: Using stable file IDs - some may be outdated")

    downloaded = 0
    for dataset_name, file_id in dataset_file_ids.items():
        output_file = DATA_DIR / f"tdc_{dataset_name}.csv"

        if output_file.exists():
            print(f"✓ {dataset_name} already exists")
            downloaded += 1
            continue

        url = f"{base_url}{file_id}"
        print(f"Downloading {dataset_name}...")

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # Save the file
            with open(output_file, 'wb') as f:
                f.write(response.content)

            downloaded += 1
            print(f"  → Saved to {output_file.name}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print(f"\nDownloaded {downloaded}/{len(dataset_file_ids)} datasets")
    return downloaded > 0


def download_admet_benchmark():
    """
    Download the ADMET benchmark group which includes multiple datasets
    with predefined train/validation/test splits.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from tdc.benchmark_group import admet_group

        print("Downloading ADMET Benchmark Group...")
        group = admet_group(path=str(DATA_DIR))

        # Get all predictions tasks
        predictions = group.get_train_valid_split()

        print(f"ADMET Benchmark includes {len(predictions)} datasets")

        # Save each split
        for name, splits in predictions.items():
            train_file = DATA_DIR / f"admet_benchmark_{name}_train.csv"
            valid_file = DATA_DIR / f"admet_benchmark_{name}_valid.csv"
            test_file = DATA_DIR / f"admet_benchmark_{name}_test.csv"

            if train_file.exists():
                print(f"✓ {name} splits already exist")
                continue

            train_df = pd.DataFrame(splits['train'])
            valid_df = pd.DataFrame(splits['valid'])

            train_df.to_csv(train_file, index=False)
            valid_df.to_csv(valid_file, index=False)

            print(f"  → {name}: {len(train_df)} train, {len(valid_df)} valid")

        return True

    except Exception as e:
        print(f"Error downloading ADMET benchmark: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download TDC ADMET datasets for NEST-DRUG")
    parser.add_argument('--method', choices=['auto', 'direct', 'benchmark'], default='auto',
                       help='Download method: auto (try PyTDC first), direct (Harvard Dataverse), benchmark (group)')

    args = parser.parse_args()

    if args.method == 'benchmark':
        success = download_admet_benchmark()
    elif args.method == 'direct':
        success = download_tdc_direct()
    else:
        success = download_tdc_datasets()

    sys.exit(0 if success else 1)
