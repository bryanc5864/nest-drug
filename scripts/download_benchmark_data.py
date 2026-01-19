#!/usr/bin/env python3
"""
Download All Benchmark Data for NEST-DRUG Validation

Downloads data for:
1. LIT-PCBA (15 targets with real experimental inactives)
2. DUD-E (property-matched decoys)
3. DRD2 from ChEMBL (GPCR DMTA replay)
4. hERG from TDC (safety endpoint)
5. Tox21 from MoleculeNet (multi-task)
6. MoleculeNet ADMET datasets
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import zipfile
import gzip
import shutil
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def download_file(url, dest_path, desc=None):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


# =============================================================================
# 1. LIT-PCBA
# =============================================================================

def download_litpcba(output_dir):
    """Download LIT-PCBA benchmark dataset."""
    print("\n" + "="*60)
    print("Downloading LIT-PCBA")
    print("="*60)

    litpcba_dir = Path(output_dir) / "litpcba"
    litpcba_dir.mkdir(parents=True, exist_ok=True)

    zip_path = litpcba_dir / "LIT-PCBA.zip"

    if (litpcba_dir / "LIT-PCBA").exists():
        print("LIT-PCBA already downloaded")
        return litpcba_dir / "LIT-PCBA"

    url = "https://drugdesign.unistra.fr/LIT-PCBA/Files/LIT-PCBA.zip"

    print(f"Downloading from {url}")
    try:
        download_file(url, zip_path, "LIT-PCBA.zip")

        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(litpcba_dir)

        os.remove(zip_path)
        print(f"LIT-PCBA extracted to {litpcba_dir / 'LIT-PCBA'}")

    except Exception as e:
        print(f"Error downloading LIT-PCBA: {e}")
        print("You may need to download manually from:")
        print("  https://drugdesign.unistra.fr/LIT-PCBA/")
        return None

    return litpcba_dir / "LIT-PCBA"


# =============================================================================
# 2. DUD-E
# =============================================================================

def download_dude(output_dir, targets=None):
    """Download DUD-E benchmark datasets."""
    print("\n" + "="*60)
    print("Downloading DUD-E")
    print("="*60)

    dude_dir = Path(output_dir) / "dude"
    dude_dir.mkdir(parents=True, exist_ok=True)

    if targets is None:
        targets = ['egfr', 'drd2', 'jak2', 'adrb2', 'esr1', 'pparg', 'hdac2', 'bace1']

    base_url = "http://dude.docking.org/targets"

    for target in targets:
        target_dir = dude_dir / target
        target_dir.mkdir(exist_ok=True)

        # Check if already downloaded
        if (target_dir / "actives_final.smi").exists():
            print(f"{target}: Already downloaded")
            continue

        print(f"\nDownloading {target}...")

        for file_type in ['actives_final', 'decoys_final']:
            # Try .smi first, then .mol2.gz
            smi_url = f"{base_url}/{target}/{file_type}.smi"
            mol2_url = f"{base_url}/{target}/{file_type}.mol2.gz"

            try:
                # Try SMILES file
                response = requests.head(smi_url)
                if response.status_code == 200:
                    download_file(smi_url, target_dir / f"{file_type}.smi", f"{target}/{file_type}.smi")
                else:
                    # Download mol2.gz
                    gz_path = target_dir / f"{file_type}.mol2.gz"
                    download_file(mol2_url, gz_path, f"{target}/{file_type}.mol2.gz")

                    # Decompress
                    mol2_path = target_dir / f"{file_type}.mol2"
                    with gzip.open(gz_path, 'rb') as f_in:
                        with open(mol2_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(gz_path)

            except Exception as e:
                print(f"  Error downloading {file_type}: {e}")

    print(f"\nDUD-E data saved to {dude_dir}")
    return dude_dir


# =============================================================================
# 3. DRD2 from ChEMBL
# =============================================================================

def download_drd2_chembl(output_dir):
    """Download DRD2 and dopamine receptor family data from ChEMBL."""
    print("\n" + "="*60)
    print("Downloading DRD2 from ChEMBL")
    print("="*60)

    drd2_dir = Path(output_dir) / "drd2"
    drd2_dir.mkdir(parents=True, exist_ok=True)

    output_file = drd2_dir / "drd2_chembl.csv"

    if output_file.exists():
        print("DRD2 ChEMBL data already downloaded")
        return output_file

    try:
        from chembl_webresource_client.new_client import new_client

        activity = new_client.activity

        # Download DRD2 (CHEMBL217)
        print("Fetching DRD2 activities...")
        drd2_acts = activity.filter(
            target_chembl_id='CHEMBL217',
            standard_type__in=['Ki', 'IC50', 'Kd', 'EC50'],
            standard_relation='=',
            standard_units='nM'
        )

        drd2_df = pd.DataFrame(list(drd2_acts))
        drd2_df = drd2_df[drd2_df['pchembl_value'].notna()]
        drd2_df = drd2_df[drd2_df['canonical_smiles'].notna()]
        drd2_df['target'] = 'DRD2'

        print(f"  DRD2: {len(drd2_df)} records")

        # Download selectivity panel
        selectivity_targets = {
            'DRD1': 'CHEMBL2056',
            'DRD3': 'CHEMBL234',
            'DRD4': 'CHEMBL219'
        }

        all_data = [drd2_df]

        for name, chembl_id in selectivity_targets.items():
            print(f"Fetching {name} activities...")
            acts = activity.filter(
                target_chembl_id=chembl_id,
                standard_type__in=['Ki', 'IC50'],
                standard_relation='=',
                pchembl_value__isnull=False
            )
            df = pd.DataFrame(list(acts))
            df = df[df['canonical_smiles'].notna()]
            df['target'] = name
            all_data.append(df)
            print(f"  {name}: {len(df)} records")

        # Combine all
        combined = pd.concat(all_data, ignore_index=True)

        # Save
        combined.to_csv(output_file, index=False)
        print(f"\nDRD2 data saved to {output_file}")
        print(f"Total records: {len(combined)}")

        return output_file

    except ImportError:
        print("chembl_webresource_client not installed")
        print("Install with: pip install chembl_webresource_client")
        return None
    except Exception as e:
        print(f"Error downloading DRD2: {e}")
        return None


# =============================================================================
# 4. hERG from TDC
# =============================================================================

def download_herg_tdc(output_dir):
    """Download hERG safety data from TDC."""
    print("\n" + "="*60)
    print("Downloading hERG from TDC")
    print("="*60)

    herg_dir = Path(output_dir) / "herg"
    herg_dir.mkdir(parents=True, exist_ok=True)

    output_file = herg_dir / "herg_tdc.csv"

    if output_file.exists():
        print("hERG TDC data already downloaded")
        return output_file

    try:
        from tdc.single_pred import Tox

        print("Loading hERG dataset from TDC...")
        data = Tox(name='hERG')
        df = data.get_data()

        # Rename columns
        df = df.rename(columns={'Drug': 'smiles', 'Y': 'is_blocker'})

        # Save
        df.to_csv(output_file, index=False)

        print(f"hERG data saved to {output_file}")
        print(f"  Total: {len(df)}")
        print(f"  Blockers: {df['is_blocker'].sum()}")
        print(f"  Non-blockers: {(df['is_blocker'] == 0).sum()}")

        return output_file

    except ImportError:
        print("TDC not installed")
        print("Install with: pip install PyTDC")
        return None
    except Exception as e:
        print(f"Error downloading hERG: {e}")
        return None


# =============================================================================
# 5. Tox21 from TDC/MoleculeNet
# =============================================================================

def download_tox21(output_dir):
    """Download Tox21 multi-task dataset."""
    print("\n" + "="*60)
    print("Downloading Tox21")
    print("="*60)

    tox21_dir = Path(output_dir) / "tox21"
    tox21_dir.mkdir(parents=True, exist_ok=True)

    output_file = tox21_dir / "tox21_combined.csv"

    if output_file.exists():
        print("Tox21 data already downloaded")
        return output_file

    try:
        from tdc.single_pred import Tox

        tox21_endpoints = [
            'hERG', 'AMES', 'DILI', 'Skin Reaction', 'Carcinogens_Lagunin',
            'ClinTox', 'hERG_Karim'
        ]

        # Try to get Tox21 specifically
        print("Loading Tox21 endpoints from TDC...")

        all_data = []

        for endpoint in tox21_endpoints:
            try:
                data = Tox(name=endpoint)
                df = data.get_data()
                df = df.rename(columns={'Drug': 'smiles', 'Y': endpoint.lower()})
                all_data.append(df[['smiles', endpoint.lower()]])
                print(f"  {endpoint}: {len(df)} compounds")
            except:
                pass

        if all_data:
            # Merge on SMILES
            combined = all_data[0]
            for df in all_data[1:]:
                combined = combined.merge(df, on='smiles', how='outer')

            combined.to_csv(output_file, index=False)
            print(f"\nTox21 data saved to {output_file}")
            print(f"Total compounds: {len(combined)}")
            return output_file

    except ImportError:
        print("TDC not installed")
    except Exception as e:
        print(f"Error with TDC: {e}")

    # Fallback: try DeepChem
    try:
        print("Trying DeepChem MoleculeNet...")
        from deepchem.molnet import load_tox21

        tasks, datasets, transformers = load_tox21(featurizer='Raw', split='scaffold')
        train, valid, test = datasets

        # Extract SMILES and labels
        all_smiles = list(train.ids) + list(valid.ids) + list(test.ids)
        all_y = np.vstack([train.y, valid.y, test.y])

        df = pd.DataFrame({'smiles': all_smiles})
        for i, task in enumerate(tasks):
            df[task] = all_y[:, i]

        df.to_csv(output_file, index=False)
        print(f"\nTox21 data saved to {output_file}")
        print(f"Tasks: {tasks}")
        print(f"Total compounds: {len(df)}")
        return output_file

    except ImportError:
        print("DeepChem not installed")
        print("Install with: pip install deepchem")
    except Exception as e:
        print(f"Error with DeepChem: {e}")

    return None


# =============================================================================
# 6. MoleculeNet ADMET
# =============================================================================

def download_moleculenet_admet(output_dir):
    """Download MoleculeNet ADMET datasets."""
    print("\n" + "="*60)
    print("Downloading MoleculeNet ADMET")
    print("="*60)

    admet_dir = Path(output_dir) / "moleculenet"
    admet_dir.mkdir(parents=True, exist_ok=True)

    datasets_info = {
        'bbbp': {'task': 'classification', 'metric': 'ROC-AUC'},
        'clintox': {'task': 'classification', 'metric': 'ROC-AUC'},
        'hiv': {'task': 'classification', 'metric': 'ROC-AUC'},
        'sider': {'task': 'classification', 'metric': 'ROC-AUC'},
        'tox21': {'task': 'classification', 'metric': 'ROC-AUC'},
        'lipo': {'task': 'regression', 'metric': 'RMSE'},
        'esol': {'task': 'regression', 'metric': 'RMSE'},
        'freesolv': {'task': 'regression', 'metric': 'RMSE'},
    }

    try:
        import deepchem as dc
        from deepchem import molnet

        for name, info in datasets_info.items():
            output_file = admet_dir / f"{name}.csv"

            if output_file.exists():
                print(f"{name}: Already downloaded")
                continue

            print(f"\nLoading {name}...")

            try:
                loader = getattr(molnet, f"load_{name}")
                tasks, datasets, transformers = loader(featurizer='Raw', split='scaffold')
                train, valid, test = datasets

                # Combine all splits
                all_smiles = list(train.ids) + list(valid.ids) + list(test.ids)
                all_y = np.vstack([train.y, valid.y, test.y])

                # Create split labels
                split_labels = (['train'] * len(train.ids) +
                               ['valid'] * len(valid.ids) +
                               ['test'] * len(test.ids))

                df = pd.DataFrame({'smiles': all_smiles, 'split': split_labels})

                for i, task in enumerate(tasks):
                    df[task] = all_y[:, i]

                df.to_csv(output_file, index=False)
                print(f"  Saved: {output_file}")
                print(f"  Size: {len(df)}, Tasks: {tasks}")

            except Exception as e:
                print(f"  Error loading {name}: {e}")

        return admet_dir

    except ImportError:
        print("DeepChem not installed")
        print("Install with: pip install deepchem")
        return None


# =============================================================================
# 7. ZINC 250K (additional decoys)
# =============================================================================

def download_zinc(output_dir):
    """Download ZINC 250K for additional decoys."""
    print("\n" + "="*60)
    print("Downloading ZINC 250K")
    print("="*60)

    zinc_dir = Path(output_dir) / "zinc"
    zinc_dir.mkdir(parents=True, exist_ok=True)

    output_file = zinc_dir / "zinc_250k.csv"

    if output_file.exists():
        print("ZINC 250K already downloaded")
        return output_file

    url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"

    try:
        print(f"Downloading from {url}")
        df = pd.read_csv(url)
        df.to_csv(output_file, index=False)
        print(f"ZINC 250K saved to {output_file}")
        print(f"  Compounds: {len(df)}")
        return output_file
    except Exception as e:
        print(f"Error downloading ZINC: {e}")
        return None


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Download benchmark data for NEST-DRUG validation')
    parser.add_argument('--output-dir', type=str, default='data/external',
                       help='Output directory for downloaded data')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['all'],
                       choices=['all', 'litpcba', 'dude', 'drd2', 'herg', 'tox21', 'moleculenet', 'zinc'],
                       help='Datasets to download')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets
    if 'all' in datasets:
        datasets = ['litpcba', 'dude', 'drd2', 'herg', 'tox21', 'moleculenet', 'zinc']

    results = {}

    if 'litpcba' in datasets:
        results['litpcba'] = download_litpcba(output_dir)

    if 'dude' in datasets:
        results['dude'] = download_dude(output_dir)

    if 'drd2' in datasets:
        results['drd2'] = download_drd2_chembl(output_dir)

    if 'herg' in datasets:
        results['herg'] = download_herg_tdc(output_dir)

    if 'tox21' in datasets:
        results['tox21'] = download_tox21(output_dir)

    if 'moleculenet' in datasets:
        results['moleculenet'] = download_moleculenet_admet(output_dir)

    if 'zinc' in datasets:
        results['zinc'] = download_zinc(output_dir)

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)

    for name, path in results.items():
        status = "✓" if path else "✗"
        print(f"  {status} {name}: {path}")

    print(f"\nData saved to: {output_dir}")


if __name__ == '__main__':
    main()
