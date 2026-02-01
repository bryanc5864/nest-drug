"""
Data Loaders for NEST-DRUG Benchmarks

Provides unified interfaces for loading:
- LIT-PCBA
- DUD-E
- DRD2 (ChEMBL)
- hERG (TDC)
- Tox21
- MoleculeNet ADMET
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from typing import Dict, List, Optional, Tuple


# =============================================================================
# LIT-PCBA Loader
# =============================================================================

LITPCBA_TARGETS = [
    'ADRB2', 'ALDH1', 'ESR1_ago', 'ESR1_ant', 'FEN1',
    'GBA', 'IDH1', 'KAT2A', 'MAPK1', 'MTORC1',
    'OPRK1', 'PKM2', 'PPARG', 'TP53', 'VDR'
]


def load_litpcba_target(target_name: str, base_path: str = "data/external/litpcba/LIT-PCBA") -> pd.DataFrame:
    """
    Load a single LIT-PCBA target.

    Args:
        target_name: Name of target (e.g., 'ADRB2')
        base_path: Path to LIT-PCBA directory

    Returns:
        DataFrame with columns: smiles, id, is_active, target
    """
    target_path = Path(base_path) / target_name

    # Try different file naming conventions
    actives_files = [
        target_path / "actives_final.smi",
        target_path / "actives.smi"
    ]
    inactives_files = [
        target_path / "inactives_final.smi",
        target_path / "inactives.smi"
    ]

    # Load actives
    actives = None
    for f in actives_files:
        if f.exists():
            actives = pd.read_csv(f, sep=r'\s+', header=None, names=['smiles', 'id'], engine='python')
            break

    if actives is None:
        raise FileNotFoundError(f"Could not find actives file for {target_name}")

    actives['is_active'] = 1

    # Load inactives
    inactives = None
    for f in inactives_files:
        if f.exists():
            inactives = pd.read_csv(f, sep=r'\s+', header=None, names=['smiles', 'id'], engine='python')
            break

    if inactives is None:
        raise FileNotFoundError(f"Could not find inactives file for {target_name}")

    inactives['is_active'] = 0

    # Combine
    combined = pd.concat([actives, inactives], ignore_index=True)
    combined['target'] = target_name

    # Validate SMILES
    combined['valid'] = combined['smiles'].apply(
        lambda x: Chem.MolFromSmiles(str(x)) is not None if pd.notna(x) else False
    )
    combined = combined[combined['valid']].drop(columns=['valid'])

    return combined


def load_all_litpcba(base_path: str = "data/external/litpcba/LIT-PCBA",
                     targets: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Load all LIT-PCBA targets.

    Args:
        base_path: Path to LIT-PCBA directory
        targets: List of targets to load (None = all)

    Returns:
        Dictionary mapping target name to DataFrame
    """
    if targets is None:
        targets = LITPCBA_TARGETS

    data = {}
    for target in targets:
        try:
            data[target] = load_litpcba_target(target, base_path)
            print(f"  {target}: {data[target]['is_active'].sum()} actives, "
                  f"{(data[target]['is_active']==0).sum()} inactives")
        except Exception as e:
            print(f"  {target}: Error - {e}")

    return data


# =============================================================================
# DUD-E Loader
# =============================================================================

DUDE_TARGETS = ['egfr', 'drd2', 'adrb2', 'bace1', 'esr1', 'hdac2', 'jak2', 'pparg', 'cyp3a4', 'fxa']


def load_dude_target(target_name: str, base_path: str = "data/external/dude") -> pd.DataFrame:
    """
    Load a single DUD-E target.

    Args:
        target_name: Name of target (lowercase, e.g., 'egfr')
        base_path: Path to DUD-E directory

    Returns:
        DataFrame with columns: smiles, id, is_active, target
    """
    target_path = Path(base_path) / target_name

    # Load actives
    actives_file = target_path / "actives_final.smi"
    if actives_file.exists():
        actives = pd.read_csv(actives_file, sep=' ', header=None, names=['smiles', 'id'])
    else:
        raise FileNotFoundError(f"Could not find actives for {target_name}")

    actives['is_active'] = 1

    # Load decoys
    decoys_file = target_path / "decoys_final.smi"
    if decoys_file.exists():
        decoys = pd.read_csv(decoys_file, sep=' ', header=None, names=['smiles', 'id'])
    else:
        raise FileNotFoundError(f"Could not find decoys for {target_name}")

    decoys['is_active'] = 0

    # Combine
    combined = pd.concat([actives, decoys], ignore_index=True)
    combined['target'] = target_name

    return combined


def load_all_dude(base_path: str = "data/external/dude",
                  targets: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Load all DUD-E targets."""
    if targets is None:
        targets = DUDE_TARGETS

    data = {}
    for target in targets:
        try:
            data[target] = load_dude_target(target, base_path)
            print(f"  {target}: {data[target]['is_active'].sum()} actives, "
                  f"{(data[target]['is_active']==0).sum()} decoys")
        except Exception as e:
            print(f"  {target}: Error - {e}")

    return data


# =============================================================================
# DRD2 ChEMBL Loader
# =============================================================================

def load_drd2_chembl(data_path: str = "data/external/drd2/drd2_chembl.csv") -> pd.DataFrame:
    """
    Load DRD2 data from ChEMBL.

    Returns:
        DataFrame with columns: smiles, pActivity, year, target, is_active
    """
    df = pd.read_csv(data_path)

    # Standardize columns
    df = df.rename(columns={
        'canonical_smiles': 'smiles',
        'pchembl_value': 'pActivity',
        'document_year': 'year'
    })

    # Define activity threshold (pKi >= 7 = Ki < 100 nM)
    df['is_active'] = (df['pActivity'] >= 7.0).astype(int)

    return df


def setup_drd2_dmta_replay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Setup DRD2 data for DMTA replay experiment.

    Args:
        df: DRD2 DataFrame

    Returns:
        DataFrame with round_id column added
    """
    # Sort by year
    df = df.sort_values('year').reset_index(drop=True)

    # Create round mapping from years
    year_min = df['year'].min()
    df['round_id'] = (df['year'] - year_min).astype(int)

    # Filter to DRD2 only for primary task
    drd2_df = df[df['target'] == 'DRD2'].copy()

    print(f"DRD2 DMTA setup:")
    print(f"  Compounds: {len(drd2_df)}")
    print(f"  Year range: {drd2_df['year'].min()} - {drd2_df['year'].max()}")
    print(f"  Rounds: {drd2_df['round_id'].nunique()}")
    print(f"  Actives: {drd2_df['is_active'].sum()}")

    return drd2_df


# =============================================================================
# hERG Loader
# =============================================================================

def load_herg(data_path: str = "data/external/herg/herg_tdc.csv") -> pd.DataFrame:
    """
    Load hERG safety data.

    Returns:
        DataFrame with columns: smiles, is_blocker
    """
    df = pd.read_csv(data_path)

    # Rename if needed
    if 'Drug' in df.columns:
        df = df.rename(columns={'Drug': 'smiles', 'Y': 'is_blocker'})

    print(f"hERG dataset:")
    print(f"  Total: {len(df)}")
    print(f"  Blockers: {df['is_blocker'].sum()}")
    print(f"  Non-blockers: {(df['is_blocker']==0).sum()}")

    return df


# =============================================================================
# Tox21 Loader
# =============================================================================

def load_tox21(data_path: str = "data/external/tox21/tox21_combined.csv") -> Tuple[pd.DataFrame, List[str]]:
    """
    Load Tox21 multi-task data.

    Returns:
        Tuple of (DataFrame, list of task names)
    """
    df = pd.read_csv(data_path)

    # Get task columns (everything except smiles and split)
    task_cols = [c for c in df.columns if c not in ['smiles', 'split', 'mol_id']]

    print(f"Tox21 dataset:")
    print(f"  Compounds: {len(df)}")
    print(f"  Tasks: {len(task_cols)}")

    for task in task_cols[:5]:  # Show first 5
        n_pos = (df[task] == 1).sum()
        print(f"    {task}: {n_pos} positives")

    return df, task_cols


# =============================================================================
# MoleculeNet ADMET Loader
# =============================================================================

def load_moleculenet_dataset(name: str,
                             base_path: str = "data/external/moleculenet") -> Tuple[pd.DataFrame, List[str], str]:
    """
    Load a MoleculeNet ADMET dataset.

    Args:
        name: Dataset name (e.g., 'bbbp', 'lipo')
        base_path: Path to MoleculeNet directory

    Returns:
        Tuple of (DataFrame, task names, task type)
    """
    data_path = Path(base_path) / f"{name}.csv"
    df = pd.read_csv(data_path)

    # Get task columns
    non_task_cols = ['smiles', 'split', 'mol_id']
    task_cols = [c for c in df.columns if c not in non_task_cols]

    # Determine task type
    classification_datasets = ['bbbp', 'clintox', 'hiv', 'sider', 'tox21']
    task_type = 'classification' if name in classification_datasets else 'regression'

    print(f"{name}: {len(df)} compounds, {len(task_cols)} tasks, {task_type}")

    return df, task_cols, task_type


def load_all_moleculenet(base_path: str = "data/external/moleculenet",
                         datasets: Optional[List[str]] = None) -> Dict:
    """Load all MoleculeNet ADMET datasets."""
    if datasets is None:
        datasets = ['bbbp', 'clintox', 'hiv', 'sider', 'lipo', 'esol', 'freesolv']

    data = {}
    for name in datasets:
        try:
            df, tasks, task_type = load_moleculenet_dataset(name, base_path)
            data[name] = {
                'df': df,
                'tasks': tasks,
                'task_type': task_type
            }
        except Exception as e:
            print(f"  {name}: Error - {e}")

    return data


# =============================================================================
# ZINC Loader (for decoys)
# =============================================================================

def load_zinc_decoys(data_path: str = "data/external/zinc/zinc_250k.csv",
                     n_samples: Optional[int] = None) -> List[str]:
    """
    Load ZINC compounds as potential decoys.

    Args:
        data_path: Path to ZINC CSV
        n_samples: Number of samples to return (None = all)

    Returns:
        List of SMILES strings
    """
    df = pd.read_csv(data_path)

    smiles_col = 'smiles' if 'smiles' in df.columns else df.columns[0]
    smiles = df[smiles_col].dropna().tolist()

    if n_samples and n_samples < len(smiles):
        import random
        smiles = random.sample(smiles, n_samples)

    print(f"ZINC: {len(smiles)} compounds loaded")

    return smiles
