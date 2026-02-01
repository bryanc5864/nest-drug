#!/usr/bin/env python3
"""
Data Leakage Check: ChEMBL Training vs DUD-E Evaluation

Critical audit: Are DUD-E actives present in the ChEMBL training set?
If so, high DUD-E AUC could reflect memorization, not generalization.

Checks:
1. Exact SMILES overlap between training and DUD-E actives
2. Canonical SMILES overlap (RDKit canonicalization)
3. InChIKey overlap (structure-based, handles SMILES variations)
4. Per-target breakdown: which targets have leakage?
5. Training data statistics per DUD-E target

Usage:
    python scripts/experiments/data_leakage_check.py \
        --output results/experiments/data_leakage_check
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem.inchi import MolFromInchi, InchiToInchiKey, MolToInchi


DUDE_TARGETS = ['egfr', 'drd2', 'adrb2', 'bace1', 'esr1', 'hdac2', 'jak2', 'pparg', 'cyp3a4', 'fxa']

# Map DUD-E target names to ChEMBL target names (as they appear in training data)
DUDE_TO_CHEMBL_NAMES = {
    'egfr': ['Epidermal growth factor receptor erbB1', 'EGFR', 'Epidermal growth factor receptor'],
    'drd2': ['Dopamine D2 receptor', 'DRD2', 'Dopamine receptor D2'],
    'adrb2': ['Beta-2 adrenergic receptor', 'ADRB2', 'Beta-2 adrenergic receptor'],
    'bace1': ['Beta-secretase 1', 'BACE1', 'BACE-1', 'Beta-site APP cleaving enzyme 1',
              'Beta-site amyloid precursor protein cleaving enzyme 1'],
    'esr1': ['Estrogen receptor alpha', 'ESR1', 'Estrogen receptor', 'ERalpha'],
    'hdac2': ['Histone deacetylase 2', 'HDAC2'],
    'jak2': ['Tyrosine-protein kinase JAK2', 'JAK2', 'Janus kinase 2'],
    'pparg': ['Peroxisome proliferator-activated receptor gamma', 'PPARG', 'PPARgamma',
              'Peroxisome proliferator activated receptor gamma'],
    'cyp3a4': ['Cytochrome P450 3A4', 'CYP3A4'],
    'fxa': ['Coagulation factor X', 'FXA', 'Factor Xa', 'Coagulation factor Xa'],
}

# ChEMBL target IDs for DUD-E targets
DUDE_TO_CHEMBL_ID = {
    'egfr': 'CHEMBL203',
    'drd2': 'CHEMBL217',
    'adrb2': 'CHEMBL210',
    'bace1': 'CHEMBL4822',
    'esr1': 'CHEMBL206',
    'hdac2': 'CHEMBL1871',
    'jak2': 'CHEMBL2971',
    'pparg': 'CHEMBL235',
    'cyp3a4': 'CHEMBL340',
    'fxa': 'CHEMBL244',
}


def canonicalize(smi):
    """Canonicalize SMILES via RDKit."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def smiles_to_inchikey(smi):
    """Convert SMILES to InChIKey for structure-based matching."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        inchi = MolToInchi(mol)
        if inchi is None:
            return None
        return InchiToInchiKey(inchi)
    except Exception:
        return None


def load_dude_target(target, data_dir='data/external/dude'):
    """Load DUD-E target actives and decoys."""
    target_dir = Path(data_dir) / target
    actives_file = target_dir / "actives_final.smi"
    if not actives_file.exists():
        return None, None

    actives = []
    with open(actives_file) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                actives.append(parts[0])

    decoys = []
    decoys_file = target_dir / "decoys_final.smi"
    with open(decoys_file) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                decoys.append(parts[0])

    return actives, decoys


def load_training_data():
    """Load all ChEMBL training data sources."""
    dfs = []

    # Primary training file
    primary = Path('data/processed/portfolio/chembl_potency_all.parquet')
    if primary.exists():
        df = pd.read_parquet(primary)
        df['source'] = 'portfolio_all'
        dfs.append(df)
        print(f"  Portfolio all: {len(df)} records, {df['smiles'].nunique()} unique SMILES")

    # Raw ChEMBL v2
    raw_v2 = Path('data/raw/chembl_v2/chembl_v2_all.parquet')
    if raw_v2.exists():
        df = pd.read_parquet(raw_v2)
        df['source'] = 'chembl_v2'
        # Standardize column names
        if 'target_name' not in df.columns and 'target_id' in df.columns:
            df['target_name'] = df.get('target_name', '')
        dfs.append(df)
        print(f"  ChEMBL v2 raw: {len(df)} records, {df['smiles'].nunique()} unique SMILES")

    # Raw ChEMBL v2 category files
    for cat_file in ['proteases.parquet', 'ion_channels.parquet', 'cyps.parquet']:
        path = Path('data/raw/chembl_v2') / cat_file
        if path.exists():
            df = pd.read_parquet(path)
            df['source'] = f'chembl_v2_{cat_file.replace(".parquet", "")}'
            dfs.append(df)
            print(f"  {cat_file}: {len(df)} records")

    # Enriched v2
    enriched = Path('data/raw/chembl_v2_enriched/chembl_v2_all.parquet')
    if enriched.exists():
        df = pd.read_parquet(enriched)
        df['source'] = 'chembl_v2_enriched'
        dfs.append(df)
        print(f"  Enriched v2: {len(df)} records")

    return dfs


def run_leakage_check(data_dir, training_dfs):
    """Check overlap between training data and DUD-E evaluation data."""
    print("\n" + "="*70)
    print("DATA LEAKAGE CHECK: ChEMBL Training vs DUD-E Evaluation")
    print("="*70)

    # Build master training SMILES sets
    print("\nBuilding training compound sets...")
    all_train_smiles_raw = set()
    all_train_smiles_canonical = set()
    all_train_inchikeys = set()
    per_target_train_smiles = defaultdict(set)
    per_target_train_canonical = defaultdict(set)

    for df in training_dfs:
        source = df['source'].iloc[0] if 'source' in df.columns and len(df) > 0 else '?'
        print(f"  Processing {source} ({len(df)} records)...")

        # Collect raw SMILES
        smiles_col = df['smiles'].dropna().astype(str)
        all_train_smiles_raw.update(smiles_col)

        # Canonicalize unique SMILES (much faster than row-by-row)
        unique_smiles = smiles_col.unique()
        print(f"    Canonicalizing {len(unique_smiles)} unique SMILES...")
        canon_map = {}
        for smi in tqdm(unique_smiles, desc=f"    {source}", leave=False):
            can = canonicalize(smi)
            if can:
                canon_map[smi] = can
                all_train_smiles_canonical.add(can)

        # Track per-target using vectorized filtering
        target_name_col = df['target_name'].astype(str) if 'target_name' in df.columns else pd.Series([''] * len(df))
        target_id_col = df['target_chembl_id'].astype(str) if 'target_chembl_id' in df.columns else pd.Series([''] * len(df))

        for dude_target, chembl_names in DUDE_TO_CHEMBL_NAMES.items():
            name_mask = target_name_col.isin(chembl_names)
            chembl_id = DUDE_TO_CHEMBL_ID.get(dude_target, '')
            id_mask = target_id_col == chembl_id
            mask = name_mask | id_mask

            matched_smiles = smiles_col[mask].unique()
            per_target_train_smiles[dude_target].update(matched_smiles)
            for smi in matched_smiles:
                can = canon_map.get(smi)
                if can:
                    per_target_train_canonical[dude_target].add(can)

    print(f"\n  Total unique raw SMILES in training: {len(all_train_smiles_raw)}")
    print(f"  Total unique canonical SMILES in training: {len(all_train_smiles_canonical)}")

    # Also compute InChIKeys for a sample (full set too slow for 1.3M)
    print("\n  Computing InChIKeys for training set...")
    for smi in tqdm(list(all_train_smiles_canonical)[:50000], desc="  InChIKeys", leave=False):
        ik = smiles_to_inchikey(smi)
        if ik:
            all_train_inchikeys.add(ik)
    print(f"  InChIKeys computed for {len(all_train_inchikeys)} training compounds (sampled)")

    # Check each DUD-E target
    results = {}

    for target in tqdm(DUDE_TARGETS, desc="DUD-E Targets"):
        actives, decoys = load_dude_target(target, data_dir)
        if actives is None:
            continue

        # Canonicalize DUD-E actives
        dude_raw = set(actives)
        dude_canonical = set()
        dude_inchikeys = set()

        for smi in actives:
            can = canonicalize(smi)
            if can:
                dude_canonical.add(can)
            ik = smiles_to_inchikey(smi)
            if ik:
                dude_inchikeys.add(ik)

        # Also canonicalize decoys
        decoy_canonical = set()
        for smi in decoys:
            can = canonicalize(smi)
            if can:
                decoy_canonical.add(can)

        # Check overlaps
        raw_overlap = dude_raw & all_train_smiles_raw
        canonical_overlap = dude_canonical & all_train_smiles_canonical
        inchikey_overlap = dude_inchikeys & all_train_inchikeys
        decoy_overlap = decoy_canonical & all_train_smiles_canonical

        # Per-target training overlap
        target_train_smiles = per_target_train_canonical.get(target, set())
        same_target_overlap = dude_canonical & target_train_smiles

        n_actives = len(dude_canonical)
        n_decoys = len(decoy_canonical)

        print(f"\n  {target.upper()}:")
        print(f"    DUD-E actives: {n_actives} (canonical)")
        print(f"    Raw SMILES overlap:       {len(raw_overlap):>5d} / {len(dude_raw)} "
              f"({len(raw_overlap)/max(len(dude_raw),1)*100:.1f}%)")
        print(f"    Canonical SMILES overlap:  {len(canonical_overlap):>5d} / {n_actives} "
              f"({len(canonical_overlap)/max(n_actives,1)*100:.1f}%)")
        print(f"    InChIKey overlap:          {len(inchikey_overlap):>5d} / {len(dude_inchikeys)} "
              f"({len(inchikey_overlap)/max(len(dude_inchikeys),1)*100:.1f}%)")
        print(f"    Same-target train overlap: {len(same_target_overlap):>5d} / {n_actives} "
              f"({len(same_target_overlap)/max(n_actives,1)*100:.1f}%)")
        print(f"    Decoy overlap with train:  {len(decoy_overlap):>5d} / {n_decoys} "
              f"({len(decoy_overlap)/max(n_decoys,1)*100:.1f}%)")
        print(f"    Training compounds for this target: {len(target_train_smiles)}")

        results[target] = {
            'n_dude_actives': n_actives,
            'n_dude_decoys': n_decoys,
            'n_train_for_target': len(target_train_smiles),
            'raw_smiles_overlap': len(raw_overlap),
            'canonical_smiles_overlap': len(canonical_overlap),
            'inchikey_overlap': len(inchikey_overlap),
            'same_target_overlap': len(same_target_overlap),
            'decoy_overlap': len(decoy_overlap),
            'active_leakage_pct': float(len(canonical_overlap) / max(n_actives, 1) * 100),
            'same_target_leakage_pct': float(len(same_target_overlap) / max(n_actives, 1) * 100),
            'decoy_leakage_pct': float(len(decoy_overlap) / max(n_decoys, 1) * 100),
        }

    # Summary
    if results:
        all_active_pcts = [r['active_leakage_pct'] for r in results.values()]
        all_same_pcts = [r['same_target_leakage_pct'] for r in results.values()]
        all_decoy_pcts = [r['decoy_leakage_pct'] for r in results.values()]

        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Target':<10} {'Active Leak%':>12} {'Same-Tgt%':>12} {'Decoy Leak%':>12} {'Train Size':>10}")
        print(f"  {'-'*56}")
        for target in DUDE_TARGETS:
            if target in results:
                r = results[target]
                print(f"  {target:<10} {r['active_leakage_pct']:>11.1f}% {r['same_target_leakage_pct']:>11.1f}% "
                      f"{r['decoy_leakage_pct']:>11.1f}% {r['n_train_for_target']:>10d}")
        print(f"  {'-'*56}")
        print(f"  {'MEAN':<10} {np.mean(all_active_pcts):>11.1f}% {np.mean(all_same_pcts):>11.1f}% "
              f"{np.mean(all_decoy_pcts):>11.1f}%")

        # Interpretation
        max_leak = max(all_active_pcts)
        mean_leak = np.mean(all_active_pcts)
        print(f"\n  INTERPRETATION:")
        if mean_leak > 50:
            print(f"  ⚠ SEVERE LEAKAGE: {mean_leak:.1f}% mean active overlap")
            print(f"  DUD-E results may reflect memorization, not generalization")
        elif mean_leak > 10:
            print(f"  ⚠ MODERATE LEAKAGE: {mean_leak:.1f}% mean active overlap")
            print(f"  Some DUD-E actives seen during training; results should be interpreted cautiously")
        elif mean_leak > 0:
            print(f"  MINOR LEAKAGE: {mean_leak:.1f}% mean active overlap")
            print(f"  Small fraction of DUD-E actives in training; unlikely to significantly inflate AUC")
        else:
            print(f"  NO LEAKAGE DETECTED: 0% active overlap")
            print(f"  DUD-E evaluation is clean")

        # Key insight for paper
        print(f"\n  KEY INSIGHT FOR REVIEWERS:")
        print(f"  Active-decoy leakage asymmetry matters most.")
        print(f"  If actives leak more than decoys, the model has an unfair advantage.")
        print(f"  If both leak equally, the effect is neutral.")

        results['_summary'] = {
            'mean_active_leakage_pct': float(np.mean(all_active_pcts)),
            'mean_same_target_leakage_pct': float(np.mean(all_same_pcts)),
            'mean_decoy_leakage_pct': float(np.mean(all_decoy_pcts)),
            'max_active_leakage_pct': float(max(all_active_pcts)),
            'total_train_smiles': len(all_train_smiles_canonical),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Data Leakage Check')
    parser.add_argument('--data-dir', type=str, default='data/external/dude')
    parser.add_argument('--output', type=str, default='results/experiments/data_leakage_check')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    training_dfs = load_training_data()

    results = run_leakage_check(args.data_dir, training_dfs)

    # Save
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'description': 'Data leakage check: ChEMBL training vs DUD-E evaluation overlap',
        'note': 'Addresses reviewer concern M2/E3 about train-test contamination',
        'results': results,
    }

    output_file = output_dir / 'data_leakage_results.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
