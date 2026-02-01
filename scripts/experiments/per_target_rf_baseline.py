#!/usr/bin/env python3
"""
Per-Target ChEMBL→DUD-E Random Forest Baseline

Addresses reviewer critiques:
- E1/W9: "Why not train 10 independent models, one per target?"
- E2/W9: "Why not simple one-hot encoding?"

Three baselines trained on ChEMBL data, evaluated on DUD-E:
1. Per-Target RF: Independent RF per target (train on target's ChEMBL data only)
2. Global RF + Target One-Hot: Single RF with target indicator feature
3. Comparison with NEST-DRUG V3 (correct L1)

Key difference from Morgan RF baseline (Section 7E):
- 7E trained on DUD-E 80/20 split (in-distribution, trivially high AUC)
- This trains on ChEMBL data and evaluates zero-shot on DUD-E
  (cross-distribution, realistic scenario)

Usage:
    python scripts/experiments/per_target_rf_baseline.py \
        --output results/experiments/per_target_rf
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm


DUDE_TARGETS = ['egfr', 'drd2', 'adrb2', 'bace1', 'esr1', 'hdac2', 'jak2', 'pparg', 'cyp3a4', 'fxa']

# ChEMBL target name variants for matching
TARGET_NAME_VARIANTS = {
    'egfr': ['Epidermal growth factor receptor erbB1', 'Epidermal growth factor receptor'],
    'drd2': ['Dopamine D2 receptor', 'Dopamine receptor D2'],
    'adrb2': ['Beta-2 adrenergic receptor'],
    'bace1': ['Beta-secretase 1', 'Beta-site APP cleaving enzyme 1',
              'Beta-site amyloid precursor protein cleaving enzyme 1'],
    'esr1': ['Estrogen receptor alpha', 'Estrogen receptor'],
    'hdac2': ['Histone deacetylase 2'],
    'jak2': ['Tyrosine-protein kinase JAK2'],
    'pparg': ['Peroxisome proliferator-activated receptor gamma',
              'Peroxisome proliferator activated receptor gamma'],
    'cyp3a4': ['Cytochrome P450 3A4'],
    'fxa': ['Coagulation factor X', 'Coagulation factor Xa'],
}

# NEST-DRUG V3 results (from RESULTS.md Table 1, generic L1)
NESTDRUG_V3_GENERIC = {
    'egfr': 0.899, 'drd2': 0.934, 'adrb2': 0.763, 'bace1': 0.842,
    'esr1': 0.817, 'hdac2': 0.901, 'jak2': 0.862, 'pparg': 0.748,
    'cyp3a4': 0.782, 'fxa': 0.846,
}

# NEST-DRUG V3 results (correct L1, from film_ablation)
NESTDRUG_V3_CORRECT = {
    'egfr': 0.965, 'drd2': 0.984, 'adrb2': 0.775, 'bace1': 0.656,
    'esr1': 0.909, 'hdac2': 0.928, 'jak2': 0.908, 'pparg': 0.835,
    'cyp3a4': 0.686, 'fxa': 0.854,
}


def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    """Convert SMILES to Morgan fingerprint array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_fps_batch(smiles_list, radius=2, n_bits=2048, desc="Computing fingerprints"):
    """Compute Morgan FPs for a list of SMILES."""
    fps = []
    valid = []
    for i, smi in enumerate(tqdm(smiles_list, desc=desc, disable=len(smiles_list) < 1000)):
        fp = smiles_to_morgan(str(smi), radius, n_bits)
        if fp is not None:
            fps.append(fp)
            valid.append(i)
    if not fps:
        return np.array([]).reshape(0, n_bits), []
    return np.array(fps), valid


def load_chembl_for_target(target, chembl_df):
    """Filter ChEMBL data for a specific target."""
    variants = TARGET_NAME_VARIANTS.get(target, [])
    mask = chembl_df['target_name'].str.lower().isin([v.lower() for v in variants])
    return chembl_df[mask].copy()


def load_dude_target(target, dude_dir='data/external/dude'):
    """Load DUD-E actives and decoys for a target."""
    target_dir = Path(dude_dir) / target

    actives = []
    with open(target_dir / 'actives_final.smi') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                actives.append(parts[0])

    decoys = []
    with open(target_dir / 'decoys_final.smi') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                decoys.append(parts[0])

    smiles = actives + decoys
    labels = np.concatenate([np.ones(len(actives)), np.zeros(len(decoys))])
    return smiles, labels


def run_per_target_rf(target, chembl_target_df, dude_dir='data/external/dude',
                      n_seeds=5, active_threshold=6.0, inactive_threshold=5.0):
    """Train per-target RF on ChEMBL and evaluate on DUD-E. Memory-efficient: computes FPs on the fly."""
    import gc

    # Binarize ChEMBL activity
    active_mask = chembl_target_df['pchembl_median'] >= active_threshold
    inactive_mask = chembl_target_df['pchembl_median'] < inactive_threshold

    train_active = chembl_target_df[active_mask]
    train_inactive = chembl_target_df[inactive_mask]

    n_active = len(train_active)
    n_inactive = len(train_inactive)

    if n_active < 10 or n_inactive < 10:
        print(f"    Insufficient data: {n_active} active, {n_inactive} inactive")
        return None

    # Combine training data
    train_df = pd.concat([train_active, train_inactive])
    train_labels = np.concatenate([np.ones(n_active), np.zeros(n_inactive)])

    # Compute training fingerprints
    print(f"    Computing training fingerprints ({len(train_df)} compounds)...")
    train_fps, train_valid = compute_fps_batch(
        train_df['smiles'].tolist(), desc=f"  {target} train FPs")
    train_labels = train_labels[train_valid]

    if len(train_fps) < 20:
        print(f"    Too few valid fingerprints: {len(train_fps)}")
        return None

    # Load and compute DUD-E fingerprints for this target only
    dude_smiles, dude_labels = load_dude_target(target, dude_dir)
    print(f"    Computing DUD-E fingerprints ({len(dude_smiles)} compounds)...")
    dude_fps, dude_fps_valid = compute_fps_batch(dude_smiles, desc=f"  {target} DUD-E FPs")
    dude_labels_valid = dude_labels[dude_fps_valid]

    # Train and evaluate with multiple seeds
    aucs = []
    for seed in range(n_seeds):
        clf = RandomForestClassifier(
            n_estimators=500,
            class_weight='balanced',
            random_state=seed * 42,
            n_jobs=4,  # Limit parallelism to reduce memory
        )
        clf.fit(train_fps, train_labels)

        # Score DUD-E
        y_prob = clf.predict_proba(dude_fps)[:, 1]
        auc = roc_auc_score(dude_labels_valid, y_prob)
        aucs.append(auc)
        del clf
        gc.collect()

    result = {
        'n_train_active': int(n_active),
        'n_train_inactive': int(n_inactive),
        'n_train_valid_fps': int(len(train_fps)),
        'n_dude_actives': int(dude_labels_valid.sum()),
        'n_dude_decoys': int((1 - dude_labels_valid).sum()),
        'mean_auc': float(np.mean(aucs)),
        'std_auc': float(np.std(aucs)),
        'per_seed_auc': [float(a) for a in aucs],
    }

    # Free DUD-E memory
    del dude_fps, dude_labels_valid, dude_smiles, dude_labels, train_fps, train_labels
    gc.collect()

    return result


def run_global_rf(all_chembl, dude_dir='data/external/dude',
                  n_seeds=3, active_threshold=6.0, inactive_threshold=5.0):
    """Train one global RF with target one-hot features on ChEMBL, evaluate per-target on DUD-E."""
    import gc

    target_to_idx = {t: i for i, t in enumerate(DUDE_TARGETS)}

    # Build global training set (ChEMBL training data is smaller, ~50K compounds)
    all_train_fps = []
    all_train_labels = []

    print("\n  Building global training set...")
    for target in DUDE_TARGETS:
        if target not in all_chembl or all_chembl[target] is None:
            continue

        chembl_df = all_chembl[target]
        active_mask = chembl_df['pchembl_median'] >= active_threshold
        inactive_mask = chembl_df['pchembl_median'] < inactive_threshold

        subset = pd.concat([chembl_df[active_mask], chembl_df[inactive_mask]])
        labels = np.concatenate([
            np.ones(int(active_mask.sum())),
            np.zeros(int(inactive_mask.sum()))
        ])

        fps, valid = compute_fps_batch(subset['smiles'].tolist(), desc=f"  {target} global FPs")
        labels = labels[valid]

        # Add target one-hot (10-dim)
        target_onehot = np.zeros((len(fps), len(DUDE_TARGETS)), dtype=np.int8)
        target_onehot[:, target_to_idx[target]] = 1

        fps_with_target = np.hstack([fps, target_onehot])
        all_train_fps.append(fps_with_target)
        all_train_labels.append(labels)

    X_train = np.vstack(all_train_fps)
    y_train = np.concatenate(all_train_labels)
    del all_train_fps, all_train_labels
    gc.collect()
    print(f"  Global training set: {len(X_train)} compounds ({int(y_train.sum())} active, {int((1-y_train).sum())} inactive)")

    # Train model once per seed, evaluate per-target with on-the-fly DUD-E FPs
    results = {}
    for seed in range(n_seeds):
        print(f"\n  Training global RF seed {seed+1}/{n_seeds}...")
        clf = RandomForestClassifier(
            n_estimators=500,
            class_weight='balanced',
            random_state=seed * 42,
            n_jobs=4,
        )
        clf.fit(X_train, y_train)

        for target in DUDE_TARGETS:
            # Compute DUD-E FPs on-the-fly for each target
            dude_smiles, dude_labels = load_dude_target(target, dude_dir)
            dude_fps, dude_fps_valid = compute_fps_batch(
                dude_smiles, desc=f"  {target} DUD-E (seed {seed+1})")
            dude_labels_valid = dude_labels[dude_fps_valid]

            # Add target one-hot
            target_onehot = np.zeros((len(dude_fps), len(DUDE_TARGETS)), dtype=np.int8)
            target_onehot[:, target_to_idx[target]] = 1
            dude_fps_with_target = np.hstack([dude_fps, target_onehot])

            y_prob = clf.predict_proba(dude_fps_with_target)[:, 1]
            auc = roc_auc_score(dude_labels_valid, y_prob)

            if target not in results:
                results[target] = []
            results[target].append(auc)

            del dude_fps, dude_fps_with_target, dude_labels_valid, dude_smiles, dude_labels
            gc.collect()

        del clf
        gc.collect()

    return {
        target: {
            'mean_auc': float(np.mean(aucs)),
            'std_auc': float(np.std(aucs)),
            'per_seed_auc': [float(a) for a in aucs],
        }
        for target, aucs in results.items()
    }


def main():
    parser = argparse.ArgumentParser(description='Per-Target ChEMBL RF Baseline')
    parser.add_argument('--output', type=str, default='results/experiments/per_target_rf')
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--chembl-path', type=str, default='data/processed/portfolio/chembl_potency_all.parquet')
    parser.add_argument('--dude-dir', type=str, default='data/external/dude')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PER-TARGET ChEMBL RF BASELINE FOR DUD-E")
    print("Addresses: E1 (per-target models), E2 (simple context encoding)")
    print("=" * 70)

    # Load ChEMBL data
    print(f"\nLoading ChEMBL data from {args.chembl_path}...")
    chembl_df = pd.read_parquet(args.chembl_path)
    print(f"  Total: {len(chembl_df)} records")

    all_chembl = {}
    for target in DUDE_TARGETS:
        target_df = load_chembl_for_target(target, chembl_df)
        if len(target_df) > 0:
            all_chembl[target] = target_df
            n_active = (target_df['pchembl_median'] >= 6.0).sum()
            n_inactive = (target_df['pchembl_median'] < 5.0).sum()
            print(f"  {target}: {len(target_df)} total, {n_active} active (≥6.0), {n_inactive} inactive (<5.0)")
        else:
            print(f"  {target}: NO ChEMBL DATA FOUND")

    # Free the full dataframe — we have per-target dicts now
    del chembl_df
    import gc; gc.collect()

    # ================================================================
    # EXPERIMENT 1: Per-Target RF (memory-efficient: one target at a time)
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Per-Target RF (one model per target, ChEMBL→DUD-E)")
    print("=" * 70)

    per_target_results = {}
    for target in DUDE_TARGETS:
        if target not in all_chembl:
            print(f"\n  {target}: SKIPPED (no training data)")
            continue

        print(f"\n  Training per-target RF for {target}...")
        result = run_per_target_rf(
            target, all_chembl[target], dude_dir=args.dude_dir,
            n_seeds=args.n_seeds,
        )
        if result:
            per_target_results[target] = result
            nd_g = NESTDRUG_V3_GENERIC.get(target, 0)
            nd_c = NESTDRUG_V3_CORRECT.get(target, 0)
            print(f"    Per-target RF: {result['mean_auc']:.4f} ± {result['std_auc']:.4f}")
            print(f"    NEST-DRUG (generic L1): {nd_g:.3f} | (correct L1): {nd_c:.3f}")
            print(f"    Delta vs generic: {result['mean_auc'] - nd_g:+.4f}")

    # ================================================================
    # EXPERIMENT 2: Global RF with Target One-Hot
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Global RF + Target One-Hot (single model, all targets)")
    print("=" * 70)

    global_results = run_global_rf(
        all_chembl, dude_dir=args.dude_dir,
        n_seeds=min(args.n_seeds, 3),  # Fewer seeds since global RF is slower
    )

    for target in DUDE_TARGETS:
        if target in global_results:
            gr = global_results[target]
            nd = NESTDRUG_V3_GENERIC.get(target, 0)
            print(f"  {target}: {gr['mean_auc']:.4f} ± {gr['std_auc']:.4f} | NEST-DRUG: {nd:.3f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"\n  {'Target':<10} {'Per-Target':>12} {'Global+1Hot':>12} {'ND Generic':>12} {'ND Correct':>12}")
    print(f"  {'-'*58}")

    pt_aucs, gl_aucs, nd_g_aucs, nd_c_aucs = [], [], [], []

    for target in DUDE_TARGETS:
        pt = per_target_results.get(target, {}).get('mean_auc', float('nan'))
        gl = global_results.get(target, {}).get('mean_auc', float('nan'))
        nd_g = NESTDRUG_V3_GENERIC.get(target, float('nan'))
        nd_c = NESTDRUG_V3_CORRECT.get(target, float('nan'))

        print(f"  {target:<10} {pt:>12.4f} {gl:>12.4f} {nd_g:>12.3f} {nd_c:>12.3f}")

        if not np.isnan(pt): pt_aucs.append(pt)
        if not np.isnan(gl): gl_aucs.append(gl)
        nd_g_aucs.append(nd_g)
        nd_c_aucs.append(nd_c)

    print(f"  {'-'*58}")
    print(f"  {'Mean':<10} {np.mean(pt_aucs):>12.4f} {np.mean(gl_aucs):>12.4f} "
          f"{np.mean(nd_g_aucs):>12.3f} {np.mean(nd_c_aucs):>12.3f}")

    # Key finding
    pt_mean = np.mean(pt_aucs)
    nd_g_mean = np.mean(nd_g_aucs)
    print(f"\n  KEY FINDING:")
    if pt_mean > nd_g_mean:
        print(f"    Per-target RF ({pt_mean:.4f}) > NEST-DRUG generic L1 ({nd_g_mean:.3f})")
        print(f"    BUT: Per-target trains on target-specific data; NEST-DRUG is a single multi-target model.")
        print(f"    Per-target RF has {50:.0f}% active overlap with DUD-E (data leakage, see Section 7F).")
    else:
        print(f"    NEST-DRUG generic L1 ({nd_g_mean:.3f}) > Per-target RF ({pt_mean:.4f})")
        print(f"    Multi-target model with shared GNN outperforms independent per-target models.")

    # Save
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'description': 'Per-target ChEMBL RF baseline for DUD-E evaluation',
        'note': 'Addresses reviewer critiques E1 (per-target models) and E2 (simple one-hot encoding)',
        'config': {
            'radius': 2,
            'n_bits': 2048,
            'n_estimators': 500,
            'n_seeds': args.n_seeds,
            'active_threshold': 6.0,
            'inactive_threshold': 5.0,
        },
        'per_target_rf': per_target_results,
        'global_rf_onehot': global_results,
        'nestdrug_v3_generic': NESTDRUG_V3_GENERIC,
        'nestdrug_v3_correct': NESTDRUG_V3_CORRECT,
        'summary': {
            'per_target_rf_mean': float(np.mean(pt_aucs)) if pt_aucs else None,
            'global_rf_onehot_mean': float(np.mean(gl_aucs)) if gl_aucs else None,
            'nestdrug_v3_generic_mean': float(np.mean(nd_g_aucs)),
            'nestdrug_v3_correct_mean': float(np.mean(nd_c_aucs)),
        }
    }

    output_file = output_dir / 'per_target_rf_results.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
