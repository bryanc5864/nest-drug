#!/usr/bin/env python3
"""
Morgan Fingerprint + Random Forest Baseline for DUD-E

Non-neural baseline: Morgan circular fingerprints (radius=2, 2048 bits)
with sklearn RandomForestClassifier. Provides the essential classical
ML comparison for publication.

Usage:
    python scripts/experiments/morgan_rf_baseline.py \
        --output results/experiments/morgan_rf_baseline \
        --n-seeds 5
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import AllChem


DUDE_TARGETS = ['egfr', 'drd2', 'adrb2', 'bace1', 'esr1', 'hdac2', 'jak2', 'pparg', 'cyp3a4', 'fxa']


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


def smiles_to_morgan(smiles_list, radius=2, n_bits=2048):
    """Convert SMILES to Morgan fingerprint numpy array."""
    fps = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros(n_bits, dtype=np.int8)
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            valid_indices.append(i)

    if not fps:
        return None, []

    return np.array(fps), valid_indices


def run_rf_baseline(data_dir, n_seeds=5, n_estimators=500, radius=2, n_bits=2048):
    """Run Morgan FP + RF on all DUD-E targets with multiple seeds."""
    all_results = {}

    for target in tqdm(DUDE_TARGETS, desc="Targets"):
        actives, decoys = load_dude_target(target, data_dir)
        if actives is None:
            print(f"  Skipping {target} - no data")
            continue

        # Build full dataset
        all_smiles = actives + decoys
        labels = np.array([1] * len(actives) + [0] * len(decoys))

        # Convert to fingerprints
        X, valid_indices = smiles_to_morgan(all_smiles, radius=radius, n_bits=n_bits)
        if X is None:
            print(f"  Skipping {target} - no valid molecules")
            continue

        y = labels[valid_indices]
        n_actives = int(y.sum())
        n_decoys = int((y == 0).sum())

        print(f"\n  {target.upper()}: {n_actives} actives, {n_decoys} decoys, {X.shape[1]} features")

        seed_aucs = []

        for seed in range(n_seeds):
            # Train/test split: use 80% for train, 20% for test
            rng = np.random.RandomState(seed)
            n_total = len(y)
            indices = rng.permutation(n_total)
            n_train = int(0.8 * n_total)
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            if len(set(y_test)) < 2:
                continue

            # Train RF
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=seed,
                class_weight='balanced',
            )
            clf.fit(X_train, y_train)

            # Predict
            y_prob = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            seed_aucs.append(auc)

            print(f"    seed {seed}: AUC = {auc:.4f} (train={len(y_train)}, test={len(y_test)})")

        if seed_aucs:
            all_results[target] = {
                'n_actives': n_actives,
                'n_decoys': n_decoys,
                'n_seeds': len(seed_aucs),
                'mean_auc': float(np.mean(seed_aucs)),
                'std_auc': float(np.std(seed_aucs)),
                'min_auc': float(np.min(seed_aucs)),
                'max_auc': float(np.max(seed_aucs)),
                'per_seed_auc': [float(a) for a in seed_aucs],
            }

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Morgan FP + RF Baseline on DUD-E')
    parser.add_argument('--data-dir', type=str, default='data/external/dude',
                        help='DUD-E data directory')
    parser.add_argument('--output', type=str, default='results/experiments/morgan_rf_baseline',
                        help='Output directory')
    parser.add_argument('--n-seeds', type=int, default=5, help='Number of random seeds')
    parser.add_argument('--n-estimators', type=int, default=500, help='RF trees')
    parser.add_argument('--radius', type=int, default=2, help='Morgan FP radius')
    parser.add_argument('--n-bits', type=int, default=2048, help='Morgan FP bits')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("MORGAN FINGERPRINT + RANDOM FOREST BASELINE")
    print(f"Radius={args.radius}, Bits={args.n_bits}, Trees={args.n_estimators}, Seeds={args.n_seeds}")
    print("="*70)

    results = run_rf_baseline(
        args.data_dir,
        n_seeds=args.n_seeds,
        n_estimators=args.n_estimators,
        radius=args.radius,
        n_bits=args.n_bits,
    )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Target':<10} {'Mean AUC':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-"*46)

    all_means = []
    for target in DUDE_TARGETS:
        if target in results:
            r = results[target]
            print(f"{target:<10} {r['mean_auc']:>10.4f} {r['std_auc']:>8.4f} {r['min_auc']:>8.4f} {r['max_auc']:>8.4f}")
            all_means.append(r['mean_auc'])

    if all_means:
        print("-"*46)
        print(f"{'MEAN':<10} {np.mean(all_means):>10.4f} {np.std(all_means):>8.4f}")

    # Save
    output_data = {
        'method': 'Morgan FP + Random Forest',
        'config': {
            'radius': args.radius,
            'n_bits': args.n_bits,
            'n_estimators': args.n_estimators,
            'n_seeds': args.n_seeds,
            'split': '80/20 random',
            'class_weight': 'balanced',
        },
        'timestamp': datetime.now().isoformat(),
        'per_target': results,
        'summary': {
            'mean_auc': float(np.mean(all_means)) if all_means else None,
            'std_auc': float(np.std(all_means)) if all_means else None,
            'n_targets': len(all_means),
        },
    }

    output_file = output_dir / 'morgan_rf_results.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
