#!/usr/bin/env python3
"""Murcko-scaffold split robustness check.

Reviewer rSJL: temporal splits can preserve scaffold continuity. We
test whether the FiLM benefit holds under a strict scaffold split,
where all training scaffolds are excluded from test.

Comparison: same model architecture, same training budget; the only
difference is whether we split by molecule (random) or by scaffold.
"""
import os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

# Reuse model/data utils from fair_concat
from rebuttal_fair_concat import (
    CondModel, MolDataset, collate, DUDE_TARGETS, train_one
)


def murcko_scaffold(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None: return None
        s = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(s)
    except Exception:
        return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-records', type=int, default=40000)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--out', default='/tmp/nest_rebuttal/scaffold_split.json')
    args = ap.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    full = pd.read_parquet('/tmp/nest_train_split.parquet')
    val_pool = pd.read_parquet('/tmp/nest_val_split.parquet')

    # Restrict to DUD-E + top-50 targets
    keep = list(DUDE_TARGETS.keys())
    extra = full['target_chembl_id'].value_counts().head(50).index.tolist()
    keep = list(dict.fromkeys(keep + extra))
    full = full[full['target_chembl_id'].isin(keep)].dropna(subset=['pchembl_median']).copy()

    # Subsample to manageable size
    full = full.groupby('target_chembl_id', group_keys=False).apply(
        lambda x: x.sample(min(len(x), max(args.n_records // len(keep), 100)), random_state=42)
    ).reset_index(drop=True)
    print(f'targets={len(keep)}  total={len(full)}')

    # Compute scaffolds
    print('computing Murcko scaffolds...')
    full['scaffold'] = full['smiles'].apply(murcko_scaffold)
    full = full[full['scaffold'].notna()].copy()
    print(f'after scaffold filter: {len(full)}, unique scaffolds: {full["scaffold"].nunique()}')

    # Two splits:
    # (A) random: stratify by target, random 90/10
    rng = np.random.RandomState(42)
    full = full.sample(frac=1, random_state=42).reset_index(drop=True)
    n_tot = len(full)
    n_val = n_tot // 10
    rand_train = full.iloc[n_val:].reset_index(drop=True)
    rand_val = full.iloc[:n_val].reset_index(drop=True)

    # (B) scaffold: hold out 10% of scaffolds entirely
    scaffolds = list(full['scaffold'].unique())
    rng.shuffle(scaffolds)
    n_val_scaf = max(1, len(scaffolds) // 10)
    val_scaffolds = set(scaffolds[:n_val_scaf])
    sc_val = full[full['scaffold'].isin(val_scaffolds)].reset_index(drop=True)
    sc_train = full[~full['scaffold'].isin(val_scaffolds)].reset_index(drop=True)
    print(f'random  train={len(rand_train)}  val={len(rand_val)}')
    print(f'scaffold train={len(sc_train)}  val={len(sc_val)}  val_scaffolds={n_val_scaf}')

    prog_map = {t: i for i, t in enumerate(sorted(keep))}

    out = {'random': {}, 'scaffold': {}}
    for split_name, (tr, vl) in [('random', (rand_train, rand_val)),
                                 ('scaffold', (sc_train, sc_val))]:
        for fusion in ['none', 'film']:
            print(f'\n=== {split_name} / {fusion} ===')
            r = train_one(fusion, tr, vl, prog_map, device,
                          epochs=args.epochs, batch=256, seed=0)
            out[split_name][fusion] = r

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)

    print('\n=== SUMMARY ===')
    for s in ['random', 'scaffold']:
        for fusion in ['none', 'film']:
            r = out[s][fusion]
            print(f'  {s:>9}  {fusion:>5}   AUC = {r["auc"]:.4f}')
    none_rand = out['random']['none']['auc']
    film_rand = out['random']['film']['auc']
    none_sc = out['scaffold']['none']['auc']
    film_sc = out['scaffold']['film']['auc']
    print()
    print(f'Random:    Δ FiLM-none = {film_rand - none_rand:+.4f}')
    print(f'Scaffold:  Δ FiLM-none = {film_sc - none_sc:+.4f}')
    print(f'AUC drop random→scaffold (FiLM): {film_rand - film_sc:+.4f}')


if __name__ == '__main__':
    main()
