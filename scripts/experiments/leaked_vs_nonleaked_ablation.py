#!/usr/bin/env python3
"""
Leaked vs Non-Leaked DUD-E Ablation

Directly answers reviewer question: "Does the L1 improvement hold on the
~50% of actives that are NOT leaked from ChEMBL?"

Approach:
1. Load ChEMBL training SMILES and canonicalize
2. For each DUD-E target, split actives into leaked (in ChEMBL) vs non-leaked
3. Score all compounds with correct L1 and generic L1
4. Compute ROC-AUC separately for leaked actives vs decoys and non-leaked actives vs decoys
5. Report L1 delta for each subset

Usage:
    python scripts/experiments/leaked_vs_nonleaked_ablation.py \
        --checkpoint results/v3/best_model.pt \
        --output results/experiments/leaked_vs_nonleaked \
        --gpu 0
"""

import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from rdkit import Chem

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph
from torch.cuda.amp import autocast


DUDE_TARGETS = ['egfr', 'drd2', 'adrb2', 'bace1', 'esr1', 'hdac2', 'jak2', 'pparg', 'cyp3a4', 'fxa']

DUDE_TO_PROGRAM_ID = {
    'egfr': 1606, 'drd2': 1448, 'adrb2': 580, 'bace1': 516,
    'esr1': 1628, 'hdac2': 2177, 'jak2': 4780, 'pparg': 3307,
    'cyp3a4': 810, 'fxa': 1103,
}


def canonicalize(smi):
    """Canonicalize SMILES via RDKit."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def load_training_smiles():
    """Load all ChEMBL training SMILES and return set of canonical SMILES."""
    print("Loading ChEMBL training data...")
    all_smiles = set()

    sources = [
        'data/processed/portfolio/chembl_potency_all.parquet',
        'data/raw/chembl_v2/chembl_v2_all.parquet',
        'data/raw/chembl_v2/proteases.parquet',
        'data/raw/chembl_v2/ion_channels.parquet',
        'data/raw/chembl_v2/cyps.parquet',
        'data/raw/chembl_v2_enriched/chembl_v2_all.parquet',
    ]

    for src in sources:
        path = Path(src)
        if path.exists():
            df = pd.read_parquet(path)
            if 'smiles' in df.columns:
                smiles_list = df['smiles'].dropna().unique()
                print(f"  {path.name}: {len(smiles_list)} unique SMILES")
                for smi in smiles_list:
                    canon = canonicalize(smi)
                    if canon:
                        all_smiles.add(canon)

    print(f"  Total unique canonical training SMILES: {len(all_smiles)}")
    return all_smiles


def load_model(checkpoint_path, device):
    """Load NEST-DRUG model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    config = checkpoint.get('config', {})

    num_programs = config.get('n_programs', config.get('num_programs',
        state_dict['context_module.program_embeddings.embeddings.weight'].shape[0]))
    num_assays = config.get('n_assays', config.get('num_assays',
        state_dict['context_module.assay_embeddings.embeddings.weight'].shape[0]))
    num_rounds = config.get('n_rounds', config.get('num_rounds',
        state_dict['context_module.round_embeddings.embeddings.weight'].shape[0]))

    endpoint_names = []
    for key in state_dict.keys():
        if 'prediction_heads.heads.' in key and '.mlp.0.weight' in key:
            name = key.split('prediction_heads.heads.')[1].split('.mlp')[0]
            endpoint_names.append(name)

    endpoints = {name: {'type': 'regression', 'weight': 1.0} for name in endpoint_names}

    model = create_nest_drug(
        num_programs=num_programs,
        num_assays=num_assays,
        num_rounds=num_rounds,
        endpoints=endpoints,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def prepare_batch(smiles_list, device):
    """Convert SMILES list to batched graph tensors."""
    graphs = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        g = smiles_to_graph(smi)
        if g is not None:
            graphs.append(g)
            valid_indices.append(i)
    if not graphs:
        return None, []

    node_features = torch.cat([g['node_features'] for g in graphs], dim=0).to(device)
    edge_index_list = []
    edge_features_list = []
    batch_indices = []
    offset = 0
    for idx, g in enumerate(graphs):
        edge_index_list.append(g['edge_index'] + offset)
        edge_features_list.append(g['edge_features'])
        batch_indices.extend([idx] * g['num_atoms'])
        offset += g['num_atoms']

    edge_index = torch.cat(edge_index_list, dim=1).to(device)
    edge_features = torch.cat(edge_features_list, dim=0).to(device)
    batch_tensor = torch.tensor(batch_indices, dtype=torch.long, device=device)

    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_features': edge_features,
        'batch': batch_tensor,
        'n_mols': len(graphs),
    }, valid_indices


def score_compounds(model, smiles_list, device, program_id, batch_size=256):
    """Score compounds with a specific program_id."""
    model.eval()
    all_scores = []
    all_valid = []

    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i+batch_size]
            batch_data, valid_indices = prepare_batch(batch_smiles, device)
            if batch_data is None:
                continue

            n_mols = batch_data['n_mols']
            program_ids = torch.full((n_mols,), program_id, dtype=torch.long, device=device)
            assay_ids = torch.zeros(n_mols, dtype=torch.long, device=device)
            round_ids = torch.zeros(n_mols, dtype=torch.long, device=device)

            with autocast(enabled=True):
                predictions = model(
                    node_features=batch_data['node_features'],
                    edge_index=batch_data['edge_index'],
                    edge_features=batch_data['edge_features'],
                    batch=batch_data['batch'],
                    program_ids=program_ids,
                    assay_ids=assay_ids,
                    round_ids=round_ids,
                )

            if 'pchembl_median' in predictions:
                scores = predictions['pchembl_median'].cpu().numpy().flatten()
            else:
                first_key = [k for k in predictions.keys() if k not in ['h_mol', 'h_mod']][0]
                scores = predictions[first_key].cpu().numpy().flatten()

            all_scores.extend([float(s) for s in scores])
            all_valid.extend([i + vi for vi in valid_indices])

    return np.array(all_scores), np.array(all_valid)


def load_dude_target(target, data_dir='data/external/dude'):
    """Load DUD-E actives and decoys for a target."""
    target_dir = Path(data_dir) / target
    actives, decoys = [], []
    with open(target_dir / 'actives_final.smi') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                actives.append(parts[0])
    with open(target_dir / 'decoys_final.smi') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                decoys.append(parts[0])
    return actives, decoys


def main():
    parser = argparse.ArgumentParser(description='Leaked vs Non-Leaked DUD-E Ablation')
    parser.add_argument('--checkpoint', type=str, default='results/v3/best_model.pt')
    parser.add_argument('--output', type=str, default='results/experiments/leaked_vs_nonleaked')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Step 1: Load training SMILES
    training_smiles = load_training_smiles()

    # Step 2: Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)

    # Step 3: For each target, split and evaluate
    results = {}

    for target in DUDE_TARGETS:
        print(f"\n{'='*60}")
        print(f"TARGET: {target.upper()}")
        print(f"{'='*60}")

        actives, decoys = load_dude_target(target)
        all_smiles = actives + decoys
        n_actives = len(actives)
        n_decoys = len(decoys)

        # Classify actives as leaked or non-leaked
        leaked_active_idx = []
        nonleaked_active_idx = []
        for i, smi in enumerate(actives):
            canon = canonicalize(smi)
            if canon and canon in training_smiles:
                leaked_active_idx.append(i)
            else:
                nonleaked_active_idx.append(i)

        n_leaked = len(leaked_active_idx)
        n_nonleaked = len(nonleaked_active_idx)
        leak_pct = n_leaked / n_actives * 100 if n_actives > 0 else 0

        print(f"  Actives: {n_actives} ({n_leaked} leaked [{leak_pct:.1f}%], {n_nonleaked} non-leaked)")
        print(f"  Decoys: {n_decoys}")

        # Score with correct L1 and generic L1
        pid = DUDE_TO_PROGRAM_ID[target]

        print(f"  Scoring with correct L1 (pid={pid})...")
        scores_correct, valid_correct = score_compounds(model, all_smiles, device, pid)

        print(f"  Scoring with generic L1 (pid=0)...")
        scores_generic, valid_generic = score_compounds(model, all_smiles, device, 0)

        # Build labels: 1 for actives, 0 for decoys
        labels = np.array([1]*n_actives + [0]*n_decoys)

        # For each condition, compute AUC on: all, leaked-only, non-leaked-only
        target_results = {}
        for condition, scores, valid_idx in [
            ('correct_l1', scores_correct, valid_correct),
            ('generic_l1', scores_generic, valid_generic),
        ]:
            valid_labels = labels[valid_idx]
            valid_smiles_idx = valid_idx  # indices into all_smiles

            # Which valid indices are leaked actives, non-leaked actives, decoys?
            leaked_set = set(leaked_active_idx)
            nonleaked_set = set(nonleaked_active_idx)
            decoy_range = set(range(n_actives, n_actives + n_decoys))

            is_leaked_active = np.array([idx in leaked_set for idx in valid_idx])
            is_nonleaked_active = np.array([idx in nonleaked_set for idx in valid_idx])
            is_decoy = np.array([idx in decoy_range for idx in valid_idx])

            # AUC: all actives vs all decoys
            mask_all = is_leaked_active | is_nonleaked_active | is_decoy
            if mask_all.sum() > 0 and valid_labels[mask_all].sum() > 0:
                auc_all = float(roc_auc_score(valid_labels[mask_all], scores[mask_all]))
            else:
                auc_all = None

            # AUC: leaked actives vs decoys
            mask_leaked = is_leaked_active | is_decoy
            labels_leaked = valid_labels[mask_leaked]
            if n_leaked > 0 and labels_leaked.sum() > 0:
                auc_leaked = float(roc_auc_score(labels_leaked, scores[mask_leaked]))
            else:
                auc_leaked = None

            # AUC: non-leaked actives vs decoys
            mask_nonleaked = is_nonleaked_active | is_decoy
            labels_nonleaked = valid_labels[mask_nonleaked]
            if n_nonleaked > 0 and labels_nonleaked.sum() > 0:
                auc_nonleaked = float(roc_auc_score(labels_nonleaked, scores[mask_nonleaked]))
            else:
                auc_nonleaked = None

            # Mean scores for each group
            mean_leaked = float(scores[is_leaked_active].mean()) if is_leaked_active.sum() > 0 else None
            mean_nonleaked = float(scores[is_nonleaked_active].mean()) if is_nonleaked_active.sum() > 0 else None
            mean_decoy = float(scores[is_decoy].mean()) if is_decoy.sum() > 0 else None

            target_results[condition] = {
                'auc_all': auc_all,
                'auc_leaked': auc_leaked,
                'auc_nonleaked': auc_nonleaked,
                'mean_score_leaked': mean_leaked,
                'mean_score_nonleaked': mean_nonleaked,
                'mean_score_decoy': mean_decoy,
            }

        # Compute deltas (correct - generic)
        deltas = {}
        for subset in ['auc_all', 'auc_leaked', 'auc_nonleaked']:
            c = target_results['correct_l1'][subset]
            g = target_results['generic_l1'][subset]
            if c is not None and g is not None:
                deltas[subset] = c - g
            else:
                deltas[subset] = None

        results[target] = {
            'n_actives': n_actives,
            'n_leaked': n_leaked,
            'n_nonleaked': n_nonleaked,
            'leak_pct': leak_pct,
            'n_decoys': n_decoys,
            'correct_l1': target_results['correct_l1'],
            'generic_l1': target_results['generic_l1'],
            'delta_all': deltas['auc_all'],
            'delta_leaked': deltas['auc_leaked'],
            'delta_nonleaked': deltas['auc_nonleaked'],
        }

        print(f"\n  {'Subset':<15} {'Correct L1':>11} {'Generic L1':>11} {'Delta':>8}")
        print(f"  {'-'*47}")
        for subset, key in [('All', 'auc_all'), ('Leaked', 'auc_leaked'), ('Non-leaked', 'auc_nonleaked')]:
            c = target_results['correct_l1'][key]
            g = target_results['generic_l1'][key]
            d = deltas[key]
            c_str = f"{c:.4f}" if c is not None else "N/A"
            g_str = f"{g:.4f}" if g is not None else "N/A"
            d_str = f"{d:+.4f}" if d is not None else "N/A"
            print(f"  {subset:<15} {c_str:>11} {g_str:>11} {d_str:>8}")

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY: L1 Effect on Leaked vs Non-Leaked Actives")
    print(f"{'='*80}")
    print(f"\n  {'Target':<10} {'Leak%':>6} {'Delta All':>10} {'Delta Leaked':>13} {'Delta NonLeak':>14} {'Consistent?':>12}")
    print(f"  {'-'*65}")

    all_deltas = []
    leaked_deltas = []
    nonleaked_deltas = []

    for target in DUDE_TARGETS:
        r = results[target]
        d_all = r['delta_all']
        d_leak = r['delta_leaked']
        d_nonleak = r['delta_nonleaked']

        if d_all is not None: all_deltas.append(d_all)
        if d_leak is not None: leaked_deltas.append(d_leak)
        if d_nonleak is not None: nonleaked_deltas.append(d_nonleak)

        # Check consistency: same sign for leaked and non-leaked
        if d_leak is not None and d_nonleak is not None:
            consistent = "Yes" if (d_leak > 0) == (d_nonleak > 0) else "DIFFER"
        else:
            consistent = "N/A"

        d_all_s = f"{d_all:+.4f}" if d_all is not None else "N/A"
        d_leak_s = f"{d_leak:+.4f}" if d_leak is not None else "N/A"
        d_nonleak_s = f"{d_nonleak:+.4f}" if d_nonleak is not None else "N/A"

        print(f"  {target:<10} {r['leak_pct']:>5.1f}% {d_all_s:>10} {d_leak_s:>13} {d_nonleak_s:>14} {consistent:>12}")

    print(f"  {'-'*65}")
    mean_all = np.mean(all_deltas) if all_deltas else 0
    mean_leaked = np.mean(leaked_deltas) if leaked_deltas else 0
    mean_nonleaked = np.mean(nonleaked_deltas) if nonleaked_deltas else 0
    print(f"  {'Mean':<10} {'':>6} {mean_all:>+10.4f} {mean_leaked:>+13.4f} {mean_nonleaked:>+14.4f}")

    print(f"\n  KEY FINDING:")
    if nonleaked_deltas:
        n_positive_nonleaked = sum(1 for d in nonleaked_deltas if d > 0)
        print(f"    Non-leaked actives: L1 effect = {mean_nonleaked:+.4f} ({n_positive_nonleaked}/{len(nonleaked_deltas)} positive)")
        if mean_nonleaked > 0:
            print(f"    → L1 improvement HOLDS on genuinely unseen compounds")
            print(f"    → This is NOT memorization — context provides genuine transfer signal")
        else:
            print(f"    → L1 improvement does NOT hold on unseen compounds")
            print(f"    → The effect may partly reflect memorization")

    # Save results
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'description': 'Leaked vs non-leaked DUD-E L1 ablation',
        'note': 'Answers: does L1 improvement hold on actives NOT in ChEMBL training?',
        'config': {
            'checkpoint': args.checkpoint,
            'n_training_smiles': len(training_smiles),
        },
        'per_target': results,
        'summary': {
            'mean_delta_all': mean_all,
            'mean_delta_leaked': mean_leaked,
            'mean_delta_nonleaked': mean_nonleaked,
            'n_targets_with_nonleaked': len(nonleaked_deltas),
            'n_positive_nonleaked': sum(1 for d in nonleaked_deltas if d > 0) if nonleaked_deltas else 0,
        }
    }

    output_file = output_dir / 'leaked_vs_nonleaked_results.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
