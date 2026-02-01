#!/usr/bin/env python3
"""
LIT-PCBA Benchmark with L1 Ablation for NEST-DRUG

Addresses reviewer critique: "Evaluate on a benchmark without data leakage
and structural bias" — LIT-PCBA has real experimental inactives from PubChem,
not property-matched decoys.

Runs:
1. Generic L1 (program_id=0) for all 15 targets
2. Correct L1 for overlapping targets (ADRB2, ESR1_ago, ESR1_ant, PPARG)
3. Reports ROC-AUC, BEDROC, Enrichment Factors

LIT-PCBA 15 targets with realistic active rates (0.04-0.27%).
3 targets overlap with DUD-E: ADRB2, ESR1 (ago+ant), PPARG

Usage:
    python scripts/experiments/litpcba_benchmark.py \
        --checkpoint results/v3/best_model.pt \
        --output results/experiments/litpcba_benchmark \
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
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph
from src.benchmarks.data_loaders import load_litpcba_target, LITPCBA_TARGETS
from torch.cuda.amp import autocast


# DUD-E program IDs that overlap with LIT-PCBA targets
LITPCBA_TO_PROGRAM_ID = {
    'ADRB2': 580,       # Beta-2 adrenergic receptor
    'ESR1_ago': 1628,    # Estrogen receptor alpha (agonist)
    'ESR1_ant': 1628,    # Estrogen receptor alpha (antagonist) — same protein
    'PPARG': 3307,       # PPARgamma
    # Other LIT-PCBA targets don't have direct DUD-E program IDs
    # MAPK1 (ERK2), ALDH1, FEN1, GBA, IDH1, KAT2A, MTORC1, OPRK1, PKM2, TP53, VDR
}


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


def score_compounds(model, smiles_list, device, program_id=0, batch_size=256):
    """Score compounds with specified program_id."""
    model.eval()
    all_scores = []
    all_valid = []

    with torch.no_grad():
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Scoring", leave=False):
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


def compute_enrichment_factor(y_true, y_score, fraction=0.01):
    """Compute enrichment factor at given fraction."""
    n = len(y_true)
    n_actives = y_true.sum()
    if n_actives == 0 or n == 0:
        return 0.0

    n_selected = max(1, int(n * fraction))
    top_indices = np.argsort(y_score)[::-1][:n_selected]
    n_actives_found = y_true[top_indices].sum()

    expected = n_actives / n * n_selected
    if expected == 0:
        return 0.0
    return float(n_actives_found / expected)


def main():
    parser = argparse.ArgumentParser(description='LIT-PCBA Benchmark with L1 Ablation')
    parser.add_argument('--checkpoint', type=str, default='results/v3/best_model.pt')
    parser.add_argument('--data-dir', type=str, default='data/external/litpcba')
    parser.add_argument('--output', type=str, default='results/experiments/litpcba_benchmark')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("LIT-PCBA BENCHMARK WITH L1 ABLATION")
    print("Addresses: Evaluate on benchmark without DUD-E bias/leakage")
    print("=" * 70)
    print(f"Device: {device}")

    # Check data availability
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"\nERROR: LIT-PCBA data not found at {data_dir}")
        print("Download from: https://drugdesign.unistra.fr/LIT-PCBA/")
        print(f"Extract to: {data_dir}")
        sys.exit(1)

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)

    # Run benchmark
    results = {}
    available_targets = []

    for target in LITPCBA_TARGETS:
        try:
            df = load_litpcba_target(target, str(data_dir))
            available_targets.append(target)
        except Exception as e:
            print(f"  {target}: Skipping ({e})")
            continue

    if not available_targets:
        print("ERROR: No LIT-PCBA targets found.")
        sys.exit(1)

    print(f"\nFound {len(available_targets)}/{len(LITPCBA_TARGETS)} targets")

    for target in available_targets:
        print(f"\n{'='*60}")
        print(f"TARGET: {target}")
        print(f"{'='*60}")

        df = load_litpcba_target(target, str(data_dir))
        n_actives = int(df['is_active'].sum())
        n_total = len(df)
        print(f"  Compounds: {n_total:,} ({n_actives} actives, {n_total-n_actives:,} inactives)")
        print(f"  Active rate: {n_actives/n_total*100:.3f}%")

        smiles_list = df['smiles'].tolist()
        labels = df['is_active'].values

        target_results = {}

        # Condition 1: Generic L1 (pid=0)
        print(f"  Scoring with generic L1 (pid=0)...")
        scores_generic, valid_generic = score_compounds(
            model, smiles_list, device, program_id=0, batch_size=args.batch_size)
        valid_labels_g = labels[valid_generic]

        if valid_labels_g.sum() > 0 and (valid_labels_g == 0).sum() > 0:
            auc_generic = float(roc_auc_score(valid_labels_g, scores_generic))
            ap_generic = float(average_precision_score(valid_labels_g, scores_generic))
            ef1_generic = compute_enrichment_factor(valid_labels_g, scores_generic, 0.01)
            ef5_generic = compute_enrichment_factor(valid_labels_g, scores_generic, 0.05)
        else:
            auc_generic = ap_generic = ef1_generic = ef5_generic = None

        target_results['generic_l1'] = {
            'roc_auc': auc_generic,
            'avg_precision': ap_generic,
            'ef_1pct': ef1_generic,
            'ef_5pct': ef5_generic,
            'n_valid': int(len(valid_generic)),
        }

        print(f"    Generic L1: AUC={auc_generic:.4f}, AP={ap_generic:.4f}, EF@1%={ef1_generic:.1f}x" if auc_generic else "    Generic L1: N/A")

        # Condition 2: Correct L1 (if available for this target)
        if target in LITPCBA_TO_PROGRAM_ID:
            pid = LITPCBA_TO_PROGRAM_ID[target]
            print(f"  Scoring with correct L1 (pid={pid})...")
            scores_correct, valid_correct = score_compounds(
                model, smiles_list, device, program_id=pid, batch_size=args.batch_size)
            valid_labels_c = labels[valid_correct]

            if valid_labels_c.sum() > 0 and (valid_labels_c == 0).sum() > 0:
                auc_correct = float(roc_auc_score(valid_labels_c, scores_correct))
                ap_correct = float(average_precision_score(valid_labels_c, scores_correct))
                ef1_correct = compute_enrichment_factor(valid_labels_c, scores_correct, 0.01)
                ef5_correct = compute_enrichment_factor(valid_labels_c, scores_correct, 0.05)
            else:
                auc_correct = ap_correct = ef1_correct = ef5_correct = None

            target_results['correct_l1'] = {
                'roc_auc': auc_correct,
                'avg_precision': ap_correct,
                'ef_1pct': ef1_correct,
                'ef_5pct': ef5_correct,
                'n_valid': int(len(valid_correct)),
                'program_id': pid,
            }

            if auc_correct and auc_generic:
                delta = auc_correct - auc_generic
                print(f"    Correct L1: AUC={auc_correct:.4f}, AP={ap_correct:.4f}, EF@1%={ef1_correct:.1f}x (delta={delta:+.4f})")
                target_results['delta_auc'] = delta
            else:
                target_results['delta_auc'] = None
        else:
            target_results['correct_l1'] = None
            target_results['delta_auc'] = None

        results[target] = {
            'n_total': n_total,
            'n_actives': n_actives,
            'active_rate': n_actives / n_total,
            'has_l1': target in LITPCBA_TO_PROGRAM_ID,
            **target_results,
        }

    # Summary
    print(f"\n{'='*70}")
    print("LIT-PCBA SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Target':<12} {'Actives':>8} {'Rate':>7} {'Generic AUC':>12} {'Correct AUC':>12} {'Delta':>8}")
    print(f"  {'-'*60}")

    generic_aucs = []
    correct_aucs = []
    deltas = []

    for target in available_targets:
        r = results[target]
        g_auc = r['generic_l1']['roc_auc']
        c_auc = r['correct_l1']['roc_auc'] if r['correct_l1'] else None
        delta = r['delta_auc']

        g_str = f"{g_auc:.4f}" if g_auc else "N/A"
        c_str = f"{c_auc:.4f}" if c_auc else "—"
        d_str = f"{delta:+.4f}" if delta else "—"

        print(f"  {target:<12} {r['n_actives']:>8} {r['active_rate']*100:>6.2f}% {g_str:>12} {c_str:>12} {d_str:>8}")

        if g_auc: generic_aucs.append(g_auc)
        if c_auc: correct_aucs.append(c_auc)
        if delta: deltas.append(delta)

    print(f"  {'-'*60}")
    if generic_aucs:
        print(f"  {'Mean':<12} {'':>8} {'':>7} {np.mean(generic_aucs):>12.4f}", end="")
        if correct_aucs:
            print(f" {np.mean(correct_aucs):>12.4f} {np.mean(deltas):>+8.4f}")
        else:
            print()

    # Save
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'description': 'LIT-PCBA benchmark with L1 ablation',
        'note': 'Real inactives benchmark — no DUD-E structural bias or decoy issues',
        'config': {
            'checkpoint': args.checkpoint,
            'data_dir': str(data_dir),
            'n_targets_available': len(available_targets),
            'n_targets_with_l1': sum(1 for t in available_targets if t in LITPCBA_TO_PROGRAM_ID),
        },
        'per_target': results,
        'summary': {
            'mean_generic_auc': float(np.mean(generic_aucs)) if generic_aucs else None,
            'std_generic_auc': float(np.std(generic_aucs)) if generic_aucs else None,
            'mean_correct_auc': float(np.mean(correct_aucs)) if correct_aucs else None,
            'mean_delta': float(np.mean(deltas)) if deltas else None,
            'n_targets': len(available_targets),
        }
    }

    output_file = output_dir / 'litpcba_results.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
