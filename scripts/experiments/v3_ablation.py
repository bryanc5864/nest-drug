#!/usr/bin/env python3
"""
V3 L1 Ablation: Compare V3 performance WITH vs WITHOUT L1 context on DUD-E.

This is the critical experiment to validate that L1 context actually helps.

Usage:
    python scripts/experiments/v3_ablation.py --gpu 0
"""

import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph


def load_model(checkpoint_path, device):
    """Load V3 model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    config = checkpoint.get('config', {})

    num_programs = config.get('n_programs', 5123)
    num_assays = config.get('n_assays', 100)
    num_rounds = config.get('n_rounds', 20)

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

    return model, config


def load_dude_target(data_dir, target):
    """Load DUD-E target data."""
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

    decoys_file = target_dir / "decoys_final.smi"
    decoys = []
    with open(decoys_file) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                decoys.append(parts[0])

    return actives, decoys


def predict_batch(model, smiles_list, device, program_id=0, use_l1=True, max_samples=2000):
    """Run predictions with or without L1 context."""
    model.eval()
    predictions = []

    # Limit samples for speed
    if len(smiles_list) > max_samples:
        indices = np.random.choice(len(smiles_list), max_samples, replace=False)
        smiles_list = [smiles_list[i] for i in indices]

    with torch.no_grad():
        for smi in smiles_list:
            g = smiles_to_graph(smi)
            if g is None:
                predictions.append(None)
                continue

            node_features = g['node_features'].to(device)
            edge_index = g['edge_index'].to(device)
            edge_features = g['edge_features'].to(device)
            batch = torch.zeros(g['num_atoms'], dtype=torch.long, device=device)

            # L1 context: use different program_id per target, or 0 for all
            if use_l1:
                prog_id = program_id
            else:
                prog_id = 0  # Same context for everything

            program_ids = torch.tensor([prog_id], dtype=torch.long, device=device)
            assay_ids = torch.zeros(1, dtype=torch.long, device=device)
            round_ids = torch.zeros(1, dtype=torch.long, device=device)

            preds = model(node_features, edge_index, edge_features, batch,
                         program_ids, assay_ids, round_ids)
            pred = list(preds.values())[0].squeeze().cpu().numpy()
            predictions.append(float(pred) if np.ndim(pred) == 0 else float(pred[0]))

    return predictions, smiles_list


def run_ablation(model, data_dir, device, targets, max_samples=2000):
    """Run L1 ablation on DUD-E targets."""
    results = {}

    # Assign different L1 IDs to different targets
    target_to_l1 = {t: i for i, t in enumerate(targets)}

    for target in tqdm(targets, desc="Targets"):
        actives, decoys = load_dude_target(data_dir, target)
        if actives is None:
            print(f"  Skipping {target} - no data")
            continue

        # Subsample for speed
        n_actives = min(len(actives), max_samples // 2)
        n_decoys = min(len(decoys), max_samples // 2)

        np.random.seed(42)
        actives_sample = list(np.random.choice(actives, n_actives, replace=False))
        decoys_sample = list(np.random.choice(decoys, n_decoys, replace=False))

        all_smiles = actives_sample + decoys_sample
        labels = [1] * len(actives_sample) + [0] * len(decoys_sample)

        # WITH L1 (different context per target)
        preds_with_l1, valid_smiles = predict_batch(
            model, all_smiles, device,
            program_id=target_to_l1[target],
            use_l1=True,
            max_samples=len(all_smiles)
        )

        # WITHOUT L1 (same context=0 for all)
        preds_no_l1, _ = predict_batch(
            model, all_smiles, device,
            program_id=0,
            use_l1=False,
            max_samples=len(all_smiles)
        )

        # Filter None predictions
        valid_idx = [i for i, p in enumerate(preds_with_l1) if p is not None]
        preds_with = [preds_with_l1[i] for i in valid_idx]
        preds_without = [preds_no_l1[i] for i in valid_idx]
        valid_labels = [labels[i] for i in valid_idx]

        if len(set(valid_labels)) < 2:
            print(f"  Skipping {target} - not enough classes")
            continue

        auc_with = roc_auc_score(valid_labels, preds_with)
        auc_without = roc_auc_score(valid_labels, preds_without)

        results[target] = {
            'with_l1': auc_with,
            'without_l1': auc_without,
            'delta': auc_with - auc_without,
            'n_samples': len(valid_labels),
            'n_actives': sum(valid_labels),
        }

        print(f"  {target}: with_L1={auc_with:.4f}, without_L1={auc_without:.4f}, delta={auc_with-auc_without:+.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="V3 L1 Ablation on DUD-E")
    parser.add_argument('--checkpoint', type=str, default='results/v3/best_model.pt',
                        help='V3 checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/external/dude',
                        help='DUD-E data directory')
    parser.add_argument('--output', type=str, default='results/experiments/v3_ablation',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--max-samples', type=int, default=2000,
                        help='Max samples per target (for speed)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading V3 model: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    print(f"Config: {config}")

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # DUD-E targets
    targets = ['egfr', 'drd2', 'adrb2', 'bace1', 'esr1', 'hdac2', 'jak2', 'pparg', 'cyp3a4', 'fxa']

    print("\n" + "="*60)
    print("V3 L1 ABLATION: WITH vs WITHOUT L1 CONTEXT")
    print("="*60)

    results = run_ablation(model, args.data_dir, device, targets, args.max_samples)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    with_l1_aucs = [r['with_l1'] for r in results.values()]
    without_l1_aucs = [r['without_l1'] for r in results.values()]
    deltas = [r['delta'] for r in results.values()]

    print(f"Mean AUC WITH L1:    {np.mean(with_l1_aucs):.4f}")
    print(f"Mean AUC WITHOUT L1: {np.mean(without_l1_aucs):.4f}")
    print(f"Mean Delta:          {np.mean(deltas):+.4f}")
    print(f"Targets improved:    {sum(1 for d in deltas if d > 0)}/{len(deltas)}")

    # Save results
    output_file = output_dir / 'v3_ablation_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'config': config,
            'per_target': results,
            'summary': {
                'mean_with_l1': float(np.mean(with_l1_aucs)),
                'mean_without_l1': float(np.mean(without_l1_aucs)),
                'mean_delta': float(np.mean(deltas)),
                'std_delta': float(np.std(deltas)),
                'targets_improved': sum(1 for d in deltas if d > 0),
                'total_targets': len(deltas),
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
