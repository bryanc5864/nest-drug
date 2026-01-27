#!/usr/bin/env python3
"""
Statistical Significance Tests for NEST-DRUG L1 Ablation.

Runs the L1 ablation experiment with multiple random seeds to compute
confidence intervals and paired t-test p-values.

Usage:
    python scripts/experiments/statistical_significance.py --checkpoint results/v3/best_model.pt --gpu 0 --n-seeds 5
"""

import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph


DUDE_TO_V3_PROGRAM_ID = {
    'egfr': 1606, 'drd2': 1448, 'adrb2': 580, 'bace1': 516,
    'esr1': 1628, 'hdac2': 2177, 'jak2': 4780, 'pparg': 3307,
    'cyp3a4': 810, 'fxa': 1103,
}


def load_model(checkpoint_path, device):
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
        num_programs=num_programs, num_assays=num_assays,
        num_rounds=num_rounds, endpoints=endpoints,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, num_programs


def load_dude_target(data_dir, target):
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
    with open(target_dir / "decoys_final.smi") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                decoys.append(parts[0])

    return actives, decoys


def evaluate_seed(model, actives, decoys, device, program_id, seed, max_samples=2000):
    """Run evaluation with a specific random seed for subsampling."""
    np.random.seed(seed)

    n_actives = min(len(actives), max_samples // 2)
    n_decoys = min(len(decoys), max_samples // 2)

    actives_sample = list(np.random.choice(actives, n_actives, replace=False))
    decoys_sample = list(np.random.choice(decoys, n_decoys, replace=False))

    all_smiles = actives_sample + decoys_sample
    labels = [1] * len(actives_sample) + [0] * len(decoys_sample)

    predictions = []
    valid_labels = []

    with torch.no_grad():
        for smi, label in zip(all_smiles, labels):
            g = smiles_to_graph(smi)
            if g is None:
                continue

            node_features = g['node_features'].to(device)
            edge_index = g['edge_index'].to(device)
            edge_features = g['edge_features'].to(device)
            batch = torch.zeros(g['num_atoms'], dtype=torch.long, device=device)

            program_ids = torch.tensor([program_id], dtype=torch.long, device=device)
            assay_ids = torch.zeros(1, dtype=torch.long, device=device)
            round_ids = torch.zeros(1, dtype=torch.long, device=device)

            preds = model(node_features, edge_index, edge_features, batch,
                         program_ids, assay_ids, round_ids)
            pred = list(preds.values())[0].squeeze().cpu().numpy()
            predictions.append(float(pred) if np.ndim(pred) == 0 else float(pred[0]))
            valid_labels.append(label)

    if len(set(valid_labels)) < 2:
        return None

    return roc_auc_score(valid_labels, predictions)


def main():
    parser = argparse.ArgumentParser(description="Statistical Significance Tests")
    parser.add_argument('--checkpoint', type=str, default='results/v3/best_model.pt')
    parser.add_argument('--data-dir', type=str, default='data/external/dude')
    parser.add_argument('--output', type=str, default='results/experiments/statistical_significance')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--max-samples', type=int, default=2000)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, num_programs = load_model(args.checkpoint, device)
    print(f"Model loaded: {num_programs} programs")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = list(DUDE_TO_V3_PROGRAM_ID.keys())
    seeds = list(range(args.n_seeds))

    all_results = {}

    for target in tqdm(targets, desc="Targets"):
        actives, decoys = load_dude_target(args.data_dir, target)
        if actives is None:
            continue

        correct_pid = DUDE_TO_V3_PROGRAM_ID[target] if num_programs >= 5123 else 0

        correct_aucs = []
        generic_aucs = []

        for seed in seeds:
            auc_correct = evaluate_seed(model, actives, decoys, device, correct_pid, seed, args.max_samples)
            auc_generic = evaluate_seed(model, actives, decoys, device, 0, seed, args.max_samples)

            if auc_correct is not None and auc_generic is not None:
                correct_aucs.append(auc_correct)
                generic_aucs.append(auc_generic)

        if len(correct_aucs) < 2:
            continue

        correct_aucs = np.array(correct_aucs)
        generic_aucs = np.array(generic_aucs)
        deltas = correct_aucs - generic_aucs

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(correct_aucs, generic_aucs)

        all_results[target] = {
            'correct_l1_id': correct_pid,
            'correct_aucs': correct_aucs.tolist(),
            'generic_aucs': generic_aucs.tolist(),
            'mean_correct': float(correct_aucs.mean()),
            'std_correct': float(correct_aucs.std()),
            'mean_generic': float(generic_aucs.mean()),
            'std_generic': float(generic_aucs.std()),
            'mean_delta': float(deltas.mean()),
            'std_delta': float(deltas.std()),
            'ci_95_lower': float(deltas.mean() - 1.96 * deltas.std() / np.sqrt(len(deltas))),
            'ci_95_upper': float(deltas.mean() + 1.96 * deltas.std() / np.sqrt(len(deltas))),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_005': bool(p_value < 0.05),
            'n_seeds': len(correct_aucs),
        }

        print(f"  {target}: delta={deltas.mean():+.4f} ± {deltas.std():.4f}, p={p_value:.4f} {'*' if p_value < 0.05 else ''}")

    # Summary
    all_deltas = [r['mean_delta'] for r in all_results.values()]
    all_pvals = [r['p_value'] for r in all_results.values()]
    n_sig = sum(1 for r in all_results.values() if r['significant_005'])

    summary = {
        'mean_delta': float(np.mean(all_deltas)),
        'std_delta': float(np.std(all_deltas)),
        'n_significant_005': n_sig,
        'total_targets': len(all_results),
        'n_seeds': args.n_seeds,
    }

    print(f"\n{'='*60}")
    print(f"SUMMARY (n_seeds={args.n_seeds})")
    print(f"{'='*60}")
    print(f"Mean delta: {summary['mean_delta']:+.4f} ± {summary['std_delta']:.4f}")
    print(f"Significant (p<0.05): {n_sig}/{len(all_results)} targets")

    # Save
    results_path = output_dir / 'statistical_significance_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'n_seeds': args.n_seeds,
            'per_target': all_results,
            'summary': summary,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
