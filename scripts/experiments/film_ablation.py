#!/usr/bin/env python3
"""
FiLM Ablation Study for NEST-DRUG

Proves that FiLM conditioning is necessary by comparing against simpler alternatives.
Does NOT retrain — modifies the forward pass at inference time to ablate FiLM.

Three conditions:
1. FiLM (baseline): Normal forward pass — h_mod = γ(context) * h_mol + β(context)
2. No Context: Skip FiLM entirely — h_mod = h_mol (molecular embedding only)
3. Additive: Replace FiLM with simple addition — h_mod = h_mol + β(context) (skip gamma)

Usage:
    python scripts/experiments/film_ablation.py \
        --checkpoint results/v3/best_model.pt \
        --output results/experiments/film_ablation \
        --gpu 0
"""

import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph
from torch.cuda.amp import autocast


# Target-specific program IDs from V3 training mapping
DUDE_TO_PROGRAM_ID = {
    'egfr': 1606,
    'drd2': 1448,
    'adrb2': 580,
    'bace1': 516,
    'esr1': 1628,
    'hdac2': 2177,
    'jak2': 4780,
    'pparg': 3307,
    'cyp3a4': 810,
    'fxa': 1103,
}

DUDE_TARGETS = ['egfr', 'drd2', 'adrb2', 'bace1', 'esr1', 'hdac2', 'jak2', 'pparg', 'cyp3a4', 'fxa']


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
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

    return model, config, num_programs


def load_dude_target(target, data_dir='data/external/dude'):
    """Load DUD-E target actives and decoys."""
    target_dir = Path(data_dir) / target

    actives = []
    actives_file = target_dir / "actives_final.smi"
    if not actives_file.exists():
        return None, None
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


def score_with_condition(model, smiles_list, device, condition, program_id=0, batch_size=256):
    """
    Score compounds under a specific FiLM ablation condition.

    Conditions:
    - 'film': Normal FiLM forward pass (baseline)
    - 'no_context': Skip FiLM, use h_mol directly
    - 'additive': Use h_mol + beta(context) instead of gamma*h_mol + beta
    """
    model.eval()
    all_scores = []
    all_valid = []

    # Save original forward method
    original_forward = model.context_module.forward

    if condition == 'no_context':
        # Monkey-patch: skip FiLM entirely, return h_mol unchanged
        def no_context_forward(h_mol, program_ids, assay_ids, round_ids):
            return h_mol
        model.context_module.forward = no_context_forward

    elif condition == 'additive':
        # Monkey-patch: use additive instead of multiplicative
        # h_mod = h_mol + beta(context) instead of gamma * h_mol + beta
        def additive_forward(h_mol, program_ids, assay_ids, round_ids):
            z_program = model.context_module.program_embeddings(program_ids)
            z_assay = model.context_module.assay_embeddings(assay_ids)
            z_round = model.context_module.round_embeddings(round_ids)
            context = torch.cat([z_program, z_assay, z_round], dim=-1)
            context = model.context_module.context_interaction(context)
            # Only use beta (shift), skip gamma (scale)
            beta = model.context_module.film.beta_net(context)
            return h_mol + beta
        model.context_module.forward = additive_forward

    # else condition == 'film': use original forward (no patching needed)

    try:
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

                # Get first prediction head output
                if 'pchembl_median' in predictions:
                    scores = predictions['pchembl_median'].cpu().numpy().flatten()
                else:
                    first_key = [k for k in predictions.keys() if k not in ['h_mol', 'h_mod']][0]
                    scores = predictions[first_key].cpu().numpy().flatten()

                all_scores.extend([float(s) for s in scores])
                all_valid.extend([i + vi for vi in valid_indices])

    finally:
        # Always restore original forward
        model.context_module.forward = original_forward

    return all_scores, all_valid


def run_film_ablation(model, device, output_dir, num_programs, data_dir='data/external/dude',
                      batch_size=256, max_samples=None):
    """Run FiLM ablation on all DUD-E targets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = ['film', 'no_context', 'additive']
    results = {c: {} for c in conditions}

    for target in tqdm(DUDE_TARGETS, desc="Targets"):
        actives, decoys = load_dude_target(target, data_dir)
        if actives is None:
            print(f"  Skipping {target} - no data")
            continue

        # Subsample if requested
        if max_samples:
            n_act = min(len(actives), max_samples // 2)
            n_dec = min(len(decoys), max_samples // 2)
            np.random.seed(42)
            actives = list(np.random.choice(actives, n_act, replace=False))
            decoys = list(np.random.choice(decoys, n_dec, replace=False))

        all_smiles = actives + decoys
        labels = [1] * len(actives) + [0] * len(decoys)

        # Get correct program ID for this target
        correct_pid = DUDE_TO_PROGRAM_ID.get(target, 0)
        if num_programs < 5123:
            correct_pid = 0  # Fallback for smaller models

        print(f"\n  {target.upper()} ({len(actives)} actives, {len(decoys)} decoys, L1={correct_pid})")

        for condition in conditions:
            # For FiLM and Additive, use correct program ID; for No Context, program ID doesn't matter
            pid = correct_pid if condition != 'no_context' else 0

            scores, valid_indices = score_with_condition(
                model, all_smiles, device, condition,
                program_id=pid, batch_size=batch_size
            )

            # Align scores with labels
            valid_labels = [labels[vi] for vi in valid_indices]

            if len(set(valid_labels)) < 2:
                print(f"    {condition}: skipped (insufficient classes)")
                continue

            auc = roc_auc_score(valid_labels, scores)
            results[condition][target] = float(auc)

            print(f"    {condition:12s}: ROC-AUC = {auc:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='FiLM Ablation Study for NEST-DRUG')
    parser.add_argument('--checkpoint', type=str, default='results/v3/best_model.pt',
                        help='Model checkpoint path')
    parser.add_argument('--data-dir', type=str, default='data/external/dude',
                        help='DUD-E data directory')
    parser.add_argument('--output', type=str, default='results/experiments/film_ablation',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for scoring')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples per target (None = use all)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load model
    print("\nLoading model...")
    model, config, num_programs = load_model(args.checkpoint, device)
    print(f"Config: {config}")
    print(f"Programs: {num_programs}")

    # Run ablation
    print("\n" + "="*70)
    print("FiLM ABLATION STUDY")
    print("Conditions: FiLM (baseline) | No Context | Additive")
    print("="*70)

    results = run_film_ablation(
        model, device, args.output, num_programs,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    # Compute summary
    summary = {}
    for condition in ['film', 'no_context', 'additive']:
        aucs = list(results[condition].values())
        if aucs:
            summary[f'{condition}_mean'] = float(np.mean(aucs))
            summary[f'{condition}_std'] = float(np.std(aucs))
            summary[f'{condition}_median'] = float(np.median(aucs))

    # Compute deltas
    film_targets = set(results['film'].keys())
    for alt in ['no_context', 'additive']:
        alt_targets = set(results[alt].keys())
        common = film_targets & alt_targets
        if common:
            deltas = [results['film'][t] - results[alt][t] for t in common]
            summary[f'film_vs_{alt}_mean_delta'] = float(np.mean(deltas))
            summary[f'film_vs_{alt}_wins'] = sum(1 for d in deltas if d > 0)
            summary[f'film_vs_{alt}_total'] = len(deltas)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Condition':<15} {'Mean AUC':>10} {'Std':>8} {'Median':>10}")
    print("-"*45)
    for condition in ['film', 'no_context', 'additive']:
        mean_key = f'{condition}_mean'
        if mean_key in summary:
            print(f"{condition:<15} {summary[mean_key]:>10.4f} {summary[f'{condition}_std']:>8.4f} {summary[f'{condition}_median']:>10.4f}")

    print()
    for alt in ['no_context', 'additive']:
        delta_key = f'film_vs_{alt}_mean_delta'
        if delta_key in summary:
            wins = summary[f'film_vs_{alt}_wins']
            total = summary[f'film_vs_{alt}_total']
            print(f"FiLM vs {alt}: delta = {summary[delta_key]:+.4f}, wins {wins}/{total} targets")

    # Per-target comparison table
    print(f"\n{'Target':<10} {'FiLM':>8} {'No Ctx':>8} {'Additive':>8} {'F-NC':>8} {'F-Add':>8}")
    print("-"*52)
    for target in DUDE_TARGETS:
        film_auc = results['film'].get(target, float('nan'))
        nc_auc = results['no_context'].get(target, float('nan'))
        add_auc = results['additive'].get(target, float('nan'))
        d1 = film_auc - nc_auc if not (np.isnan(film_auc) or np.isnan(nc_auc)) else float('nan')
        d2 = film_auc - add_auc if not (np.isnan(film_auc) or np.isnan(add_auc)) else float('nan')
        print(f"{target:<10} {film_auc:>8.4f} {nc_auc:>8.4f} {add_auc:>8.4f} {d1:>+8.4f} {d2:>+8.4f}")

    # Save results
    output_data = {
        'checkpoint': args.checkpoint,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'program_id_mapping': DUDE_TO_PROGRAM_ID,
        'results': results,
        'summary': summary,
    }

    output_file = Path(args.output) / 'film_ablation_results.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
