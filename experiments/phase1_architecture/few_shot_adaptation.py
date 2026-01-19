#!/usr/bin/env python3
"""
Few-Shot Adaptation Test for NEST-DRUG (Experiment 1.4)

Goal: Prove nested architecture enables fast adaptation to new targets.

Design:
1. Use DUD-E targets as "new" targets (not seen during ChEMBL pretraining)
2. Provide small support set (N = 10, 25, 50, 100 actives)
3. Adapt L1 context only (freeze everything else)
4. Compare to zero-shot and full fine-tuning

Success criteria: L1 adaptation achieves ≥80% of full fine-tuning with 10x fewer steps.

Usage:
    python experiments/phase1_architecture/few_shot_adaptation.py \
        --checkpoint results/phase1/ablation_dude/L0_L1_seed0_best.pt \
        --device cuda
"""

import argparse
import sys
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph
from src.benchmarks.data_loaders import load_all_dude


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = create_nest_drug(
        num_programs=config['n_programs'],
        num_assays=config['n_assays'],
        num_rounds=config['n_rounds'],
    )

    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('base_model.') for k in state_dict.keys()):
        state_dict = {k.replace('base_model.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)

    return model, config


def prepare_data(df, support_size: int, seed: int = 42):
    """Split data into support and query sets."""
    np.random.seed(seed)

    actives = df[df['is_active'] == 1].copy()
    decoys = df[df['is_active'] == 0].copy()

    # Shuffle actives
    actives = actives.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Support set: N actives + N decoys (balanced)
    support_actives = actives.iloc[:support_size]
    support_decoys = decoys.sample(n=support_size, random_state=seed)
    support_df = pd.concat([support_actives, support_decoys]).sample(frac=1, random_state=seed)

    # Query set: remaining actives + remaining decoys
    query_actives = actives.iloc[support_size:]
    query_decoys = decoys.drop(support_decoys.index)
    query_df = pd.concat([query_actives, query_decoys])

    return support_df, query_df


def smiles_to_batch(smiles_list: List[str], device: torch.device):
    """Convert SMILES list to batched tensors."""
    graphs = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        graph = smiles_to_graph(smi)
        if graph is not None:
            graphs.append(graph)
            valid_indices.append(i)

    if not graphs:
        return None, None

    # Collate
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


def evaluate_model(model, smiles_list, y_true, device, l1_id=0, batch_size=256):
    """Evaluate model on data."""
    model.eval()
    all_scores = []
    all_valid = []

    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i+batch_size]
            batch_data, valid_idx = smiles_to_batch(batch_smiles, device)

            if batch_data is None:
                all_valid.extend([False] * len(batch_smiles))
                continue

            # Track which are valid
            valid_mask = [False] * len(batch_smiles)
            for idx in valid_idx:
                valid_mask[idx] = True
            all_valid.extend(valid_mask)

            n_mols = batch_data['n_mols']
            program_ids = torch.full((n_mols,), l1_id, dtype=torch.long, device=device)
            assay_ids = torch.zeros(n_mols, dtype=torch.long, device=device)
            round_ids = torch.zeros(n_mols, dtype=torch.long, device=device)

            with autocast(enabled=True):
                predictions = model(
                    batch_data['node_features'],
                    batch_data['edge_index'],
                    batch_data['edge_features'],
                    batch_data['batch'],
                    program_ids, assay_ids, round_ids
                )

            pred = predictions.get('pActivity', list(predictions.values())[0])
            all_scores.extend(pred.cpu().numpy().flatten().tolist())

    # Filter to valid only
    y_true_valid = [y for y, v in zip(y_true, all_valid) if v]
    scores_valid = [s for s, v in zip(all_scores, all_valid) if v]

    if len(set(y_true_valid)) < 2:
        return 0.5  # Can't compute AUC

    return roc_auc_score(y_true_valid, scores_valid)


def train_l1_adaptation(model, support_df, device, l1_id, n_epochs=50, lr=1e-3):
    """Train only the L1 embedding for the new target."""
    model.train()

    # Freeze everything except L1 embedding
    for param in model.parameters():
        param.requires_grad = False

    # Get the L1 embedding parameter and enable gradients
    l1_weight = model.context_module.program_embeddings.embeddings.weight
    l1_weight.requires_grad = True

    optimizer = optim.Adam([l1_weight], lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    smiles_list = support_df['smiles'].tolist()
    y_true = support_df['is_active'].values.astype(np.float32)

    # Convert to batch
    batch_data, valid_idx = smiles_to_batch(smiles_list, device)
    if batch_data is None:
        return model

    y_true_valid = torch.tensor([y_true[i] for i in valid_idx], dtype=torch.float32, device=device)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        n_mols = batch_data['n_mols']
        program_ids = torch.full((n_mols,), l1_id, dtype=torch.long, device=device)
        assay_ids = torch.zeros(n_mols, dtype=torch.long, device=device)
        round_ids = torch.zeros(n_mols, dtype=torch.long, device=device)

        with autocast(enabled=True):
            predictions = model(
                batch_data['node_features'],
                batch_data['edge_index'],
                batch_data['edge_features'],
                batch_data['batch'],
                program_ids, assay_ids, round_ids
            )

            pred = predictions.get('pActivity', list(predictions.values())[0]).squeeze()
            loss = criterion(pred, y_true_valid)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Unfreeze for next use
    for param in model.parameters():
        param.requires_grad = True

    return model


def train_full_finetune(model, support_df, device, l1_id, n_epochs=50, lr=1e-4):
    """Full fine-tuning baseline."""
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    smiles_list = support_df['smiles'].tolist()
    y_true = support_df['is_active'].values.astype(np.float32)

    batch_data, valid_idx = smiles_to_batch(smiles_list, device)
    if batch_data is None:
        return model

    y_true_valid = torch.tensor([y_true[i] for i in valid_idx], dtype=torch.float32, device=device)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        n_mols = batch_data['n_mols']
        program_ids = torch.full((n_mols,), l1_id, dtype=torch.long, device=device)
        assay_ids = torch.zeros(n_mols, dtype=torch.long, device=device)
        round_ids = torch.zeros(n_mols, dtype=torch.long, device=device)

        with autocast(enabled=True):
            predictions = model(
                batch_data['node_features'],
                batch_data['edge_index'],
                batch_data['edge_features'],
                batch_data['batch'],
                program_ids, assay_ids, round_ids
            )

            pred = predictions.get('pActivity', list(predictions.values())[0]).squeeze()
            loss = criterion(pred, y_true_valid)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return model


def run_few_shot_experiment(
    model_orig,
    target_name: str,
    target_df,
    device: torch.device,
    support_sizes: List[int] = [10, 25, 50, 100],
    n_seeds: int = 3,
    n_epochs_l1: int = 100,
    n_epochs_full: int = 100,
):
    """Run few-shot adaptation experiment for one target."""
    print(f"\n{'='*60}")
    print(f"Target: {target_name.upper()}")
    print(f"{'='*60}")

    results = {
        'target': target_name,
        'n_actives': int(target_df['is_active'].sum()),
        'n_decoys': int((~target_df['is_active'].astype(bool)).sum()),
        'support_sizes': {},
    }

    # Get new L1 ID (use next available)
    n_programs = model_orig.context_module.program_embeddings.num_contexts
    new_l1_id = n_programs  # Will add new embedding

    for support_size in support_sizes:
        if support_size > results['n_actives'] // 2:
            print(f"\n  Skipping N={support_size} (not enough actives)")
            continue

        print(f"\n  Support size: N={support_size}")

        seed_results = {
            'zero_shot': [],
            'l1_adapted': [],
            'full_finetune': [],
        }

        for seed in range(n_seeds):
            # Prepare data split
            support_df, query_df = prepare_data(target_df, support_size, seed=seed)

            query_smiles = query_df['smiles'].tolist()
            query_labels = query_df['is_active'].values

            # 1. Zero-shot (L1=0)
            model_zero = copy.deepcopy(model_orig)
            auc_zero = evaluate_model(model_zero, query_smiles, query_labels, device, l1_id=0)
            seed_results['zero_shot'].append(auc_zero)

            # 2. L1 adaptation
            model_l1 = copy.deepcopy(model_orig)
            # Add new L1 embedding
            model_l1.context_module.add_program(1)
            model_l1 = model_l1.to(device)
            # Train only L1
            model_l1 = train_l1_adaptation(model_l1, support_df, device, new_l1_id, n_epochs=n_epochs_l1)
            auc_l1 = evaluate_model(model_l1, query_smiles, query_labels, device, l1_id=new_l1_id)
            seed_results['l1_adapted'].append(auc_l1)

            # 3. Full fine-tuning (same L1 ID)
            model_full = copy.deepcopy(model_orig)
            model_full.context_module.add_program(1)
            model_full = model_full.to(device)
            model_full = train_full_finetune(model_full, support_df, device, new_l1_id, n_epochs=n_epochs_full)
            auc_full = evaluate_model(model_full, query_smiles, query_labels, device, l1_id=new_l1_id)
            seed_results['full_finetune'].append(auc_full)

            print(f"    Seed {seed}: zero={auc_zero:.3f}, L1={auc_l1:.3f}, full={auc_full:.3f}")

        # Aggregate
        results['support_sizes'][support_size] = {
            'zero_shot': {
                'mean': float(np.mean(seed_results['zero_shot'])),
                'std': float(np.std(seed_results['zero_shot'])),
            },
            'l1_adapted': {
                'mean': float(np.mean(seed_results['l1_adapted'])),
                'std': float(np.std(seed_results['l1_adapted'])),
            },
            'full_finetune': {
                'mean': float(np.mean(seed_results['full_finetune'])),
                'std': float(np.std(seed_results['full_finetune'])),
            },
        }

        # Calculate improvement
        zero_mean = results['support_sizes'][support_size]['zero_shot']['mean']
        l1_mean = results['support_sizes'][support_size]['l1_adapted']['mean']
        full_mean = results['support_sizes'][support_size]['full_finetune']['mean']

        l1_improvement = l1_mean - zero_mean
        full_improvement = full_mean - zero_mean
        efficiency = l1_improvement / full_improvement if full_improvement > 0 else 0

        results['support_sizes'][support_size]['l1_improvement'] = float(l1_improvement)
        results['support_sizes'][support_size]['full_improvement'] = float(full_improvement)
        results['support_sizes'][support_size]['efficiency'] = float(efficiency)

        print(f"    Mean: zero={zero_mean:.3f}, L1={l1_mean:.3f} (+{l1_improvement:.3f}), "
              f"full={full_mean:.3f} (+{full_improvement:.3f}), efficiency={efficiency:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='results/phase1/ablation_dude/L0_L1_seed0_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='results/phase1/few_shot',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--targets', type=str, nargs='+',
                        default=['egfr', 'drd2', 'jak2', 'bace1', 'cyp3a4'],
                        help='Targets to test')
    parser.add_argument('--support-sizes', type=int, nargs='+',
                        default=[10, 25, 50, 100],
                        help='Support set sizes to test')
    parser.add_argument('--n-seeds', type=int, default=3,
                        help='Number of random seeds')
    parser.add_argument('--n-epochs-l1', type=int, default=100,
                        help='Epochs for L1 adaptation')
    parser.add_argument('--n-epochs-full', type=int, default=100,
                        help='Epochs for full fine-tuning')
    args = parser.parse_args()

    # Import pandas here to avoid slow import at top
    global pd
    import pandas as pd

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, config = load_checkpoint(Path(args.checkpoint), device)

    print(f"\nModel config:")
    print(f"  use_l1: {config.get('use_l1', 'N/A')}")
    print(f"  n_programs: {config.get('n_programs', 'N/A')}")

    # Load DUD-E data
    print("\nLoading DUD-E data...")
    dude_data = load_all_dude()

    if not dude_data:
        print("ERROR: No DUD-E data found!")
        return

    print(f"Available targets: {list(dude_data.keys())}")

    # Run experiments
    all_results = {}

    for target in args.targets:
        if target not in dude_data:
            print(f"\nSkipping {target}: not in DUD-E data")
            continue

        results = run_few_shot_experiment(
            model,
            target,
            dude_data[target],
            device,
            support_sizes=args.support_sizes,
            n_seeds=args.n_seeds,
            n_epochs_l1=args.n_epochs_l1,
            n_epochs_full=args.n_epochs_full,
        )
        all_results[target] = results

    # Save results
    results_file = output_dir / 'few_shot_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Summary
    print("\n" + "="*70)
    print("FEW-SHOT ADAPTATION SUMMARY")
    print("="*70)

    print(f"\n{'Target':<10} {'N':<6} {'Zero-Shot':<12} {'L1-Adapted':<12} {'Full-FT':<12} {'Efficiency':<10}")
    print("-"*70)

    for target, results in all_results.items():
        for n, data in results['support_sizes'].items():
            zero = data['zero_shot']['mean']
            l1 = data['l1_adapted']['mean']
            full = data['full_finetune']['mean']
            eff = data['efficiency']
            print(f"{target:<10} {n:<6} {zero:<12.3f} {l1:<12.3f} {full:<12.3f} {eff:<10.1%}")

    # Overall assessment
    print("\n" + "-"*70)
    print("ASSESSMENT")
    print("-"*70)

    efficiencies = []
    for target, results in all_results.items():
        for n, data in results['support_sizes'].items():
            efficiencies.append(data['efficiency'])

    mean_efficiency = np.mean(efficiencies)
    print(f"\nMean efficiency (L1 vs Full): {mean_efficiency:.1%}")

    if mean_efficiency >= 0.8:
        print("✓ SUCCESS: L1 adaptation achieves ≥80% of full fine-tuning")
    else:
        print(f"✗ Below target: L1 adaptation at {mean_efficiency:.1%} of full fine-tuning")


if __name__ == '__main__':
    main()
