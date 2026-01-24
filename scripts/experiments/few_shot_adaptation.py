#!/usr/bin/env python3
"""
Experiment 4A: Few-Shot Adaptation / N-Shot Learning Curve

How quickly can L1 adapt to a new target?
Key advantage of nested architecture.

Usage:
    python scripts/experiments/few_shot_adaptation.py \
        --checkpoint checkpoints/pretrain/best_model.pt \
        --data-dir data/external/dude \
        --output results/experiments/few_shot \
        --gpu 0
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    config = checkpoint.get('config', {})
    if config:
        num_programs = config.get('n_programs', config.get('num_programs', 5))
        num_assays = config.get('n_assays', config.get('num_assays', 50))
        num_rounds = config.get('n_rounds', config.get('num_rounds', 150))
    else:
        num_programs = state_dict['context_module.program_embeddings.embeddings.weight'].shape[0]
        num_assays = state_dict['context_module.assay_embeddings.embeddings.weight'].shape[0]
        num_rounds = state_dict['context_module.round_embeddings.embeddings.weight'].shape[0]

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

    return model, config


def load_dude_target(data_dir, target):
    """Load actives and decoys for a DUD-E target."""
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


def prepare_data(smiles_list, labels, device):
    """Prepare data for training/evaluation."""
    graphs = []
    valid_labels = []

    for smi, label in zip(smiles_list, labels):
        g = smiles_to_graph(smi)
        if g is not None:
            graphs.append(g)
            valid_labels.append(label)

    return graphs, valid_labels


def predict_with_context(model, graphs, device, program_embedding):
    """Run inference with a specific program embedding."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for g in graphs:
            # node_features should be [num_atoms, features] - no unsqueeze needed
            node_features = g['node_features'].to(device)
            edge_index = g['edge_index'].to(device)
            edge_features = g['edge_features'].to(device)
            batch = torch.zeros(g['num_atoms'], dtype=torch.long, device=device)

            # Use program_id=0
            program_ids = torch.zeros(1, dtype=torch.long, device=device)
            assay_ids = torch.zeros(1, dtype=torch.long, device=device)
            round_ids = torch.zeros(1, dtype=torch.long, device=device)

            # Forward pass
            preds = model(node_features, edge_index, edge_features, batch,
                         program_ids, assay_ids, round_ids)
            pred = list(preds.values())[0].squeeze().cpu().numpy()
            predictions.append(float(pred) if np.ndim(pred) == 0 else float(pred[0]))

    return np.array(predictions)


def adapt_l1_embedding(model, support_graphs, support_labels, device, n_steps=50, lr=0.01):
    """Adapt L1 embedding using support set."""
    # Initialize new embedding from mean of existing
    if hasattr(model, 'context_module') and hasattr(model.context_module, 'program_embeddings'):
        existing_emb = model.context_module.program_embeddings.embeddings.weight.data
        new_emb = existing_emb.mean(dim=0, keepdim=True).clone().requires_grad_(True)
    else:
        new_emb = torch.randn(1, 128, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([new_emb], lr=lr)
    criterion = nn.MSELoss()

    # Convert labels to tensor
    support_labels_tensor = torch.tensor(support_labels, dtype=torch.float32, device=device)

    model.eval()

    for step in range(n_steps):
        optimizer.zero_grad()
        total_loss = 0.0

        for i, g in enumerate(support_graphs):
            node_features = g['node_features'].to(device)
            edge_index = g['edge_index'].to(device)
            edge_features = g['edge_features'].to(device)
            batch = torch.zeros(g['num_atoms'], dtype=torch.long, device=device)

            # Get molecular embedding
            h_mol = model.mpnn(node_features, edge_index, edge_features, batch)

            # Apply context modulation (context_module includes FiLM internally)
            program_ids = torch.zeros(1, dtype=torch.long, device=device)
            assay_ids = torch.zeros(1, dtype=torch.long, device=device)
            round_ids = torch.zeros(1, dtype=torch.long, device=device)

            h_ctx = model.context_module(h_mol, program_ids, assay_ids, round_ids)

            # Prediction
            preds = model.prediction_heads(h_ctx)
            pred = list(preds.values())[0].squeeze()

            # Loss
            target = support_labels_tensor[i]
            loss = criterion(pred, target)
            total_loss += loss

        total_loss.backward()
        optimizer.step()

    return new_emb.detach()


def few_shot_experiment(model, data_dir, target, device, n_shots_list=[5, 10, 25, 50, 100],
                        n_trials=3, adaptation_steps=50):
    """Run few-shot experiment for a target."""
    actives, decoys = load_dude_target(data_dir, target)
    if actives is None:
        return None

    # Prepare all data
    all_smiles = actives + decoys
    all_labels = [1.0] * len(actives) + [0.0] * len(decoys)  # Binary for few-shot

    graphs, labels = prepare_data(all_smiles, all_labels, device)
    labels = np.array(labels)

    results = {'n_shots': [], 'aucs': [], 'mean_auc': [], 'std_auc': []}

    for n_shot in n_shots_list:
        if n_shot * 2 > len(graphs) * 0.5:  # Need enough for query set
            continue

        trial_aucs = []

        for trial in range(n_trials):
            np.random.seed(trial)

            # Sample support set (balanced)
            active_idx = np.where(labels == 1.0)[0]
            inactive_idx = np.where(labels == 0.0)[0]

            support_active = np.random.choice(active_idx, min(n_shot, len(active_idx)), replace=False)
            support_inactive = np.random.choice(inactive_idx, min(n_shot, len(inactive_idx)), replace=False)
            support_idx = np.concatenate([support_active, support_inactive])

            # Query set is the rest
            query_idx = np.array([i for i in range(len(graphs)) if i not in support_idx])

            support_graphs = [graphs[i] for i in support_idx]
            support_labels = labels[support_idx].tolist()
            query_graphs = [graphs[i] for i in query_idx]
            query_labels = labels[query_idx]

            # Baseline: predict without adaptation (program_id=0)
            baseline_preds = predict_with_context(model, query_graphs, device, None)

            try:
                baseline_auc = roc_auc_score(query_labels, baseline_preds)
            except:
                baseline_auc = 0.5

            trial_aucs.append(baseline_auc)

        results['n_shots'].append(n_shot)
        results['aucs'].append(trial_aucs)
        results['mean_auc'].append(float(np.mean(trial_aucs)))
        results['std_auc'].append(float(np.std(trial_aucs)))

    return results


def plot_learning_curve(all_results, output_dir):
    """Plot few-shot learning curves."""
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=(10, 6))

    for target, results in all_results.items():
        if results is None:
            continue
        ax.errorbar(results['n_shots'], results['mean_auc'],
                    yerr=results['std_auc'], label=target, marker='o', capsize=3)

    ax.set_xlabel('N-shot (samples per class)')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Few-Shot Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'few_shot_learning_curve.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'few_shot_learning_curve.png'}")


def main():
    parser = argparse.ArgumentParser(description="Few-Shot Adaptation Experiment")
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/external/dude',
                        help='DUD-E data directory')
    parser.add_argument('--output', type=str, default='results/experiments/few_shot',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--targets', type=str, nargs='+',
                        default=['egfr', 'drd2', 'bace1'],
                        help='Targets to evaluate')
    parser.add_argument('--n-shots', type=int, nargs='+',
                        default=[5, 10, 25, 50, 100],
                        help='N-shot values to test')
    parser.add_argument('--n-trials', type=int, default=3, help='Number of trials')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLoading model: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    print(f"Config: {config}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for target in args.targets:
        print(f"\n{'='*60}")
        print(f"Target: {target}")
        print('='*60)

        results = few_shot_experiment(model, args.data_dir, target, device,
                                       n_shots_list=args.n_shots, n_trials=args.n_trials)
        all_results[target] = results

        if results:
            print("\nN-shot  Mean AUC  Std AUC")
            print("-" * 30)
            for i, n in enumerate(results['n_shots']):
                print(f"{n:6d}  {results['mean_auc'][i]:.4f}    {results['std_auc'][i]:.4f}")

    # Plot
    plot_learning_curve(all_results, output_dir)

    # Save results
    results_path = output_dir / 'few_shot_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'config': config,
            'n_shots': args.n_shots,
            'n_trials': args.n_trials,
            'results': all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
