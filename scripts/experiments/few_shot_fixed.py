#!/usr/bin/env python3
"""
Few-Shot Adaptation (FIXED): Actually adapts L1 embeddings.

Compares:
- Zero-shot: Use generic program_id=0
- L1-adapted: Learn a new L1 embedding from support set
- Correct L1: Use the correct program ID from training (if available)

Usage:
    python scripts/experiments/few_shot_fixed.py --gpu 0
"""

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


# Correct program IDs from V3's training
DUDE_TO_V3_PROGRAM_ID = {
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


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    config = checkpoint.get('config', {})

    num_programs = config.get('n_programs', config.get('num_programs', 5123))
    num_assays = config.get('n_assays', config.get('num_assays', 100))
    num_rounds = config.get('n_rounds', config.get('num_rounds', 20))

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


def prepare_graphs(smiles_list):
    """Convert SMILES to graphs."""
    graphs = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        g = smiles_to_graph(smi)
        if g is not None:
            graphs.append(g)
            valid_indices.append(i)
    return graphs, valid_indices


def predict_with_program_id(model, graphs, device, program_id=0):
    """Run inference with a specific program ID."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for g in graphs:
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

    return np.array(predictions)


def predict_with_custom_embedding(model, graphs, device, custom_embedding):
    """Run inference with a custom L1 embedding (bypassing the lookup).

    Replicates the full NestedContextModule forward pass:
    1. custom L1 embedding (128d) + L2=0 (64d) + L3=0 (32d) → concat (224d)
    2. context_interaction MLP (224d → 224d)
    3. FiLM: gamma_net/beta_net → h_mod = gamma * h_mol + beta
    4. prediction_heads(h_mod)
    """
    model.eval()
    predictions = []

    ctx = model.context_module

    with torch.no_grad():
        # Get default L2 and L3 embeddings (id=0)
        assay_ids = torch.zeros(1, dtype=torch.long, device=device)
        round_ids = torch.zeros(1, dtype=torch.long, device=device)
        z_assay = ctx.assay_embeddings(assay_ids)   # [1, 64]
        z_round = ctx.round_embeddings(round_ids)    # [1, 32]

        for g in graphs:
            node_features = g['node_features'].to(device)
            edge_index = g['edge_index'].to(device)
            edge_features = g['edge_features'].to(device)
            batch = torch.zeros(g['num_atoms'], dtype=torch.long, device=device)

            # Get molecular embedding from MPNN
            h_mol = model.mpnn(node_features, edge_index, edge_features, batch)

            # Build full context vector: L1 (custom) + L2 (zero) + L3 (zero)
            context = torch.cat([custom_embedding, z_assay, z_round], dim=-1)  # [1, 224]

            # Apply context_interaction MLP
            context = ctx.context_interaction(context)  # [1, 224]

            # Apply FiLM modulation
            h_mod = ctx.film(h_mol, context)  # [1, 512]

            # Prediction heads
            preds = model.prediction_heads(h_mod)
            pred = list(preds.values())[0].squeeze().cpu().numpy()
            predictions.append(float(pred) if np.ndim(pred) == 0 else float(pred[0]))

    return np.array(predictions)


def adapt_l1_embedding(model, support_graphs, support_labels, device, n_steps=100, lr=0.01):
    """
    Adapt a new L1 embedding using the support set.

    Learns a new 128-dim L1 embedding that minimizes BCE loss on the support set,
    using the full context pipeline: L1 + L2(0) + L3(0) → context_interaction → FiLM.
    """
    ctx = model.context_module

    # Initialize from mean of existing embeddings
    with torch.no_grad():
        existing_emb = ctx.program_embeddings.embeddings.weight.data
        new_emb = existing_emb.mean(dim=0, keepdim=True).clone().to(device)

    new_emb = new_emb.requires_grad_(True)
    optimizer = torch.optim.Adam([new_emb], lr=lr)

    # Convert labels to tensor
    labels_tensor = torch.tensor(support_labels, dtype=torch.float32, device=device)

    model.eval()  # Keep model frozen

    # Get default L2 and L3 embeddings (id=0) - these are constant
    with torch.no_grad():
        assay_ids = torch.zeros(1, dtype=torch.long, device=device)
        round_ids = torch.zeros(1, dtype=torch.long, device=device)
        z_assay = ctx.assay_embeddings(assay_ids)   # [1, 64]
        z_round = ctx.round_embeddings(round_ids)    # [1, 32]

    for step in range(n_steps):
        optimizer.zero_grad()
        total_loss = 0.0

        for i, g in enumerate(support_graphs):
            node_features = g['node_features'].to(device)
            edge_index = g['edge_index'].to(device)
            edge_features = g['edge_features'].to(device)
            batch = torch.zeros(g['num_atoms'], dtype=torch.long, device=device)

            # Forward through MPNN (frozen)
            with torch.no_grad():
                h_mol = model.mpnn(node_features, edge_index, edge_features, batch)

            # Build full context: custom L1 + default L2 + default L3
            context = torch.cat([new_emb, z_assay, z_round], dim=-1)  # [1, 224]

            # Apply context_interaction (frozen weights, but grad flows through new_emb)
            context = ctx.context_interaction(context)  # [1, 224]

            # Apply FiLM (frozen weights, grad flows through context)
            h_mod = ctx.film(h_mol, context)  # [1, 512]

            # Prediction
            preds = model.prediction_heads(h_mod)
            pred = list(preds.values())[0].squeeze()

            # Binary cross-entropy loss
            target = labels_tensor[i]
            loss = nn.functional.binary_cross_entropy_with_logits(
                pred.unsqueeze(0),
                target.unsqueeze(0)
            )
            total_loss += loss

        total_loss = total_loss / len(support_graphs)
        total_loss.backward()
        optimizer.step()

    return new_emb.detach()


def few_shot_experiment(model, data_dir, target, device, n_shots_list=[10, 25, 50],
                        n_trials=3, adaptation_steps=100):
    """Run few-shot experiment for a target."""
    actives, decoys = load_dude_target(data_dir, target)
    if actives is None:
        return None

    # Limit data size for speed
    max_actives = min(500, len(actives))
    max_decoys = min(500, len(decoys))

    np.random.seed(42)
    actives = list(np.random.choice(actives, max_actives, replace=False))
    decoys = list(np.random.choice(decoys, max_decoys, replace=False))

    all_smiles = actives + decoys
    all_labels = [1.0] * len(actives) + [0.0] * len(decoys)

    graphs, valid_indices = prepare_graphs(all_smiles)
    labels = np.array([all_labels[i] for i in valid_indices])

    # Get correct program ID if available
    correct_pid = DUDE_TO_V3_PROGRAM_ID.get(target, 0)

    results = {
        'n_shots': [],
        'zero_shot_auc': [],
        'correct_l1_auc': [],
        'adapted_auc': [],
        'zero_shot_std': [],
        'adapted_std': [],
    }

    for n_shot in n_shots_list:
        if n_shot * 2 > len(graphs) * 0.3:
            continue

        trial_zero_shot = []
        trial_correct_l1 = []
        trial_adapted = []

        for trial in range(n_trials):
            np.random.seed(trial + 100)

            # Sample balanced support set
            active_idx = np.where(labels == 1.0)[0]
            inactive_idx = np.where(labels == 0.0)[0]

            n_per_class = min(n_shot, len(active_idx), len(inactive_idx))
            support_active = np.random.choice(active_idx, n_per_class, replace=False)
            support_inactive = np.random.choice(inactive_idx, n_per_class, replace=False)
            support_idx = np.concatenate([support_active, support_inactive])

            query_idx = np.array([i for i in range(len(graphs)) if i not in support_idx])

            support_graphs = [graphs[i] for i in support_idx]
            support_labels = labels[support_idx].tolist()
            query_graphs = [graphs[i] for i in query_idx]
            query_labels = labels[query_idx]

            # 1. Zero-shot (program_id=0)
            zero_shot_preds = predict_with_program_id(model, query_graphs, device, program_id=0)
            try:
                zero_shot_auc = roc_auc_score(query_labels, zero_shot_preds)
            except:
                zero_shot_auc = 0.5
            trial_zero_shot.append(zero_shot_auc)

            # 2. Correct L1 (use training program ID)
            correct_preds = predict_with_program_id(model, query_graphs, device, program_id=correct_pid)
            try:
                correct_auc = roc_auc_score(query_labels, correct_preds)
            except:
                correct_auc = 0.5
            trial_correct_l1.append(correct_auc)

            # 3. Adapted L1 (learn new embedding from support)
            adapted_emb = adapt_l1_embedding(
                model, support_graphs, support_labels, device,
                n_steps=adaptation_steps, lr=0.01
            )
            adapted_preds = predict_with_custom_embedding(model, query_graphs, device, adapted_emb)
            try:
                adapted_auc = roc_auc_score(query_labels, adapted_preds)
            except:
                adapted_auc = 0.5
            trial_adapted.append(adapted_auc)

        results['n_shots'].append(n_shot)
        results['zero_shot_auc'].append(float(np.mean(trial_zero_shot)))
        results['correct_l1_auc'].append(float(np.mean(trial_correct_l1)))
        results['adapted_auc'].append(float(np.mean(trial_adapted)))
        results['zero_shot_std'].append(float(np.std(trial_zero_shot)))
        results['adapted_std'].append(float(np.std(trial_adapted)))

    return results


def main():
    parser = argparse.ArgumentParser(description="Few-Shot Adaptation (Fixed)")
    parser.add_argument('--checkpoint', type=str, default='results/v3/best_model.pt',
                        help='Model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/external/dude',
                        help='DUD-E data directory')
    parser.add_argument('--output', type=str, default='results/experiments/few_shot_fixed',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--targets', type=str, nargs='+',
                        default=['egfr', 'drd2', 'bace1'],
                        help='Targets to evaluate')
    parser.add_argument('--n-shots', type=int, nargs='+',
                        default=[10, 25, 50],
                        help='N-shot values')
    parser.add_argument('--n-trials', type=int, default=3, help='Number of trials')
    parser.add_argument('--adapt-steps', type=int, default=100, help='Adaptation steps')
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
        print(f"Target: {target.upper()} (correct L1 ID: {DUDE_TO_V3_PROGRAM_ID.get(target, 0)})")
        print('='*60)

        results = few_shot_experiment(
            model, args.data_dir, target, device,
            n_shots_list=args.n_shots,
            n_trials=args.n_trials,
            adaptation_steps=args.adapt_steps
        )
        all_results[target] = results

        if results:
            print(f"\n{'N-shot':<8} {'Zero-shot':<12} {'Correct L1':<12} {'Adapted':<12} {'Delta':<10}")
            print("-" * 55)
            for i, n in enumerate(results['n_shots']):
                delta = results['adapted_auc'][i] - results['zero_shot_auc'][i]
                print(f"{n:<8} {results['zero_shot_auc'][i]:<12.4f} {results['correct_l1_auc'][i]:<12.4f} {results['adapted_auc'][i]:<12.4f} {delta:+.4f}")

    # Save results
    results_path = output_dir / 'few_shot_fixed_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'config': config,
            'n_trials': args.n_trials,
            'adapt_steps': args.adapt_steps,
            'results': all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
