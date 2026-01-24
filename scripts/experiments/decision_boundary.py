#!/usr/bin/env python3
"""
Experiment 2C: Decision Boundary Analysis

What separates actives from inactives in embedding space?
Shows learned chemical intuition.

Usage:
    python scripts/experiments/decision_boundary.py \
        --checkpoint checkpoints/pretrain/best_model.pt \
        --data-dir data/external/dude \
        --output results/experiments/decision_boundary \
        --gpu 0
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


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
    model.eval()

    return model, config


def get_embeddings(model, smiles_list, device, batch_size=64, program_id=0):
    """Extract molecular embeddings from model."""
    embeddings = []
    valid_idx = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Extracting embeddings"):
            batch_smiles = smiles_list[i:i+batch_size]

            # Process batch
            batch_graphs = []
            batch_idx = []
            for j, smi in enumerate(batch_smiles):
                g = smiles_to_graph(smi)
                if g is not None:
                    batch_graphs.append(g)
                    batch_idx.append(i + j)

            if not batch_graphs:
                continue

            # Collate
            node_features = []
            edge_indices = []
            edge_features = []
            batch_indices = []
            offset = 0

            for k, g in enumerate(batch_graphs):
                node_features.append(g['node_features'])
                edge_indices.append(g['edge_index'] + offset)
                edge_features.append(g['edge_features'])
                batch_indices.extend([k] * g['num_atoms'])
                offset += g['num_atoms']

            node_features = torch.cat(node_features, dim=0).to(device)
            edge_index = torch.cat(edge_indices, dim=1).to(device)
            edge_features = torch.cat(edge_features, dim=0).to(device)
            batch = torch.tensor(batch_indices, dtype=torch.long, device=device)

            n_mols = len(batch_graphs)
            program_ids = torch.full((n_mols,), program_id, dtype=torch.long, device=device)
            assay_ids = torch.zeros(n_mols, dtype=torch.long, device=device)
            round_ids = torch.zeros(n_mols, dtype=torch.long, device=device)

            # Get embeddings (before prediction head)
            # context_module takes h_mol and applies FiLM internally
            h_mol = model.mpnn(node_features, edge_index, edge_features, batch)
            h_contextualized = model.context_module(h_mol, program_ids, assay_ids, round_ids)

            embeddings.append(h_contextualized.cpu().numpy())
            valid_idx.extend(batch_idx)

    return np.vstack(embeddings), valid_idx


def load_dude_target(data_dir, target):
    """Load actives and decoys for a DUD-E target."""
    target_dir = Path(data_dir) / target

    # Load actives
    actives_file = target_dir / "actives_final.smi"
    if not actives_file.exists():
        return None, None

    actives = []
    with open(actives_file) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                actives.append(parts[0])

    # Load decoys
    decoys_file = target_dir / "decoys_final.smi"
    decoys = []
    with open(decoys_file) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                decoys.append(parts[0])

    return actives, decoys


def visualize_embeddings(embeddings, labels, output_path, method='tsne', title=""):
    """Visualize embeddings with dimensionality reduction."""
    if method == 'tsne' and HAS_SKLEARN:
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    elif method == 'pca' and HAS_SKLEARN:
        reducer = PCA(n_components=2)
    elif method == 'umap' and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        print(f"Method {method} not available")
        return

    print(f"Running {method.upper()}...")
    emb_2d = reducer.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {'active': 'red', 'inactive': 'blue'}
    for label in ['inactive', 'active']:
        mask = np.array(labels) == label
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=colors[label], label=label, alpha=0.5, s=20)

    ax.legend()
    ax.set_title(f"{title} - {method.upper()}")
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def compute_separation_metrics(embeddings, labels):
    """Compute metrics for class separation."""
    labels = np.array(labels)
    active_mask = labels == 'active'
    inactive_mask = labels == 'inactive'

    active_emb = embeddings[active_mask]
    inactive_emb = embeddings[inactive_mask]

    # Centroid distance
    active_centroid = active_emb.mean(axis=0)
    inactive_centroid = inactive_emb.mean(axis=0)
    centroid_dist = np.linalg.norm(active_centroid - inactive_centroid)

    # Within-class variance
    active_var = np.mean(np.var(active_emb, axis=0))
    inactive_var = np.mean(np.var(inactive_emb, axis=0))

    # Fisher's discriminant ratio
    between_var = centroid_dist ** 2
    within_var = active_var + inactive_var
    fisher_ratio = between_var / (within_var + 1e-10)

    return {
        'centroid_distance': float(centroid_dist),
        'active_variance': float(active_var),
        'inactive_variance': float(inactive_var),
        'fisher_ratio': float(fisher_ratio),
    }


def main():
    parser = argparse.ArgumentParser(description="Decision Boundary Analysis")
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/external/dude',
                        help='DUD-E data directory')
    parser.add_argument('--output', type=str, default='results/experiments/decision_boundary',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--targets', type=str, nargs='+',
                        default=['egfr', 'drd2', 'bace1'],
                        help='DUD-E targets to analyze')
    parser.add_argument('--max-samples', type=int, default=1000,
                        help='Max samples per class')
    parser.add_argument('--program-id', type=int, default=0, help='Program context')
    parser.add_argument('--method', type=str, default='tsne',
                        choices=['tsne', 'pca', 'umap'], help='Reduction method')
    args = parser.parse_args()

    if not HAS_SKLEARN:
        print("ERROR: scikit-learn required. Install with: pip install scikit-learn")
        sys.exit(1)

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
        print(f"Processing: {target}")
        print('='*60)

        actives, decoys = load_dude_target(args.data_dir, target)
        if actives is None:
            print(f"Skipping {target}: data not found")
            continue

        # Sample
        if len(actives) > args.max_samples:
            np.random.seed(42)
            actives = list(np.random.choice(actives, args.max_samples, replace=False))
        if len(decoys) > args.max_samples:
            np.random.seed(42)
            decoys = list(np.random.choice(decoys, args.max_samples, replace=False))

        print(f"Actives: {len(actives)}, Decoys: {len(decoys)}")

        # Combine
        all_smiles = actives + decoys
        all_labels = ['active'] * len(actives) + ['inactive'] * len(decoys)

        # Get embeddings
        embeddings, valid_idx = get_embeddings(model, all_smiles, device,
                                                program_id=args.program_id)

        # Filter labels to valid
        valid_labels = [all_labels[i] for i in valid_idx]

        print(f"Valid embeddings: {len(valid_labels)}")

        # Compute separation metrics
        metrics = compute_separation_metrics(embeddings, valid_labels)
        print(f"Fisher ratio: {metrics['fisher_ratio']:.4f}")
        print(f"Centroid distance: {metrics['centroid_distance']:.4f}")

        # Visualize
        viz_path = output_dir / f"{target}_{args.method}.png"
        visualize_embeddings(embeddings, valid_labels, viz_path,
                            method=args.method, title=target.upper())

        all_results[target] = {
            'n_actives': sum(1 for l in valid_labels if l == 'active'),
            'n_decoys': sum(1 for l in valid_labels if l == 'inactive'),
            'embedding_dim': embeddings.shape[1],
            'metrics': metrics,
        }

    # Save results
    results_path = output_dir / 'decision_boundary_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'config': config,
            'program_id': args.program_id,
            'method': args.method,
            'results': all_results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    for target, res in all_results.items():
        print(f"{target}: Fisher={res['metrics']['fisher_ratio']:.3f}, "
              f"Centroid dist={res['metrics']['centroid_distance']:.3f}")

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
