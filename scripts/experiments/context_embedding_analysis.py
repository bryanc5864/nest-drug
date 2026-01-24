#!/usr/bin/env python3
"""
Experiment 1C: Context Embedding Visualization

t-SNE/UMAP of learned L1 embeddings colored by target class.
Shows if model learned meaningful target representations.

Usage:
    python scripts/experiments/context_embedding_analysis.py \
        --checkpoint checkpoints/pretrain/best_model.pt \
        --output results/experiments/context_embeddings \
        --gpu 0
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist, squareform
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


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

    program_mapping = checkpoint.get('program_mapping', {})

    return model, config, program_mapping


def extract_embeddings(model):
    """Extract all context embeddings from model."""
    embeddings = {}

    if hasattr(model, 'context_module'):
        ctx = model.context_module

        if hasattr(ctx, 'program_embeddings'):
            embeddings['L1_program'] = ctx.program_embeddings.embeddings.weight.detach().cpu().numpy()

        if hasattr(ctx, 'assay_embeddings'):
            embeddings['L2_assay'] = ctx.assay_embeddings.embeddings.weight.detach().cpu().numpy()

        if hasattr(ctx, 'round_embeddings'):
            embeddings['L3_round'] = ctx.round_embeddings.embeddings.weight.detach().cpu().numpy()

    return embeddings


def compute_embedding_statistics(embeddings):
    """Compute statistics for embeddings."""
    stats = {}

    for name, emb in embeddings.items():
        norms = np.linalg.norm(emb, axis=1)

        # Pairwise cosine similarities
        emb_norm = emb / (norms[:, np.newaxis] + 1e-10)
        cosine_sim = np.dot(emb_norm, emb_norm.T)
        np.fill_diagonal(cosine_sim, 0)  # Exclude self-similarity

        # Pairwise distances
        distances = squareform(pdist(emb, metric='euclidean'))

        stats[name] = {
            'n_embeddings': emb.shape[0],
            'embedding_dim': emb.shape[1],
            'mean_norm': float(np.mean(norms)),
            'std_norm': float(np.std(norms)),
            'min_norm': float(np.min(norms)),
            'max_norm': float(np.max(norms)),
            'mean_cosine_sim': float(np.mean(cosine_sim[np.triu_indices(len(cosine_sim), k=1)])),
            'mean_distance': float(np.mean(distances[np.triu_indices(len(distances), k=1)])),
            'variance_explained_by_pc1': 0.0,  # Will be computed below
        }

        # PCA variance explained
        if emb.shape[0] > 2:
            pca = PCA(n_components=min(5, emb.shape[0], emb.shape[1]))
            pca.fit(emb)
            stats[name]['variance_explained_by_pc1'] = float(pca.explained_variance_ratio_[0])
            stats[name]['variance_explained_top5'] = float(sum(pca.explained_variance_ratio_))

    return stats


def visualize_embeddings(embeddings, output_dir, program_mapping=None):
    """Create visualizations of embeddings."""
    output_dir = Path(output_dir)

    for name, emb in embeddings.items():
        n_emb = emb.shape[0]

        if n_emb < 3:
            print(f"Skipping {name}: too few embeddings ({n_emb})")
            continue

        # PCA
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        pca = PCA(n_components=2)
        emb_pca = pca.fit_transform(emb)

        axes[0].scatter(emb_pca[:, 0], emb_pca[:, 1], c=range(n_emb), cmap='viridis', s=100)
        for i in range(min(n_emb, 20)):  # Label first 20
            axes[0].annotate(str(i), (emb_pca[i, 0], emb_pca[i, 1]), fontsize=8)
        axes[0].set_title(f'{name} - PCA')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

        # t-SNE (if enough samples)
        if n_emb >= 5:
            perplexity = min(30, n_emb - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            emb_tsne = tsne.fit_transform(emb)

            axes[1].scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=range(n_emb), cmap='viridis', s=100)
            for i in range(min(n_emb, 20)):
                axes[1].annotate(str(i), (emb_tsne[i, 0], emb_tsne[i, 1]), fontsize=8)
            axes[1].set_title(f'{name} - t-SNE')

        # Cosine similarity heatmap
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb_norm = emb / (norms + 1e-10)
        cosine_sim = np.dot(emb_norm, emb_norm.T)

        im = axes[2].imshow(cosine_sim, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[2].set_title(f'{name} - Cosine Similarity')
        plt.colorbar(im, ax=axes[2])

        plt.tight_layout()
        plt.savefig(output_dir / f'{name}_visualization.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir / f'{name}_visualization.png'}")

        # Dendrogram for hierarchical clustering
        if n_emb >= 3 and n_emb <= 100:
            fig, ax = plt.subplots(figsize=(12, 6))
            linkage_matrix = linkage(emb, method='ward')
            dendrogram(linkage_matrix, ax=ax, labels=[str(i) for i in range(n_emb)])
            ax.set_title(f'{name} - Hierarchical Clustering')
            plt.tight_layout()
            plt.savefig(output_dir / f'{name}_dendrogram.png', dpi=150)
            plt.close()
            print(f"Saved: {output_dir / f'{name}_dendrogram.png'}")


def main():
    parser = argparse.ArgumentParser(description="Context Embedding Analysis")
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--output', type=str, default='results/experiments/context_embeddings',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    args = parser.parse_args()

    if not HAS_SKLEARN:
        print("ERROR: scikit-learn required. Install with: pip install scikit-learn")
        sys.exit(1)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLoading model: {args.checkpoint}")
    model, config, program_mapping = load_model(args.checkpoint, device)
    print(f"Config: {config}")
    print(f"Program mapping: {len(program_mapping)} entries")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings = extract_embeddings(model)

    for name, emb in embeddings.items():
        print(f"  {name}: {emb.shape}")

    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_embedding_statistics(embeddings)

    # Print summary
    print(f"\n{'='*60}")
    print("EMBEDDING ANALYSIS RESULTS")
    print('='*60)

    for name, s in stats.items():
        print(f"\n{name}:")
        print(f"  Shape: ({s['n_embeddings']}, {s['embedding_dim']})")
        print(f"  Norm: {s['mean_norm']:.4f} ± {s['std_norm']:.4f}")
        print(f"  Mean cosine similarity: {s['mean_cosine_sim']:.4f}")
        print(f"  PC1 variance explained: {s['variance_explained_by_pc1']:.1%}")

    # Create visualizations
    print("\nCreating visualizations...")
    visualize_embeddings(embeddings, output_dir, program_mapping)

    # Save results
    results_path = output_dir / 'embedding_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'config': config,
            'program_mapping': program_mapping,
            'statistics': stats,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
