#!/usr/bin/env python3
"""
Context Embedding Analysis for NEST-DRUG

Goal: Understand what context embeddings represent.

Key questions:
- Do L1 embeddings cluster by target class?
- Are similar targets close in embedding space?
- How do L2/L3 embeddings relate to L1?

Usage:
    python experiments/phase1_architecture/context_embeddings.py \
        --checkpoint results/phase1/ablation_dude/L0_L1_L2_seed0_best.pt
"""

import argparse
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug


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
    model.eval()

    return model, config


def extract_embeddings(model) -> Dict[str, np.ndarray]:
    """Extract all context embeddings from model."""
    context_module = model.context_module

    embeddings = {
        'L1_program': context_module.program_embeddings.embeddings.weight.detach().cpu().numpy(),
        'L2_assay': context_module.assay_embeddings.embeddings.weight.detach().cpu().numpy(),
        'L3_round': context_module.round_embeddings.embeddings.weight.detach().cpu().numpy(),
    }

    return embeddings


def analyze_embedding_statistics(embeddings: Dict[str, np.ndarray]) -> Dict:
    """Compute basic statistics for embeddings."""
    results = {}

    for name, emb in embeddings.items():
        n_contexts, dim = emb.shape

        # Basic stats
        mean_norm = np.linalg.norm(emb, axis=1).mean()
        std_norm = np.linalg.norm(emb, axis=1).std()

        # Pairwise cosine similarities
        if n_contexts > 1:
            cos_sims = []
            for i in range(n_contexts):
                for j in range(i+1, n_contexts):
                    sim = 1 - cosine(emb[i], emb[j])
                    cos_sims.append(sim)
            cos_sims = np.array(cos_sims)
            mean_cos_sim = cos_sims.mean()
            std_cos_sim = cos_sims.std()
            min_cos_sim = cos_sims.min()
            max_cos_sim = cos_sims.max()
        else:
            mean_cos_sim = std_cos_sim = min_cos_sim = max_cos_sim = 0.0

        # Euclidean distances
        if n_contexts > 1:
            distances = pdist(emb, metric='euclidean')
            mean_dist = distances.mean()
            std_dist = distances.std()
        else:
            mean_dist = std_dist = 0.0

        results[name] = {
            'n_contexts': n_contexts,
            'embedding_dim': dim,
            'mean_norm': float(mean_norm),
            'std_norm': float(std_norm),
            'mean_cosine_sim': float(mean_cos_sim),
            'std_cosine_sim': float(std_cos_sim),
            'min_cosine_sim': float(min_cos_sim),
            'max_cosine_sim': float(max_cos_sim),
            'mean_euclidean_dist': float(mean_dist),
            'std_euclidean_dist': float(std_dist),
        }

    return results


def analyze_l1_structure(embeddings: np.ndarray) -> Dict:
    """Analyze structure of L1 (program) embeddings."""
    n_programs, dim = embeddings.shape

    # Compute pairwise distances
    dist_matrix = squareform(pdist(embeddings, metric='euclidean'))
    cos_sim_matrix = np.zeros((n_programs, n_programs))

    for i in range(n_programs):
        for j in range(n_programs):
            if i == j:
                cos_sim_matrix[i, j] = 1.0
            else:
                cos_sim_matrix[i, j] = 1 - cosine(embeddings[i], embeddings[j])

    # Hierarchical clustering
    linkage_matrix = linkage(embeddings, method='ward')

    # Find natural clusters (using distance threshold)
    # Try different numbers of clusters
    cluster_results = {}
    for n_clusters in [2, 3, 4, 5]:
        if n_clusters <= n_programs:
            clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            cluster_results[f'{n_clusters}_clusters'] = clusters.tolist()

    # PCA for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(3, dim, n_programs))
    pca_coords = pca.fit_transform(embeddings)
    explained_var = pca.explained_variance_ratio_

    return {
        'distance_matrix': dist_matrix.tolist(),
        'cosine_similarity_matrix': cos_sim_matrix.tolist(),
        'clustering': cluster_results,
        'pca_coords': pca_coords.tolist(),
        'pca_explained_variance': explained_var.tolist(),
    }


def analyze_cross_level_relationships(embeddings: Dict[str, np.ndarray]) -> Dict:
    """Analyze relationships between different context levels."""
    results = {}

    # Compare L1 and L2 embedding spaces
    l1 = embeddings['L1_program']
    l2 = embeddings['L2_assay']
    l3 = embeddings['L3_round']

    # Check if L2/L3 embeddings are more uniform (less specialized)
    l1_var = np.var(l1, axis=0).mean()
    l2_var = np.var(l2, axis=0).mean()
    l3_var = np.var(l3, axis=0).mean()

    results['variance_per_level'] = {
        'L1_program': float(l1_var),
        'L2_assay': float(l2_var),
        'L3_round': float(l3_var),
    }

    # Effective dimensionality (via PCA)
    from sklearn.decomposition import PCA

    for name, emb in embeddings.items():
        if emb.shape[0] > 1:
            pca = PCA()
            pca.fit(emb)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            # Dimensions needed for 90% variance
            dim_90 = np.argmax(cumvar >= 0.90) + 1
            dim_95 = np.argmax(cumvar >= 0.95) + 1
            results[f'{name}_effective_dim_90'] = int(dim_90)
            results[f'{name}_effective_dim_95'] = int(dim_95)

    return results


def plot_embeddings(embeddings: Dict[str, np.ndarray], output_dir: Path):
    """Create visualizations of embeddings."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: L1 program embeddings - PCA
    l1 = embeddings['L1_program']
    n_programs = l1.shape[0]

    if n_programs > 1:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        l1_2d = pca.fit_transform(l1)

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(l1_2d[:, 0], l1_2d[:, 1], s=200, c=range(n_programs), cmap='tab10')

        for i in range(n_programs):
            ax.annotate(f'P{i}', (l1_2d[i, 0], l1_2d[i, 1]), fontsize=12, ha='center', va='center')

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        ax.set_title('L1 (Program) Embeddings - PCA')
        plt.colorbar(scatter, label='Program ID')
        plt.tight_layout()
        plt.savefig(output_dir / 'l1_pca.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir / 'l1_pca.png'}")

    # Plot 2: L1 cosine similarity heatmap
    if n_programs > 1:
        cos_sim = np.zeros((n_programs, n_programs))
        for i in range(n_programs):
            for j in range(n_programs):
                if i == j:
                    cos_sim[i, j] = 1.0
                else:
                    cos_sim[i, j] = 1 - cosine(l1[i], l1[j])

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cos_sim, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_xticks(range(n_programs))
        ax.set_yticks(range(n_programs))
        ax.set_xticklabels([f'P{i}' for i in range(n_programs)])
        ax.set_yticklabels([f'P{i}' for i in range(n_programs)])
        ax.set_title('L1 (Program) Cosine Similarity')
        plt.colorbar(im)

        # Add text annotations
        for i in range(n_programs):
            for j in range(n_programs):
                ax.text(j, i, f'{cos_sim[i,j]:.2f}', ha='center', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / 'l1_cosine_similarity.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir / 'l1_cosine_similarity.png'}")

    # Plot 3: Hierarchical clustering dendrogram
    if n_programs > 2:
        linkage_matrix = linkage(l1, method='ward')

        fig, ax = plt.subplots(figsize=(12, 6))
        dendrogram(linkage_matrix, labels=[f'P{i}' for i in range(n_programs)], ax=ax)
        ax.set_title('L1 (Program) Hierarchical Clustering')
        ax.set_ylabel('Distance')
        plt.tight_layout()
        plt.savefig(output_dir / 'l1_dendrogram.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir / 'l1_dendrogram.png'}")

    # Plot 4: Embedding norms per level
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (name, emb) in enumerate(embeddings.items()):
        norms = np.linalg.norm(emb, axis=1)
        axes[idx].hist(norms, bins=20, alpha=0.7, edgecolor='black')
        axes[idx].axvline(norms.mean(), color='r', linestyle='--', label=f'Mean: {norms.mean():.3f}')
        axes[idx].set_xlabel('L2 Norm')
        axes[idx].set_ylabel('Count')
        axes[idx].set_title(f'{name} Embedding Norms')
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'embedding_norms.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'embedding_norms.png'}")

    # Plot 5: L2 and L3 sample visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (name, ax) in enumerate([('L2_assay', axes[0]), ('L3_round', axes[1])]):
        emb = embeddings[name]
        if emb.shape[0] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            emb_2d = pca.fit_transform(emb)

            scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], s=50, alpha=0.6, c=range(len(emb)), cmap='viridis')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
            ax.set_title(f'{name} Embeddings (n={len(emb)})')
            plt.colorbar(scatter, ax=ax, label='Context ID')

    plt.tight_layout()
    plt.savefig(output_dir / 'l2_l3_pca.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'l2_l3_pca.png'}")


def analyze_single_checkpoint(checkpoint_path: Path, device: torch.device, output_dir: Path, no_plots: bool = False):
    """Analyze a single checkpoint and return results."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {checkpoint_path.name}")
    print(f"{'='*60}")

    model, config = load_checkpoint(checkpoint_path, device)

    print(f"\nModel config:")
    print(f"  use_l1: {config.get('use_l1', 'N/A')}")
    print(f"  use_l2: {config.get('use_l2', 'N/A')}")
    print(f"  use_l3: {config.get('use_l3', 'N/A')}")

    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings = extract_embeddings(model)

    for name, emb in embeddings.items():
        print(f"  {name}: shape {emb.shape}")

    # Analyze statistics
    print("\nEmbedding Statistics:")
    stats = analyze_embedding_statistics(embeddings)

    for name, s in stats.items():
        print(f"  {name}: cos_sim={s['mean_cosine_sim']:.3f}, norm={s['mean_norm']:.3f}")

    # Analyze L1 structure
    l1_analysis = analyze_l1_structure(embeddings['L1_program'])

    # Cross-level analysis
    cross_level = analyze_cross_level_relationships(embeddings)

    print(f"\nVariance: L1={cross_level['variance_per_level']['L1_program']:.4f}, "
          f"L2={cross_level['variance_per_level']['L2_assay']:.4f}, "
          f"L3={cross_level['variance_per_level']['L3_round']:.4f}")

    # Create plots for this checkpoint
    if not no_plots:
        try:
            checkpoint_name = checkpoint_path.stem.replace('_seed0_best', '')
            plot_dir = output_dir / checkpoint_name
            plot_embeddings(embeddings, plot_dir)
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")

    return {
        'config': config,
        'statistics': stats,
        'l1_analysis': {
            'pca_explained_variance': l1_analysis['pca_explained_variance'],
            'clustering': l1_analysis['clustering'],
        },
        'cross_level': cross_level,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str, default='results/phase1/ablation_dude',
                        help='Directory containing checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Single checkpoint path (overrides --checkpoint-dir)')
    parser.add_argument('--output', type=str, default='results/phase1/context_embeddings',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        # Single checkpoint mode
        checkpoint_path = Path(args.checkpoint)
        results = analyze_single_checkpoint(checkpoint_path, device, output_dir, args.no_plots)
        all_results = {checkpoint_path.stem: results}
    else:
        # Multi-checkpoint mode
        checkpoint_dir = Path(args.checkpoint_dir)
        conditions = ['L0', 'L0_L1', 'L0_L1_L2', 'L0_L1_L2_L3']
        all_results = {}

        for cond in conditions:
            checkpoint_path = checkpoint_dir / f"{cond}_seed0_best.pt"
            if checkpoint_path.exists():
                results = analyze_single_checkpoint(checkpoint_path, device, output_dir, args.no_plots)
                all_results[cond] = results
            else:
                print(f"\nSkipping {cond}: checkpoint not found")

    # Save all results
    results_file = output_dir / 'context_embedding_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Comparative summary
    print("\n" + "="*70)
    print("COMPARATIVE SUMMARY")
    print("="*70)

    print(f"\n{'Condition':<15} {'L1 cos_sim':<12} {'L1 var':<12} {'L2 var':<12} {'L3 var':<12}")
    print("-"*70)

    for cond, results in all_results.items():
        l1_cos = results['statistics']['L1_program']['mean_cosine_sim']
        l1_var = results['cross_level']['variance_per_level']['L1_program']
        l2_var = results['cross_level']['variance_per_level']['L2_assay']
        l3_var = results['cross_level']['variance_per_level']['L3_round']
        print(f"{cond:<15} {l1_cos:<12.4f} {l1_var:<12.6f} {l2_var:<12.6f} {l3_var:<12.6f}")

    # Key insights
    print("\n" + "-"*70)
    print("KEY INSIGHTS")
    print("-"*70)

    # Check differentiation across conditions
    if 'L0' in all_results and 'L0_L1' in all_results:
        l0_cos = all_results['L0']['statistics']['L1_program']['mean_cosine_sim']
        l1_cos = all_results['L0_L1']['statistics']['L1_program']['mean_cosine_sim']

        print(f"\nL1 embedding differentiation:")
        print(f"  L0 (no context): cos_sim = {l0_cos:.4f}")
        print(f"  L0+L1 (with L1): cos_sim = {l1_cos:.4f}")

        if l1_cos < l0_cos:
            print(f"  ✓ L1 context makes programs MORE differentiated (Δ={l0_cos-l1_cos:.4f})")
        else:
            print(f"  ⚠ L1 context makes programs LESS differentiated")

    # Check variance hierarchy
    for cond, results in all_results.items():
        l1_var = results['cross_level']['variance_per_level']['L1_program']
        l2_var = results['cross_level']['variance_per_level']['L2_assay']
        l3_var = results['cross_level']['variance_per_level']['L3_round']

        if l1_var > l2_var > l3_var:
            hierarchy = "L1 > L2 > L3 ✓"
        else:
            hierarchy = f"L1={l1_var:.4f}, L2={l2_var:.4f}, L3={l3_var:.4f}"
        print(f"\n{cond} variance hierarchy: {hierarchy}")


if __name__ == '__main__':
    main()
