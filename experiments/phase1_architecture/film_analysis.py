#!/usr/bin/env python3
"""
FiLM Modulation Analysis for NEST-DRUG

Goal: Verify FiLM is learning meaningful modulations, not staying at identity.

Key questions:
- Are γ values ≠ 1?
- Are β values ≠ 0?
- Do different contexts produce different modulations?

Usage:
    python experiments/phase1_architecture/film_analysis.py \
        --checkpoint results/phase1/ablation_dude/L0_L1_L2_seed0_best.pt
"""

import argparse
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
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

    # Load state dict (may need to unwrap from AblatedNESTDRUG)
    state_dict = checkpoint['model_state_dict']

    # Check if state dict has 'base_model.' prefix
    if any(k.startswith('base_model.') for k in state_dict.keys()):
        # Remove 'base_model.' prefix
        state_dict = {k.replace('base_model.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, config


def analyze_film_parameters(model, device) -> Dict:
    """Analyze the FiLM modulation parameters."""

    context_module = model.context_module
    film = context_module.film

    # Get all context embeddings
    n_programs = context_module.program_embeddings.num_contexts
    n_assays = context_module.assay_embeddings.num_contexts
    n_rounds = context_module.round_embeddings.num_contexts

    print(f"\nContext dimensions:")
    print(f"  Programs (L1): {n_programs}")
    print(f"  Assays (L2): {n_assays}")
    print(f"  Rounds (L3): {n_rounds}")

    results = {
        'n_programs': n_programs,
        'n_assays': n_assays,
        'n_rounds': n_rounds,
        'analyses': {}
    }

    # Analyze γ and β for all programs (with assay=0, round=0)
    print("\n" + "="*60)
    print("ANALYSIS 1: Modulation across Programs (L1)")
    print("="*60)

    all_gammas = []
    all_betas = []

    with torch.no_grad():
        for prog_id in range(n_programs):
            # Create context IDs
            program_ids = torch.tensor([prog_id], device=device)
            assay_ids = torch.tensor([0], device=device)
            round_ids = torch.tensor([0], device=device)

            # Get context vector
            context = context_module.get_context_vector(program_ids, assay_ids, round_ids)

            # Get FiLM parameters
            gamma, beta = film.get_modulation_params(context)

            all_gammas.append(gamma.cpu().numpy().flatten())
            all_betas.append(beta.cpu().numpy().flatten())

    all_gammas = np.array(all_gammas)  # [n_programs, feature_dim]
    all_betas = np.array(all_betas)

    # Statistics
    gamma_mean = all_gammas.mean()
    gamma_std = all_gammas.std()
    gamma_min = all_gammas.min()
    gamma_max = all_gammas.max()
    gamma_dev_from_1 = np.abs(all_gammas - 1.0).mean()

    beta_mean = all_betas.mean()
    beta_std = all_betas.std()
    beta_min = all_betas.min()
    beta_max = all_betas.max()
    beta_dev_from_0 = np.abs(all_betas).mean()

    print(f"\nγ (scale) statistics:")
    print(f"  Mean: {gamma_mean:.4f} (should be ~1 if identity)")
    print(f"  Std:  {gamma_std:.4f}")
    print(f"  Range: [{gamma_min:.4f}, {gamma_max:.4f}]")
    print(f"  Mean |γ - 1|: {gamma_dev_from_1:.4f}")

    print(f"\nβ (shift) statistics:")
    print(f"  Mean: {beta_mean:.4f} (should be ~0 if identity)")
    print(f"  Std:  {beta_std:.4f}")
    print(f"  Range: [{beta_min:.4f}, {beta_max:.4f}]")
    print(f"  Mean |β|: {beta_dev_from_0:.4f}")

    # Cross-program variability
    gamma_var_across_programs = all_gammas.var(axis=0).mean()  # Variance across programs
    beta_var_across_programs = all_betas.var(axis=0).mean()

    print(f"\nCross-program variability:")
    print(f"  γ variance (avg over dims): {gamma_var_across_programs:.6f}")
    print(f"  β variance (avg over dims): {beta_var_across_programs:.6f}")

    # Is FiLM learning?
    is_learning_gamma = gamma_dev_from_1 > 0.01
    is_learning_beta = beta_dev_from_0 > 0.01
    is_context_dependent = gamma_var_across_programs > 0.0001 or beta_var_across_programs > 0.0001

    print(f"\n✓ FiLM γ deviating from 1: {is_learning_gamma} (dev={gamma_dev_from_1:.4f})")
    print(f"✓ FiLM β deviating from 0: {is_learning_beta} (dev={beta_dev_from_0:.4f})")
    print(f"✓ Modulation is context-dependent: {is_context_dependent}")

    results['analyses']['programs'] = {
        'gamma_mean': float(gamma_mean),
        'gamma_std': float(gamma_std),
        'gamma_min': float(gamma_min),
        'gamma_max': float(gamma_max),
        'gamma_dev_from_1': float(gamma_dev_from_1),
        'beta_mean': float(beta_mean),
        'beta_std': float(beta_std),
        'beta_min': float(beta_min),
        'beta_max': float(beta_max),
        'beta_dev_from_0': float(beta_dev_from_0),
        'gamma_var_across_contexts': float(gamma_var_across_programs),
        'beta_var_across_contexts': float(beta_var_across_programs),
        'is_learning_gamma': is_learning_gamma,
        'is_learning_beta': is_learning_beta,
        'is_context_dependent': is_context_dependent,
    }

    # Analyze specific programs
    print("\n" + "-"*60)
    print("Per-program γ and β (first 10 feature dimensions)")
    print("-"*60)

    for prog_id in range(min(5, n_programs)):
        gamma_sample = all_gammas[prog_id, :10]
        beta_sample = all_betas[prog_id, :10]
        print(f"\nProgram {prog_id}:")
        print(f"  γ[:10]: {np.array2string(gamma_sample, precision=3, suppress_small=True)}")
        print(f"  β[:10]: {np.array2string(beta_sample, precision=3, suppress_small=True)}")

    # Pairwise program distances
    print("\n" + "-"*60)
    print("Pairwise FiLM distance between programs (Euclidean)")
    print("-"*60)

    # Combine gamma and beta for distance calculation
    film_vectors = np.concatenate([all_gammas, all_betas], axis=1)

    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(film_vectors))

    print("\nDistance matrix (programs 0-4):")
    print("      ", end="")
    for i in range(min(5, n_programs)):
        print(f"P{i:1d}     ", end="")
    print()

    for i in range(min(5, n_programs)):
        print(f"P{i}: ", end="")
        for j in range(min(5, n_programs)):
            print(f"{distances[i,j]:.3f}  ", end="")
        print()

    results['analyses']['pairwise_distances'] = distances.tolist()

    # Analyze assays (sample)
    print("\n" + "="*60)
    print("ANALYSIS 2: Modulation across Assays (L2) - sample")
    print("="*60)

    sample_assays = min(10, n_assays)
    assay_gammas = []
    assay_betas = []

    with torch.no_grad():
        for assay_id in range(sample_assays):
            program_ids = torch.tensor([0], device=device)
            assay_ids = torch.tensor([assay_id], device=device)
            round_ids = torch.tensor([0], device=device)

            context = context_module.get_context_vector(program_ids, assay_ids, round_ids)
            gamma, beta = film.get_modulation_params(context)

            assay_gammas.append(gamma.cpu().numpy().flatten())
            assay_betas.append(beta.cpu().numpy().flatten())

    assay_gammas = np.array(assay_gammas)
    assay_betas = np.array(assay_betas)

    assay_gamma_var = assay_gammas.var(axis=0).mean()
    assay_beta_var = assay_betas.var(axis=0).mean()

    print(f"\nAssay variability (first {sample_assays} assays):")
    print(f"  γ variance: {assay_gamma_var:.6f}")
    print(f"  β variance: {assay_beta_var:.6f}")

    results['analyses']['assays_sample'] = {
        'n_sampled': sample_assays,
        'gamma_var': float(assay_gamma_var),
        'beta_var': float(assay_beta_var),
    }

    return results, all_gammas, all_betas


def plot_film_analysis(all_gammas: np.ndarray, all_betas: np.ndarray, output_dir: Path):
    """Create visualizations of FiLM parameters."""

    output_dir.mkdir(parents=True, exist_ok=True)

    n_programs = all_gammas.shape[0]
    feature_dim = all_gammas.shape[1]

    # Plot 1: Distribution of γ and β
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(all_gammas.flatten(), bins=50, alpha=0.7, label='γ values')
    axes[0].axvline(x=1.0, color='r', linestyle='--', label='Identity (γ=1)')
    axes[0].set_xlabel('γ value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of FiLM Scale (γ)')
    axes[0].legend()

    axes[1].hist(all_betas.flatten(), bins=50, alpha=0.7, label='β values', color='orange')
    axes[1].axvline(x=0.0, color='r', linestyle='--', label='Identity (β=0)')
    axes[1].set_xlabel('β value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of FiLM Shift (β)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'film_distributions.png', dpi=150)
    plt.close()

    print(f"\nSaved: {output_dir / 'film_distributions.png'}")

    # Plot 2: Heatmap of γ and β per program (first 50 dims)
    dims_to_show = min(50, feature_dim)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    im1 = axes[0].imshow(all_gammas[:, :dims_to_show], aspect='auto', cmap='RdBu_r',
                          vmin=0.8, vmax=1.2)
    axes[0].set_xlabel('Feature Dimension')
    axes[0].set_ylabel('Program ID')
    axes[0].set_title('FiLM γ (scale) per Program')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(all_betas[:, :dims_to_show], aspect='auto', cmap='RdBu_r',
                          vmin=-0.5, vmax=0.5)
    axes[1].set_xlabel('Feature Dimension')
    axes[1].set_ylabel('Program ID')
    axes[1].set_title('FiLM β (shift) per Program')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(output_dir / 'film_heatmaps.png', dpi=150)
    plt.close()

    print(f"Saved: {output_dir / 'film_heatmaps.png'}")

    # Plot 3: Per-dimension variance across programs
    gamma_dim_var = all_gammas.var(axis=0)
    beta_dim_var = all_betas.var(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes[0].bar(range(dims_to_show), gamma_dim_var[:dims_to_show], alpha=0.7)
    axes[0].set_xlabel('Feature Dimension')
    axes[0].set_ylabel('Variance')
    axes[0].set_title('γ Variance across Programs (per dimension)')

    axes[1].bar(range(dims_to_show), beta_dim_var[:dims_to_show], alpha=0.7, color='orange')
    axes[1].set_xlabel('Feature Dimension')
    axes[1].set_ylabel('Variance')
    axes[1].set_title('β Variance across Programs (per dimension)')

    plt.tight_layout()
    plt.savefig(output_dir / 'film_variance.png', dpi=150)
    plt.close()

    print(f"Saved: {output_dir / 'film_variance.png'}")


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

    # Run analysis
    results, all_gammas, all_betas = analyze_film_parameters(model, device)
    results['config'] = config

    # Create plots for this checkpoint
    if not no_plots:
        try:
            checkpoint_name = checkpoint_path.stem.replace('_seed0_best', '')
            plot_dir = output_dir / checkpoint_name
            plot_film_analysis(all_gammas, all_betas, plot_dir)
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str, default='results/phase1/ablation_dude',
                        help='Directory containing checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Single checkpoint path (overrides --checkpoint-dir)')
    parser.add_argument('--output', type=str, default='results/phase1/film_analysis',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plotting (useful for headless servers)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays in nested dicts to lists for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

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
    results_file = output_dir / 'film_analysis_results.json'
    results_serializable = convert_to_json_serializable(all_results)

    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Comparative summary
    print("\n" + "="*70)
    print("COMPARATIVE SUMMARY")
    print("="*70)

    print(f"\n{'Condition':<15} {'|γ-1|':<10} {'|β|':<10} {'γ var':<12} {'β var':<12} {'Learning?':<10}")
    print("-"*70)

    for cond, results in all_results.items():
        prog = results['analyses']['programs']
        learning = "✓" if (prog['is_learning_gamma'] or prog['is_learning_beta']) else "✗"
        print(f"{cond:<15} {prog['gamma_dev_from_1']:<10.4f} {prog['beta_dev_from_0']:<10.4f} "
              f"{prog['gamma_var_across_contexts']:<12.6f} {prog['beta_var_across_contexts']:<12.6f} {learning:<10}")

    # Key insights
    print("\n" + "-"*70)
    print("KEY INSIGHTS")
    print("-"*70)

    # Check if context models learn more than L0
    if 'L0' in all_results and len(all_results) > 1:
        l0_gamma_dev = all_results['L0']['analyses']['programs']['gamma_dev_from_1']
        l0_beta_dev = all_results['L0']['analyses']['programs']['beta_dev_from_0']

        for cond, results in all_results.items():
            if cond == 'L0':
                continue
            prog = results['analyses']['programs']
            gamma_diff = prog['gamma_dev_from_1'] - l0_gamma_dev
            beta_diff = prog['beta_dev_from_0'] - l0_beta_dev

            if gamma_diff > 0.001 or beta_diff > 0.001:
                print(f"\n{cond} vs L0:")
                print(f"  Δ|γ-1|: {gamma_diff:+.4f}")
                print(f"  Δ|β|:   {beta_diff:+.4f}")
                if prog['gamma_var_across_contexts'] > 0.0001:
                    print(f"  γ is context-dependent (var={prog['gamma_var_across_contexts']:.6f})")
                if prog['beta_var_across_contexts'] > 0.0001:
                    print(f"  β is context-dependent (var={prog['beta_var_across_contexts']:.6f})")


if __name__ == '__main__':
    main()
