#!/usr/bin/env python3
"""
Experiment 1B: FiLM Deviation Analysis

Check if γ and β actually deviated from identity after training.
If γ≈1, β≈0 still, FiLM isn't learning anything.

Usage:
    python scripts/experiments/film_analysis.py \
        --checkpoint checkpoints/pretrain/best_model.pt \
        --output results/experiments/film_analysis \
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

    return model, {'num_programs': num_programs, 'num_assays': num_assays, 'num_rounds': num_rounds}


def analyze_film_parameters(model):
    """Analyze FiLM gamma and beta parameters."""
    results = {}

    # Find FiLM layers in the model
    film_params = {}
    for name, param in model.named_parameters():
        if 'film' in name.lower() or 'gamma' in name.lower() or 'beta' in name.lower():
            film_params[name] = param.detach().cpu().numpy()

    # If no explicit FiLM params, check the film module
    if hasattr(model, 'film'):
        film = model.film
        for name, param in film.named_parameters():
            film_params[f'film.{name}'] = param.detach().cpu().numpy()

    if not film_params:
        # Try to extract from state dict directly
        state_dict = model.state_dict()
        for name, param in state_dict.items():
            if 'film' in name.lower():
                film_params[name] = param.cpu().numpy()

    for name, param in film_params.items():
        is_gamma = 'gamma' in name.lower() or 'scale' in name.lower()
        is_beta = 'beta' in name.lower() or 'shift' in name.lower() or 'bias' in name.lower()

        analysis = {
            'shape': list(param.shape),
            'mean': float(np.mean(param)),
            'std': float(np.std(param)),
            'min': float(np.min(param)),
            'max': float(np.max(param)),
        }

        if is_gamma:
            analysis['deviation_from_1'] = float(np.mean(np.abs(param - 1.0)))
            analysis['is_learning'] = analysis['deviation_from_1'] > 0.1
            analysis['type'] = 'gamma'
        elif is_beta:
            analysis['deviation_from_0'] = float(np.mean(np.abs(param)))
            analysis['is_learning'] = analysis['deviation_from_0'] > 0.05
            analysis['type'] = 'beta'
        else:
            analysis['type'] = 'unknown'

        results[name] = analysis

    return results


def analyze_context_modulation(model, config):
    """Analyze how context embeddings modulate the network."""
    results = {}

    # Get context embeddings
    if hasattr(model, 'context_module'):
        ctx = model.context_module

        # Program embeddings (L1)
        if hasattr(ctx, 'program_embeddings'):
            prog_emb = ctx.program_embeddings.embeddings.weight.detach().cpu().numpy()
            results['program_embeddings'] = {
                'shape': list(prog_emb.shape),
                'mean_norm': float(np.mean(np.linalg.norm(prog_emb, axis=1))),
                'std_norm': float(np.std(np.linalg.norm(prog_emb, axis=1))),
                'variance_across_programs': float(np.mean(np.var(prog_emb, axis=0))),
            }

        # Assay embeddings (L2)
        if hasattr(ctx, 'assay_embeddings'):
            assay_emb = ctx.assay_embeddings.embeddings.weight.detach().cpu().numpy()
            results['assay_embeddings'] = {
                'shape': list(assay_emb.shape),
                'mean_norm': float(np.mean(np.linalg.norm(assay_emb, axis=1))),
                'variance_across_assays': float(np.mean(np.var(assay_emb, axis=0))),
            }

        # Round embeddings (L3)
        if hasattr(ctx, 'round_embeddings'):
            round_emb = ctx.round_embeddings.embeddings.weight.detach().cpu().numpy()
            results['round_embeddings'] = {
                'shape': list(round_emb.shape),
                'mean_norm': float(np.mean(np.linalg.norm(round_emb, axis=1))),
                'variance_across_rounds': float(np.mean(np.var(round_emb, axis=0))),
            }

    return results


def plot_film_analysis(film_results, output_dir):
    """Create visualizations of FiLM parameters."""
    output_dir = Path(output_dir)

    # Separate gamma and beta
    gammas = {k: v for k, v in film_results.items() if v.get('type') == 'gamma'}
    betas = {k: v for k, v in film_results.items() if v.get('type') == 'beta'}

    if gammas or betas:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Gamma distribution
        if gammas:
            gamma_devs = [v['deviation_from_1'] for v in gammas.values()]
            gamma_names = [k.split('.')[-1] for k in gammas.keys()]
            axes[0].bar(range(len(gamma_devs)), gamma_devs)
            axes[0].axhline(y=0.1, color='r', linestyle='--', label='Learning threshold')
            axes[0].set_ylabel('Deviation from 1.0')
            axes[0].set_title('Gamma (Scale) Parameters')
            axes[0].legend()

        # Beta distribution
        if betas:
            beta_devs = [v['deviation_from_0'] for v in betas.values()]
            beta_names = [k.split('.')[-1] for k in betas.keys()]
            axes[1].bar(range(len(beta_devs)), beta_devs)
            axes[1].axhline(y=0.05, color='r', linestyle='--', label='Learning threshold')
            axes[1].set_ylabel('Deviation from 0.0')
            axes[1].set_title('Beta (Shift) Parameters')
            axes[1].legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'film_deviations.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir / 'film_deviations.png'}")


def main():
    parser = argparse.ArgumentParser(description="FiLM Deviation Analysis")
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--output', type=str, default='results/experiments/film_analysis',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLoading model: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    print(f"Config: {config}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze FiLM parameters
    print("\nAnalyzing FiLM parameters...")
    film_results = analyze_film_parameters(model)

    # Analyze context modulation
    print("Analyzing context modulation...")
    context_results = analyze_context_modulation(model, config)

    # Summary
    print(f"\n{'='*60}")
    print("FiLM ANALYSIS RESULTS")
    print('='*60)

    n_learning_gamma = sum(1 for v in film_results.values() if v.get('type') == 'gamma' and v.get('is_learning', False))
    n_learning_beta = sum(1 for v in film_results.values() if v.get('type') == 'beta' and v.get('is_learning', False))
    n_gamma = sum(1 for v in film_results.values() if v.get('type') == 'gamma')
    n_beta = sum(1 for v in film_results.values() if v.get('type') == 'beta')

    print(f"\nFiLM Parameters:")
    print(f"  Gamma learning: {n_learning_gamma}/{n_gamma}")
    print(f"  Beta learning: {n_learning_beta}/{n_beta}")

    for name, analysis in film_results.items():
        status = "LEARNING" if analysis.get('is_learning', False) else "NOT LEARNING"
        if analysis.get('type') == 'gamma':
            print(f"  {name}: dev={analysis['deviation_from_1']:.4f} [{status}]")
        elif analysis.get('type') == 'beta':
            print(f"  {name}: dev={analysis['deviation_from_0']:.4f} [{status}]")

    print(f"\nContext Embeddings:")
    for name, analysis in context_results.items():
        print(f"  {name}: shape={analysis['shape']}, mean_norm={analysis['mean_norm']:.4f}")

    # Create plots
    plot_film_analysis(film_results, output_dir)

    # Save results
    results_path = output_dir / 'film_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'config': config,
            'film_parameters': film_results,
            'context_embeddings': context_results,
            'summary': {
                'gamma_learning': n_learning_gamma,
                'gamma_total': n_gamma,
                'beta_learning': n_learning_beta,
                'beta_total': n_beta,
            }
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
