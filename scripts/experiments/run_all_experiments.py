#!/usr/bin/env python3
"""
Master Runner: Run All Experiments on Multiple Model Versions

Runs all Phase 1-4 experiments on V1 baseline, V2 (expanded data), and V3 (fine-tuned).

Usage:
    # Run all experiments on all models
    python scripts/experiments/run_all_experiments.py --gpu 0

    # Run specific model only
    python scripts/experiments/run_all_experiments.py --models v1 --gpu 0

    # Run specific experiments only
    python scripts/experiments/run_all_experiments.py --experiments film context_embedding --gpu 0

    # Quick test mode (minimal samples)
    python scripts/experiments/run_all_experiments.py --quick-test --gpu 0
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Model configurations
MODEL_CONFIGS = {
    'v1': {
        'name': 'V1-Original',
        'checkpoint': 'checkpoints/pretrain/best_model.pt',
        'description': 'Original pretrained model (5 programs, 50 assays, 150 rounds)',
    },
    'v2': {
        'name': 'V2-Expanded',
        'checkpoint': 'results/v2_full/best_model.pt',
        'description': 'V2 trained from scratch with expanded data (5123 programs)',
    },
    'v3': {
        'name': 'V3-FineTuned',
        'checkpoint': 'results/v3/best_model.pt',
        'description': 'V3 fine-tuned from V1 backbone',
    },
}

# Experiment configurations
EXPERIMENTS = {
    # Phase 1: Model Diagnostics
    'film': {
        'name': '1B: FiLM Deviation Analysis',
        'script': 'scripts/experiments/film_analysis.py',
        'args': [],
        'requires_data': False,
    },
    'context_embedding': {
        'name': '1C: Context Embedding Visualization',
        'script': 'scripts/experiments/context_embedding_analysis.py',
        'args': [],
        'requires_data': False,
    },

    # Phase 2: Attribution Analysis (require captum: pip install captum)
    'integrated_gradients': {
        'name': '2A: Integrated Gradients',
        'script': 'scripts/experiments/integrated_gradients.py',
        'args': [],  # Uses default example molecules
        'quick_args': ['--n-steps', '20'],  # Fewer steps for quick test
        'requires_data': False,
        'requires_package': 'captum',
    },
    'context_attribution': {
        'name': '2B: Context-Conditional Attribution',
        'script': 'scripts/experiments/context_conditional_attribution.py',
        'args': ['--program-ids', '0', '1', '2', '3', '4'],
        'quick_args': ['--program-ids', '0', '1', '--n-steps', '20'],
        'requires_data': False,
        'requires_package': 'captum',
    },
    'decision_boundary': {
        'name': '2C: Decision Boundary Visualization',
        'script': 'scripts/experiments/decision_boundary.py',
        'args': ['--data-dir', 'data/external/dude', '--targets', 'egfr', 'drd2', 'bace1'],
        'quick_args': ['--data-dir', 'data/external/dude', '--targets', 'egfr', '--max-samples', '200'],
        'requires_data': True,
    },

    # Phase 3: Generalization Tests
    'tdc_benchmark': {
        'name': '3A: TDC Benchmark',
        'script': 'scripts/experiments/tdc_benchmark.py',
        'args': ['--datasets', 'hERG', 'AMES', 'BBB'],
        'quick_args': ['--datasets', 'hERG'],
        'requires_data': False,  # Downloads automatically
    },
    'temporal_split': {
        'name': '3B: Temporal Split',
        'script': 'scripts/experiments/temporal_split.py',
        'args': ['--data', 'data/processed/portfolio/chembl_potency_all.parquet'],
        'quick_args': ['--data', 'data/processed/portfolio/chembl_potency_all.parquet', '--max-test', '500'],
        'requires_data': True,
    },
    'cross_target': {
        'name': '3C: Cross-Target Zero-Shot',
        'script': 'scripts/experiments/cross_target_zeroshot.py',
        'args': ['--data-dir', 'data/external/dude'],
        'quick_args': ['--data-dir', 'data/external/dude'],
        'requires_data': True,
    },

    # Phase 4: Few-Shot
    'few_shot': {
        'name': '4A: Few-Shot Adaptation',
        'script': 'scripts/experiments/few_shot_adaptation.py',
        'args': ['--data-dir', 'data/external/dude', '--targets', 'egfr', 'drd2', 'bace1'],
        'quick_args': ['--data-dir', 'data/external/dude', '--targets', 'egfr', '--n-shots', '5', '10', '--n-trials', '1'],
        'requires_data': True,
    },
}


def check_checkpoint_exists(checkpoint_path):
    """Check if checkpoint file exists."""
    return Path(checkpoint_path).exists()


def check_data_exists(experiment_key):
    """Check if required data exists for experiment."""
    exp = EXPERIMENTS[experiment_key]
    if not exp['requires_data']:
        return True

    # Check common data paths
    dude_path = Path('data/external/dude')
    chembl_path = Path('data/processed/portfolio/chembl_potency_all.parquet')

    if 'dude' in str(exp.get('args', [])):
        return dude_path.exists()
    if 'chembl' in str(exp.get('args', [])):
        return chembl_path.exists()

    return True


def check_package_installed(package_name):
    """Check if a Python package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def run_experiment(exp_key, model_key, gpu, quick_test=False, base_output='results/experiments'):
    """Run a single experiment on a specific model."""
    exp = EXPERIMENTS[exp_key]
    model = MODEL_CONFIGS[model_key]

    # Check checkpoint
    if not check_checkpoint_exists(model['checkpoint']):
        print(f"  SKIP: Checkpoint not found: {model['checkpoint']}")
        return {'status': 'skipped', 'reason': 'checkpoint_not_found'}

    # Check required package
    if 'requires_package' in exp and not check_package_installed(exp['requires_package']):
        print(f"  SKIP: Missing package: {exp['requires_package']} (pip install {exp['requires_package']})")
        return {'status': 'skipped', 'reason': f"missing_package_{exp['requires_package']}"}

    # Check data
    if not check_data_exists(exp_key):
        print(f"  SKIP: Required data not found for {exp_key}")
        return {'status': 'skipped', 'reason': 'data_not_found'}

    # Build command
    output_dir = f"{base_output}/{exp_key}/{model_key}"
    args = exp.get('quick_args', exp['args']) if quick_test else exp['args']

    # Use conda run to ensure nest environment is used
    cmd = [
        'conda', 'run', '-n', 'nest',
        'python', exp['script'],
        '--checkpoint', model['checkpoint'],
        '--output', output_dir,
        '--gpu', str(gpu),
    ] + args

    print(f"  Running: {' '.join(cmd[:5])}...")

    # Get project root directory
    project_root = Path(__file__).parent.parent.parent

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800 if not quick_test else 300,  # 30min or 5min for quick
            cwd=project_root,  # Run from project root
        )

        if result.returncode == 0:
            print(f"  SUCCESS: Results saved to {output_dir}")
            return {'status': 'success', 'output_dir': output_dir}
        else:
            print(f"  FAILED: {result.stderr[:200]}")
            return {'status': 'failed', 'error': result.stderr[:500]}

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: Experiment took too long")
        return {'status': 'timeout'}
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return {'status': 'error', 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description="Run All Experiments")
    parser.add_argument('--models', nargs='+', default=['v1', 'v2', 'v3'],
                        choices=['v1', 'v2', 'v3'], help='Models to evaluate')
    parser.add_argument('--experiments', nargs='+', default=None,
                        choices=list(EXPERIMENTS.keys()), help='Experiments to run')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--output', type=str, default='results/experiments',
                        help='Base output directory')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test mode with minimal samples')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    args = parser.parse_args()

    experiments_to_run = args.experiments or list(EXPERIMENTS.keys())

    print("=" * 70)
    print("NEST-DRUG Experiment Runner")
    print("=" * 70)
    print(f"Models: {args.models}")
    print(f"Experiments: {experiments_to_run}")
    print(f"GPU: {args.gpu}")
    print(f"Quick test: {args.quick_test}")
    print(f"Output: {args.output}")
    print("=" * 70)

    # Check which models are available
    print("\nModel Status:")
    for model_key in args.models:
        model = MODEL_CONFIGS[model_key]
        exists = check_checkpoint_exists(model['checkpoint'])
        status = "FOUND" if exists else "NOT FOUND"
        print(f"  {model['name']}: {status}")
        print(f"    Path: {model['checkpoint']}")

    if args.dry_run:
        print("\n[DRY RUN - Commands that would be executed:]")
        for model_key in args.models:
            model = MODEL_CONFIGS[model_key]
            print(f"\n{model['name']}:")
            for exp_key in experiments_to_run:
                exp = EXPERIMENTS[exp_key]
                output_dir = f"{args.output}/{exp_key}/{model_key}"
                cmd_args = exp.get('quick_args', exp['args']) if args.quick_test else exp['args']
                cmd = ['python', exp['script'], '--checkpoint', model['checkpoint'],
                       '--output', output_dir, '--gpu', str(args.gpu)] + cmd_args
                print(f"  {exp['name']}:")
                print(f"    {' '.join(cmd)}")
        return

    # Run experiments
    all_results = {}
    start_time = datetime.now()

    for model_key in args.models:
        model = MODEL_CONFIGS[model_key]
        print(f"\n{'='*70}")
        print(f"MODEL: {model['name']}")
        print(f"{'='*70}")

        model_results = {}

        for exp_key in experiments_to_run:
            exp = EXPERIMENTS[exp_key]
            print(f"\n[{exp['name']}]")

            result = run_experiment(
                exp_key, model_key, args.gpu,
                quick_test=args.quick_test,
                base_output=args.output
            )
            model_results[exp_key] = result

        all_results[model_key] = model_results

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Duration: {duration:.1f}s")

    # Create summary table
    print(f"\n{'Experiment':<30} ", end='')
    for model_key in args.models:
        print(f"{model_key:<12}", end='')
    print()
    print("-" * (30 + 12 * len(args.models)))

    for exp_key in experiments_to_run:
        exp = EXPERIMENTS[exp_key]
        print(f"{exp['name'][:28]:<30} ", end='')
        for model_key in args.models:
            status = all_results.get(model_key, {}).get(exp_key, {}).get('status', 'N/A')
            symbol = {'success': '✓', 'failed': '✗', 'skipped': '-', 'timeout': '⏱', 'error': '!'}
            print(f"{symbol.get(status, '?'):<12}", end='')
        print()

    # Save results
    results_path = Path(args.output) / 'run_summary.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'models': args.models,
            'experiments': experiments_to_run,
            'quick_test': args.quick_test,
            'results': all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
