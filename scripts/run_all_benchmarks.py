#!/usr/bin/env python3
"""
NEST-DRUG Comprehensive Benchmark Suite

Coordinator script to run all validation experiments:
1. LIT-PCBA (Critical) - Real experimental inactives
2. DUD-E (Critical) - Property-matched decoys
3. hERG (High) - Safety/ion channel
4. DRD2 DMTA (High) - GPCR generalization [if implemented]
5. Tox21 (Medium) - Multi-task toxicity [if implemented]
6. MoleculeNet ADMET (Medium) - ADMET benchmarks [if implemented]

Usage:
    python scripts/run_all_benchmarks.py --checkpoint checkpoints/pretrain/best_model.pt
    python scripts/run_all_benchmarks.py --checkpoint checkpoints/pretrain/best_model.pt --benchmarks litpcba dude
    python scripts/run_all_benchmarks.py --download-only  # Just download data
"""

import argparse
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Available benchmarks
BENCHMARKS = {
    'litpcba': {
        'name': 'LIT-PCBA',
        'description': '15 targets with real experimental inactives (REQUIRES MANUAL DOWNLOAD)',
        'priority': 'Critical',
        'script': 'scripts/benchmarks/run_litpcba.py',
        'data_key': 'litpcba',
        'estimated_time': '4-6 hrs'
    },
    'dude': {
        'name': 'DUD-E',
        'description': '10 targets with property-matched decoys',
        'priority': 'Critical',
        'script': 'scripts/benchmarks/run_dude.py',
        'data_key': 'dude',
        'estimated_time': '4-6 hrs'
    },
    'herg': {
        'name': 'hERG Safety',
        'description': 'Cardiac safety endpoint (ion channel)',
        'priority': 'High',
        'script': 'scripts/benchmarks/run_herg.py',
        'data_key': 'herg',
        'estimated_time': '2-3 hrs'
    },
    'hts': {
        'name': 'HTS Comparison',
        'description': 'EGFR HTS simulation with comprehensive metrics',
        'priority': 'Complete',
        'script': 'scripts/run_hts_comparison.py',
        'data_key': None,  # Uses existing EGFR data
        'estimated_time': '1-2 hrs'
    }
}


def download_data(benchmarks=None):
    """Download data for specified benchmarks."""
    print("\n" + "="*70)
    print("DOWNLOADING BENCHMARK DATA")
    print("="*70)

    if benchmarks is None:
        benchmarks = list(BENCHMARKS.keys())

    # Map benchmark names to data keys
    data_keys = [BENCHMARKS[b]['data_key'] for b in benchmarks if BENCHMARKS[b]['data_key']]
    data_keys = list(set(data_keys))  # Unique

    if not data_keys:
        print("No data to download.")
        return

    cmd = ['python', 'scripts/download_benchmark_data.py', '--datasets'] + data_keys

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    return result.returncode == 0


def run_benchmark(benchmark_name, checkpoint, gpu=0, extra_args=None):
    """Run a single benchmark."""
    if benchmark_name not in BENCHMARKS:
        print(f"Unknown benchmark: {benchmark_name}")
        return None

    info = BENCHMARKS[benchmark_name]

    print(f"\n{'='*70}")
    print(f"RUNNING: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"Priority: {info['priority']}")
    print(f"Estimated time: {info['estimated_time']}")
    print(f"{'='*70}")

    script = info['script']
    output_dir = f"results/benchmarks/{benchmark_name}"

    cmd = [
        'python', script,
        '--checkpoint', checkpoint,
        '--output', output_dir,
        '--gpu', str(gpu)
    ]

    if extra_args:
        cmd.extend(extra_args)

    print(f"\nCommand: {' '.join(cmd)}")

    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=False)
    end_time = datetime.now()

    duration = (end_time - start_time).total_seconds()

    return {
        'benchmark': benchmark_name,
        'success': result.returncode == 0,
        'duration_seconds': duration,
        'output_dir': output_dir
    }


def aggregate_results(results_dirs):
    """Aggregate results from all benchmarks."""
    print("\n" + "="*70)
    print("AGGREGATING RESULTS")
    print("="*70)

    all_metrics = {}

    for name, output_dir in results_dirs.items():
        summary_file = Path(output_dir) / "summary.csv"
        results_file = Path(output_dir) / "all_results.json"

        if summary_file.exists():
            df = pd.read_csv(summary_file, index_col=0)
            all_metrics[name] = {
                'mean_roc_auc': df['roc_auc'].mean() if 'roc_auc' in df.columns else None,
                'mean_bedroc': df['bedroc_20'].mean() if 'bedroc_20' in df.columns else None,
                'mean_ef_1pct': df['ef_1%_ef'].mean() if 'ef_1%_ef' in df.columns else None,
                'n_targets': len(df)
            }
        elif results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            if 'vs_metrics' in data:
                all_metrics[name] = {
                    'roc_auc': data['vs_metrics'].get('roc_auc'),
                    'bedroc': data['vs_metrics'].get('bedroc_20'),
                    'n_targets': 1
                }

    return all_metrics


def print_final_summary(run_results, metrics):
    """Print final summary of all benchmarks."""
    print("\n" + "="*70)
    print("BENCHMARK SUITE SUMMARY")
    print("="*70)

    print("\n--- Run Status ---")
    for result in run_results:
        status = "✓" if result['success'] else "✗"
        duration = f"{result['duration_seconds']/60:.1f} min"
        print(f"  {status} {result['benchmark']}: {duration}")

    print("\n--- Performance Metrics ---")
    print(f"{'Benchmark':<15} {'ROC-AUC':>10} {'BEDROC':>10} {'EF@1%':>10} {'Targets':>10}")
    print("-" * 60)

    for name, m in metrics.items():
        roc = f"{m.get('mean_roc_auc', m.get('roc_auc', 'N/A')):.4f}" if m.get('mean_roc_auc') or m.get('roc_auc') else "N/A"
        bed = f"{m.get('mean_bedroc', m.get('bedroc', 'N/A')):.4f}" if m.get('mean_bedroc') or m.get('bedroc') else "N/A"
        ef = f"{m.get('mean_ef_1pct', 'N/A'):.1f}x" if m.get('mean_ef_1pct') else "N/A"
        n = m.get('n_targets', 'N/A')
        print(f"{name:<15} {roc:>10} {bed:>10} {ef:>10} {n:>10}")


def main():
    parser = argparse.ArgumentParser(
        description='NEST-DRUG Comprehensive Benchmark Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  python scripts/run_all_benchmarks.py --checkpoint checkpoints/pretrain/best_model.pt

  # Run specific benchmarks
  python scripts/run_all_benchmarks.py --checkpoint checkpoints/pretrain/best_model.pt --benchmarks litpcba herg

  # Download data only
  python scripts/run_all_benchmarks.py --download-only

  # List available benchmarks
  python scripts/run_all_benchmarks.py --list
        """
    )
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       choices=list(BENCHMARKS.keys()),
                       help='Specific benchmarks to run (default: all)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU to use')
    parser.add_argument('--download-only', action='store_true',
                       help='Only download data, do not run benchmarks')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip data download step')
    parser.add_argument('--list', action='store_true',
                       help='List available benchmarks and exit')
    args = parser.parse_args()

    # List benchmarks
    if args.list:
        print("\nAvailable Benchmarks:")
        print("="*70)
        for key, info in BENCHMARKS.items():
            print(f"\n{key}:")
            print(f"  Name: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Priority: {info['priority']}")
            print(f"  Estimated time: {info['estimated_time']}")
        return

    # Determine which benchmarks to run
    # Default excludes litpcba (requires manual download)
    default_benchmarks = ['dude', 'herg', 'hts']
    benchmarks = args.benchmarks or default_benchmarks

    print("\n" + "="*70)
    print("NEST-DRUG COMPREHENSIVE BENCHMARK SUITE")
    print("="*70)
    print(f"\nBenchmarks to run: {', '.join(benchmarks)}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"GPU: {args.gpu}")

    # Download data
    if not args.skip_download:
        download_data(benchmarks)

    if args.download_only:
        print("\nData download complete. Exiting.")
        return

    # Check checkpoint
    if not args.checkpoint:
        print("\nERROR: --checkpoint is required to run benchmarks")
        return

    if not Path(args.checkpoint).exists():
        print(f"\nERROR: Checkpoint not found: {args.checkpoint}")
        return

    # Run benchmarks
    run_results = []
    results_dirs = {}

    for benchmark in benchmarks:
        result = run_benchmark(benchmark, args.checkpoint, args.gpu)
        if result:
            run_results.append(result)
            if result['success']:
                results_dirs[benchmark] = result['output_dir']

    # Aggregate and summarize
    metrics = aggregate_results(results_dirs)
    print_final_summary(run_results, metrics)

    # Save overall summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': args.checkpoint,
        'benchmarks_run': benchmarks,
        'run_results': run_results,
        'metrics': metrics
    }

    summary_path = Path("results/benchmarks/overall_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nOverall summary saved to: {summary_path}")

    print("\n" + "="*70)
    print("BENCHMARK SUITE COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
