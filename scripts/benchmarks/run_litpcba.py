#!/usr/bin/env python3
"""
LIT-PCBA Benchmark for NEST-DRUG

LIT-PCBA is the gold-standard virtual screening benchmark with REAL experimental
inactives from PubChem (not property-matched decoys).

15 targets across multiple protein classes:
- GPCRs: ADRB2, OPRK1
- Kinases: MAPK1, MTORC1
- Nuclear Receptors: ESR1, PPARG, VDR
- Enzymes: ALDH1, FEN1, GBA, IDH1, KAT2A, PKM2
- Other: TP53

Active rates: 0.08% - 0.27% (realistic HTS conditions)
"""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from benchmarks.data_loaders import load_litpcba_target, load_all_litpcba, LITPCBA_TARGETS
from benchmarks.metrics import calculate_all_vs_metrics, print_metrics_summary
from models.nest_drug import create_nest_drug
from training.data_utils import smiles_to_graph
from torch.cuda.amp import autocast


def score_compounds(model, smiles_list, device, batch_size=256):
    """Score compounds using NEST-DRUG model."""
    model.eval()
    scores = []
    valid_indices = []

    with torch.no_grad():
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Scoring"):
            batch_smiles = smiles_list[i:i+batch_size]
            batch_graphs = []
            batch_valid = []

            for j, smi in enumerate(batch_smiles):
                graph = smiles_to_graph(smi)
                if graph is not None:
                    batch_graphs.append(graph)
                    batch_valid.append(i + j)

            if not batch_graphs:
                continue

            # Collate batch
            node_features = torch.cat([g['node_features'] for g in batch_graphs], dim=0).to(device)
            edge_index_list = []
            edge_features_list = []
            batch_indices = []
            offset = 0

            for idx, g in enumerate(batch_graphs):
                edge_index_list.append(g['edge_index'] + offset)
                edge_features_list.append(g['edge_features'])
                batch_indices.extend([idx] * g['num_atoms'])
                offset += g['num_atoms']

            edge_index = torch.cat(edge_index_list, dim=1).to(device)
            edge_features = torch.cat(edge_features_list, dim=0).to(device)
            batch_tensor = torch.tensor(batch_indices, dtype=torch.long, device=device)

            # Context IDs (all zeros for inference)
            n_mols = len(batch_graphs)
            program_ids = torch.zeros(n_mols, dtype=torch.long, device=device)
            assay_ids = torch.zeros(n_mols, dtype=torch.long, device=device)
            round_ids = torch.zeros(n_mols, dtype=torch.long, device=device)

            with autocast(enabled=True):
                predictions = model(
                    node_features=node_features,
                    edge_index=edge_index,
                    edge_features=edge_features,
                    batch=batch_tensor,
                    program_ids=program_ids,
                    assay_ids=assay_ids,
                    round_ids=round_ids,
                )

            # Use pchembl_median prediction as score
            if 'pchembl_median' in predictions:
                batch_scores = predictions['pchembl_median'].cpu().numpy().flatten()
            else:
                first_key = [k for k in predictions.keys() if k not in ['h_mol', 'h_mod']][0]
                batch_scores = predictions[first_key].cpu().numpy().flatten()

            scores.extend([float(s) for s in batch_scores])
            valid_indices.extend(batch_valid)

    return scores, valid_indices


def run_litpcba_benchmark(model, device, output_dir, targets=None, batch_size=256):
    """
    Run NEST-DRUG on all LIT-PCBA targets.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if targets is None:
        targets = LITPCBA_TARGETS

    # Load all targets
    print("\n" + "="*70)
    print("LOADING LIT-PCBA DATA")
    print("="*70)

    litpcba_data = load_all_litpcba(targets=targets)

    if not litpcba_data:
        print("ERROR: No LIT-PCBA data found. Run download script first.")
        return None

    # Run benchmark on each target
    all_results = {}

    for target_name, df in litpcba_data.items():
        print(f"\n{'='*70}")
        print(f"BENCHMARKING: {target_name}")
        print(f"{'='*70}")

        n_actives = df['is_active'].sum()
        n_total = len(df)
        print(f"  Compounds: {n_total:,}")
        print(f"  Actives: {n_actives}")
        print(f"  Active rate: {n_actives/n_total*100:.3f}%")

        # Score all compounds
        smiles_list = df['smiles'].tolist()
        scores, valid_indices = score_compounds(model, smiles_list, device, batch_size)

        # Create scored dataframe
        scored_df = df.iloc[valid_indices].copy()
        scored_df['pred_score'] = scores

        # Calculate all metrics
        y_true = scored_df['is_active'].values
        y_score = scored_df['pred_score'].values

        metrics = calculate_all_vs_metrics(y_true, y_score, name=target_name)
        all_results[target_name] = metrics

        # Print summary
        print_metrics_summary(metrics, title=f"{target_name} Results")

        # Save per-target results
        scored_df.to_parquet(output_dir / f"{target_name}_scored.parquet")

    # Aggregate results
    print("\n" + "="*70)
    print("AGGREGATE RESULTS")
    print("="*70)

    summary_df = pd.DataFrame(all_results).T

    # Calculate means
    print(f"\nMean ROC-AUC:    {summary_df['roc_auc'].mean():.4f} ± {summary_df['roc_auc'].std():.4f}")
    print(f"Mean BEDROC:     {summary_df['bedroc_20'].mean():.4f} ± {summary_df['bedroc_20'].std():.4f}")
    print(f"Mean EF@1%:      {summary_df['ef_1%_ef'].mean():.2f}x ± {summary_df['ef_1%_ef'].std():.2f}")
    print(f"Mean AP:         {summary_df['average_precision'].mean():.4f} ± {summary_df['average_precision'].std():.4f}")
    print(f"Mean AUAC:       {summary_df['auac'].mean():.4f} ± {summary_df['auac'].std():.4f}")

    # Save summary
    summary_df.to_csv(output_dir / "summary.csv")

    # Save full results as JSON
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate visualizations
    generate_litpcba_figures(summary_df, output_dir)

    return all_results, summary_df


def generate_litpcba_figures(summary_df, output_dir):
    """Generate visualization figures for LIT-PCBA results."""
    output_dir = Path(output_dir)

    # Figure 1: ROC-AUC and BEDROC bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    targets = summary_df.index.tolist()
    x = np.arange(len(targets))

    # ROC-AUC
    ax1 = axes[0]
    bars1 = ax1.bar(x, summary_df['roc_auc'], color='steelblue', alpha=0.8)
    ax1.axhline(y=0.5, color='red', linestyle='--', label='Random')
    ax1.axhline(y=summary_df['roc_auc'].mean(), color='green', linestyle='-', label=f"Mean: {summary_df['roc_auc'].mean():.3f}")
    ax1.set_xlabel('Target')
    ax1.set_ylabel('ROC-AUC')
    ax1.set_title('ROC-AUC by Target')
    ax1.set_xticks(x)
    ax1.set_xticklabels(targets, rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # BEDROC
    ax2 = axes[1]
    bars2 = ax2.bar(x, summary_df['bedroc_20'], color='coral', alpha=0.8)
    ax2.axhline(y=summary_df['bedroc_20'].mean(), color='green', linestyle='-', label=f"Mean: {summary_df['bedroc_20'].mean():.3f}")
    ax2.set_xlabel('Target')
    ax2.set_ylabel('BEDROC (α=20)')
    ax2.set_title('BEDROC by Target')
    ax2.set_xticks(x)
    ax2.set_xticklabels(targets, rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "roc_bedroc_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Enrichment factors
    fig, ax = plt.subplots(figsize=(12, 5))

    ef_cols = ['ef_0.1%_ef', 'ef_1%_ef', 'ef_5%_ef', 'ef_10%_ef']
    ef_labels = ['EF@0.1%', 'EF@1%', 'EF@5%', 'EF@10%']
    width = 0.2

    for i, (col, label) in enumerate(zip(ef_cols, ef_labels)):
        if col in summary_df.columns:
            ax.bar(x + i*width, summary_df[col], width, label=label, alpha=0.8)

    ax.set_xlabel('Target')
    ax.set_ylabel('Enrichment Factor')
    ax.set_title('Enrichment Factors by Target')
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(targets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "enrichment_factors.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFigures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='LIT-PCBA Benchmark for NEST-DRUG')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/external/litpcba/LIT-PCBA',
                       help='Path to LIT-PCBA data directory')
    parser.add_argument('--output', type=str, default='results/benchmarks/litpcba',
                       help='Output directory')
    parser.add_argument('--targets', type=str, nargs='+', default=None,
                       help='Specific targets to run (default: all)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for scoring')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU to use')
    args = parser.parse_args()

    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # Detect endpoints
    endpoint_names = []
    for key in state_dict.keys():
        if 'prediction_heads.heads.' in key and '.mlp.0.weight' in key:
            name = key.split('prediction_heads.heads.')[1].split('.mlp')[0]
            endpoint_names.append(name)

    endpoints = {name: {'type': 'regression', 'weight': 1.0} for name in endpoint_names}
    print(f"Endpoints: {endpoint_names}")

    model = create_nest_drug(
        num_programs=5,
        num_assays=50,
        num_rounds=150,
        endpoints=endpoints,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Run benchmark
    result = run_litpcba_benchmark(
        model=model,
        device=device,
        output_dir=args.output,
        targets=args.targets,
        batch_size=args.batch_size
    )

    if result is None:
        print("\n" + "="*70)
        print("LIT-PCBA BENCHMARK FAILED - No data available")
        print("="*70)
        print("\nPlease download LIT-PCBA manually from:")
        print("  https://drugdesign.unistra.fr/LIT-PCBA/")
        print(f"Extract to: data/external/litpcba/LIT-PCBA/")
        sys.exit(1)

    results, summary = result
    print("\n" + "="*70)
    print("LIT-PCBA BENCHMARK COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
