#!/usr/bin/env python3
"""
hERG Safety Benchmark for NEST-DRUG

hERG (human Ether-à-go-go Related Gene) channel blocking is a critical
safety endpoint - compounds that block hERG can cause cardiac arrhythmias.

This benchmark tests binary classification:
- Blocker (is_blocker=1): IC50 < 10 µM (dangerous)
- Non-blocker (is_blocker=0): IC50 >= 10 µM (safe)

For drug discovery, HIGH SENSITIVITY is critical - we must catch all blockers.
"""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from benchmarks.data_loaders import load_herg
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

            if 'pchembl_median' in predictions:
                batch_scores = predictions['pchembl_median'].cpu().numpy().flatten()
            else:
                first_key = [k for k in predictions.keys() if k not in ['h_mol', 'h_mod']][0]
                batch_scores = predictions[first_key].cpu().numpy().flatten()

            scores.extend([float(s) for s in batch_scores])
            valid_indices.extend(batch_valid)

    return scores, valid_indices


def run_herg_benchmark(model, device, output_dir, batch_size=256):
    """Run hERG safety prediction benchmark."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("LOADING hERG DATA")
    print("="*70)

    herg_df = load_herg()

    print(f"\n{'='*70}")
    print("SCORING COMPOUNDS")
    print(f"{'='*70}")

    smiles_list = herg_df['smiles'].tolist()
    scores, valid_indices = score_compounds(model, smiles_list, device, batch_size)

    scored_df = herg_df.iloc[valid_indices].copy()
    scored_df['pred_score'] = scores

    # For hERG, higher pActivity = stronger blocker = more dangerous
    # So we use the score directly to predict blocking
    y_true = scored_df['is_blocker'].values
    y_score = scored_df['pred_score'].values

    print(f"\n{'='*70}")
    print("hERG CLASSIFICATION RESULTS")
    print(f"{'='*70}")

    # Calculate VS metrics (treating blockers as "actives")
    metrics = calculate_all_vs_metrics(y_true, y_score, name="hERG")
    print_metrics_summary(metrics, title="hERG Safety Prediction")

    # Classification at multiple thresholds
    print(f"\n{'='*60}")
    print("CLASSIFICATION AT DIFFERENT THRESHOLDS")
    print('='*60)

    results_by_threshold = {}

    for threshold_pct in [10, 20, 30, 50]:
        # Use percentile-based threshold
        threshold = np.percentile(y_score, 100 - threshold_pct)
        y_pred = (y_score >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for blockers
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for non-blockers
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-10)

        results_by_threshold[f'top_{threshold_pct}%'] = {
            'threshold': float(threshold),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'f1': float(f1),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }

        print(f"\nTop {threshold_pct}% as predicted blockers (threshold={threshold:.2f}):")
        print(f"  Sensitivity (catch blockers): {sensitivity*100:.1f}%")
        print(f"  Specificity (spare non-blockers): {specificity*100:.1f}%")
        print(f"  Precision: {precision*100:.1f}%")
        print(f"  F1: {f1:.3f}")
        print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Save results
    all_results = {
        'vs_metrics': metrics,
        'classification_thresholds': results_by_threshold
    }

    with open(output_dir / "herg_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    scored_df.to_parquet(output_dir / "herg_scored.parquet")

    # Generate figures
    generate_herg_figures(scored_df, metrics, results_by_threshold, output_dir)

    return all_results


def generate_herg_figures(scored_df, metrics, thresholds, output_dir):
    """Generate hERG visualization figures."""
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Score distribution by class
    ax = axes[0, 0]
    blocker_scores = scored_df[scored_df['is_blocker'] == 1]['pred_score']
    nonblocker_scores = scored_df[scored_df['is_blocker'] == 0]['pred_score']
    ax.hist(nonblocker_scores, bins=50, alpha=0.6, label=f'Non-blockers (n={len(nonblocker_scores)})', density=True)
    ax.hist(blocker_scores, bins=50, alpha=0.6, label=f'Blockers (n={len(blocker_scores)})', density=True)
    ax.set_xlabel('Predicted Score')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution by Class')
    ax.legend()

    # Sensitivity vs Specificity trade-off
    ax = axes[0, 1]
    sens = [thresholds[k]['sensitivity'] for k in sorted(thresholds.keys())]
    spec = [thresholds[k]['specificity'] for k in sorted(thresholds.keys())]
    labels = [k.replace('top_', '').replace('%', '') for k in sorted(thresholds.keys())]
    ax.plot(spec, sens, 'bo-', markersize=10)
    for i, label in enumerate(labels):
        ax.annotate(f'{label}%', (spec[i], sens[i]), textcoords='offset points', xytext=(5, 5))
    ax.set_xlabel('Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('Sensitivity-Specificity Trade-off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [1, 0], 'k--', alpha=0.3)
    ax.grid(True, alpha=0.3)

    # F1 by threshold
    ax = axes[1, 0]
    f1_scores = [thresholds[k]['f1'] for k in sorted(thresholds.keys())]
    x = range(len(labels))
    ax.bar(x, f1_scores, color='steelblue', alpha=0.8)
    ax.set_xlabel('Threshold (top X%)')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score by Threshold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}%' for l in labels])

    # Summary metrics
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
hERG Safety Prediction Summary
==============================

ROC-AUC:          {metrics['roc_auc']:.4f}
BEDROC (α=20):    {metrics['bedroc_20']:.4f}
Average Precision: {metrics['average_precision']:.4f}

Classification at Top 20%:
  Sensitivity:    {thresholds['top_20%']['sensitivity']*100:.1f}%
  Specificity:    {thresholds['top_20%']['specificity']*100:.1f}%
  F1 Score:       {thresholds['top_20%']['f1']:.3f}

Note: For safety, HIGH SENSITIVITY is critical
(catch all potential blockers)
"""
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output_dir / "herg_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFigures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='hERG Safety Benchmark for NEST-DRUG')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-path', type=str, default='data/external/herg/herg_tdc.csv')
    parser.add_argument('--output', type=str, default='results/benchmarks/herg')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']

    endpoint_names = []
    for key in state_dict.keys():
        if 'prediction_heads.heads.' in key and '.mlp.0.weight' in key:
            name = key.split('prediction_heads.heads.')[1].split('.mlp')[0]
            endpoint_names.append(name)

    endpoints = {name: {'type': 'regression', 'weight': 1.0} for name in endpoint_names}

    # Get model config from checkpoint if available, otherwise use defaults
    config = checkpoint.get('config', {})
    num_programs = config.get('n_programs', config.get('num_programs', 5))
    num_assays = config.get('n_assays', config.get('num_assays', 50))
    num_rounds = config.get('n_rounds', config.get('num_rounds', 150))

    model = create_nest_drug(
        num_programs=num_programs,
        num_assays=num_assays,
        num_rounds=num_rounds,
        endpoints=endpoints,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    run_herg_benchmark(model, device, args.output, args.batch_size)

    print("\n" + "="*70)
    print("hERG BENCHMARK COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
