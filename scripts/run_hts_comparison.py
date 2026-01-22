#!/usr/bin/env python3
"""
HTS Comparison Experiment - Comprehensive Metrics

Creates a realistic HTS-like library by diluting EGFR actives with decoys,
then tests NEST-DRUG's ability to enrich for actives.

Reports ALL gold-standard virtual screening metrics:
- Enrichment Factor (EF) at multiple thresholds
- ROC-AUC
- BEDROC (early enrichment)
- Precision-Recall / Average Precision
- Enrichment curves with visualization
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import requests
import random
import json
from datetime import datetime

# ML metrics
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

# Plotting
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.nest_drug import create_nest_drug
from training.data_utils import smiles_to_graph
from torch.cuda.amp import autocast


# =============================================================================
# COMPREHENSIVE METRICS FUNCTIONS
# =============================================================================

def calculate_bedroc(active_ranks, n_total, alpha=20.0):
    """
    Calculate BEDROC (Boltzmann-Enhanced Discrimination of ROC).

    BEDROC emphasizes early enrichment, which is what matters in practice.
    alpha=20.0 emphasizes top ~8% of ranked list
    alpha=80.5 emphasizes top ~2% (more stringent)

    Returns value between 0 and 1.
    """
    n_actives = len(active_ranks)
    if n_actives == 0:
        return 0.0

    # Convert to 0-indexed ranks if not already
    ranks = np.array(active_ranks)

    # Calculate sum of exponentials
    sum_exp = np.sum(np.exp(-alpha * ranks / n_total))

    # Random expectation
    ra = n_actives / n_total
    random_sum = ra * (1 - np.exp(-alpha)) / (np.exp(alpha / n_total) - 1)

    # Perfect enrichment (all actives at top)
    perfect_sum = (1 - np.exp(-alpha * n_actives / n_total)) / (1 - np.exp(-alpha / n_total))

    # BEDROC formula
    bedroc = (sum_exp - random_sum) / (perfect_sum - random_sum)

    return max(0, min(1, bedroc))


def calculate_rie(active_ranks, n_total, alpha=20.0):
    """
    Calculate RIE (Robust Initial Enhancement).
    Similar to BEDROC but unbounded.
    """
    n_actives = len(active_ranks)
    if n_actives == 0:
        return 0.0

    ranks = np.array(active_ranks)
    sum_exp = np.sum(np.exp(-alpha * ranks / n_total))

    # Random expectation
    ra = n_actives / n_total
    random_sum = ra * n_total * (1 - np.exp(-alpha)) / alpha

    rie = sum_exp / random_sum
    return rie


def calculate_enrichment_factors(library, score_col, active_col, percentages=[0.1, 0.5, 1, 2, 5, 10]):
    """
    Calculate enrichment factors at multiple thresholds.
    Standard reporting: EF_0.1%, EF_1%, EF_5%, EF_10%
    """
    library_sorted = library.sort_values(score_col, ascending=False).reset_index(drop=True)
    n_total = len(library)
    n_actives = int(library[active_col].sum())
    baseline_rate = n_actives / n_total

    results = {}
    for pct in percentages:
        n_selected = max(1, int(n_total * pct / 100))
        selected = library_sorted.head(n_selected)
        hits = int(selected[active_col].sum())
        hit_rate = hits / n_selected
        ef = hit_rate / baseline_rate if baseline_rate > 0 else 0

        # Maximum possible EF at this threshold
        max_hits = min(n_actives, n_selected)
        max_ef = (max_hits / n_selected) / baseline_rate if baseline_rate > 0 else 0

        results[f'EF_{pct}%'] = {
            'percentage': pct,
            'n_selected': n_selected,
            'hits': hits,
            'hit_rate': hit_rate,
            'enrichment_factor': ef,
            'max_possible_ef': max_ef,
            'normalized_ef': ef / max_ef if max_ef > 0 else 0
        }

    return results


def calculate_roc_metrics(library, score_col, active_col):
    """Calculate ROC-AUC and related metrics."""
    y_true = library[active_col].values.astype(int)
    y_score = library[score_col].values

    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Calculate partial AUC (focus on low FPR region)
    # pAUC at 10% FPR
    idx_10 = np.searchsorted(fpr, 0.10)
    pauc_10 = np.trapz(tpr[:idx_10+1], fpr[:idx_10+1]) / 0.10 if idx_10 > 0 else 0

    # pAUC at 5% FPR
    idx_5 = np.searchsorted(fpr, 0.05)
    pauc_5 = np.trapz(tpr[:idx_5+1], fpr[:idx_5+1]) / 0.05 if idx_5 > 0 else 0

    return {
        'roc_auc': auc,
        'partial_auc_10': pauc_10,
        'partial_auc_5': pauc_5,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }


def calculate_precision_recall(library, score_col, active_col):
    """
    Precision-Recall analysis - better for imbalanced datasets.
    """
    y_true = library[active_col].values.astype(int)
    y_score = library[score_col].values

    ap = average_precision_score(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # Baseline (random) AP = prevalence
    baseline_ap = y_true.mean()

    # F1 scores at each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else None

    return {
        'average_precision': ap,
        'baseline_ap': baseline_ap,
        'ap_lift': ap / baseline_ap if baseline_ap > 0 else 0,
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds
    }


def calculate_enrichment_curve(library, score_col, active_col):
    """
    Calculate enrichment curve data.
    X-axis: % of library screened
    Y-axis: % of actives found (recovery)
    """
    library_sorted = library.sort_values(score_col, ascending=False).reset_index(drop=True)
    n_total = len(library)
    n_actives = int(library[active_col].sum())

    # Calculate cumulative actives found
    cumsum = library_sorted[active_col].cumsum().values

    # Normalize
    x = np.arange(1, n_total + 1) / n_total * 100  # % screened
    y = cumsum / n_actives * 100  # % actives found (recovery)

    # Calculate Area Under Accumulation Curve (AUAC)
    auac = np.trapz(y, x) / (100 * 100)  # Normalized to 0-1

    # Random baseline AUAC = 0.5
    # Perfect AUAC depends on active rate

    # Key recovery points
    recovery_points = {}
    for pct in [0.1, 0.5, 1, 2, 5, 10, 20]:
        idx = max(0, int(n_total * pct / 100) - 1)
        recovery_points[f'recovery_{pct}%'] = float(y[idx])

    return {
        'x_pct_screened': x,
        'y_pct_recovered': y,
        'auac': auac,
        'recovery_points': recovery_points
    }


def calculate_hit_rate_curve(library, score_col, active_col, budgets=[50, 100, 200, 500, 1000, 2000, 5000, 10000]):
    """
    Report hit rate at multiple selection budgets.
    """
    library_sorted = library.sort_values(score_col, ascending=False)
    baseline_rate = library[active_col].mean()
    n_total = len(library)

    results = []
    for k in budgets:
        if k > n_total:
            continue
        selected = library_sorted.head(k)
        hits = int(selected[active_col].sum())
        hit_rate = hits / k
        ef = hit_rate / baseline_rate if baseline_rate > 0 else 0

        results.append({
            'budget': k,
            'hits': hits,
            'hit_rate': hit_rate,
            'enrichment_factor': ef,
            'pct_library': k / n_total * 100
        })

    return results


def run_comprehensive_analysis(scored_df, output_dir):
    """
    Run ALL comprehensive metrics and generate visualizations.
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE VIRTUAL SCREENING METRICS")
    print("=" * 80)

    score_col = 'predicted_score'
    active_col = 'is_active'

    n_total = len(scored_df)
    n_actives = int(scored_df[active_col].sum())
    baseline_rate = n_actives / n_total

    print(f"\nLibrary Statistics:")
    print(f"  Total compounds: {n_total:,}")
    print(f"  Total actives: {n_actives:,}")
    print(f"  Baseline active rate: {baseline_rate*100:.2f}%")

    results = {
        'library_stats': {
            'n_total': n_total,
            'n_actives': n_actives,
            'baseline_rate': baseline_rate
        }
    }

    # -------------------------------------------------------------------------
    # 1. ENRICHMENT FACTORS
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("1. ENRICHMENT FACTORS (EF)")
    print("-" * 60)

    ef_results = calculate_enrichment_factors(
        scored_df, score_col, active_col,
        percentages=[0.1, 0.5, 1, 2, 5, 10, 20]
    )
    results['enrichment_factors'] = ef_results

    print(f"\n{'Threshold':<12} {'Selected':>10} {'Hits':>8} {'Hit Rate':>10} {'EF':>8} {'Max EF':>8} {'Norm EF':>8}")
    print("-" * 74)
    for key, data in ef_results.items():
        print(f"{key:<12} {data['n_selected']:>10,} {data['hits']:>8,} {data['hit_rate']*100:>9.1f}% {data['enrichment_factor']:>8.1f} {data['max_possible_ef']:>8.1f} {data['normalized_ef']:>8.2f}")

    # -------------------------------------------------------------------------
    # 2. ROC-AUC
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("2. ROC-AUC (Receiver Operating Characteristic)")
    print("-" * 60)

    roc_results = calculate_roc_metrics(scored_df, score_col, active_col)
    results['roc'] = {
        'roc_auc': roc_results['roc_auc'],
        'partial_auc_10': roc_results['partial_auc_10'],
        'partial_auc_5': roc_results['partial_auc_5']
    }

    print(f"\n  ROC-AUC:          {roc_results['roc_auc']:.4f}")
    print(f"  Partial AUC @10%: {roc_results['partial_auc_10']:.4f}")
    print(f"  Partial AUC @5%:  {roc_results['partial_auc_5']:.4f}")

    auc_interpretation = "Outstanding" if roc_results['roc_auc'] > 0.95 else \
                         "Excellent" if roc_results['roc_auc'] > 0.90 else \
                         "Good" if roc_results['roc_auc'] > 0.80 else \
                         "Acceptable" if roc_results['roc_auc'] > 0.70 else "Poor"
    print(f"  Interpretation:   {auc_interpretation}")

    # -------------------------------------------------------------------------
    # 3. BEDROC (Early Enrichment)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("3. BEDROC (Boltzmann-Enhanced Discrimination of ROC)")
    print("-" * 60)

    # Get active ranks
    library_sorted = scored_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    active_ranks = library_sorted[library_sorted[active_col] == 1].index.tolist()

    bedroc_20 = calculate_bedroc(active_ranks, n_total, alpha=20.0)
    bedroc_80 = calculate_bedroc(active_ranks, n_total, alpha=80.5)
    rie_20 = calculate_rie(active_ranks, n_total, alpha=20.0)

    results['bedroc'] = {
        'bedroc_alpha_20': bedroc_20,
        'bedroc_alpha_80.5': bedroc_80,
        'rie_alpha_20': rie_20
    }

    print(f"\n  BEDROC (α=20.0):   {bedroc_20:.4f}  (emphasizes top ~8%)")
    print(f"  BEDROC (α=80.5):   {bedroc_80:.4f}  (emphasizes top ~2%)")
    print(f"  RIE (α=20.0):      {rie_20:.4f}")

    bedroc_interpretation = "Excellent" if bedroc_20 > 0.8 else \
                            "Good" if bedroc_20 > 0.6 else \
                            "Acceptable" if bedroc_20 > 0.4 else "Poor"
    print(f"  Interpretation:    {bedroc_interpretation}")

    # -------------------------------------------------------------------------
    # 4. PRECISION-RECALL
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("4. PRECISION-RECALL ANALYSIS")
    print("-" * 60)

    pr_results = calculate_precision_recall(scored_df, score_col, active_col)
    results['precision_recall'] = {
        'average_precision': pr_results['average_precision'],
        'baseline_ap': pr_results['baseline_ap'],
        'ap_lift': pr_results['ap_lift'],
        'best_f1': pr_results['best_f1']
    }

    print(f"\n  Average Precision: {pr_results['average_precision']:.4f}")
    print(f"  Baseline AP:       {pr_results['baseline_ap']:.4f}")
    print(f"  AP Lift:           {pr_results['ap_lift']:.1f}x better than random")
    print(f"  Best F1 Score:     {pr_results['best_f1']:.4f}")

    # -------------------------------------------------------------------------
    # 5. ENRICHMENT CURVE
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("5. ENRICHMENT CURVE (Recovery Analysis)")
    print("-" * 60)

    ec_results = calculate_enrichment_curve(scored_df, score_col, active_col)
    results['enrichment_curve'] = {
        'auac': ec_results['auac'],
        'recovery_points': ec_results['recovery_points']
    }

    print(f"\n  AUAC (Area Under Accumulation Curve): {ec_results['auac']:.4f}")
    print(f"  Random baseline AUAC: 0.5000")
    print(f"\n  Recovery at key thresholds:")
    for key, val in ec_results['recovery_points'].items():
        pct = key.replace('recovery_', '').replace('%', '')
        print(f"    Screen {pct:>4}% → Find {val:>5.1f}% of actives")

    # -------------------------------------------------------------------------
    # 6. HIT RATE CURVE
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("6. HIT RATE AT SELECTION BUDGETS")
    print("-" * 60)

    hr_results = calculate_hit_rate_curve(
        scored_df, score_col, active_col,
        budgets=[50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    )
    results['hit_rate_curve'] = hr_results

    print(f"\n{'Budget':>10} {'Hits':>8} {'Hit Rate':>10} {'EF':>8} {'% Library':>10}")
    print("-" * 50)
    for r in hr_results:
        print(f"{r['budget']:>10,} {r['hits']:>8,} {r['hit_rate']*100:>9.1f}% {r['enrichment_factor']:>8.1f} {r['pct_library']:>9.2f}%")

    # -------------------------------------------------------------------------
    # 7. GENERATE VISUALIZATIONS
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("7. GENERATING VISUALIZATIONS")
    print("-" * 60)

    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: ROC Curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(roc_results['fpr'], roc_results['tpr'], 'b-', linewidth=2,
             label=f'NEST-DRUG (AUC={roc_results["roc_auc"]:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.500)')
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Plot 2: ROC Curve (Zoomed to low FPR)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(roc_results['fpr'], roc_results['tpr'], 'b-', linewidth=2, label='NEST-DRUG')
    ax2.plot([0, 0.1], [0, 0.1], 'k--', linewidth=1, label='Random')
    ax2.set_xlabel('False Positive Rate', fontsize=11)
    ax2.set_ylabel('True Positive Rate', fontsize=11)
    ax2.set_title('ROC Curve (Low FPR Region)', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_xlim(0, 0.1)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Precision-Recall Curve
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(pr_results['recall'], pr_results['precision'], 'g-', linewidth=2,
             label=f'NEST-DRUG (AP={pr_results["average_precision"]:.3f})')
    ax3.axhline(y=pr_results['baseline_ap'], color='k', linestyle='--',
                label=f'Random (AP={pr_results["baseline_ap"]:.3f})')
    ax3.set_xlabel('Recall', fontsize=11)
    ax3.set_ylabel('Precision', fontsize=11)
    ax3.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Enrichment Curve
    ax4 = fig.add_subplot(gs[1, 0])
    x_ec = ec_results['x_pct_screened']
    y_ec = ec_results['y_pct_recovered']
    ax4.plot(x_ec, y_ec, 'b-', linewidth=2, label=f'NEST-DRUG (AUAC={ec_results["auac"]:.3f})')
    ax4.plot([0, 100], [0, 100], 'k--', linewidth=1, label='Random (AUAC=0.500)')
    # Perfect enrichment line
    perfect_x = [0, baseline_rate * 100, 100]
    perfect_y = [0, 100, 100]
    ax4.plot(perfect_x, perfect_y, 'g:', linewidth=1, label='Perfect')
    ax4.set_xlabel('% of Library Screened', fontsize=11)
    ax4.set_ylabel('% of Actives Found', fontsize=11)
    ax4.set_title('Enrichment Curve (Recovery)', fontsize=12, fontweight='bold')
    ax4.legend(loc='lower right')
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3)
    # Add key points
    for pct in [1, 5, 10]:
        idx = max(0, int(n_total * pct / 100) - 1)
        recovery = y_ec[idx]
        ax4.scatter([pct], [recovery], s=50, c='red', zorder=5)
        ax4.annotate(f'{recovery:.0f}%', (pct, recovery), textcoords='offset points',
                     xytext=(5, 5), fontsize=9)

    # Plot 5: Enrichment Curve (Zoomed to early)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(x_ec, y_ec, 'b-', linewidth=2, label='NEST-DRUG')
    ax5.plot([0, 10], [0, 10], 'k--', linewidth=1, label='Random')
    ax5.set_xlabel('% of Library Screened', fontsize=11)
    ax5.set_ylabel('% of Actives Found', fontsize=11)
    ax5.set_title('Enrichment Curve (Top 10%)', fontsize=12, fontweight='bold')
    ax5.legend(loc='lower right')
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 100)
    ax5.grid(True, alpha=0.3)
    for pct in [0.5, 1, 2, 5]:
        idx = max(0, int(n_total * pct / 100) - 1)
        recovery = y_ec[idx]
        ax5.scatter([pct], [recovery], s=50, c='red', zorder=5)
        ax5.annotate(f'{recovery:.0f}%', (pct, recovery), textcoords='offset points',
                     xytext=(5, 5), fontsize=9)

    # Plot 6: Enrichment Factor Bar Chart
    ax6 = fig.add_subplot(gs[1, 2])
    ef_keys = list(ef_results.keys())
    ef_values = [ef_results[k]['enrichment_factor'] for k in ef_keys]
    ef_max = [ef_results[k]['max_possible_ef'] for k in ef_keys]
    x_pos = np.arange(len(ef_keys))
    bars = ax6.bar(x_pos, ef_values, color='steelblue', alpha=0.8, label='Achieved EF')
    ax6.bar(x_pos, ef_max, color='lightgray', alpha=0.5, label='Max Possible EF')
    ax6.bar(x_pos, ef_values, color='steelblue', alpha=0.8)  # Redraw on top
    ax6.set_xlabel('Threshold', fontsize=11)
    ax6.set_ylabel('Enrichment Factor', fontsize=11)
    ax6.set_title('Enrichment Factors at Thresholds', fontsize=12, fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([k.replace('EF_', '') for k in ef_keys], rotation=45)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for i, v in enumerate(ef_values):
        ax6.text(i, v + 1, f'{v:.0f}x', ha='center', fontsize=9)

    # Plot 7: Hit Rate vs Budget
    ax7 = fig.add_subplot(gs[2, 0])
    budgets = [r['budget'] for r in hr_results]
    hit_rates = [r['hit_rate'] * 100 for r in hr_results]
    ax7.semilogx(budgets, hit_rates, 'bo-', linewidth=2, markersize=8)
    ax7.axhline(y=baseline_rate * 100, color='k', linestyle='--', label=f'Baseline ({baseline_rate*100:.1f}%)')
    ax7.set_xlabel('Selection Budget (compounds)', fontsize=11)
    ax7.set_ylabel('Hit Rate (%)', fontsize=11)
    ax7.set_title('Hit Rate vs Selection Budget', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    for i, (b, hr) in enumerate(zip(budgets, hit_rates)):
        ax7.annotate(f'{hr:.0f}%', (b, hr), textcoords='offset points',
                     xytext=(0, 8), fontsize=9, ha='center')

    # Plot 8: Score Distribution
    ax8 = fig.add_subplot(gs[2, 1])
    active_scores = scored_df[scored_df[active_col] == 1][score_col]
    inactive_scores = scored_df[scored_df[active_col] == 0][score_col]
    ax8.hist(inactive_scores, bins=50, alpha=0.6, color='gray', label=f'Inactives (n={len(inactive_scores):,})', density=True)
    ax8.hist(active_scores, bins=50, alpha=0.6, color='green', label=f'Actives (n={len(active_scores):,})', density=True)
    ax8.set_xlabel('Predicted Score', fontsize=11)
    ax8.set_ylabel('Density', fontsize=11)
    ax8.set_title('Score Distribution by Class', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Plot 9: Summary Statistics Table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    summary_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['─' * 20, '─' * 15, '─' * 15],
        ['ROC-AUC', f'{roc_results["roc_auc"]:.4f}', auc_interpretation],
        ['BEDROC (α=20)', f'{bedroc_20:.4f}', bedroc_interpretation],
        ['Average Precision', f'{pr_results["average_precision"]:.4f}', f'{pr_results["ap_lift"]:.0f}x lift'],
        ['AUAC', f'{ec_results["auac"]:.4f}', 'vs 0.5 random'],
        ['─' * 20, '─' * 15, '─' * 15],
        ['EF @ 0.1%', f'{ef_results["EF_0.1%"]["enrichment_factor"]:.1f}x', ''],
        ['EF @ 1%', f'{ef_results["EF_1%"]["enrichment_factor"]:.1f}x', ''],
        ['EF @ 5%', f'{ef_results["EF_5%"]["enrichment_factor"]:.1f}x', ''],
        ['─' * 20, '─' * 15, '─' * 15],
        ['Recovery @ 1%', f'{ec_results["recovery_points"]["recovery_1%"]:.1f}%', ''],
        ['Recovery @ 5%', f'{ec_results["recovery_points"]["recovery_5%"]:.1f}%', ''],
        ['Recovery @ 10%', f'{ec_results["recovery_points"]["recovery_10%"]:.1f}%', ''],
    ]

    table_text = '\n'.join([f'{row[0]:<22} {row[1]:<15} {row[2]:<15}' for row in summary_data])
    ax9.text(0.1, 0.95, 'SUMMARY METRICS', fontsize=14, fontweight='bold',
             transform=ax9.transAxes, verticalalignment='top', fontfamily='monospace')
    ax9.text(0.1, 0.85, table_text, fontsize=10,
             transform=ax9.transAxes, verticalalignment='top', fontfamily='monospace')

    # Overall title
    fig.suptitle('NEST-DRUG HTS Comparison: Comprehensive Virtual Screening Metrics',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    fig_path = output_dir / 'comprehensive_metrics.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Saved: {fig_path}")

    # Also save individual high-res figures
    # ROC curve
    fig_roc, ax = plt.subplots(figsize=(8, 6))
    ax.plot(roc_results['fpr'], roc_results['tpr'], 'b-', linewidth=2.5,
            label=f'NEST-DRUG (AUC = {roc_results["roc_auc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - NEST-DRUG HTS Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'roc_curve.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'roc_curve.png'}")

    # Enrichment curve
    fig_ec, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_ec, y_ec, 'b-', linewidth=2.5, label=f'NEST-DRUG (AUAC = {ec_results["auac"]:.3f})')
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1.5, label='Random (AUAC = 0.500)')
    ax.plot(perfect_x, perfect_y, 'g:', linewidth=1.5, label='Perfect')
    ax.set_xlabel('% of Library Screened', fontsize=12)
    ax.set_ylabel('% of Actives Found (Recovery)', fontsize=12)
    ax.set_title('Enrichment Curve - NEST-DRUG HTS Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    for pct in [1, 5, 10, 20]:
        idx = max(0, int(n_total * pct / 100) - 1)
        recovery = y_ec[idx]
        ax.scatter([pct], [recovery], s=80, c='red', zorder=5)
        ax.annotate(f'{recovery:.0f}%', (pct, recovery), textcoords='offset points',
                    xytext=(8, 5), fontsize=10, fontweight='bold')
    plt.savefig(output_dir / 'enrichment_curve.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'enrichment_curve.png'}")

    # -------------------------------------------------------------------------
    # 8. PRINT FINAL SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│                    NEST-DRUG HTS COMPARISON RESULTS                      │
├─────────────────────────────────────────────────────────────────────────┤
│  Library: {n_total:,} compounds ({n_actives:,} actives, {baseline_rate*100:.2f}% baseline)      │
├─────────────────────────────────────────────────────────────────────────┤
│  DISCRIMINATION METRICS                                                  │
│    ROC-AUC:           {roc_results['roc_auc']:.4f}  ({auc_interpretation})                          │
│    BEDROC (α=20):     {bedroc_20:.4f}  ({bedroc_interpretation})                          │
│    Average Precision: {pr_results['average_precision']:.4f}  ({pr_results['ap_lift']:.0f}x better than random)             │
├─────────────────────────────────────────────────────────────────────────┤
│  ENRICHMENT FACTORS                                                      │
│    EF @ 0.1%:  {ef_results['EF_0.1%']['enrichment_factor']:>6.1f}x  (top {ef_results['EF_0.1%']['n_selected']:,} compounds)                   │
│    EF @ 1%:    {ef_results['EF_1%']['enrichment_factor']:>6.1f}x  (top {ef_results['EF_1%']['n_selected']:,} compounds)                   │
│    EF @ 5%:    {ef_results['EF_5%']['enrichment_factor']:>6.1f}x  (top {ef_results['EF_5%']['n_selected']:,} compounds)                  │
│    EF @ 10%:   {ef_results['EF_10%']['enrichment_factor']:>6.1f}x  (top {ef_results['EF_10%']['n_selected']:,} compounds)                 │
├─────────────────────────────────────────────────────────────────────────┤
│  RECOVERY (% of actives found)                                          │
│    @ 1% screened:   {ec_results['recovery_points']['recovery_1%']:>5.1f}%                                          │
│    @ 5% screened:   {ec_results['recovery_points']['recovery_5%']:>5.1f}%                                          │
│    @ 10% screened:  {ec_results['recovery_points']['recovery_10%']:>5.1f}%                                          │
└─────────────────────────────────────────────────────────────────────────┘
""")

    return results


def download_zinc_250k():
    """Download ZINC 250K drug-like compounds."""
    cache_path = Path("data/decoys/zinc_250k.csv")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Loading cached ZINC data from {cache_path}")
        df = pd.read_csv(cache_path)
        return df['smiles'].tolist()

    print("Downloading ZINC 250K dataset...")
    url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"

    try:
        df = pd.read_csv(url)
        df.to_csv(cache_path, index=False)
        print(f"Downloaded {len(df)} ZINC compounds")
        return df['smiles'].tolist()
    except Exception as e:
        print(f"Failed to download ZINC: {e}")
        return None


def download_chembl_decoys():
    """Download ChEMBL-derived decoys from MoleculeNet."""
    cache_path = Path("data/decoys/chembl_decoys.csv")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Loading cached ChEMBL decoys from {cache_path}")
        df = pd.read_csv(cache_path)
        return df['smiles'].tolist()

    print("Downloading MoleculeNet datasets for decoys...")
    all_smiles = set()

    # Multiple MoleculeNet datasets with many compounds
    datasets = [
        ("HIV", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"),
        ("BBBP", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"),
        ("Tox21", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"),
        ("SIDER", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz"),
        ("ClinTox", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"),
        ("BACE", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"),
    ]

    for name, url in datasets:
        try:
            print(f"  Downloading {name}...")
            if url.endswith('.gz'):
                df = pd.read_csv(url, compression='gzip')
            else:
                df = pd.read_csv(url)

            # Find SMILES column
            smiles_col = None
            for col in df.columns:
                if 'smiles' in col.lower():
                    smiles_col = col
                    break
            if smiles_col is None and 'mol' in df.columns[0].lower():
                smiles_col = df.columns[0]

            if smiles_col:
                smiles = df[smiles_col].dropna().tolist()
                all_smiles.update(smiles)
                print(f"    Got {len(smiles)} compounds from {name}")
        except Exception as e:
            print(f"    Failed to download {name}: {e}")

    # Save cache
    if all_smiles:
        pd.DataFrame({'smiles': list(all_smiles)}).to_csv(cache_path, index=False)
        print(f"Downloaded {len(all_smiles)} ChEMBL/MoleculeNet compounds total")

    return list(all_smiles)


def download_pubchem_decoys():
    """Download PubChem diverse compounds."""
    cache_path = Path("data/decoys/pubchem_decoys.csv")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Loading cached PubChem decoys from {cache_path}")
        df = pd.read_csv(cache_path)
        return df['smiles'].tolist()

    print("Downloading GDB-13 subset for additional decoys...")
    # GDB-13 sample - small molecules that are definitely not EGFR actives
    url = "https://raw.githubusercontent.com/GLambard/Molecules_Dataset_Collection/master/latest/ESOL.csv"

    try:
        df = pd.read_csv(url)
        smiles_col = [c for c in df.columns if 'smiles' in c.lower()][0] if any('smiles' in c.lower() for c in df.columns) else df.columns[0]
        smiles = df[smiles_col].dropna().tolist()
        pd.DataFrame({'smiles': smiles}).to_csv(cache_path, index=False)
        print(f"Downloaded {len(smiles)} additional decoy compounds")
        return smiles
    except Exception as e:
        print(f"Failed to download additional decoys: {e}")
        return []


def generate_random_decoys(n_decoys, seed=42):
    """
    Generate random drug-like SMILES as decoys.
    Uses simple molecular templates that are unlikely to be EGFR actives.
    """
    print(f"Generating {n_decoys} random decoy molecules...")
    random.seed(seed)
    np.random.seed(seed)

    # Common drug-like scaffolds (unlikely to be kinase inhibitors)
    templates = [
        # Simple alkanes/alcohols
        "CCCCCCCC", "CCCCCCCCC", "CCCCCCCCCC",
        "CCCCCCO", "CCCCCCCO", "CCCCCCCCO",
        # Simple aromatics
        "c1ccccc1", "c1ccc(C)cc1", "c1ccc(O)cc1", "c1ccc(N)cc1",
        "c1ccc(F)cc1", "c1ccc(Cl)cc1", "c1ccc(Br)cc1",
        # Simple heterocycles
        "c1ccncc1", "c1ccoc1", "c1ccsc1", "c1cnc2ccccc2n1",
        # Esters/amides
        "CCOC(=O)CC", "CC(=O)NCC", "CCNC(=O)CC",
        # Ethers
        "CCOCC", "CCOCCOC", "c1ccc(OCC)cc1",
    ]

    # Substituents to add variety
    substituents = [
        "", "C", "CC", "CCC", "C(C)C", "C(C)(C)C",
        "O", "OC", "OCC", "N", "NC", "NCC", "N(C)C",
        "F", "Cl", "Br", "C#N", "C(=O)O", "C(=O)N",
        "c1ccccc1", "c1ccncc1", "c1ccc(O)cc1",
    ]

    decoys = set()
    attempts = 0
    max_attempts = n_decoys * 10

    while len(decoys) < n_decoys and attempts < max_attempts:
        template = random.choice(templates)
        n_subs = random.randint(0, 3)

        smiles = template
        for _ in range(n_subs):
            sub = random.choice(substituents)
            if sub:
                # Simple concatenation (not chemically rigorous but fast)
                if random.random() > 0.5:
                    smiles = smiles + sub
                else:
                    smiles = sub + smiles

        # Validate SMILES
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and 5 <= mol.GetNumAtoms() <= 50:
                canonical = Chem.MolToSmiles(mol)
                decoys.add(canonical)
        except:
            pass

        attempts += 1

        if len(decoys) % 10000 == 0 and len(decoys) > 0:
            print(f"  Generated {len(decoys)} valid decoys...")

    print(f"Generated {len(decoys)} valid decoy molecules")
    return list(decoys)


def create_hts_library(actives_path, target_active_rate=0.01, max_decoys=None, max_actives=None):
    """
    Create HTS-like library by diluting actives with decoys.

    Args:
        actives_path: Path to EGFR actives CSV
        target_active_rate: Desired fraction of actives (default 1%)
        max_decoys: Maximum number of decoys (for faster testing)
        max_actives: Maximum number of actives to use (to achieve target rate with limited decoys)

    Returns:
        DataFrame with combined library
    """
    print("=" * 70)
    print("CREATING HTS-COMPARABLE LIBRARY")
    print("=" * 70)

    # Load actives
    print(f"\nLoading actives from {actives_path}")
    actives_df = pd.read_csv(actives_path)

    # Define activity threshold (pActivity >= 6.0 = IC50 < 1 µM)
    activity_threshold = 6.0
    actives_df['is_active'] = (actives_df['pActivity'] >= activity_threshold).astype(int)

    n_actives = actives_df['is_active'].sum()
    n_total = len(actives_df)
    original_rate = n_actives / n_total

    print(f"  Total compounds: {n_total}")
    print(f"  Actives (pActivity >= {activity_threshold}): {n_actives}")
    print(f"  Original active rate: {original_rate*100:.1f}%")

    # Optionally subsample actives to achieve target rate with available decoys
    if max_actives and n_actives > max_actives:
        print(f"\n  Subsampling to {max_actives} actives to achieve target rate")
        # Keep all inactives, sample actives
        active_mask = actives_df['is_active'] == 1
        active_rows = actives_df[active_mask].sample(n=max_actives, random_state=42)
        inactive_rows = actives_df[~active_mask]
        actives_df = pd.concat([active_rows, inactive_rows], ignore_index=True)
        n_actives = max_actives
        n_total = len(actives_df)
        print(f"  New total: {n_total}, actives: {n_actives}")

    # Calculate decoys needed
    n_total_needed = int(n_actives / target_active_rate)
    n_decoys_needed = n_total_needed - n_total

    if max_decoys:
        n_decoys_needed = min(n_decoys_needed, max_decoys)
        n_total_needed = n_total + n_decoys_needed

    print(f"\nTarget active rate: {target_active_rate*100:.1f}%")
    print(f"Decoys needed: {n_decoys_needed:,}")
    print(f"Total library size: {n_total_needed:,}")

    # Collect decoys from multiple sources
    print("\n--- Collecting Decoy Compounds ---")
    all_decoy_smiles = set()

    # Source 1: ZINC 250K
    zinc_smiles = download_zinc_250k()
    if zinc_smiles:
        all_decoy_smiles.update(zinc_smiles)
        print(f"  ZINC 250K: {len(zinc_smiles)} compounds")

    # Source 2: MoleculeNet datasets (ChEMBL-derived)
    chembl_smiles = download_chembl_decoys()
    if chembl_smiles:
        all_decoy_smiles.update(chembl_smiles)
        print(f"  MoleculeNet: {len(chembl_smiles)} compounds")

    # Source 3: Additional datasets
    pubchem_smiles = download_pubchem_decoys()
    if pubchem_smiles:
        all_decoy_smiles.update(pubchem_smiles)
        print(f"  Additional: {len(pubchem_smiles)} compounds")

    # Remove any that might overlap with actives
    active_smiles_set = set(actives_df['smiles'].tolist())
    all_decoy_smiles = all_decoy_smiles - active_smiles_set

    print(f"\n  Total unique decoys available: {len(all_decoy_smiles):,}")

    # Check if we have enough
    if len(all_decoy_smiles) >= n_decoys_needed:
        print(f"  Using {n_decoys_needed:,} decoys from downloaded sources")
        decoy_smiles = random.sample(list(all_decoy_smiles), n_decoys_needed)
    else:
        # Generate additional random decoys
        remaining = n_decoys_needed - len(all_decoy_smiles)
        print(f"  Need {remaining:,} more decoys, generating random molecules...")

        generated = generate_random_decoys(remaining)
        decoy_smiles = list(all_decoy_smiles) + generated
        decoy_smiles = decoy_smiles[:n_decoys_needed]

    # Create decoy dataframe
    decoy_df = pd.DataFrame({
        'smiles': decoy_smiles,
        'pActivity': 4.0,  # Inactive (IC50 > 100 µM)
        'is_active': 0,
        'source': 'decoy',
        'program_id': 0,
        'assay_id': 0,
        'round_id': 0,
    })

    # Prepare actives dataframe
    actives_df['source'] = 'egfr_active'
    if 'program_id' not in actives_df.columns:
        actives_df['program_id'] = 0
    if 'assay_id' not in actives_df.columns:
        actives_df['assay_id'] = 0
    if 'round_id' not in actives_df.columns:
        actives_df['round_id'] = 0

    # Select common columns
    common_cols = ['smiles', 'pActivity', 'is_active', 'source', 'program_id', 'assay_id', 'round_id']
    actives_subset = actives_df[common_cols].copy()

    # Combine and shuffle
    combined = pd.concat([actives_subset, decoy_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    actual_rate = combined['is_active'].mean()
    print(f"\n✓ Combined library created:")
    print(f"  Total compounds: {len(combined):,}")
    print(f"  Actives: {combined['is_active'].sum():,}")
    print(f"  Actual active rate: {actual_rate*100:.2f}%")

    return combined


def score_library(model, library_df, device, batch_size=256):
    """
    Score all compounds in library using NEST-DRUG.
    """
    print("\n" + "=" * 70)
    print("SCORING LIBRARY WITH NEST-DRUG")
    print("=" * 70)

    model.eval()
    scores = []
    valid_indices = []

    smiles_list = library_df['smiles'].tolist()

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

            # Context IDs (all zeros for scoring)
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
                # Fallback to first endpoint
                first_key = [k for k in predictions.keys() if k not in ['h_mol', 'h_mod']][0]
                batch_scores = predictions[first_key].cpu().numpy().flatten()

            scores.extend(batch_scores)
            valid_indices.extend(batch_valid)

    # Create scored dataframe
    scored_df = library_df.iloc[valid_indices].copy()
    # Ensure scores are proper floats (fix pandas nlargest type error)
    scored_df['predicted_score'] = [float(s) for s in scores]

    print(f"\n✓ Scored {len(scored_df):,} / {len(library_df):,} compounds")
    print(f"  Score range: {min(scores):.2f} - {max(scores):.2f}")

    return scored_df


def run_selection_experiment(scored_df, selection_budgets=[100, 500, 1000, 5000]):
    """
    Run compound selection at various budgets and calculate enrichment.
    """
    print("\n" + "=" * 70)
    print("SELECTION EXPERIMENT RESULTS")
    print("=" * 70)

    baseline_rate = scored_df['is_active'].mean()
    print(f"\nBaseline (random) hit rate: {baseline_rate*100:.2f}%")

    results = []

    for budget in selection_budgets:
        if budget > len(scored_df):
            continue

        # Select top-K by predicted score
        selected = scored_df.nlargest(budget, 'predicted_score')

        n_hits = selected['is_active'].sum()
        hit_rate = n_hits / budget
        enrichment = hit_rate / baseline_rate if baseline_rate > 0 else 0

        result = {
            'budget': budget,
            'hits': n_hits,
            'hit_rate': hit_rate,
            'enrichment': enrichment,
        }
        results.append(result)

        print(f"\nBudget: {budget:,} compounds")
        print(f"  Hits found: {n_hits:,}")
        print(f"  Hit rate: {hit_rate*100:.1f}%")
        print(f"  Enrichment factor: {enrichment:.1f}x")

        if enrichment >= 50:
            print(f"  ✓ Claim supported: >{int(enrichment)}x better than random!")

    return results


def main():
    parser = argparse.ArgumentParser(description='HTS Comparison Experiment')
    parser.add_argument('--actives', type=str,
                       default='data/processed/programs/program_egfr_augmented.csv',
                       help='Path to EGFR actives')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/pretrain/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--target-rate', type=float, default=0.01,
                       help='Target active rate (default: 0.01 = 1%%)')
    parser.add_argument('--max-decoys', type=int, default=None,
                       help='Maximum decoys (for faster testing)')
    parser.add_argument('--max-actives', type=int, default=None,
                       help='Maximum actives to use (subsample to achieve target rate with limited decoys)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU to use')
    parser.add_argument('--output', type=str, default='results/hts_comparison',
                       help='Output directory')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create HTS-like library
    library_df = create_hts_library(
        actives_path=args.actives,
        target_active_rate=args.target_rate,
        max_decoys=args.max_decoys,
        max_actives=args.max_actives,
    )

    # Load model
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

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

    # Score library
    scored_df = score_library(model, library_df, device)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run COMPREHENSIVE analysis with all metrics
    results = run_comprehensive_analysis(scored_df, output_dir)

    # Save scored library (convert mixed-type columns to string for parquet compatibility)
    save_df = scored_df.copy()
    for col in ['program_id', 'assay_id', 'source']:
        if col in save_df.columns:
            save_df[col] = save_df[col].astype(str)
    save_df.to_parquet(output_dir / 'scored_library.parquet')

    # Save comprehensive results as JSON
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    results_serializable = convert_to_serializable(results)
    with open(output_dir / 'comprehensive_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)

    # Save enrichment factors as CSV
    ef_data = []
    for key, data in results['enrichment_factors'].items():
        ef_data.append({
            'threshold': key,
            'percentage': data['percentage'],
            'n_selected': data['n_selected'],
            'hits': data['hits'],
            'hit_rate': data['hit_rate'],
            'enrichment_factor': data['enrichment_factor'],
            'max_possible_ef': data['max_possible_ef'],
            'normalized_ef': data['normalized_ef']
        })
    pd.DataFrame(ef_data).to_csv(output_dir / 'enrichment_factors.csv', index=False)

    # Save hit rate curve as CSV
    pd.DataFrame(results['hit_rate_curve']).to_csv(output_dir / 'hit_rate_curve.csv', index=False)

    # Print final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to {output_dir}/")
    print(f"  - comprehensive_metrics.png (9-panel visualization)")
    print(f"  - roc_curve.png (high-res ROC curve)")
    print(f"  - enrichment_curve.png (high-res enrichment curve)")
    print(f"  - comprehensive_results.json (all metrics)")
    print(f"  - enrichment_factors.csv (EF at all thresholds)")
    print(f"  - hit_rate_curve.csv (hit rates at budgets)")
    print(f"  - scored_library.parquet (full scored library)")

    # Final assessment
    ef_1pct = results['enrichment_factors']['EF_1%']['enrichment_factor']
    roc_auc = results['roc']['roc_auc']
    bedroc = results['bedroc']['bedroc_alpha_20']

    print("\n" + "=" * 80)
    print("FINAL ASSESSMENT")
    print("=" * 80)

    if ef_1pct >= 50 and roc_auc >= 0.90 and bedroc >= 0.80:
        print("\n✓ OUTSTANDING: All metrics exceed state-of-the-art benchmarks!")
    elif ef_1pct >= 30 and roc_auc >= 0.85 and bedroc >= 0.60:
        print("\n✓ EXCELLENT: Strong performance across all metrics")
    elif ef_1pct >= 20 and roc_auc >= 0.80:
        print("\n✓ GOOD: Solid virtual screening performance")
    else:
        print(f"\n⚠ Results below target - consider model improvements")


if __name__ == '__main__':
    main()
