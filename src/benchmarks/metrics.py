"""
Comprehensive Virtual Screening Metrics for NEST-DRUG Benchmarks

Includes:
- Enrichment Factor (EF) at multiple thresholds
- ROC-AUC and partial AUC
- BEDROC (early enrichment)
- Precision-Recall metrics
- AUAC (Area Under Accumulation Curve)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from scipy import stats


def calculate_bedroc(active_ranks, n_total, alpha=20.0):
    """
    Calculate BEDROC (Boltzmann-Enhanced Discrimination of ROC).

    BEDROC emphasizes early enrichment, which is what matters in practice.
    alpha=20.0 emphasizes top ~8% of ranked list
    alpha=80.5 emphasizes top ~2% (more stringent)

    Args:
        active_ranks: List of 0-indexed ranks where actives appear
        n_total: Total number of compounds
        alpha: Exponential decay parameter

    Returns:
        BEDROC score between 0 and 1
    """
    n_actives = len(active_ranks)
    if n_actives == 0:
        return 0.0

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


def calculate_enrichment_factors(y_true, y_score, percentages=[0.1, 0.5, 1, 2, 5, 10, 20]):
    """
    Calculate enrichment factors at multiple thresholds.

    Args:
        y_true: Binary labels (1=active, 0=inactive)
        y_score: Predicted scores (higher = more likely active)
        percentages: List of percentages to calculate EF at

    Returns:
        Dictionary with EF results at each percentage
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # Sort by score descending
    sorted_idx = np.argsort(-y_score)
    y_true_sorted = y_true[sorted_idx]

    n_total = len(y_true)
    n_actives = y_true.sum()
    baseline_rate = n_actives / n_total

    results = {}
    for pct in percentages:
        n_selected = max(1, int(n_total * pct / 100))
        hits = y_true_sorted[:n_selected].sum()
        hit_rate = hits / n_selected
        ef = hit_rate / baseline_rate if baseline_rate > 0 else 0

        # Maximum possible EF at this threshold
        max_hits = min(n_actives, n_selected)
        max_ef = (max_hits / n_selected) / baseline_rate if baseline_rate > 0 else 0

        results[f'EF_{pct}%'] = {
            'percentage': pct,
            'n_selected': n_selected,
            'hits': int(hits),
            'hit_rate': float(hit_rate),
            'enrichment_factor': float(ef),
            'max_possible_ef': float(max_ef),
            'normalized_ef': float(ef / max_ef) if max_ef > 0 else 0
        }

    return results


def calculate_roc_metrics(y_true, y_score):
    """
    Calculate ROC-AUC and partial AUC metrics.
    """
    y_true = np.array(y_true).astype(int)
    y_score = np.array(y_score)

    # ROC-AUC
    try:
        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
    except:
        return {'roc_auc': 0.5, 'partial_auc_10': 0.5, 'partial_auc_5': 0.5}

    # Partial AUC at 10% FPR
    idx_10 = np.searchsorted(fpr, 0.10)
    pauc_10 = np.trapz(tpr[:idx_10+1], fpr[:idx_10+1]) / 0.10 if idx_10 > 0 else 0

    # Partial AUC at 5% FPR
    idx_5 = np.searchsorted(fpr, 0.05)
    pauc_5 = np.trapz(tpr[:idx_5+1], fpr[:idx_5+1]) / 0.05 if idx_5 > 0 else 0

    return {
        'roc_auc': float(auc),
        'partial_auc_10': float(pauc_10),
        'partial_auc_5': float(pauc_5),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    }


def calculate_precision_recall_metrics(y_true, y_score):
    """
    Calculate precision-recall metrics.
    """
    y_true = np.array(y_true).astype(int)
    y_score = np.array(y_score)

    try:
        ap = average_precision_score(y_true, y_score)
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    except:
        return {'average_precision': 0.0, 'baseline_ap': 0.0, 'ap_lift': 0.0}

    # Baseline AP = prevalence
    baseline_ap = y_true.mean()

    # Best F1
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]

    return {
        'average_precision': float(ap),
        'baseline_ap': float(baseline_ap),
        'ap_lift': float(ap / baseline_ap) if baseline_ap > 0 else 0,
        'best_f1': float(best_f1),
        'precision': precision.tolist(),
        'recall': recall.tolist()
    }


def calculate_enrichment_curve(y_true, y_score):
    """
    Calculate enrichment curve data.
    X-axis: % of library screened
    Y-axis: % of actives found (recovery)
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # Sort by score descending
    sorted_idx = np.argsort(-y_score)
    y_true_sorted = y_true[sorted_idx]

    n_total = len(y_true)
    n_actives = int(y_true.sum())

    # Cumulative actives found
    cumsum = np.cumsum(y_true_sorted)

    # Normalize
    x = np.arange(1, n_total + 1) / n_total * 100  # % screened
    y = cumsum / n_actives * 100 if n_actives > 0 else cumsum  # % actives found

    # AUAC (Area Under Accumulation Curve)
    auac = np.trapz(y, x) / (100 * 100)

    # Recovery at key thresholds
    recovery_points = {}
    for pct in [0.1, 0.5, 1, 2, 5, 10, 20]:
        idx = max(0, int(n_total * pct / 100) - 1)
        recovery_points[f'recovery_{pct}%'] = float(y[idx])

    return {
        'auac': float(auac),
        'recovery_points': recovery_points,
        'n_actives': n_actives,
        'n_total': n_total
    }


def calculate_all_vs_metrics(y_true, y_score, name=""):
    """
    Calculate all virtual screening metrics.

    Args:
        y_true: Binary labels (1=active, 0=inactive)
        y_score: Predicted scores
        name: Optional name for the target

    Returns:
        Dictionary with all metrics
    """
    y_true = np.array(y_true).astype(int)
    y_score = np.array(y_score).astype(float)

    n_total = len(y_true)
    n_actives = int(y_true.sum())
    baseline_rate = n_actives / n_total if n_total > 0 else 0

    # Get sorted indices and active ranks
    sorted_idx = np.argsort(-y_score)
    y_true_sorted = y_true[sorted_idx]
    active_ranks = np.where(y_true_sorted == 1)[0].tolist()

    results = {
        'name': name,
        'n_total': n_total,
        'n_actives': n_actives,
        'baseline_rate': baseline_rate
    }

    # ROC metrics
    roc = calculate_roc_metrics(y_true, y_score)
    results['roc_auc'] = roc['roc_auc']
    results['partial_auc_10'] = roc['partial_auc_10']
    results['partial_auc_5'] = roc['partial_auc_5']

    # BEDROC
    results['bedroc_20'] = calculate_bedroc(active_ranks, n_total, alpha=20.0)
    results['bedroc_80'] = calculate_bedroc(active_ranks, n_total, alpha=80.5)
    results['rie_20'] = calculate_rie(active_ranks, n_total, alpha=20.0)

    # Precision-Recall
    pr = calculate_precision_recall_metrics(y_true, y_score)
    results['average_precision'] = pr['average_precision']
    results['ap_lift'] = pr['ap_lift']
    results['best_f1'] = pr['best_f1']

    # Enrichment factors
    ef = calculate_enrichment_factors(y_true, y_score)
    for key, data in ef.items():
        results[f'{key.lower()}_ef'] = data['enrichment_factor']
        results[f'{key.lower()}_hits'] = data['hits']
        results[f'{key.lower()}_hitrate'] = data['hit_rate']

    # Enrichment curve
    ec = calculate_enrichment_curve(y_true, y_score)
    results['auac'] = ec['auac']
    for key, val in ec['recovery_points'].items():
        results[key] = val

    return results


def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics for ADMET prediction.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Remove NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {}

    # Basic metrics
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    # Correlations
    pearson_r, pearson_p = stats.pearsonr(y_pred, y_true)
    spearman_r, spearman_p = stats.spearmanr(y_pred, y_true)

    return {
        'n_samples': len(y_true),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r)
    }


def print_metrics_summary(results, title="Metrics Summary"):
    """Pretty print metrics summary."""
    print(f"\n{'='*60}")
    print(title)
    print('='*60)

    if 'name' in results and results['name']:
        print(f"Target: {results['name']}")

    print(f"\nLibrary: {results.get('n_total', 'N/A'):,} compounds")
    print(f"Actives: {results.get('n_actives', 'N/A'):,}")
    print(f"Baseline: {results.get('baseline_rate', 0)*100:.2f}%")

    print(f"\nDiscrimination Metrics:")
    print(f"  ROC-AUC:      {results.get('roc_auc', 0):.4f}")
    print(f"  BEDROC (α=20): {results.get('bedroc_20', 0):.4f}")
    print(f"  Avg Precision: {results.get('average_precision', 0):.4f}")
    print(f"  AP Lift:       {results.get('ap_lift', 0):.1f}x")

    print(f"\nEnrichment Factors:")
    for pct in ['0.1', '1', '5', '10']:
        key = f'ef_{pct}%_ef'
        if key in results:
            print(f"  EF @ {pct}%:  {results[key]:.1f}x")

    print(f"\nRecovery:")
    for pct in ['1', '5', '10']:
        key = f'recovery_{pct}%'
        if key in results:
            print(f"  @ {pct}% screened: {results[key]:.1f}%")

    print(f"\nAUAC: {results.get('auac', 0):.4f}")
