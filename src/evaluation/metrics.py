#!/usr/bin/env python3
"""
Evaluation Metrics for NEST-DRUG

Implements metrics for assessing DMTA cycle performance:
- Enrichment Factor (EF)
- Hit Rate
- Area Under ROC Curve (AUC)
- Temporal performance metrics
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def compute_enrichment_factor(
    predictions: torch.Tensor,
    actives: torch.Tensor,
    top_fraction: float = 0.01,
) -> float:
    """
    Compute enrichment factor at top X%.

    EF = (hits_in_top_X / total_in_top_X) / (total_hits / total_compounds)

    Args:
        predictions: Model predictions (higher = better)
        actives: Binary labels (1 = active/hit)
        top_fraction: Fraction to consider (default: 1%)

    Returns:
        Enrichment factor value
    """
    n_total = len(predictions)
    n_actives = actives.sum().item()

    if n_actives == 0:
        return 0.0

    # Get top fraction
    n_top = max(1, int(n_total * top_fraction))
    top_indices = torch.argsort(predictions, descending=True)[:n_top]

    # Count actives in top
    hits_in_top = actives[top_indices].sum().item()

    # Random baseline
    random_rate = n_actives / n_total

    # Enrichment factor
    if random_rate > 0:
        ef = (hits_in_top / n_top) / random_rate
    else:
        ef = 0.0

    return ef


def compute_hit_rate(
    predictions: torch.Tensor,
    actives: torch.Tensor,
    top_k: int = 100,
) -> float:
    """
    Compute hit rate in top-k predictions.

    Args:
        predictions: Model predictions
        actives: Binary labels
        top_k: Number of top predictions to consider

    Returns:
        Hit rate (fraction of actives in top-k)
    """
    n_top = min(top_k, len(predictions))
    top_indices = torch.argsort(predictions, descending=True)[:n_top]
    hits = actives[top_indices].sum().item()
    return hits / n_top


def compute_auc(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute Area Under ROC Curve.

    Args:
        predictions: Model predictions
        targets: Binary labels

    Returns:
        AUC score (0.5 = random, 1.0 = perfect)
    """
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    if len(np.unique(targets)) < 2:
        return 0.5  # Can't compute AUC with single class

    try:
        return roc_auc_score(targets, predictions)
    except ValueError:
        return 0.5


def compute_pr_auc(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute Area Under Precision-Recall Curve.

    More informative than ROC-AUC for imbalanced data.
    """
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    if len(np.unique(targets)) < 2:
        return targets.mean()

    try:
        precision, recall, _ = precision_recall_curve(targets, predictions)
        return auc(recall, precision)
    except ValueError:
        return targets.mean()


def compute_rmse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        predictions: Model predictions
        targets: Ground truth values
        mask: Optional mask for valid entries

    Returns:
        RMSE value
    """
    if mask is not None:
        predictions = predictions[mask.bool()]
        targets = targets[mask.bool()]

    if len(predictions) == 0:
        return float('nan')

    mse = ((predictions - targets) ** 2).mean()
    return torch.sqrt(mse).item()


def compute_mae(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute Mean Absolute Error.
    """
    if mask is not None:
        predictions = predictions[mask.bool()]
        targets = targets[mask.bool()]

    if len(predictions) == 0:
        return float('nan')

    return (predictions - targets).abs().mean().item()


def compute_r2(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute R-squared (coefficient of determination).
    """
    if mask is not None:
        predictions = predictions[mask.bool()]
        targets = targets[mask.bool()]

    if len(predictions) < 2:
        return float('nan')

    ss_res = ((targets - predictions) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()

    if ss_tot == 0:
        return 0.0

    return (1 - ss_res / ss_tot).item()


def compute_temporal_metrics(
    round_results: List[Dict],
    metric_name: str = 'hit_rate',
) -> Dict[str, float]:
    """
    Compute temporal aggregates over DMTA rounds.

    Args:
        round_results: List of per-round metric dictionaries
        metric_name: Name of metric to aggregate

    Returns:
        Dictionary with temporal statistics
    """
    values = [r.get(metric_name, 0) for r in round_results if metric_name in r]

    if len(values) == 0:
        return {}

    values = np.array(values)

    return {
        f'{metric_name}_mean': float(values.mean()),
        f'{metric_name}_std': float(values.std()),
        f'{metric_name}_min': float(values.min()),
        f'{metric_name}_max': float(values.max()),
        f'{metric_name}_first_half': float(values[:len(values)//2].mean()) if len(values) > 1 else float(values.mean()),
        f'{metric_name}_second_half': float(values[len(values)//2:].mean()) if len(values) > 1 else float(values.mean()),
        f'{metric_name}_trend': float(np.polyfit(range(len(values)), values, 1)[0]) if len(values) > 1 else 0.0,
    }


def compute_calibration_error(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error for uncertainty estimates.

    Well-calibrated models should have prediction intervals
    that contain the true value at the expected frequency.

    Args:
        predictions: Mean predictions
        uncertainties: Predicted standard deviations
        targets: Ground truth values
        n_bins: Number of confidence bins

    Returns:
        ECE value (lower is better)
    """
    # Compute z-scores
    z_scores = torch.abs(targets - predictions) / (uncertainties + 1e-6)

    # Expected coverage at different confidence levels
    confidence_levels = torch.linspace(0.1, 0.99, n_bins)
    z_thresholds = torch.tensor([1.645, 1.96, 2.576])  # ~90%, 95%, 99%

    ece = 0.0
    for i, conf in enumerate(confidence_levels):
        # z-score for this confidence level (approximation)
        z_thresh = -torch.log(torch.tensor(1 - conf)) * 1.2

        # Observed coverage
        observed = (z_scores < z_thresh).float().mean()

        # ECE contribution
        ece += torch.abs(observed - conf)

    return (ece / n_bins).item()


def compute_ranking_correlation(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute Spearman rank correlation.

    Measures how well the model preserves ranking order.
    """
    from scipy.stats import spearmanr

    predictions = predictions.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()

    if len(predictions) < 2:
        return 0.0

    corr, _ = spearmanr(predictions, targets)
    return float(corr) if not np.isnan(corr) else 0.0


class MetricsTracker:
    """
    Utility class for tracking metrics across rounds.
    """

    def __init__(self):
        self.round_metrics: List[Dict[str, float]] = []
        self.cumulative_hits = 0
        self.cumulative_tested = 0

    def add_round(
        self,
        round_id: int,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        actives: Optional[torch.Tensor] = None,
        uncertainties: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute and store metrics for a round.

        Args:
            round_id: DMTA round number
            predictions: Model predictions
            targets: Ground truth values
            actives: Binary active labels (for hit rate)
            uncertainties: Predicted uncertainties

        Returns:
            Dictionary of metrics for this round
        """
        metrics = {'round_id': round_id}

        # Regression metrics
        metrics['rmse'] = compute_rmse(predictions, targets)
        metrics['mae'] = compute_mae(predictions, targets)
        metrics['r2'] = compute_r2(predictions, targets)
        metrics['spearman'] = compute_ranking_correlation(predictions, targets)

        # Classification/ranking metrics (if actives provided)
        if actives is not None:
            metrics['hit_rate_100'] = compute_hit_rate(predictions, actives, top_k=100)
            metrics['hit_rate_50'] = compute_hit_rate(predictions, actives, top_k=50)
            metrics['ef_1pct'] = compute_enrichment_factor(predictions, actives, top_fraction=0.01)
            metrics['ef_5pct'] = compute_enrichment_factor(predictions, actives, top_fraction=0.05)
            metrics['auc'] = compute_auc(predictions, actives)

            # Cumulative tracking
            top_50_idx = torch.argsort(predictions, descending=True)[:50]
            self.cumulative_hits += actives[top_50_idx].sum().item()
            self.cumulative_tested += 50
            metrics['cumulative_hit_rate'] = self.cumulative_hits / self.cumulative_tested

        # Calibration metrics (if uncertainties provided)
        if uncertainties is not None:
            metrics['calibration_error'] = compute_calibration_error(
                predictions, uncertainties, targets
            )

        self.round_metrics.append(metrics)
        return metrics

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics across all rounds."""
        if len(self.round_metrics) == 0:
            return {}

        summary = {}

        # Aggregate each metric
        metric_names = set()
        for rm in self.round_metrics:
            metric_names.update(rm.keys())
        metric_names.discard('round_id')

        for name in metric_names:
            temporal = compute_temporal_metrics(self.round_metrics, name)
            summary.update(temporal)

        return summary


if __name__ == '__main__':
    # Test metrics
    print("Testing Evaluation Metrics...")

    # Dummy data
    n = 1000
    predictions = torch.randn(n)
    targets = predictions + torch.randn(n) * 0.5  # Correlated with noise
    actives = (targets > targets.median()).float()

    print("\nRegression Metrics:")
    print(f"  RMSE: {compute_rmse(predictions, targets):.4f}")
    print(f"  MAE: {compute_mae(predictions, targets):.4f}")
    print(f"  R²: {compute_r2(predictions, targets):.4f}")
    print(f"  Spearman: {compute_ranking_correlation(predictions, targets):.4f}")

    print("\nRanking Metrics:")
    print(f"  Hit Rate @100: {compute_hit_rate(predictions, actives, 100):.4f}")
    print(f"  EF @1%: {compute_enrichment_factor(predictions, actives, 0.01):.2f}")
    print(f"  EF @5%: {compute_enrichment_factor(predictions, actives, 0.05):.2f}")
    print(f"  AUC: {compute_auc(predictions, actives):.4f}")

    # Test tracker
    print("\nMetrics Tracker:")
    tracker = MetricsTracker()

    for round_id in range(10):
        predictions = torch.randn(100)
        targets = predictions + torch.randn(100) * (0.5 + round_id * 0.05)
        actives = (targets > targets.median()).float()

        metrics = tracker.add_round(round_id, predictions, targets, actives)
        print(f"  Round {round_id}: RMSE={metrics['rmse']:.3f}, Hit@100={metrics['hit_rate_100']:.3f}")

    summary = tracker.get_summary()
    print(f"\nSummary:")
    print(f"  Mean RMSE: {summary.get('rmse_mean', 'N/A'):.4f}")
    print(f"  Mean Hit Rate: {summary.get('hit_rate_100_mean', 'N/A'):.4f}")
    print(f"  RMSE Trend: {summary.get('rmse_trend', 'N/A'):.4f}")

    print("\nMetrics tests complete!")
