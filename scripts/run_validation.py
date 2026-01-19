#!/usr/bin/env python3
"""
Run Final Validation on Best Checkpoint

Loads the best model and computes comprehensive validation metrics.
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.nest_drug import create_nest_drug
from training.data_utils import PortfolioDataLoader
from torch.cuda.amp import autocast


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray = None):
    """Compute comprehensive regression metrics."""
    if mask is not None:
        mask = mask.astype(bool).flatten()
        predictions = predictions.flatten()[mask]
        targets = targets.flatten()[mask]
    else:
        predictions = predictions.flatten()
        targets = targets.flatten()

    if len(predictions) == 0:
        return {}

    # Basic metrics
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    # R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    # Correlations
    pearson_r, pearson_p = stats.pearsonr(predictions, targets)
    spearman_r, spearman_p = stats.spearmanr(predictions, targets)

    # Percentile errors
    abs_errors = np.abs(predictions - targets)
    p50_error = np.percentile(abs_errors, 50)
    p90_error = np.percentile(abs_errors, 90)
    p95_error = np.percentile(abs_errors, 95)

    # Within threshold accuracy
    within_05 = np.mean(abs_errors <= 0.5) * 100
    within_10 = np.mean(abs_errors <= 1.0) * 100

    return {
        'n_samples': len(predictions),
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'p50_error': p50_error,
        'p90_error': p90_error,
        'p95_error': p95_error,
        'within_0.5': within_05,
        'within_1.0': within_10,
        'mean_pred': np.mean(predictions),
        'std_pred': np.std(predictions),
        'mean_target': np.mean(targets),
        'std_target': np.std(targets),
    }


def main():
    parser = argparse.ArgumentParser(description='Run validation on best checkpoint')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/pretrain/best_model.pt',
                       help='Path to checkpoint')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to validation data (default: use temp val split)')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Detect endpoints from checkpoint
    state_dict = checkpoint['model_state_dict']
    endpoint_names = []
    for key in state_dict.keys():
        if 'prediction_heads.heads.' in key and '.mlp.0.weight' in key:
            name = key.split('prediction_heads.heads.')[1].split('.mlp')[0]
            endpoint_names.append(name)

    print(f"Detected endpoints: {endpoint_names}")

    # Create model with detected endpoints
    endpoints = {name: {'type': 'regression', 'weight': 1.0} for name in endpoint_names}
    model = create_nest_drug(
        num_programs=5,
        num_assays=50,
        num_rounds=150,
        endpoints=endpoints,
    )

    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load validation data
    if args.data:
        val_path = args.data
    else:
        import tempfile
        val_path = Path(tempfile.gettempdir()) / "nest_val_split.parquet"
        if not val_path.exists():
            print("ERROR: Validation split not found. Please provide --data path.")
            sys.exit(1)

    print(f"\nLoading validation data: {val_path}")
    val_loader = PortfolioDataLoader(
        data_path=str(val_path),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Run inference
    print(f"\nRunning validation on {len(val_loader.dataset):,} samples...")

    all_predictions = {name: [] for name in endpoint_names}
    all_targets = {name: [] for name in endpoint_names}
    all_masks = {name: [] for name in endpoint_names}

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch is None:
                continue

            # Move to device
            node_features = batch['node_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_features = batch['edge_features'].to(device)
            batch_idx_tensor = batch['batch'].to(device)
            program_ids = batch['program_ids'].to(device)
            assay_ids = batch['assay_ids'].to(device)
            round_ids = batch['round_ids'].to(device)

            with autocast(enabled=True):
                predictions = model(
                    node_features=node_features,
                    edge_index=edge_index,
                    edge_features=edge_features,
                    batch=batch_idx_tensor,
                    program_ids=program_ids,
                    assay_ids=assay_ids,
                    round_ids=round_ids,
                )

            # Collect predictions
            for name in endpoint_names:
                if name in predictions and name in batch['endpoints']:
                    all_predictions[name].append(predictions[name].cpu().numpy())
                    all_targets[name].append(batch['endpoints'][name].numpy())
                    all_masks[name].append(batch['masks'][name].numpy())

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1}/{len(val_loader)} batches...")

    # Compute metrics per endpoint
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    global_preds = []
    global_targets = []

    for name in endpoint_names:
        if len(all_predictions[name]) == 0:
            continue

        preds = np.concatenate(all_predictions[name], axis=0)
        targets = np.concatenate(all_targets[name], axis=0)
        masks = np.concatenate(all_masks[name], axis=0)

        metrics = compute_metrics(preds, targets, masks)

        if metrics:
            print(f"\n{name}:")
            print(f"  Samples:     {metrics['n_samples']:,}")
            print(f"  R²:          {metrics['r2']:.4f}")
            print(f"  Pearson r:   {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.2e})")
            print(f"  Spearman ρ:  {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.2e})")
            print(f"  RMSE:        {metrics['rmse']:.4f}")
            print(f"  MAE:         {metrics['mae']:.4f}")
            print(f"  P50 Error:   {metrics['p50_error']:.4f}")
            print(f"  P90 Error:   {metrics['p90_error']:.4f}")
            print(f"  Within ±0.5: {metrics['within_0.5']:.1f}%")
            print(f"  Within ±1.0: {metrics['within_1.0']:.1f}%")

            # Accumulate for global metrics
            mask_bool = masks.astype(bool).flatten()
            global_preds.extend(preds.flatten()[mask_bool])
            global_targets.extend(targets.flatten()[mask_bool])

    # Global metrics
    if global_preds:
        global_preds = np.array(global_preds)
        global_targets = np.array(global_targets)
        global_metrics = compute_metrics(global_preds, global_targets)

        print("\n" + "=" * 70)
        print("GLOBAL METRICS (all endpoints combined)")
        print("=" * 70)
        print(f"  Total Samples: {global_metrics['n_samples']:,}")
        print(f"  R²:            {global_metrics['r2']:.4f}")
        print(f"  Pearson r:     {global_metrics['pearson_r']:.4f}")
        print(f"  Spearman ρ:    {global_metrics['spearman_r']:.4f}")
        print(f"  RMSE:          {global_metrics['rmse']:.4f}")
        print(f"  MAE:           {global_metrics['mae']:.4f}")
        print(f"  Within ±0.5:   {global_metrics['within_0.5']:.1f}%")
        print(f"  Within ±1.0:   {global_metrics['within_1.0']:.1f}%")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
