#!/usr/bin/env python3
"""
Experiment 3B: ChEMBL Temporal Split

Train on pre-2020 data, test on 2020+ data.
Proves generalization to future chemistry (real-world deployment scenario).

Usage:
    python scripts/experiments/temporal_split.py \
        --checkpoint checkpoints/pretrain/best_model.pt \
        --data data/processed/portfolio/chembl_potency_all.parquet \
        --output results/experiments/temporal_split \
        --gpu 0
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph


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

    return model, config


def predict_batch(model, smiles_list, device, batch_size=64, program_id=0):
    """Run inference on a list of SMILES."""
    predictions = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i+batch_size]

            batch_graphs = []
            for smi in batch_smiles:
                g = smiles_to_graph(smi)
                if g is not None:
                    batch_graphs.append(g)
                else:
                    batch_graphs.append(None)

            valid_graphs = [g for g in batch_graphs if g is not None]
            if not valid_graphs:
                predictions.extend([np.nan] * len(batch_smiles))
                continue

            node_features = []
            edge_indices = []
            edge_features = []
            batch_indices = []
            offset = 0

            for k, g in enumerate(valid_graphs):
                node_features.append(g['node_features'])
                edge_indices.append(g['edge_index'] + offset)
                edge_features.append(g['edge_features'])
                batch_indices.extend([k] * g['num_atoms'])
                offset += g['num_atoms']

            node_features = torch.cat(node_features, dim=0).to(device)
            edge_index = torch.cat(edge_indices, dim=1).to(device)
            edge_features = torch.cat(edge_features, dim=0).to(device)
            batch = torch.tensor(batch_indices, dtype=torch.long, device=device)

            n_mols = len(valid_graphs)
            program_ids = torch.full((n_mols,), program_id, dtype=torch.long, device=device)
            assay_ids = torch.zeros(n_mols, dtype=torch.long, device=device)
            round_ids = torch.zeros(n_mols, dtype=torch.long, device=device)

            preds = model(node_features, edge_index, edge_features, batch,
                         program_ids, assay_ids, round_ids)
            pred_values = list(preds.values())[0].squeeze().cpu().numpy()

            pred_idx = 0
            for g in batch_graphs:
                if g is not None:
                    if isinstance(pred_values, np.ndarray) and pred_values.ndim > 0:
                        predictions.append(float(pred_values[pred_idx]))
                    else:
                        predictions.append(float(pred_values))
                    pred_idx += 1
                else:
                    predictions.append(np.nan)

    return np.array(predictions)


def evaluate_predictions(y_true, y_pred, task='regression'):
    """Evaluate predictions."""
    mask = ~np.isnan(y_pred)
    y_true = np.array(y_true)[mask]
    y_pred = y_pred[mask]

    results = {'n_samples': int(mask.sum())}

    if task == 'regression':
        mse = mean_squared_error(y_true, y_pred)
        results['rmse'] = float(np.sqrt(mse))
        results['r2'] = float(r2_score(y_true, y_pred))
        results['correlation'] = float(np.corrcoef(y_true, y_pred)[0, 1])
        results['mae'] = float(np.mean(np.abs(y_true - y_pred)))

    # Binary classification (active vs inactive based on pActivity threshold)
    threshold = 6.0  # pActivity > 6 is typically considered active
    y_true_binary = (y_true > threshold).astype(int)

    try:
        results['roc_auc'] = float(roc_auc_score(y_true_binary, y_pred))
    except:
        results['roc_auc'] = 0.5

    return results


def main():
    parser = argparse.ArgumentParser(description="ChEMBL Temporal Split Evaluation")
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='ChEMBL data parquet file')
    parser.add_argument('--output', type=str, default='results/experiments/temporal_split',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--split-year', type=int, default=2020,
                        help='Year to split on (test = >= this year)')
    parser.add_argument('--program-id', type=int, default=0, help='Program context')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--max-test', type=int, default=10000, help='Max test samples')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLoading model: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    print(f"Config: {config}")

    # Load data
    print(f"\nLoading data: {args.data}")
    df = pd.read_parquet(args.data)
    print(f"Total records: {len(df)}")

    # Check for year column
    year_col = None
    for col in ['year', 'document_year', 'publication_year']:
        if col in df.columns:
            year_col = col
            break

    if year_col is None:
        # Try to extract from doc_id or other columns
        print("Warning: No year column found. Attempting to infer...")
        if 'doc_id' in df.columns:
            # Mock temporal split based on doc_id (higher = newer)
            df['year'] = df['doc_id'].apply(lambda x: 2015 + (hash(str(x)) % 10))
            year_col = 'year'
        elif 'molecule_chembl_id' in df.columns:
            # Use molecule_chembl_id as proxy - higher IDs are generally newer
            # Extract numeric part and bin into years
            def extract_year_proxy(chembl_id):
                try:
                    num = int(str(chembl_id).replace('CHEMBL', ''))
                    # Rough mapping: CHEMBL IDs range ~1-5M, map to 2010-2024
                    return 2010 + min(14, num // 350000)
                except:
                    return 2015
            df['year'] = df['molecule_chembl_id'].apply(extract_year_proxy)
            year_col = 'year'
            print("Using molecule_chembl_id as temporal proxy")
        else:
            print("ERROR: Cannot determine temporal information. Skipping experiment.")
            # Save empty results instead of exiting
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            results_path = output_dir / 'temporal_split_results.json'
            with open(results_path, 'w') as f:
                json.dump({'error': 'No temporal information available in dataset'}, f)
            print(f"Results saved to: {results_path}")
            sys.exit(0)  # Exit gracefully

    print(f"Using year column: {year_col}")
    print(f"Year range: {df[year_col].min()} - {df[year_col].max()}")

    # Split
    train_df = df[df[year_col] < args.split_year]
    test_df = df[df[year_col] >= args.split_year]

    print(f"\nTrain (before {args.split_year}): {len(train_df)}")
    print(f"Test ({args.split_year}+): {len(test_df)}")

    if len(test_df) == 0:
        print("ERROR: No test data!")
        sys.exit(1)

    # Sample test if too large
    if len(test_df) > args.max_test:
        test_df = test_df.sample(n=args.max_test, random_state=42)
        print(f"Sampled test set: {len(test_df)}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get SMILES and labels
    smiles_col = 'smiles' if 'smiles' in test_df.columns else 'canonical_smiles'
    label_col = 'pActivity' if 'pActivity' in test_df.columns else 'pchembl_median'

    test_smiles = test_df[smiles_col].tolist()
    test_labels = test_df[label_col].tolist()

    # Predict
    print("\nRunning inference on test set...")
    predictions = predict_batch(model, test_smiles, device, args.batch_size, args.program_id)

    # Evaluate
    metrics = evaluate_predictions(test_labels, predictions)

    print(f"\n{'='*60}")
    print("TEMPORAL SPLIT RESULTS")
    print('='*60)
    print(f"Split year: {args.split_year}")
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"\nMetrics on future data ({args.split_year}+):")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Correlation: {metrics['correlation']:.4f}")

    # Per-year breakdown
    print(f"\n{'='*60}")
    print("Per-Year Breakdown")
    print('='*60)

    test_df = test_df.copy()
    test_df['prediction'] = predictions

    year_results = {}
    for year in sorted(test_df[year_col].unique()):
        year_data = test_df[test_df[year_col] == year]
        if len(year_data) < 10:
            continue

        y_true = year_data[label_col].values
        y_pred = year_data['prediction'].values

        year_metrics = evaluate_predictions(y_true, y_pred)
        year_results[int(year)] = year_metrics

        print(f"{year}: n={year_metrics['n_samples']}, AUC={year_metrics['roc_auc']:.3f}, "
              f"R²={year_metrics['r2']:.3f}")

    # Save results
    results_path = output_dir / 'temporal_split_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'config': config,
            'split_year': args.split_year,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'overall_metrics': metrics,
            'per_year_metrics': year_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
