#!/usr/bin/env python3
"""
Experiment 3A: TDC Benchmark Suite

Run on Therapeutics Data Commons benchmarks - standardized, widely-cited, real experimental data.

Usage:
    python scripts/experiments/tdc_benchmark.py \
        --checkpoint checkpoints/pretrain/best_model.pt \
        --output results/experiments/tdc_benchmark \
        --gpu 0

Benchmarks:
    - hERG: Ion channel toxicity
    - AMES: Mutagenicity
    - BBB: Blood-brain barrier penetration
    - CYP2D6: CYP inhibition
    - Solubility: Aqueous solubility
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
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph

try:
    from tdc.single_pred import ADME, Tox
    HAS_TDC = True
except ImportError:
    HAS_TDC = False
    print("Warning: TDC not installed. Run: pip install PyTDC")


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

            # Filter valid
            valid_graphs = [g for g in batch_graphs if g is not None]
            if not valid_graphs:
                predictions.extend([np.nan] * len(batch_smiles))
                continue

            # Collate
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

            # Map back including NaN for invalid
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


# TDC dataset configurations
TDC_DATASETS = {
    # Toxicity
    'hERG': {'module': Tox, 'name': 'hERG', 'task': 'classification', 'target': 0.85},
    'AMES': {'module': Tox, 'name': 'AMES', 'task': 'classification', 'target': 0.83},
    'ClinTox': {'module': Tox, 'name': 'ClinTox', 'task': 'classification', 'target': 0.90},
    'DILI': {'module': Tox, 'name': 'DILI', 'task': 'classification', 'target': 0.85},

    # ADME
    'BBB': {'module': ADME, 'name': 'BBB_Martins', 'task': 'classification', 'target': 0.90},
    'CYP2D6': {'module': ADME, 'name': 'CYP2D6_Veith', 'task': 'classification', 'target': 0.75},
    'CYP3A4': {'module': ADME, 'name': 'CYP3A4_Veith', 'task': 'classification', 'target': 0.80},
    'Pgp': {'module': ADME, 'name': 'Pgp_Broccatelli', 'task': 'classification', 'target': 0.90},
    'Bioavailability': {'module': ADME, 'name': 'Bioavailability_Ma', 'task': 'classification', 'target': 0.70},
    'Solubility': {'module': ADME, 'name': 'Solubility_AqSolDB', 'task': 'regression', 'target': 0.80},
    'Lipophilicity': {'module': ADME, 'name': 'Lipophilicity_AstraZeneca', 'task': 'regression', 'target': 0.70},
}


def load_tdc_dataset(name):
    """Load a TDC dataset."""
    if not HAS_TDC:
        raise ImportError("TDC not installed")

    config = TDC_DATASETS.get(name)
    if config is None:
        raise ValueError(f"Unknown dataset: {name}")

    dataset = config['module'](name=config['name'])
    data = dataset.get_data()

    return data, config['task'], config['target']


def evaluate_classification(y_true, y_pred):
    """Evaluate classification metrics."""
    # Remove NaN
    mask = ~np.isnan(y_pred)
    y_true = np.array(y_true)[mask]
    y_pred = y_pred[mask]

    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = 0.5

    # Binary predictions (threshold at median prediction)
    threshold = np.median(y_pred)
    y_pred_binary = (y_pred > threshold).astype(int)
    acc = accuracy_score(y_true, y_pred_binary)

    return {'roc_auc': float(auc), 'accuracy': float(acc), 'n_samples': int(mask.sum())}


def evaluate_regression(y_true, y_pred):
    """Evaluate regression metrics."""
    mask = ~np.isnan(y_pred)
    y_true = np.array(y_true)[mask]
    y_pred = y_pred[mask]

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Correlation
    corr = np.corrcoef(y_true, y_pred)[0, 1]

    return {'rmse': float(rmse), 'r2': float(r2), 'correlation': float(corr), 'n_samples': int(mask.sum())}


def main():
    parser = argparse.ArgumentParser(description="TDC Benchmark Suite")
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--output', type=str, default='results/experiments/tdc_benchmark',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['hERG', 'AMES', 'BBB', 'CYP2D6', 'Solubility'],
                        help='Datasets to evaluate')
    parser.add_argument('--program-id', type=int, default=0, help='Program context')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    args = parser.parse_args()

    if not HAS_TDC:
        print("ERROR: TDC required. Install with: pip install PyTDC")
        sys.exit(1)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLoading model: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    print(f"Config: {config}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print('='*60)

        try:
            data, task, target = load_tdc_dataset(dataset_name)
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            continue

        smiles = data['Drug'].tolist()
        labels = data['Y'].tolist()

        print(f"Samples: {len(smiles)}")
        print(f"Task: {task}")
        print(f"Target benchmark: {target}")

        # Predict
        predictions = predict_batch(model, smiles, device, args.batch_size, args.program_id)

        # Evaluate
        if task == 'classification':
            metrics = evaluate_classification(labels, predictions)
            main_metric = metrics['roc_auc']
            metric_name = 'ROC-AUC'
        else:
            metrics = evaluate_regression(labels, predictions)
            main_metric = metrics['r2']
            metric_name = 'R²'

        # Compare to target
        meets_target = main_metric >= target
        status = "✓ PASS" if meets_target else "✗ FAIL"

        print(f"\n{metric_name}: {main_metric:.4f} (target: {target}) {status}")
        print(f"Metrics: {metrics}")

        all_results[dataset_name] = {
            'task': task,
            'target': target,
            'metrics': metrics,
            'meets_target': meets_target,
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    passed = sum(1 for r in all_results.values() if r['meets_target'])
    total = len(all_results)

    print(f"\nPassed: {passed}/{total}")
    print("\nDataset          Task           Score    Target   Status")
    print("-" * 60)
    for name, res in all_results.items():
        if res['task'] == 'classification':
            score = res['metrics']['roc_auc']
        else:
            score = res['metrics']['r2']
        status = "PASS" if res['meets_target'] else "FAIL"
        print(f"{name:16s} {res['task']:14s} {score:.3f}    {res['target']:.2f}     {status}")

    # Save results
    results_path = output_dir / 'tdc_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'config': config,
            'program_id': args.program_id,
            'summary': {'passed': passed, 'total': total},
            'results': all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
