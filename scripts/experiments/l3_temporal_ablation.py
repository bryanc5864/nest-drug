#!/usr/bin/env python3
"""
L3 Temporal Ablation: Does providing the correct round context (L3) improve predictions?

Tests whether the model performs better when given the temporal round ID
corresponding to when the data was generated, vs using a generic round_id=0.

Usage:
    python scripts/experiments/l3_temporal_ablation.py --checkpoint results/v3/best_model.pt --gpu 0
"""

import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph


DUDE_TO_V3_PROGRAM_ID = {
    'egfr': 1606, 'drd2': 1448, 'cyp3a4': 810, 'fxa': 1103, 'herg': 0,
}

PROGRAM_FILES = {
    'egfr': 'data/processed/programs/program_egfr_augmented.csv',
    'drd2': 'data/processed/programs/program_drd2_augmented.csv',
    'cyp3a4': 'data/processed/programs/program_cyp3a4_augmented.csv',
    'fxa': 'data/processed/programs/program_fxa_augmented.csv',
    'herg': 'data/processed/programs/program_herg_augmented.csv',
}


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    config = checkpoint.get('config', {})

    num_programs = config.get('n_programs', config.get('num_programs',
        state_dict['context_module.program_embeddings.embeddings.weight'].shape[0]))
    num_assays = config.get('n_assays', config.get('num_assays',
        state_dict['context_module.assay_embeddings.embeddings.weight'].shape[0]))
    num_rounds = config.get('n_rounds', config.get('num_rounds',
        state_dict['context_module.round_embeddings.embeddings.weight'].shape[0]))

    endpoint_names = []
    for key in state_dict.keys():
        if 'prediction_heads.heads.' in key and '.mlp.0.weight' in key:
            name = key.split('prediction_heads.heads.')[1].split('.mlp')[0]
            endpoint_names.append(name)

    endpoints = {name: {'type': 'regression', 'weight': 1.0} for name in endpoint_names}
    model = create_nest_drug(
        num_programs=num_programs, num_assays=num_assays,
        num_rounds=num_rounds, endpoints=endpoints,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, {'num_programs': num_programs, 'num_assays': num_assays, 'num_rounds': num_rounds}


def predict_batch(model, smiles_list, device, program_id=0, assay_id=0, round_id=0):
    model.eval()
    predictions = []
    with torch.no_grad():
        for smi in smiles_list:
            g = smiles_to_graph(smi)
            if g is None:
                predictions.append(None)
                continue
            node_features = g['node_features'].to(device)
            edge_index = g['edge_index'].to(device)
            edge_features = g['edge_features'].to(device)
            batch = torch.zeros(g['num_atoms'], dtype=torch.long, device=device)
            program_ids = torch.tensor([program_id], dtype=torch.long, device=device)
            assay_ids = torch.tensor([assay_id], dtype=torch.long, device=device)
            round_ids = torch.tensor([round_id], dtype=torch.long, device=device)
            preds = model(node_features, edge_index, edge_features, batch,
                         program_ids, assay_ids, round_ids)
            pred = list(preds.values())[0].squeeze().cpu().numpy()
            predictions.append(float(pred) if np.ndim(pred) == 0 else float(pred[0]))
    return predictions


def evaluate_predictions(predictions, targets, threshold=7.0):
    """Compute regression and ranking metrics."""
    valid = [(p, t) for p, t in zip(predictions, targets) if p is not None]
    if len(valid) < 10:
        return None
    preds = np.array([v[0] for v in valid])
    tgts = np.array([v[1] for v in valid])

    binary = (tgts > threshold).astype(int)

    results = {
        'n_samples': len(valid),
        'rmse': float(np.sqrt(mean_squared_error(tgts, preds))),
        'correlation': float(pearsonr(preds, tgts)[0]) if len(set(binary)) > 1 else 0.0,
    }

    if len(set(binary)) == 2:
        results['roc_auc'] = float(roc_auc_score(binary, preds))
    else:
        results['roc_auc'] = None

    return results


def main():
    parser = argparse.ArgumentParser(description="L3 Temporal Ablation")
    parser.add_argument('--checkpoint', type=str, default='results/v3/best_model.pt')
    parser.add_argument('--output', type=str, default='results/experiments/l3_ablation')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--targets', type=str, nargs='+', default=['egfr', 'drd2', 'fxa'])
    parser.add_argument('--max-per-round', type=int, default=200, help='Max samples per round')
    parser.add_argument('--threshold', type=float, default=7.0, help='Activity threshold')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, config = load_model(args.checkpoint, device)
    print(f"Config: {config}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_round_id = config['num_rounds'] - 1
    all_results = {}

    for target in args.targets:
        print(f"\n{'='*60}")
        print(f"Target: {target.upper()}")
        print(f"{'='*60}")

        data_file = PROGRAM_FILES.get(target)
        if data_file is None or not Path(data_file).exists():
            print(f"  No data for {target}")
            continue

        program_data = pd.read_csv(data_file)
        program_id = DUDE_TO_V3_PROGRAM_ID.get(target, 0)
        program_id = min(program_id, config['num_programs'] - 1)

        # Group rounds into temporal bins (5 bins)
        rounds = sorted(program_data['round_id'].unique())
        n_rounds = len(rounds)
        bin_size = max(n_rounds // 5, 1)
        bins = []
        for i in range(0, n_rounds, bin_size):
            bin_rounds = rounds[i:i+bin_size]
            bins.append(bin_rounds)
        if len(bins) > 5:
            bins[-2].extend(bins[-1])
            bins = bins[:-1]

        target_results = []

        for bin_idx, bin_rounds in enumerate(bins):
            bin_data = program_data[program_data['round_id'].isin(bin_rounds)]

            # Subsample if needed
            if len(bin_data) > args.max_per_round:
                np.random.seed(42 + bin_idx)
                bin_data = bin_data.sample(args.max_per_round)

            smiles = bin_data['smiles'].tolist()
            activities = bin_data['pActivity'].values
            representative_round = int(np.median(bin_rounds))
            clamped_round = min(representative_round, max_round_id)

            year_range = f"{int(bin_data['document_year'].min())}-{int(bin_data['document_year'].max())}"

            print(f"\n  Bin {bin_idx}: rounds {bin_rounds[0]}-{bin_rounds[-1]} ({year_range}), n={len(bin_data)}")

            # 1. With correct L3 (round-specific)
            preds_correct_l3 = predict_batch(model, smiles, device,
                                              program_id=program_id, round_id=clamped_round)
            metrics_correct = evaluate_predictions(preds_correct_l3, activities, args.threshold)

            # 2. With generic L3 (round_id=0)
            preds_generic_l3 = predict_batch(model, smiles, device,
                                              program_id=program_id, round_id=0)
            metrics_generic = evaluate_predictions(preds_generic_l3, activities, args.threshold)

            # 3. With wrong L3 (furthest round from correct)
            wrong_round = 0 if clamped_round > max_round_id // 2 else max_round_id
            preds_wrong_l3 = predict_batch(model, smiles, device,
                                            program_id=program_id, round_id=wrong_round)
            metrics_wrong = evaluate_predictions(preds_wrong_l3, activities, args.threshold)

            bin_result = {
                'bin_idx': bin_idx,
                'rounds': [int(r) for r in bin_rounds],
                'year_range': year_range,
                'representative_round': int(clamped_round),
                'n_samples': len(bin_data),
                'n_active': int((activities > args.threshold).sum()),
                'correct_l3': metrics_correct,
                'generic_l3': metrics_generic,
                'wrong_l3': metrics_wrong,
            }

            target_results.append(bin_result)

            # Print comparison
            if metrics_correct and metrics_generic:
                auc_c = metrics_correct.get('roc_auc', 'N/A')
                auc_g = metrics_generic.get('roc_auc', 'N/A')
                rmse_c = metrics_correct['rmse']
                rmse_g = metrics_generic['rmse']
                if isinstance(auc_c, float) and isinstance(auc_g, float):
                    print(f"    Correct L3: AUC={auc_c:.4f}, RMSE={rmse_c:.3f}")
                    print(f"    Generic L3: AUC={auc_g:.4f}, RMSE={rmse_g:.3f}")
                    print(f"    L3 benefit:  AUC={auc_c-auc_g:+.4f}, RMSE={rmse_g-rmse_c:+.3f}")
                else:
                    print(f"    RMSE correct={rmse_c:.3f}, generic={rmse_g:.3f}, delta={rmse_g-rmse_c:+.3f}")

        # Summary
        correct_aucs = [r['correct_l3']['roc_auc'] for r in target_results
                       if r['correct_l3'] and r['correct_l3'].get('roc_auc') is not None]
        generic_aucs = [r['generic_l3']['roc_auc'] for r in target_results
                       if r['generic_l3'] and r['generic_l3'].get('roc_auc') is not None]

        summary = {}
        if correct_aucs and generic_aucs:
            summary = {
                'mean_auc_correct_l3': float(np.mean(correct_aucs)),
                'mean_auc_generic_l3': float(np.mean(generic_aucs)),
                'mean_l3_benefit': float(np.mean(correct_aucs) - np.mean(generic_aucs)),
                'bins_improved': sum(1 for c, g in zip(correct_aucs, generic_aucs) if c > g),
                'total_bins': len(correct_aucs),
            }
            print(f"\n  Summary: correct={summary['mean_auc_correct_l3']:.4f}, generic={summary['mean_auc_generic_l3']:.4f}, delta={summary['mean_l3_benefit']:+.4f}")

        all_results[target] = {
            'program_id': program_id,
            'per_bin': target_results,
            'summary': summary,
        }

    # Save
    results_path = output_dir / 'l3_ablation_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'config': config,
            'threshold': args.threshold,
            'results': all_results,
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
