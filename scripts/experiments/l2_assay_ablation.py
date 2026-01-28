#!/usr/bin/env python3
"""
L2 Assay Context Ablation: Does providing the correct assay type (L2) improve predictions?

Tests whether the model performs better when given the correct assay ID
(binding vs functional vs ADMET) vs using a generic assay_id=0.

Usage:
    python scripts/experiments/l2_assay_ablation.py --checkpoint results/v3/best_model.pt --gpu 0
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


def classify_assay_type(activity_type):
    """Classify activity type into coarse assay categories."""
    if activity_type in ('Ki', 'Kd', 'IC50'):
        return 'binding'
    elif activity_type in ('EC50', 'GI50', 'Inhibition', 'Activity'):
        return 'functional'
    else:
        return 'other'


def evaluate_predictions(predictions, targets, threshold=7.0):
    valid = [(p, t) for p, t in zip(predictions, targets) if p is not None]
    if len(valid) < 10:
        return None
    preds = np.array([v[0] for v in valid])
    tgts = np.array([v[1] for v in valid])
    binary = (tgts > threshold).astype(int)

    results = {
        'n_samples': len(valid),
        'rmse': float(np.sqrt(mean_squared_error(tgts, preds))),
        'correlation': float(pearsonr(preds, tgts)[0]) if np.std(preds) > 0 else 0.0,
    }
    if len(set(binary)) == 2:
        results['roc_auc'] = float(roc_auc_score(binary, preds))
    else:
        results['roc_auc'] = None
    return results


def main():
    parser = argparse.ArgumentParser(description="L2 Assay Context Ablation")
    parser.add_argument('--checkpoint', type=str, default='results/v3/best_model.pt')
    parser.add_argument('--output', type=str, default='results/experiments/l2_ablation')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--targets', type=str, nargs='+', default=['egfr', 'drd2', 'fxa'])
    parser.add_argument('--max-per-group', type=int, default=300, help='Max samples per assay group')
    parser.add_argument('--threshold', type=float, default=7.0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, config = load_model(args.checkpoint, device)
    print(f"Config: {config}")
    max_assay_id = config['num_assays'] - 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

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

        # Classify assays
        program_data['assay_category'] = program_data['activity_type'].apply(classify_assay_type)
        print(f"  Assay categories: {program_data['assay_category'].value_counts().to_dict()}")
        print(f"  Unique assay_ids: {program_data['assay_id'].nunique()}")

        # Get unique assay_context IDs (pre-mapped integers)
        if 'assay_context' in program_data.columns:
            unique_assay_contexts = sorted(program_data['assay_context'].unique())
            print(f"  Unique assay_context IDs: {len(unique_assay_contexts)}")
        else:
            unique_assay_contexts = []

        target_results = []

        # Test per assay category
        for category in ['binding', 'functional', 'other']:
            cat_data = program_data[program_data['assay_category'] == category]
            if len(cat_data) < 20:
                continue

            # Subsample
            if len(cat_data) > args.max_per_group:
                np.random.seed(42)
                cat_data = cat_data.sample(args.max_per_group)

            smiles = cat_data['smiles'].tolist()
            activities = cat_data['pActivity'].values

            # Get representative assay_context for this category
            if 'assay_context' in cat_data.columns:
                representative_assay = int(cat_data['assay_context'].mode().iloc[0])
            else:
                representative_assay = 0

            clamped_assay = min(representative_assay, max_assay_id)

            print(f"\n  Category: {category} (n={len(cat_data)}, assay_context={clamped_assay})")

            # Test different L2 contexts
            assay_ids_to_test = {
                'correct': clamped_assay,
                'generic': 0,
            }

            # Also test a few other random assay IDs for comparison
            np.random.seed(123)
            random_assays = np.random.choice(min(max_assay_id + 1, 100), 3, replace=False)
            for i, ra in enumerate(random_assays):
                assay_ids_to_test[f'random_{i}'] = int(ra)

            cat_result = {
                'category': category,
                'n_samples': len(cat_data),
                'n_active': int((activities > args.threshold).sum()),
                'representative_assay_id': clamped_assay,
                'conditions': {},
            }

            for condition_name, assay_id in assay_ids_to_test.items():
                preds = predict_batch(model, smiles, device,
                                       program_id=program_id, assay_id=assay_id)
                metrics = evaluate_predictions(preds, activities, args.threshold)

                cat_result['conditions'][condition_name] = {
                    'assay_id': assay_id,
                    'metrics': metrics,
                }

                if metrics:
                    auc_str = f"AUC={metrics['roc_auc']:.4f}" if metrics.get('roc_auc') is not None else "AUC=N/A"
                    print(f"    L2={condition_name} (id={assay_id}): {auc_str}, RMSE={metrics['rmse']:.3f}, r={metrics['correlation']:.3f}")

            target_results.append(cat_result)

        # Cross-category test: predict binding data with functional L2 and vice versa
        print(f"\n  Cross-category test:")
        binding_data = program_data[program_data['assay_category'] == 'binding']
        functional_data = program_data[program_data['assay_category'] == 'functional']

        if len(binding_data) >= 50 and len(functional_data) >= 50:
            # Get representative assay IDs for each
            binding_assay = 0
            functional_assay = 1
            if 'assay_context' in program_data.columns:
                binding_assays = binding_data['assay_context'].mode()
                functional_assays = functional_data['assay_context'].mode()
                if len(binding_assays) > 0:
                    binding_assay = min(int(binding_assays.iloc[0]), max_assay_id)
                if len(functional_assays) > 0:
                    functional_assay = min(int(functional_assays.iloc[0]), max_assay_id)

            # Test binding data with binding L2 vs functional L2
            binding_sample = binding_data.sample(min(300, len(binding_data)), random_state=42)
            b_smiles = binding_sample['smiles'].tolist()
            b_acts = binding_sample['pActivity'].values

            preds_match = predict_batch(model, b_smiles, device, program_id=program_id, assay_id=binding_assay)
            preds_mismatch = predict_batch(model, b_smiles, device, program_id=program_id, assay_id=functional_assay)

            m_match = evaluate_predictions(preds_match, b_acts, args.threshold)
            m_mismatch = evaluate_predictions(preds_mismatch, b_acts, args.threshold)

            if m_match and m_mismatch:
                print(f"    Binding data + binding L2:    RMSE={m_match['rmse']:.3f}, r={m_match['correlation']:.3f}")
                print(f"    Binding data + functional L2: RMSE={m_mismatch['rmse']:.3f}, r={m_mismatch['correlation']:.3f}")

        # Summary
        summary = {}
        correct_rmses = []
        generic_rmses = []
        for r in target_results:
            correct = r['conditions'].get('correct', {}).get('metrics')
            generic = r['conditions'].get('generic', {}).get('metrics')
            if correct and generic:
                correct_rmses.append(correct['rmse'])
                generic_rmses.append(generic['rmse'])

        if correct_rmses:
            summary = {
                'mean_rmse_correct_l2': float(np.mean(correct_rmses)),
                'mean_rmse_generic_l2': float(np.mean(generic_rmses)),
                'l2_rmse_benefit': float(np.mean(generic_rmses) - np.mean(correct_rmses)),
            }

        all_results[target] = {
            'program_id': program_id,
            'per_category': target_results,
            'summary': summary,
        }

    # Save
    results_path = output_dir / 'l2_ablation_results.json'
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
