#!/usr/bin/env python3
"""
V3 L1 Ablation (CORRECT): Uses actual program IDs from V3 training.

Compares:
- CORRECT L1: Use the actual program ID for each target from training
- NO L1: Use program_id=0 for everything (no target-specific context)

Usage:
    python scripts/experiments/v3_ablation_correct.py --gpu 0
"""

import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph


# Correct program IDs from V3's training mapping
DUDE_TO_V3_PROGRAM_ID = {
    'egfr': 1606,   # "Epidermal growth factor receptor"
    'drd2': 1448,   # "Dopamine D2 receptor"
    'adrb2': 580,   # "Beta-2 adrenergic receptor"
    'bace1': 516,   # "BACE1"
    'esr1': 1628,   # "Estrogen receptor alpha"
    'hdac2': 2177,  # "Histone deacetylase 2"
    'jak2': 4780,   # "Tyrosine-protein kinase JAK2"
    'pparg': 3307,  # "Peroxisome proliferator-activated receptor gamma"
    'cyp3a4': 810,  # "CYP3A4"
    'fxa': 1103,    # "Coagulation factor X"
}


def load_model(checkpoint_path, device):
    """Load V3 model and get program mapping."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    config = checkpoint.get('config', {})
    program_mapping = checkpoint.get('program_mapping', {})

    # Infer dimensions from state_dict if not in config
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
        num_programs=num_programs,
        num_assays=num_assays,
        num_rounds=num_rounds,
        endpoints=endpoints,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config, program_mapping


def load_dude_target(data_dir, target):
    """Load DUD-E target data."""
    target_dir = Path(data_dir) / target

    actives_file = target_dir / "actives_final.smi"
    if not actives_file.exists():
        return None, None

    actives = []
    with open(actives_file) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                actives.append(parts[0])

    decoys_file = target_dir / "decoys_final.smi"
    decoys = []
    with open(decoys_file) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                decoys.append(parts[0])

    return actives, decoys


def predict_batch(model, smiles_list, device, program_id=0, max_samples=2000):
    """Run predictions with specified program ID."""
    model.eval()
    predictions = []

    if len(smiles_list) > max_samples:
        np.random.seed(42)
        indices = np.random.choice(len(smiles_list), max_samples, replace=False)
        smiles_list = [smiles_list[i] for i in indices]

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
            assay_ids = torch.zeros(1, dtype=torch.long, device=device)
            round_ids = torch.zeros(1, dtype=torch.long, device=device)

            preds = model(node_features, edge_index, edge_features, batch,
                         program_ids, assay_ids, round_ids)
            pred = list(preds.values())[0].squeeze().cpu().numpy()
            predictions.append(float(pred) if np.ndim(pred) == 0 else float(pred[0]))

    return predictions, smiles_list


def run_ablation(model, data_dir, device, targets, max_samples=2000, num_programs=5123):
    """Run L1 ablation with CORRECT program IDs."""
    results = {}

    for target in tqdm(targets, desc="Targets"):
        actives, decoys = load_dude_target(data_dir, target)
        if actives is None:
            print(f"  Skipping {target} - no data")
            continue

        # Get correct program ID (only valid for V3 with 5123 programs)
        if num_programs >= 5123:
            correct_program_id = DUDE_TO_V3_PROGRAM_ID.get(target, 0)
        else:
            # V1 only has 5 programs - test different IDs (1-4) vs baseline (0)
            # This tests if L1 affects predictions, not if "correct" helps
            correct_program_id = (list(DUDE_TO_V3_PROGRAM_ID.keys()).index(target) % (num_programs - 1)) + 1

        # Subsample for speed
        n_actives = min(len(actives), max_samples // 2)
        n_decoys = min(len(decoys), max_samples // 2)

        np.random.seed(42)
        actives_sample = list(np.random.choice(actives, n_actives, replace=False))
        decoys_sample = list(np.random.choice(decoys, n_decoys, replace=False))

        all_smiles = actives_sample + decoys_sample
        labels = [1] * len(actives_sample) + [0] * len(decoys_sample)

        # WITH CORRECT L1 (target-specific program ID from training)
        preds_correct_l1, valid_smiles = predict_batch(
            model, all_smiles, device,
            program_id=correct_program_id,
            max_samples=len(all_smiles)
        )

        # WITHOUT L1 (generic program_id=0)
        preds_no_l1, _ = predict_batch(
            model, all_smiles, device,
            program_id=0,
            max_samples=len(all_smiles)
        )

        # Filter None predictions
        valid_idx = [i for i, p in enumerate(preds_correct_l1) if p is not None]
        preds_correct = [preds_correct_l1[i] for i in valid_idx]
        preds_generic = [preds_no_l1[i] for i in valid_idx]
        valid_labels = [labels[i] for i in valid_idx]

        if len(set(valid_labels)) < 2:
            print(f"  Skipping {target} - not enough classes")
            continue

        auc_correct = roc_auc_score(valid_labels, preds_correct)
        auc_generic = roc_auc_score(valid_labels, preds_generic)

        results[target] = {
            'correct_l1_id': correct_program_id,
            'with_correct_l1': auc_correct,
            'with_generic_l1': auc_generic,
            'delta': auc_correct - auc_generic,
            'n_samples': len(valid_labels),
            'n_actives': sum(valid_labels),
        }

        print(f"  {target} (L1={correct_program_id}): correct={auc_correct:.4f}, generic={auc_generic:.4f}, delta={auc_correct-auc_generic:+.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="V3 L1 Ablation with CORRECT IDs")
    parser.add_argument('--checkpoint', type=str, default='results/v3/best_model.pt',
                        help='V3 checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/external/dude',
                        help='DUD-E data directory')
    parser.add_argument('--output', type=str, default='results/experiments/v3_ablation_correct',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--max-samples', type=int, default=2000,
                        help='Max samples per target')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading V3 model: {args.checkpoint}")
    model, config, program_mapping = load_model(args.checkpoint, device)
    print(f"Config: {config}")
    print(f"Program mapping has {len(program_mapping)} entries")

    # Verify program IDs
    print("\nTarget to Program ID mapping:")
    for target, pid in DUDE_TO_V3_PROGRAM_ID.items():
        # Find the name for this ID
        name = [k for k, v in program_mapping.items() if v == pid]
        name = name[0] if name else "NOT FOUND"
        print(f"  {target}: ID={pid} -> {name[:50]}")

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # DUD-E targets
    targets = list(DUDE_TO_V3_PROGRAM_ID.keys())

    print("\n" + "="*70)
    print("V3 L1 ABLATION: CORRECT L1 vs GENERIC L1")
    print("="*70)

    num_programs = state_dict['context_module.program_embeddings.embeddings.weight'].shape[0] if 'state_dict' in dir() else config.get('n_programs', 5123)
    # Re-read to get num_programs
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_programs = checkpoint['model_state_dict']['context_module.program_embeddings.embeddings.weight'].shape[0]
    print(f"Model has {num_programs} programs")

    results = run_ablation(model, args.data_dir, device, targets, args.max_samples, num_programs)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    correct_aucs = [r['with_correct_l1'] for r in results.values()]
    generic_aucs = [r['with_generic_l1'] for r in results.values()]
    deltas = [r['delta'] for r in results.values()]

    print(f"Mean AUC with CORRECT L1: {np.mean(correct_aucs):.4f}")
    print(f"Mean AUC with GENERIC L1: {np.mean(generic_aucs):.4f}")
    print(f"Mean Delta:               {np.mean(deltas):+.4f}")
    print(f"Targets improved:         {sum(1 for d in deltas if d > 0)}/{len(deltas)}")

    # Save results
    output_file = output_dir / 'v3_ablation_correct_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'config': config,
            'program_id_mapping': DUDE_TO_V3_PROGRAM_ID,
            'per_target': results,
            'summary': {
                'mean_correct_l1': float(np.mean(correct_aucs)),
                'mean_generic_l1': float(np.mean(generic_aucs)),
                'mean_delta': float(np.mean(deltas)),
                'std_delta': float(np.std(deltas)),
                'targets_improved': sum(1 for d in deltas if d > 0),
                'total_targets': len(deltas),
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
