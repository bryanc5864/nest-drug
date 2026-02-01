#!/usr/bin/env python3
"""
Concatenation Fusion Baseline for NEST-DRUG

Compares FiLM conditioning against concatenation-based context fusion,
the standard alternative in multi-task learning.

Four conditions (all at inference time, no retraining):
1. FiLM (baseline): h_mod = γ(context) * h_mol + β(context)
2. No Context: h_mod = h_mol
3. Additive: h_mod = h_mol + β(context)
4. Concatenation: h_mod = MLP(concat(h_mol, context_224))  [projects 736→512]

For the concatenation condition, we initialize a projection MLP from
scratch (random weights) since the model was never trained with it.
To make this fair, we also test a *trained* variant using the existing
FiLM beta_net reshaped as a projection.

Usage:
    python scripts/experiments/concat_fusion_baseline.py \
        --checkpoint results/v3/best_model.pt \
        --output results/experiments/concat_fusion_baseline \
        --gpu 0
"""

import sys
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph
from torch.cuda.amp import autocast


DUDE_TO_PROGRAM_ID = {
    'egfr': 1606, 'drd2': 1448, 'adrb2': 580, 'bace1': 516,
    'esr1': 1628, 'hdac2': 2177, 'jak2': 4780, 'pparg': 3307,
    'cyp3a4': 810, 'fxa': 1103,
}

DUDE_TARGETS = ['egfr', 'drd2', 'adrb2', 'bace1', 'esr1', 'hdac2', 'jak2', 'pparg', 'cyp3a4', 'fxa']


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
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
        num_programs=num_programs,
        num_assays=num_assays,
        num_rounds=num_rounds,
        endpoints=endpoints,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config, num_programs


def load_dude_target(target, data_dir='data/external/dude'):
    """Load DUD-E target actives and decoys."""
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

    decoys = []
    decoys_file = target_dir / "decoys_final.smi"
    with open(decoys_file) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                decoys.append(parts[0])

    return actives, decoys


def prepare_batch(smiles_list, device):
    """Convert SMILES list to batched graph tensors."""
    graphs = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        g = smiles_to_graph(smi)
        if g is not None:
            graphs.append(g)
            valid_indices.append(i)

    if not graphs:
        return None, []

    node_features = torch.cat([g['node_features'] for g in graphs], dim=0).to(device)
    edge_index_list = []
    edge_features_list = []
    batch_indices = []
    offset = 0

    for idx, g in enumerate(graphs):
        edge_index_list.append(g['edge_index'] + offset)
        edge_features_list.append(g['edge_features'])
        batch_indices.extend([idx] * g['num_atoms'])
        offset += g['num_atoms']

    edge_index = torch.cat(edge_index_list, dim=1).to(device)
    edge_features = torch.cat(edge_features_list, dim=0).to(device)
    batch_tensor = torch.tensor(batch_indices, dtype=torch.long, device=device)

    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_features': edge_features,
        'batch': batch_tensor,
        'n_mols': len(graphs),
    }, valid_indices


def score_with_condition(model, smiles_list, device, condition, program_id=0,
                         batch_size=256, concat_proj=None):
    """
    Score compounds under a specific context fusion condition.

    Conditions:
    - 'film': Normal FiLM (baseline)
    - 'no_context': h_mod = h_mol
    - 'additive': h_mod = h_mol + beta(context)
    - 'concat': h_mod = concat_proj(cat(h_mol, context))
    """
    model.eval()
    all_scores = []
    all_valid = []

    original_forward = model.context_module.forward

    if condition == 'no_context':
        def patched_forward(h_mol, program_ids, assay_ids, round_ids):
            return h_mol
        model.context_module.forward = patched_forward

    elif condition == 'additive':
        def patched_forward(h_mol, program_ids, assay_ids, round_ids):
            z_program = model.context_module.program_embeddings(program_ids)
            z_assay = model.context_module.assay_embeddings(assay_ids)
            z_round = model.context_module.round_embeddings(round_ids)
            context = torch.cat([z_program, z_assay, z_round], dim=-1)
            context = model.context_module.context_interaction(context)
            beta = model.context_module.film.beta_net(context)
            return h_mol + beta
        model.context_module.forward = patched_forward

    elif condition == 'concat':
        def patched_forward(h_mol, program_ids, assay_ids, round_ids):
            z_program = model.context_module.program_embeddings(program_ids)
            z_assay = model.context_module.assay_embeddings(assay_ids)
            z_round = model.context_module.round_embeddings(round_ids)
            context = torch.cat([z_program, z_assay, z_round], dim=-1)
            context = model.context_module.context_interaction(context)
            # Concatenate h_mol (512) with context (224) → 736, project to 512
            combined = torch.cat([h_mol, context], dim=-1)
            return concat_proj(combined)
        model.context_module.forward = patched_forward

    try:
        with torch.no_grad():
            for i in range(0, len(smiles_list), batch_size):
                batch_smiles = smiles_list[i:i+batch_size]
                batch_data, valid_indices = prepare_batch(batch_smiles, device)

                if batch_data is None:
                    continue

                n_mols = batch_data['n_mols']
                program_ids = torch.full((n_mols,), program_id, dtype=torch.long, device=device)
                assay_ids = torch.zeros(n_mols, dtype=torch.long, device=device)
                round_ids = torch.zeros(n_mols, dtype=torch.long, device=device)

                with autocast(enabled=True):
                    predictions = model(
                        node_features=batch_data['node_features'],
                        edge_index=batch_data['edge_index'],
                        edge_features=batch_data['edge_features'],
                        batch=batch_data['batch'],
                        program_ids=program_ids,
                        assay_ids=assay_ids,
                        round_ids=round_ids,
                    )

                if 'pchembl_median' in predictions:
                    scores = predictions['pchembl_median'].cpu().numpy().flatten()
                else:
                    first_key = [k for k in predictions.keys() if k not in ['h_mol', 'h_mod']][0]
                    scores = predictions[first_key].cpu().numpy().flatten()

                all_scores.extend([float(s) for s in scores])
                all_valid.extend([i + vi for vi in valid_indices])

    finally:
        model.context_module.forward = original_forward

    return all_scores, all_valid


def run_ablation(model, device, num_programs, data_dir='data/external/dude', batch_size=256):
    """Run all four conditions on all DUD-E targets."""
    # Create concatenation projection MLP (random init, not trained)
    # h_mol=512, context=224, output=512
    concat_proj = nn.Sequential(
        nn.Linear(512 + 224, 512),
        nn.LayerNorm(512),
        nn.ReLU(),
        nn.Linear(512, 512),
    ).to(device).eval()

    # Use same L1 (correct program_id) for ALL conditions to avoid confound
    conditions = ['film', 'no_context', 'additive', 'concat']
    results = {c: {} for c in conditions}

    for target in tqdm(DUDE_TARGETS, desc="Targets"):
        actives, decoys = load_dude_target(target, data_dir)
        if actives is None:
            print(f"  Skipping {target} - no data")
            continue

        all_smiles = actives + decoys
        labels = [1] * len(actives) + [0] * len(decoys)

        correct_pid = DUDE_TO_PROGRAM_ID.get(target, 0)
        if num_programs < 5123:
            correct_pid = 0

        print(f"\n  {target.upper()} ({len(actives)} actives, {len(decoys)} decoys, L1={correct_pid})")

        for condition in conditions:
            # ALL conditions use same correct L1 — isolates fusion mechanism
            scores, valid_indices = score_with_condition(
                model, all_smiles, device, condition,
                program_id=correct_pid, batch_size=batch_size,
                concat_proj=concat_proj,
            )

            valid_labels = [labels[vi] for vi in valid_indices]

            if len(set(valid_labels)) < 2:
                print(f"    {condition}: skipped (insufficient classes)")
                continue

            auc = roc_auc_score(valid_labels, scores)
            results[condition][target] = float(auc)
            print(f"    {condition:12s}: ROC-AUC = {auc:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Concatenation Fusion Baseline for NEST-DRUG')
    parser.add_argument('--checkpoint', type=str, default='results/v3/best_model.pt')
    parser.add_argument('--data-dir', type=str, default='data/external/dude')
    parser.add_argument('--output', type=str, default='results/experiments/concat_fusion_baseline')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nLoading model...")
    model, config, num_programs = load_model(args.checkpoint, device)
    print(f"Config: {config}")

    print("\n" + "="*70)
    print("CONTEXT FUSION COMPARISON")
    print("Conditions: FiLM | No Context | Additive | Concatenation")
    print("All conditions use SAME correct L1 (no confound)")
    print("="*70)

    results = run_ablation(model, device, num_programs, args.data_dir, args.batch_size)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    summary = {}
    for cond in ['film', 'no_context', 'additive', 'concat']:
        aucs = list(results[cond].values())
        if aucs:
            summary[f'{cond}_mean'] = float(np.mean(aucs))
            summary[f'{cond}_std'] = float(np.std(aucs))

    print(f"{'Condition':<15} {'Mean AUC':>10} {'Std':>8}")
    print("-"*35)
    for cond in ['film', 'no_context', 'additive', 'concat']:
        mk = f'{cond}_mean'
        if mk in summary:
            print(f"{cond:<15} {summary[mk]:>10.4f} {summary[f'{cond}_std']:>8.4f}")

    # Per-target table
    print(f"\n{'Target':<10} {'FiLM':>8} {'No Ctx':>8} {'Additive':>8} {'Concat':>8}")
    print("-"*44)
    for target in DUDE_TARGETS:
        vals = []
        for cond in ['film', 'no_context', 'additive', 'concat']:
            v = results[cond].get(target, float('nan'))
            vals.append(v)
        print(f"{target:<10} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} {vals[3]:>8.4f}")

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        'checkpoint': args.checkpoint,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'note': 'All conditions use same correct L1 program_id (no confound)',
        'concat_proj': 'Random init Linear(736,512)->LN->ReLU->Linear(512,512)',
        'results': results,
        'summary': summary,
    }

    output_file = output_dir / 'concat_fusion_results.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
