#!/usr/bin/env python3
"""
Run DUD-E benchmark on saved ablation checkpoints.

Usage:
    python scripts/run_dude_on_checkpoints.py --checkpoint-dir results/phase1/ablation --device cuda
"""

import argparse
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
from torch.cuda.amp import autocast

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph
from src.benchmarks.data_loaders import load_all_dude, DUDE_TARGETS
from src.benchmarks.metrics import calculate_all_vs_metrics


class AblatedNESTDRUG(nn.Module):
    """NEST-DRUG with configurable context levels."""

    def __init__(self, base_model, use_l1=True, use_l2=True, use_l3=True):
        super().__init__()
        self.base_model = base_model
        self.use_l1 = use_l1
        self.use_l2 = use_l2
        self.use_l3 = use_l3

    def forward(self, node_features, edge_index, edge_features, batch,
                program_ids, assay_ids, round_ids):
        if not self.use_l1:
            program_ids = torch.zeros_like(program_ids)
        if not self.use_l2:
            assay_ids = torch.zeros_like(assay_ids)
        if not self.use_l3:
            round_ids = torch.zeros_like(round_ids)

        return self.base_model(
            node_features, edge_index, edge_features, batch,
            program_ids, assay_ids, round_ids
        )


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    base_model = create_nest_drug(
        num_programs=config['n_programs'],
        num_assays=config['n_assays'],
        num_rounds=config['n_rounds'],
    )

    model = AblatedNESTDRUG(
        base_model,
        use_l1=config['use_l1'],
        use_l2=config['use_l2'],
        use_l3=config['use_l3'],
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


def score_compounds(model, smiles_list, device, l1_context_id=0, batch_size=256):
    """Score compounds using model.

    Args:
        model: The model to use for scoring
        smiles_list: List of SMILES strings
        device: Device to run on
        l1_context_id: L1 (program/target) context ID for this target
        batch_size: Batch size for inference
    """
    scores = []
    valid_mask = []

    with torch.no_grad():
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Scoring", leave=False):
            batch_smiles = smiles_list[i:i+batch_size]
            batch_graphs = []
            batch_valid = []

            for smi in batch_smiles:
                graph = smiles_to_graph(smi)
                if graph is not None:
                    batch_graphs.append(graph)
                    batch_valid.append(True)
                else:
                    batch_valid.append(False)

            if not batch_graphs:
                valid_mask.extend(batch_valid)
                scores.extend([float('nan')] * len(batch_valid))
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

            n_mols = len(batch_graphs)
            # L1: Use target-specific context ID
            program_ids = torch.full((n_mols,), l1_context_id, dtype=torch.long, device=device)
            # L2/L3: No assay or temporal info in DUD-E, use 0
            assay_ids = torch.zeros(n_mols, dtype=torch.long, device=device)
            round_ids = torch.zeros(n_mols, dtype=torch.long, device=device)

            with autocast(enabled=True):
                predictions = model(
                    node_features, edge_index, edge_features, batch_tensor,
                    program_ids, assay_ids, round_ids
                )

            pred = predictions.get('pActivity', list(predictions.values())[0])
            batch_scores = pred.cpu().numpy().flatten()

            # Map back to original order
            score_idx = 0
            for is_valid in batch_valid:
                if is_valid:
                    scores.append(float(batch_scores[score_idx]))
                    score_idx += 1
                else:
                    scores.append(float('nan'))
            valid_mask.extend(batch_valid)

    return np.array(scores), np.array(valid_mask)


def run_dude_benchmark(model, device, condition_name, use_target_l1=True):
    """Run DUD-E benchmark on a single model.

    Args:
        model: The model to evaluate
        device: Device to run on
        condition_name: Name of the ablation condition
        use_target_l1: If True, use unique L1 context per target. If False, all L1=0.
    """
    print(f"\n{'='*60}")
    print(f"DUD-E BENCHMARK: {condition_name}")
    print(f"L1 Context: {'Per-target' if use_target_l1 else 'All zeros'}")
    print(f"{'='*60}")

    # L1 context mapping: each DUD-E target gets a unique L1 ID
    L1_MAPPING = {
        'egfr': 0,
        'drd2': 1,
        'jak2': 2,
        'adrb2': 3,
        'esr1': 4,
        'pparg': 5,
        'hdac2': 6,
        'fxa': 7,
        'bace1': 8,
        'cyp3a4': 9,
    }

    dude_data = load_all_dude()
    if not dude_data:
        print("ERROR: No DUD-E data found!")
        return {}

    results = {}

    for target_name, df in dude_data.items():
        # Get L1 context for this target
        l1_id = L1_MAPPING.get(target_name, 0) if use_target_l1 else 0

        print(f"\n  {target_name.upper()} (L1={l1_id}):", end=" ")

        smiles_list = df['smiles'].tolist()
        y_true = df['is_active'].values

        scores, valid_mask = score_compounds(model, smiles_list, device, l1_context_id=l1_id)

        # Filter valid
        y_true_valid = y_true[valid_mask]
        scores_valid = scores[valid_mask]

        metrics = calculate_all_vs_metrics(y_true_valid, scores_valid, name=target_name)
        results[target_name] = metrics

        print(f"ROC-AUC={metrics['roc_auc']:.4f}, EF@1%={metrics.get('ef_1', 0):.1f}x")

    # Summary
    aucs = [r['roc_auc'] for r in results.values()]
    print(f"\n  Mean ROC-AUC: {np.mean(aucs):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str, default='results/phase1/ablation')
    parser.add_argument('--output', type=str, default='results/phase1/dude_ablation')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0, help='Which seed to use (default: 0)')
    parser.add_argument('--use-target-l1', action='store_true', default=True,
                        help='Use per-target L1 context (default: True)')
    parser.add_argument('--no-target-l1', action='store_false', dest='use_target_l1',
                        help='Use L1=0 for all targets')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"L1 Context Mode: {'Per-target' if args.use_target_l1 else 'All zeros'}")

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find checkpoints
    conditions = ['L0', 'L0_L1', 'L0_L1_L2', 'L0_L1_L2_L3']
    all_results = {}

    for cond in conditions:
        checkpoint_path = checkpoint_dir / f"{cond}_seed{args.seed}_best.pt"

        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            continue

        print(f"\nLoading {checkpoint_path}")
        model, config = load_checkpoint(checkpoint_path, device)

        results = run_dude_benchmark(model, device, cond, use_target_l1=args.use_target_l1)
        all_results[cond] = results

    # Save results
    output_file = output_dir / f"dude_results_seed{args.seed}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Condition':<15} {'Mean ROC-AUC':<15} {'Best Target':<20} {'Worst Target':<20}")
    print("-" * 70)

    for cond, results in all_results.items():
        if not results:
            continue
        aucs = {t: r['roc_auc'] for t, r in results.items()}
        mean_auc = np.mean(list(aucs.values()))
        best = max(aucs, key=aucs.get)
        worst = min(aucs, key=aucs.get)
        print(f"{cond:<15} {mean_auc:<15.4f} {best}({aucs[best]:.3f})  {worst}({aucs[worst]:.3f})")


if __name__ == '__main__':
    main()
