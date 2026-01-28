#!/usr/bin/env python3
"""
DMTA Replay Simulation: L3 Round Context Validation

Simulates Design-Make-Test-Analyze cycles on historical program data.
Compares model-guided compound selection (with/without L3 round context)
against random baseline.

Key metric: "experiments saved" - how many fewer compounds need testing
to find N hits when using model guidance + L3 context.

Usage:
    python scripts/experiments/dmta_replay.py --checkpoint results/v3/best_model.pt --gpu 0
    python scripts/experiments/dmta_replay.py --checkpoint results/v3/best_model.pt --gpu 0 --targets egfr drd2 fxa
"""

import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph


# Target-specific L1 program IDs (V3)
DUDE_TO_V3_PROGRAM_ID = {
    'egfr': 1606, 'drd2': 1448, 'cyp3a4': 810, 'fxa': 1103, 'herg': 0,
}

# Program data files
PROGRAM_FILES = {
    'egfr': 'data/processed/programs/program_egfr_augmented.csv',
    'drd2': 'data/processed/programs/program_drd2_augmented.csv',
    'cyp3a4': 'data/processed/programs/program_cyp3a4_augmented.csv',
    'fxa': 'data/processed/programs/program_fxa_augmented.csv',
    'herg': 'data/processed/programs/program_herg_augmented.csv',
}


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
        num_programs=num_programs, num_assays=num_assays,
        num_rounds=num_rounds, endpoints=endpoints,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, {'num_programs': num_programs, 'num_assays': num_assays, 'num_rounds': num_rounds}


def predict_compounds(model, smiles_list, device, program_id=0, assay_id=0, round_id=0):
    """Run predictions for a list of SMILES."""
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


def run_dmta_replay(model, program_data, device, program_id, num_rounds_model,
                    n_select=100, select_fraction=0.3, activity_threshold=7.0, seed_rounds=3):
    """
    Run DMTA replay simulation comparing:
    1. Random selection
    2. Model-guided (no L3 context, round_id=0)
    3. Model-guided (with L3 context, correct round_id)
    """
    # Sort by round
    rounds = sorted(program_data['round_id'].unique())

    # Clamp round_ids to model's max
    max_round = num_rounds_model - 1

    results_per_round = []
    tested = set()

    for round_idx, round_id in enumerate(tqdm(rounds, desc="Rounds")):
        round_data = program_data[program_data['round_id'] == round_id].copy()

        # Skip seed rounds (model "trained" on these)
        if round_idx < seed_rounds:
            tested.update(round_data['smiles'].tolist())
            continue

        # Get untested compounds from this round
        available = round_data[~round_data['smiles'].isin(tested)]
        if len(available) < 10:
            continue

        smiles_list = available['smiles'].tolist()
        activities = available['pActivity'].values
        is_active = (activities > activity_threshold).astype(int)

        n_active_available = is_active.sum()
        n_available = len(available)
        base_rate = n_active_available / max(n_available, 1)

        # Select a fraction of available compounds so model ranking matters
        n_sel = min(n_select, max(int(n_available * select_fraction), 5))

        # === 1. Random selection ===
        np.random.seed(round_idx)
        random_idx = np.random.choice(n_available, n_sel, replace=False)
        random_hits = is_active[random_idx].sum()

        # === 2. Model-guided WITHOUT L3 (round_id=0) ===
        preds_no_l3 = predict_compounds(model, smiles_list, device,
                                         program_id=program_id, round_id=0)
        valid_preds_no_l3 = np.array([p if p is not None else -999 for p in preds_no_l3])
        top_idx_no_l3 = np.argsort(valid_preds_no_l3)[-n_sel:]
        model_no_l3_hits = is_active[top_idx_no_l3].sum()

        # === 3. Model-guided WITH L3 (correct round_id) ===
        clamped_round = min(round_id, max_round)
        preds_with_l3 = predict_compounds(model, smiles_list, device,
                                           program_id=program_id, round_id=clamped_round)
        valid_preds_l3 = np.array([p if p is not None else -999 for p in preds_with_l3])
        top_idx_l3 = np.argsort(valid_preds_l3)[-n_sel:]
        model_l3_hits = is_active[top_idx_l3].sum()

        results_per_round.append({
            'round_id': int(round_id),
            'round_idx': round_idx,
            'n_available': n_available,
            'n_active_available': int(n_active_available),
            'base_rate': float(base_rate),
            'n_selected': n_sel,
            'random_hits': int(random_hits),
            'model_no_l3_hits': int(model_no_l3_hits),
            'model_l3_hits': int(model_l3_hits),
            'enrichment_random': float(random_hits / max(base_rate * n_sel, 0.01)),
            'enrichment_no_l3': float(model_no_l3_hits / max(base_rate * n_sel, 0.01)),
            'enrichment_l3': float(model_l3_hits / max(base_rate * n_sel, 0.01)),
        })

        tested.update(smiles_list)

    return results_per_round


def compute_experiments_to_n_hits(results, target_hits=50):
    """How many experiments needed to find N hits?"""
    cumulative = {'random': 0, 'model_no_l3': 0, 'model_l3': 0}
    experiments = {'random': 0, 'model_no_l3': 0, 'model_l3': 0}

    for r in results:
        for method, key in [('random', 'random_hits'), ('model_no_l3', 'model_no_l3_hits'),
                            ('model_l3', 'model_l3_hits')]:
            if cumulative[method] < target_hits:
                cumulative[method] += r[key]
                experiments[method] += r['n_selected']

    return experiments, cumulative


def main():
    parser = argparse.ArgumentParser(description="DMTA Replay Simulation")
    parser.add_argument('--checkpoint', type=str, default='results/v3/best_model.pt')
    parser.add_argument('--output', type=str, default='results/experiments/dmta_replay')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--targets', type=str, nargs='+', default=['egfr', 'drd2', 'fxa'])
    parser.add_argument('--n-select', type=int, default=100, help='Max compounds to select per round')
    parser.add_argument('--select-fraction', type=float, default=0.3, help='Fraction of available compounds to select per round')
    parser.add_argument('--activity-threshold', type=float, default=7.0, help='pActivity threshold for hits')
    parser.add_argument('--seed-rounds', type=int, default=3, help='Number of seed rounds to skip')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, config = load_model(args.checkpoint, device)
    print(f"Model config: {config}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_target_results = {}

    for target in args.targets:
        print(f"\n{'='*60}")
        print(f"Target: {target.upper()}")
        print(f"{'='*60}")

        data_file = PROGRAM_FILES.get(target)
        if data_file is None or not Path(data_file).exists():
            print(f"  No data file for {target}, skipping")
            continue

        program_data = pd.read_csv(data_file)
        print(f"  Data: {len(program_data)} records, {program_data['round_id'].nunique()} rounds")
        print(f"  Active (pActivity > {args.activity_threshold}): {(program_data['pActivity'] > args.activity_threshold).sum()}")

        program_id = DUDE_TO_V3_PROGRAM_ID.get(target, 0)
        program_id = min(program_id, config['num_programs'] - 1)
        print(f"  L1 program ID: {program_id}")

        results = run_dmta_replay(
            model, program_data, device,
            program_id=program_id,
            num_rounds_model=config['num_rounds'],
            n_select=args.n_select,
            select_fraction=args.select_fraction,
            activity_threshold=args.activity_threshold,
            seed_rounds=args.seed_rounds,
        )

        if not results:
            print(f"  No results for {target}")
            continue

        # Compute summary
        total_random = sum(r['random_hits'] for r in results)
        total_no_l3 = sum(r['model_no_l3_hits'] for r in results)
        total_l3 = sum(r['model_l3_hits'] for r in results)
        total_tested = sum(r['n_selected'] for r in results)

        exp_to_50, cum_50 = compute_experiments_to_n_hits(results, target_hits=50)

        summary = {
            'target': target,
            'program_id': program_id,
            'total_rounds': len(results),
            'total_tested': total_tested,
            'total_hits_random': total_random,
            'total_hits_no_l3': total_no_l3,
            'total_hits_l3': total_l3,
            'hit_rate_random': total_random / max(total_tested, 1),
            'hit_rate_no_l3': total_no_l3 / max(total_tested, 1),
            'hit_rate_l3': total_l3 / max(total_tested, 1),
            'enrichment_no_l3_vs_random': total_no_l3 / max(total_random, 1),
            'enrichment_l3_vs_random': total_l3 / max(total_random, 1),
            'l3_benefit_vs_no_l3': (total_l3 - total_no_l3) / max(total_no_l3, 1),
            'experiments_to_50_hits': exp_to_50,
        }

        print(f"\n  Summary:")
        print(f"    Total rounds: {summary['total_rounds']}")
        print(f"    Total tested: {summary['total_tested']}")
        print(f"    Random hits:     {total_random} ({summary['hit_rate_random']:.2%})")
        print(f"    Model (no L3):   {total_no_l3} ({summary['hit_rate_no_l3']:.2%})")
        print(f"    Model (with L3): {total_l3} ({summary['hit_rate_l3']:.2%})")
        print(f"    L3 benefit:      {summary['l3_benefit_vs_no_l3']:+.1%}")
        print(f"    Experiments to 50 hits: random={exp_to_50['random']}, no_l3={exp_to_50['model_no_l3']}, l3={exp_to_50['model_l3']}")

        all_target_results[target] = {
            'summary': summary,
            'per_round': results,
        }

    # Save results
    results_path = output_dir / 'dmta_replay_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'config': config,
            'activity_threshold': args.activity_threshold,
            'n_select': args.n_select,
            'select_fraction': args.select_fraction,
            'results': all_target_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
