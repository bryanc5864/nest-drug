#!/usr/bin/env python3
"""
DUD-E Benchmark for NEST-DRUG

DUD-E (Database of Useful Decoys: Enhanced) is the standard virtual screening
benchmark with property-matched decoys.

Targets included:
- EGFR: Kinase
- DRD2: GPCR
- JAK2: Kinase
- ADRB2: GPCR
- ESR1: Nuclear Receptor
- PPARG: Nuclear Receptor
- HDAC2: Enzyme
- BACE1: Protease
"""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from benchmarks.data_loaders import load_dude_target, load_all_dude, DUDE_TARGETS
from benchmarks.metrics import calculate_all_vs_metrics, print_metrics_summary
from models.nest_drug import create_nest_drug
from training.data_utils import smiles_to_graph
from torch.cuda.amp import autocast


def score_compounds(model, smiles_list, device, batch_size=256):
    """Score compounds using NEST-DRUG model."""
    model.eval()
    scores = []
    valid_indices = []

    with torch.no_grad():
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Scoring"):
            batch_smiles = smiles_list[i:i+batch_size]
            batch_graphs = []
            batch_valid = []

            for j, smi in enumerate(batch_smiles):
                graph = smiles_to_graph(smi)
                if graph is not None:
                    batch_graphs.append(graph)
                    batch_valid.append(i + j)

            if not batch_graphs:
                continue

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
            program_ids = torch.zeros(n_mols, dtype=torch.long, device=device)
            assay_ids = torch.zeros(n_mols, dtype=torch.long, device=device)
            round_ids = torch.zeros(n_mols, dtype=torch.long, device=device)

            with autocast(enabled=True):
                predictions = model(
                    node_features=node_features,
                    edge_index=edge_index,
                    edge_features=edge_features,
                    batch=batch_tensor,
                    program_ids=program_ids,
                    assay_ids=assay_ids,
                    round_ids=round_ids,
                )

            if 'pchembl_median' in predictions:
                batch_scores = predictions['pchembl_median'].cpu().numpy().flatten()
            else:
                first_key = [k for k in predictions.keys() if k not in ['h_mol', 'h_mod']][0]
                batch_scores = predictions[first_key].cpu().numpy().flatten()

            scores.extend([float(s) for s in batch_scores])
            valid_indices.extend(batch_valid)

    return scores, valid_indices


def run_dude_benchmark(model, device, output_dir, targets=None, batch_size=256):
    """Run NEST-DRUG on all DUD-E targets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if targets is None:
        targets = DUDE_TARGETS

    print("\n" + "="*70)
    print("LOADING DUD-E DATA")
    print("="*70)

    dude_data = load_all_dude(targets=targets)

    if not dude_data:
        print("ERROR: No DUD-E data found. Run download script first.")
        return None

    all_results = {}

    for target_name, df in dude_data.items():
        print(f"\n{'='*70}")
        print(f"BENCHMARKING: {target_name.upper()}")
        print(f"{'='*70}")

        n_actives = df['is_active'].sum()
        n_decoys = (df['is_active'] == 0).sum()
        n_total = len(df)
        print(f"  Actives: {n_actives}")
        print(f"  Decoys: {n_decoys}")
        print(f"  Active rate: {n_actives/n_total*100:.2f}%")

        smiles_list = df['smiles'].tolist()
        scores, valid_indices = score_compounds(model, smiles_list, device, batch_size)

        scored_df = df.iloc[valid_indices].copy()
        scored_df['pred_score'] = scores

        y_true = scored_df['is_active'].values
        y_score = scored_df['pred_score'].values

        metrics = calculate_all_vs_metrics(y_true, y_score, name=target_name)
        all_results[target_name] = metrics

        print_metrics_summary(metrics, title=f"{target_name.upper()} Results")

        scored_df.to_parquet(output_dir / f"{target_name}_scored.parquet")

    # Aggregate
    print("\n" + "="*70)
    print("AGGREGATE RESULTS")
    print("="*70)

    summary_df = pd.DataFrame(all_results).T

    print(f"\nMean ROC-AUC:    {summary_df['roc_auc'].mean():.4f} ± {summary_df['roc_auc'].std():.4f}")
    print(f"Mean BEDROC:     {summary_df['bedroc_20'].mean():.4f} ± {summary_df['bedroc_20'].std():.4f}")
    print(f"Mean EF@1%:      {summary_df['ef_1%_ef'].mean():.2f}x ± {summary_df['ef_1%_ef'].std():.2f}")
    print(f"Mean AUAC:       {summary_df['auac'].mean():.4f} ± {summary_df['auac'].std():.4f}")

    summary_df.to_csv(output_dir / "summary.csv")

    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate figures
    generate_dude_figures(summary_df, output_dir)

    return all_results, summary_df


def generate_dude_figures(summary_df, output_dir):
    """Generate DUD-E visualization figures."""
    output_dir = Path(output_dir)

    targets = summary_df.index.tolist()
    x = np.arange(len(targets))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ROC-AUC
    ax = axes[0, 0]
    ax.bar(x, summary_df['roc_auc'], color='steelblue', alpha=0.8)
    ax.axhline(y=0.5, color='red', linestyle='--', label='Random')
    ax.axhline(y=summary_df['roc_auc'].mean(), color='green', linestyle='-',
               label=f"Mean: {summary_df['roc_auc'].mean():.3f}")
    ax.set_ylabel('ROC-AUC')
    ax.set_title('ROC-AUC by Target')
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in targets], rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # BEDROC
    ax = axes[0, 1]
    ax.bar(x, summary_df['bedroc_20'], color='coral', alpha=0.8)
    ax.axhline(y=summary_df['bedroc_20'].mean(), color='green', linestyle='-',
               label=f"Mean: {summary_df['bedroc_20'].mean():.3f}")
    ax.set_ylabel('BEDROC (α=20)')
    ax.set_title('BEDROC by Target')
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in targets], rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # EF@1%
    ax = axes[1, 0]
    ax.bar(x, summary_df['ef_1%_ef'], color='forestgreen', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle='--', label='Random')
    ax.axhline(y=summary_df['ef_1%_ef'].mean(), color='orange', linestyle='-',
               label=f"Mean: {summary_df['ef_1%_ef'].mean():.1f}x")
    ax.set_ylabel('Enrichment Factor')
    ax.set_title('EF @ 1% by Target')
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in targets], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # AUAC
    ax = axes[1, 1]
    ax.bar(x, summary_df['auac'], color='purple', alpha=0.8)
    ax.axhline(y=0.5, color='red', linestyle='--', label='Random')
    ax.axhline(y=summary_df['auac'].mean(), color='green', linestyle='-',
               label=f"Mean: {summary_df['auac'].mean():.3f}")
    ax.set_ylabel('AUAC')
    ax.set_title('AUAC by Target')
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in targets], rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "dude_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFigures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='DUD-E Benchmark for NEST-DRUG')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='data/external/dude')
    parser.add_argument('--output', type=str, default='results/benchmarks/dude')
    parser.add_argument('--targets', type=str, nargs='+', default=None)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    endpoint_names = []
    for key in state_dict.keys():
        if 'prediction_heads.heads.' in key and '.mlp.0.weight' in key:
            name = key.split('prediction_heads.heads.')[1].split('.mlp')[0]
            endpoint_names.append(name)

    endpoints = {name: {'type': 'regression', 'weight': 1.0} for name in endpoint_names}

    # Get model config from checkpoint if available, otherwise infer from weights
    config = checkpoint.get('config', {})
    if config:
        num_programs = config.get('n_programs', config.get('num_programs', 5))
        num_assays = config.get('n_assays', config.get('num_assays', 50))
        num_rounds = config.get('n_rounds', config.get('num_rounds', 150))
    else:
        # Infer from embedding weight shapes
        num_programs = state_dict['context_module.program_embeddings.embeddings.weight'].shape[0]
        num_assays = state_dict['context_module.assay_embeddings.embeddings.weight'].shape[0]
        num_rounds = state_dict['context_module.round_embeddings.embeddings.weight'].shape[0]
        print(f"Inferred config: programs={num_programs}, assays={num_assays}, rounds={num_rounds}")

    model = create_nest_drug(
        num_programs=num_programs,
        num_assays=num_assays,
        num_rounds=num_rounds,
        endpoints=endpoints,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Run benchmark
    run_dude_benchmark(model, device, args.output, args.targets, args.batch_size)

    print("\n" + "="*70)
    print("DUD-E BENCHMARK COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
