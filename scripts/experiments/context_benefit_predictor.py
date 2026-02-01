#!/usr/bin/env python3
"""
Context Benefit Predictor: When Does L1 Help vs Hurt?

Addresses reviewer critique E4/W5: "No diagnostic is provided to predict
when context will fail."

Correlates L1 delta (correct - generic) with observable target properties
across all 10 DUD-E targets to identify predictors of context benefit:

1. Training set size (# compounds for target in ChEMBL)
2. Training-test chemical similarity (mean Tanimoto, from BACE1 analysis)
3. L1 embedding norm (from BACE1 analysis)
4. L1 embedding centroid distance
5. Mean training pActivity
6. Training pActivity std
7. Active fraction in training data
8. DUD-E active count
9. DUD-E active:decoy ratio

If strong correlations exist, practitioners can predict BEFORE evaluation
whether L1 context will help or hurt for a new target.

Usage:
    python scripts/experiments/context_benefit_predictor.py \
        --output results/experiments/context_benefit_predictor
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats


DUDE_TARGETS = ['egfr', 'drd2', 'adrb2', 'bace1', 'esr1', 'hdac2', 'jak2', 'pparg', 'cyp3a4', 'fxa']


def load_existing_results():
    """Load results from previous experiments to build feature matrix."""

    # L1 ablation results (V3, from film_ablation or statistical_significance)
    l1_deltas = {}

    film_path = Path('results/experiments/film_ablation/film_ablation_results.json')
    if film_path.exists():
        with open(film_path) as f:
            data = json.load(f)
        film_aucs = data['results']['film']
        nc_aucs = data['results']['no_context']
        for target in DUDE_TARGETS:
            if target in film_aucs and target in nc_aucs:
                l1_deltas[target] = film_aucs[target] - nc_aucs[target]
        print(f"  Loaded L1 deltas from FiLM ablation: {len(l1_deltas)} targets")

    # BACE1 analysis (has Tanimoto, embedding, training stats for 4 targets)
    bace1_path = Path('results/experiments/bace1_analysis/bace1_analysis_results.json')
    bace1_data = None
    if bace1_path.exists():
        with open(bace1_path) as f:
            bace1_data = json.load(f)
        print(f"  Loaded BACE1 analysis data")

    # Statistical significance results
    stat_path = Path('results/experiments/statistical_significance')
    stat_data = {}
    for f in stat_path.glob('*.json') if stat_path.exists() else []:
        with open(f) as fh:
            stat_data[f.stem] = json.load(fh)
    if stat_data:
        print(f"  Loaded statistical significance data: {len(stat_data)} files")

    return l1_deltas, bace1_data, stat_data


def compute_training_stats(target_name_variants):
    """Compute training data statistics for a target from ChEMBL data."""
    # Try loading portfolio data
    portfolio_path = Path('data/processed/portfolio/chembl_potency_all.parquet')
    if not portfolio_path.exists():
        return None

    df = pd.read_parquet(portfolio_path)

    # Find rows matching this target
    mask = df['target_name'].str.lower().isin([v.lower() for v in target_name_variants])
    target_df = df[mask]

    if len(target_df) == 0:
        return None

    return {
        'n_compounds': len(target_df),
        'n_unique_smiles': target_df['smiles'].nunique(),
        'n_assays': target_df['assay_chembl_id'].nunique() if 'assay_chembl_id' in target_df.columns else 0,
        'mean_pactivity': float(target_df['pchembl_median'].mean()),
        'std_pactivity': float(target_df['pchembl_median'].std()),
        'active_fraction': float((target_df['pchembl_median'] >= 6.5).mean()),
        'median_pactivity': float(target_df['pchembl_median'].median()),
    }


# ChEMBL target name variants for matching
TARGET_NAME_VARIANTS = {
    'egfr': ['Epidermal growth factor receptor erbB1', 'Epidermal growth factor receptor'],
    'drd2': ['Dopamine D2 receptor', 'Dopamine receptor D2'],
    'adrb2': ['Beta-2 adrenergic receptor'],
    'bace1': ['Beta-secretase 1', 'Beta-site APP cleaving enzyme 1',
              'Beta-site amyloid precursor protein cleaving enzyme 1'],
    'esr1': ['Estrogen receptor alpha', 'Estrogen receptor'],
    'hdac2': ['Histone deacetylase 2'],
    'jak2': ['Tyrosine-protein kinase JAK2'],
    'pparg': ['Peroxisome proliferator-activated receptor gamma',
              'Peroxisome proliferator activated receptor gamma'],
    'cyp3a4': ['Cytochrome P450 3A4'],
    'fxa': ['Coagulation factor X', 'Coagulation factor Xa'],
}


def compute_embedding_stats(checkpoint_path='results/v3/best_model.pt'):
    """Extract L1 embedding properties for all DUD-E targets."""
    import torch

    DUDE_TO_PROGRAM_ID = {
        'egfr': 1606, 'drd2': 1448, 'adrb2': 580, 'bace1': 516,
        'esr1': 1628, 'hdac2': 2177, 'jak2': 4780, 'pparg': 3307,
        'cyp3a4': 810, 'fxa': 1103,
    }

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']

    emb_weight = state_dict['context_module.program_embeddings.embeddings.weight']
    all_norms = torch.norm(emb_weight, dim=1).numpy()
    centroid = emb_weight.mean(dim=0)

    results = {}
    for target in DUDE_TARGETS:
        pid = DUDE_TO_PROGRAM_ID.get(target)
        if pid is None or pid >= emb_weight.shape[0]:
            continue

        emb = emb_weight[pid]
        norm = float(torch.norm(emb).item())
        centroid_dist = float(torch.norm(emb - centroid).item())
        norm_zscore = float((norm - all_norms.mean()) / all_norms.std())
        centroid_zscore = float((centroid_dist - all_norms.mean()) / all_norms.std())

        # Cosine similarity to all others
        cos_sims = []
        for other_target, other_pid in DUDE_TO_PROGRAM_ID.items():
            if other_target != target and other_pid < emb_weight.shape[0]:
                other_emb = emb_weight[other_pid]
                cos = float(torch.nn.functional.cosine_similarity(
                    emb.unsqueeze(0), other_emb.unsqueeze(0)).item())
                cos_sims.append(cos)

        results[target] = {
            'program_id': pid,
            'norm': norm,
            'norm_zscore': norm_zscore,
            'centroid_dist': centroid_dist,
            'centroid_dist_zscore': centroid_zscore,
            'mean_cosine_to_dude_targets': float(np.mean(cos_sims)) if cos_sims else 0,
        }

    return results


def run_correlation_analysis(l1_deltas, training_stats, embedding_stats):
    """Correlate L1 delta with target properties."""
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS: What Predicts L1 Context Benefit?")
    print("="*70)

    # Build feature matrix
    targets = []
    deltas = []
    features = {}

    feature_names = [
        'n_train_compounds', 'mean_pactivity', 'std_pactivity',
        'active_fraction', 'embedding_norm', 'embedding_norm_zscore',
        'centroid_dist', 'mean_cosine_to_others',
    ]

    for fn in feature_names:
        features[fn] = []

    for target in DUDE_TARGETS:
        if target not in l1_deltas:
            continue
        if target not in training_stats or training_stats[target] is None:
            continue
        if target not in embedding_stats:
            continue

        targets.append(target)
        deltas.append(l1_deltas[target])

        ts = training_stats[target]
        es = embedding_stats[target]

        features['n_train_compounds'].append(ts['n_compounds'])
        features['mean_pactivity'].append(ts['mean_pactivity'])
        features['std_pactivity'].append(ts['std_pactivity'])
        features['active_fraction'].append(ts['active_fraction'])
        features['embedding_norm'].append(es['norm'])
        features['embedding_norm_zscore'].append(es['norm_zscore'])
        features['centroid_dist'].append(es['centroid_dist'])
        features['mean_cosine_to_others'].append(es['mean_cosine_to_dude_targets'])

    if len(targets) < 4:
        print(f"  Only {len(targets)} targets with complete data — insufficient for correlation analysis")
        return {}

    deltas = np.array(deltas)

    print(f"\n  Targets with complete data: {len(targets)}")
    print(f"  L1 delta range: [{deltas.min():.4f}, {deltas.max():.4f}]")
    print(f"  L1 delta mean: {deltas.mean():.4f}")

    print(f"\n  {'Feature':<28} {'Pearson r':>10} {'p-value':>10} {'Spearman ρ':>10} {'p-value':>10} {'Direction':>12}")
    print(f"  {'-'*80}")

    correlations = {}

    for fn in feature_names:
        x = np.array(features[fn])
        if np.std(x) < 1e-10:
            continue

        # Pearson
        r, p_pearson = stats.pearsonr(x, deltas)
        # Spearman (rank-based, better for small N)
        rho, p_spearman = stats.spearmanr(x, deltas)

        direction = "+" if r > 0 else "-"
        sig = "***" if p_spearman < 0.01 else "**" if p_spearman < 0.05 else "*" if p_spearman < 0.1 else ""

        print(f"  {fn:<28} {r:>10.4f} {p_pearson:>10.4f} {rho:>10.4f} {p_spearman:>10.4f} {direction:>6} {sig}")

        correlations[fn] = {
            'pearson_r': float(r),
            'pearson_p': float(p_pearson),
            'spearman_rho': float(rho),
            'spearman_p': float(p_spearman),
            'values': [float(v) for v in x],
        }

    # Per-target detail table
    print(f"\n  {'Target':<10} {'L1 Δ':>8} {'N_train':>8} {'Mean pAct':>10} {'Emb Norm':>9} {'Norm z':>8}")
    print(f"  {'-'*54}")
    for i, target in enumerate(targets):
        print(f"  {target:<10} {deltas[i]:>8.4f} "
              f"{features['n_train_compounds'][i]:>8d} "
              f"{features['mean_pactivity'][i]:>10.3f} "
              f"{features['embedding_norm'][i]:>9.3f} "
              f"{features['embedding_norm_zscore'][i]:>8.2f}")

    # Key finding
    best_predictor = None
    best_p = 1.0
    for fn, corr in correlations.items():
        if corr['spearman_p'] < best_p:
            best_p = corr['spearman_p']
            best_predictor = fn

    if best_predictor:
        bc = correlations[best_predictor]
        print(f"\n  BEST PREDICTOR: {best_predictor}")
        print(f"    Spearman ρ = {bc['spearman_rho']:.4f} (p = {bc['spearman_p']:.4f})")
        print(f"    Pearson r = {bc['pearson_r']:.4f} (p = {bc['pearson_p']:.4f})")
        direction = "helps more" if bc['pearson_r'] > 0 else "helps less (or hurts)"
        print(f"    Higher {best_predictor} → L1 context {direction}")

    # Practical guideline
    print(f"\n  PRACTICAL GUIDELINE FOR REVIEWERS:")
    negative_targets = [t for i, t in enumerate(targets) if deltas[i] < 0]
    positive_targets = [t for i, t in enumerate(targets) if deltas[i] > 0]
    print(f"    Targets where L1 helps ({len(positive_targets)}): {', '.join(positive_targets)}")
    print(f"    Targets where L1 hurts ({len(negative_targets)}): {', '.join(negative_targets)}")

    if negative_targets and best_predictor:
        neg_vals = [features[best_predictor][i] for i, t in enumerate(targets) if deltas[i] < 0]
        pos_vals = [features[best_predictor][i] for i, t in enumerate(targets) if deltas[i] > 0]
        if neg_vals and pos_vals:
            print(f"    {best_predictor} for positive targets: {np.mean(pos_vals):.3f} ± {np.std(pos_vals):.3f}")
            print(f"    {best_predictor} for negative targets: {np.mean(neg_vals):.3f} ± {np.std(neg_vals):.3f}")

    return {
        'targets': targets,
        'l1_deltas': [float(d) for d in deltas],
        'correlations': correlations,
        'best_predictor': best_predictor,
        'best_predictor_p': float(best_p) if best_predictor else None,
        'n_positive': len(positive_targets),
        'n_negative': len(negative_targets),
        'positive_targets': positive_targets,
        'negative_targets': negative_targets,
    }


def main():
    parser = argparse.ArgumentParser(description='Context Benefit Predictor')
    parser.add_argument('--output', type=str, default='results/experiments/context_benefit_predictor')
    parser.add_argument('--checkpoint', type=str, default='results/v3/best_model.pt')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("CONTEXT BENEFIT PREDICTOR")
    print("When does L1 context help vs hurt?")
    print("="*70)

    # Load existing results
    print("\nLoading existing experiment results...")
    l1_deltas, bace1_data, stat_data = load_existing_results()

    # Compute training stats for all targets
    print("\nComputing training statistics per target...")
    training_stats = {}
    for target in DUDE_TARGETS:
        variants = TARGET_NAME_VARIANTS.get(target, [])
        ts = compute_training_stats(variants)
        if ts:
            training_stats[target] = ts
            print(f"  {target}: {ts['n_compounds']} compounds, mean pAct={ts['mean_pactivity']:.3f}")
        else:
            print(f"  {target}: no training data found")

    # Compute embedding stats
    print(f"\nComputing embedding statistics from {args.checkpoint}...")
    try:
        embedding_stats = compute_embedding_stats(args.checkpoint)
        for target in DUDE_TARGETS:
            if target in embedding_stats:
                es = embedding_stats[target]
                print(f"  {target}: norm={es['norm']:.3f}, z={es['norm_zscore']:.2f}")
    except Exception as e:
        print(f"  Error loading checkpoint: {e}")
        embedding_stats = {}

    # Run correlation analysis
    correlation_results = run_correlation_analysis(l1_deltas, training_stats, embedding_stats)

    # Save
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'description': 'Context benefit predictor: correlating L1 delta with target properties',
        'note': 'Addresses reviewer critique E4: when does context help vs hurt?',
        'l1_deltas': {t: float(d) for t, d in l1_deltas.items()},
        'training_stats': training_stats,
        'embedding_stats': embedding_stats,
        'correlation_analysis': correlation_results,
    }

    output_file = output_dir / 'context_benefit_predictor.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
