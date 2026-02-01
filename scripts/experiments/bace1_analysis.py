#!/usr/bin/env python3
"""
BACE1 Error Analysis for NEST-DRUG

Investigates why BACE1 is the only DUD-E target with a consistently negative L1 effect
(generic L1 outperforms correct L1 across all model versions by -10 to -13% AUC).

Four analyses:
1. Training-Test Chemical Similarity — Tanimoto between ChEMBL training set and DUD-E actives
2. L1 Embedding Analysis — Is BACE1's embedding an outlier?
3. Prediction Distribution Analysis — How does correct vs generic L1 change score distributions?
4. Training Data Statistics — Unusual data characteristics?

Usage:
    python scripts/experiments/bace1_analysis.py \
        --checkpoint results/v3/best_model.pt \
        --output results/experiments/bace1_analysis \
        --gpu 0
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
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph
from torch.cuda.amp import autocast

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("WARNING: RDKit not available, chemical similarity analysis will be skipped")


# Target-specific program IDs from V3 training
DUDE_TO_PROGRAM_ID = {
    'egfr': 1606,
    'drd2': 1448,
    'adrb2': 580,
    'bace1': 516,
    'esr1': 1628,
    'hdac2': 2177,
    'jak2': 4780,
    'pparg': 3307,
    'cyp3a4': 810,
    'fxa': 1103,
}

# Mapping from DUD-E target to ChEMBL training data target_name
DUDE_TO_CHEMBL_NAME = {
    'bace1': 'Beta-secretase 1',
    'egfr': 'Epidermal growth factor receptor erbB1',
    'drd2': 'Dopamine D2 receptor',
    'fxa': 'Coagulation factor X',
    'esr1': 'Estrogen receptor alpha',
    'jak2': 'Tyrosine-protein kinase JAK2',
    'pparg': 'Peroxisome proliferator-activated receptor gamma',
    'adrb2': 'Beta-2 adrenergic receptor',
    'hdac2': 'Histone deacetylase 2',
    'cyp3a4': 'Cytochrome P450 3A4',
}

DUDE_TARGETS = ['egfr', 'drd2', 'adrb2', 'bace1', 'esr1', 'hdac2', 'jak2', 'pparg', 'cyp3a4', 'fxa']

# Control targets to compare against BACE1
CONTROL_TARGETS = ['egfr', 'drd2', 'fxa']


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

    return model, config, state_dict


def load_dude_smiles(target, data_dir='data/external/dude'):
    """Load DUD-E target actives and decoys as SMILES lists."""
    target_dir = Path(data_dir) / target

    actives = []
    actives_file = target_dir / "actives_final.smi"
    if not actives_file.exists():
        return None, None
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


# =============================================================================
# Analysis 1: Chemical Similarity
# =============================================================================

def compute_morgan_fps(smiles_list, radius=2, n_bits=2048, max_mols=5000):
    """Compute Morgan fingerprints for a list of SMILES."""
    if not HAS_RDKIT:
        return []

    fps = []
    valid_indices = []

    # Subsample if too many
    if len(smiles_list) > max_mols:
        np.random.seed(42)
        indices = np.random.choice(len(smiles_list), max_mols, replace=False)
        smiles_list = [smiles_list[i] for i in indices]

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(fp)
            valid_indices.append(i)

    return fps


def compute_pairwise_tanimoto(fps_a, fps_b, max_pairs=50000):
    """Compute Tanimoto similarities between two sets of fingerprints."""
    if not fps_a or not fps_b:
        return []

    sims = []
    # If too many pairs, subsample
    n_pairs = len(fps_a) * len(fps_b)
    if n_pairs > max_pairs:
        n_a = min(len(fps_a), int(np.sqrt(max_pairs)))
        n_b = min(len(fps_b), int(np.sqrt(max_pairs)))
        np.random.seed(42)
        idx_a = np.random.choice(len(fps_a), n_a, replace=False)
        idx_b = np.random.choice(len(fps_b), n_b, replace=False)
        fps_a = [fps_a[i] for i in idx_a]
        fps_b = [fps_b[i] for i in idx_b]

    for fp_a in fps_a:
        bulk_sims = DataStructs.BulkTanimotoSimilarity(fp_a, fps_b)
        sims.extend(bulk_sims)

    return sims


def analysis_chemical_similarity(targets, training_data_path, dude_data_dir):
    """Analysis 1: Compare training-test chemical similarity across targets."""
    print("\n" + "="*70)
    print("ANALYSIS 1: Training-Test Chemical Similarity")
    print("="*70)

    if not HAS_RDKIT:
        print("  SKIPPED: RDKit not available")
        return {}

    # Load training data
    print("  Loading training data...")
    train_df = pd.read_parquet(training_data_path)
    results = {}

    for target in targets:
        chembl_name = DUDE_TO_CHEMBL_NAME.get(target)
        if not chembl_name:
            print(f"  {target}: no ChEMBL mapping")
            continue

        # Get training SMILES for this target
        target_train = train_df[train_df['target_name'] == chembl_name]
        train_smiles = target_train['smiles'].unique().tolist()

        # Get DUD-E actives
        actives, _ = load_dude_smiles(target, dude_data_dir)
        if actives is None:
            continue

        print(f"\n  {target.upper()}: {len(train_smiles)} train, {len(actives)} DUD-E actives")

        # Compute fingerprints
        train_fps = compute_morgan_fps(train_smiles, max_mols=2000)
        dude_fps = compute_morgan_fps(actives, max_mols=1000)

        if not train_fps or not dude_fps:
            print(f"    No valid fingerprints")
            continue

        # Compute pairwise similarities
        sims = compute_pairwise_tanimoto(train_fps, dude_fps)

        if not sims:
            continue

        sims_arr = np.array(sims)

        # Also compute intra-training similarity for reference
        intra_sims = compute_pairwise_tanimoto(train_fps[:200], train_fps[:200])
        intra_arr = np.array(intra_sims) if intra_sims else np.array([0])

        results[target] = {
            'n_train': len(train_smiles),
            'n_dude_actives': len(actives),
            'n_train_fps': len(train_fps),
            'n_dude_fps': len(dude_fps),
            'mean_tanimoto_train_vs_dude': float(sims_arr.mean()),
            'median_tanimoto_train_vs_dude': float(np.median(sims_arr)),
            'max_tanimoto_train_vs_dude': float(sims_arr.max()),
            'std_tanimoto_train_vs_dude': float(sims_arr.std()),
            'pct_above_0.5': float((sims_arr > 0.5).mean()),
            'pct_above_0.7': float((sims_arr > 0.7).mean()),
            'mean_intra_train_tanimoto': float(intra_arr.mean()),
        }

        print(f"    Mean Tanimoto (train↔DUD-E): {sims_arr.mean():.4f}")
        print(f"    Max Tanimoto:                {sims_arr.max():.4f}")
        print(f"    % > 0.5:                     {(sims_arr > 0.5).mean()*100:.1f}%")
        print(f"    Mean intra-train sim:        {intra_arr.mean():.4f}")

    return results


# =============================================================================
# Analysis 2: L1 Embedding Analysis
# =============================================================================

def analysis_embedding(state_dict):
    """Analysis 2: Check if BACE1's L1 embedding is an outlier."""
    print("\n" + "="*70)
    print("ANALYSIS 2: L1 Embedding Analysis")
    print("="*70)

    # Extract program embeddings
    embeddings = state_dict['context_module.program_embeddings.embeddings.weight'].cpu().numpy()
    n_programs, emb_dim = embeddings.shape
    print(f"  Program embeddings: {n_programs} x {emb_dim}")

    results = {}

    # Compute norms
    norms = np.linalg.norm(embeddings, axis=1)

    # Compute centroid
    centroid = embeddings.mean(axis=0)
    centroid_dists = np.linalg.norm(embeddings - centroid, axis=1)

    # Overall stats
    results['overall'] = {
        'n_programs': int(n_programs),
        'embedding_dim': int(emb_dim),
        'mean_norm': float(norms.mean()),
        'std_norm': float(norms.std()),
        'mean_centroid_dist': float(centroid_dists.mean()),
        'std_centroid_dist': float(centroid_dists.std()),
    }

    # Per DUD-E target analysis
    results['targets'] = {}
    for target, pid in DUDE_TO_PROGRAM_ID.items():
        if pid >= n_programs:
            print(f"  {target}: program_id {pid} exceeds embedding table size {n_programs}")
            continue

        emb = embeddings[pid]
        norm = float(norms[pid])
        dist_to_centroid = float(centroid_dists[pid])

        # Compute cosine similarity to all other DUD-E target embeddings
        cosine_sims = {}
        for other_target, other_pid in DUDE_TO_PROGRAM_ID.items():
            if other_target == target or other_pid >= n_programs:
                continue
            other_emb = embeddings[other_pid]
            cos = float(np.dot(emb, other_emb) / (np.linalg.norm(emb) * np.linalg.norm(other_emb) + 1e-8))
            cosine_sims[other_target] = cos

        # Z-score for norm and centroid distance
        norm_zscore = float((norm - norms.mean()) / (norms.std() + 1e-8))
        dist_zscore = float((dist_to_centroid - centroid_dists.mean()) / (centroid_dists.std() + 1e-8))

        # Nearest neighbor among DUD-E targets
        if cosine_sims:
            nn_target = max(cosine_sims, key=cosine_sims.get)
            nn_cosine = cosine_sims[nn_target]
        else:
            nn_target, nn_cosine = None, None

        # Mean cosine to other DUD-E targets
        mean_cosine = float(np.mean(list(cosine_sims.values()))) if cosine_sims else 0.0

        results['targets'][target] = {
            'program_id': pid,
            'norm': norm,
            'norm_zscore': norm_zscore,
            'dist_to_centroid': dist_to_centroid,
            'dist_zscore': dist_zscore,
            'mean_cosine_to_dude_targets': mean_cosine,
            'nearest_neighbor': nn_target,
            'nearest_cosine': nn_cosine,
            'cosine_similarities': cosine_sims,
        }

        print(f"  {target:<8s} (id={pid:>4d}): norm={norm:.4f} (z={norm_zscore:+.2f}), "
              f"centroid_dist={dist_to_centroid:.4f} (z={dist_zscore:+.2f}), "
              f"mean_cos={mean_cosine:.4f}, NN={nn_target}")

    return results


# =============================================================================
# Analysis 3: Prediction Distribution Analysis
# =============================================================================

def score_compounds(model, smiles_list, device, program_id=0, batch_size=256):
    """Score compounds with a specific program_id."""
    model.eval()
    scores = []
    valid_indices = []

    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i+batch_size]
            graphs = []
            batch_valid = []

            for j, smi in enumerate(batch_smiles):
                g = smiles_to_graph(smi)
                if g is not None:
                    graphs.append(g)
                    batch_valid.append(i + j)

            if not graphs:
                continue

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

            n_mols = len(graphs)
            program_ids = torch.full((n_mols,), program_id, dtype=torch.long, device=device)
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


def analysis_prediction_distributions(model, device, targets, dude_data_dir, batch_size=256):
    """Analysis 3: Compare prediction distributions with correct vs generic L1."""
    print("\n" + "="*70)
    print("ANALYSIS 3: Prediction Distribution Analysis")
    print("="*70)

    results = {}

    for target in targets:
        actives, decoys = load_dude_smiles(target, dude_data_dir)
        if actives is None:
            continue

        correct_pid = DUDE_TO_PROGRAM_ID.get(target, 0)
        all_smiles = actives + decoys
        labels = np.array([1] * len(actives) + [0] * len(decoys))

        print(f"\n  {target.upper()} (L1={correct_pid}, {len(actives)} actives, {len(decoys)} decoys)")

        # Score with correct L1
        scores_correct, valid_correct = score_compounds(
            model, all_smiles, device, program_id=correct_pid, batch_size=batch_size
        )

        # Score with generic L1=0
        scores_generic, valid_generic = score_compounds(
            model, all_smiles, device, program_id=0, batch_size=batch_size
        )

        # Use only indices valid in both
        valid_set = set(valid_correct) & set(valid_generic)
        if not valid_set:
            continue

        # Build aligned arrays
        correct_map = dict(zip(valid_correct, scores_correct))
        generic_map = dict(zip(valid_generic, scores_generic))

        valid_list = sorted(valid_set)
        sc = np.array([correct_map[v] for v in valid_list])
        sg = np.array([generic_map[v] for v in valid_list])
        lab = labels[valid_list]

        active_mask = lab == 1
        decoy_mask = lab == 0

        # AUC
        auc_correct = roc_auc_score(lab, sc) if len(set(lab)) > 1 else float('nan')
        auc_generic = roc_auc_score(lab, sg) if len(set(lab)) > 1 else float('nan')

        # Distribution stats
        target_result = {
            'program_id': correct_pid,
            'n_actives': int(active_mask.sum()),
            'n_decoys': int(decoy_mask.sum()),
            'auc_correct_l1': float(auc_correct),
            'auc_generic_l1': float(auc_generic),
            'auc_delta': float(auc_correct - auc_generic),
            'correct_l1': {
                'active_mean': float(sc[active_mask].mean()),
                'active_std': float(sc[active_mask].std()),
                'decoy_mean': float(sc[decoy_mask].mean()),
                'decoy_std': float(sc[decoy_mask].std()),
                'separation': float(sc[active_mask].mean() - sc[decoy_mask].mean()),
            },
            'generic_l1': {
                'active_mean': float(sg[active_mask].mean()),
                'active_std': float(sg[active_mask].std()),
                'decoy_mean': float(sg[decoy_mask].mean()),
                'decoy_std': float(sg[decoy_mask].std()),
                'separation': float(sg[active_mask].mean() - sg[decoy_mask].mean()),
            },
            'score_shift': {
                'mean_shift_actives': float((sc[active_mask] - sg[active_mask]).mean()),
                'mean_shift_decoys': float((sc[decoy_mask] - sg[decoy_mask]).mean()),
                'std_shift_actives': float((sc[active_mask] - sg[active_mask]).std()),
                'std_shift_decoys': float((sc[decoy_mask] - sg[decoy_mask]).std()),
            }
        }

        results[target] = target_result

        sep_c = target_result['correct_l1']['separation']
        sep_g = target_result['generic_l1']['separation']
        shift_a = target_result['score_shift']['mean_shift_actives']
        shift_d = target_result['score_shift']['mean_shift_decoys']

        print(f"    AUC: correct={auc_correct:.4f}, generic={auc_generic:.4f}, delta={auc_correct-auc_generic:+.4f}")
        print(f"    Separation (active-decoy mean): correct={sep_c:.4f}, generic={sep_g:.4f}")
        print(f"    Score shift (correct-generic): actives={shift_a:+.4f}, decoys={shift_d:+.4f}")

    return results


# =============================================================================
# Analysis 4: Training Data Statistics
# =============================================================================

def analysis_training_stats(targets, training_data_path):
    """Analysis 4: Compare training data characteristics across targets."""
    print("\n" + "="*70)
    print("ANALYSIS 4: Training Data Statistics")
    print("="*70)

    train_df = pd.read_parquet(training_data_path)
    results = {}

    for target in targets:
        chembl_name = DUDE_TO_CHEMBL_NAME.get(target)
        if not chembl_name:
            continue

        target_data = train_df[train_df['target_name'] == chembl_name]
        if len(target_data) == 0:
            print(f"  {target}: no training data found for '{chembl_name}'")
            continue

        n_compounds = len(target_data)
        n_unique_smiles = target_data['smiles'].nunique()
        n_assays = target_data['assay_chembl_id'].nunique() if 'assay_chembl_id' in target_data.columns else 0
        n_measurement_types = target_data['standard_type'].nunique() if 'standard_type' in target_data.columns else 0

        # pActivity distribution
        if 'pchembl_median' in target_data.columns:
            pact_col = 'pchembl_median'
        elif 'pActivity' in target_data.columns:
            pact_col = 'pActivity'
        else:
            pact_col = None

        if pact_col:
            pact_values = target_data[pact_col].dropna()
            mean_pact = float(pact_values.mean())
            std_pact = float(pact_values.std())
            median_pact = float(pact_values.median())
            # Class balance (active if pActivity >= 6.5, i.e. ~300 nM)
            n_active = int((pact_values >= 6.5).sum())
            n_inactive = int((pact_values < 6.5).sum())
            active_fraction = float(n_active / len(pact_values)) if len(pact_values) > 0 else 0
        else:
            mean_pact = std_pact = median_pact = float('nan')
            n_active = n_inactive = 0
            active_fraction = float('nan')

        results[target] = {
            'chembl_name': chembl_name,
            'n_compounds': n_compounds,
            'n_unique_smiles': n_unique_smiles,
            'n_assays': n_assays,
            'n_measurement_types': n_measurement_types,
            'mean_pactivity': mean_pact,
            'std_pactivity': std_pact,
            'median_pactivity': median_pact,
            'n_active_6.5': n_active,
            'n_inactive_6.5': n_inactive,
            'active_fraction': active_fraction,
        }

        print(f"  {target:<8s}: {n_compounds:>6d} compounds, {n_unique_smiles:>6d} unique SMILES, "
              f"{n_assays:>4d} assays, mean pAct={mean_pact:.2f}±{std_pact:.2f}, "
              f"active%={active_fraction*100:.1f}%")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='BACE1 Error Analysis for NEST-DRUG')
    parser.add_argument('--checkpoint', type=str, default='results/v3/best_model.pt',
                        help='Model checkpoint path')
    parser.add_argument('--data-dir', type=str, default='data/external/dude',
                        help='DUD-E data directory')
    parser.add_argument('--training-data', type=str,
                        default='data/processed/portfolio/chembl_potency_all.parquet',
                        help='Training data parquet file')
    parser.add_argument('--output', type=str, default='results/experiments/bace1_analysis',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--skip-similarity', action='store_true',
                        help='Skip chemical similarity analysis (slow)')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load model
    print("\nLoading model...")
    model, config, state_dict = load_model(args.checkpoint, device)
    print(f"Config: {config}")

    # Targets to analyze: BACE1 + controls
    analysis_targets = ['bace1'] + [t for t in CONTROL_TARGETS if t != 'bace1']

    all_results = {
        'checkpoint': args.checkpoint,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'analysis_targets': analysis_targets,
    }

    # Analysis 1: Chemical Similarity
    if not args.skip_similarity and HAS_RDKIT:
        sim_results = analysis_chemical_similarity(
            analysis_targets, args.training_data, args.data_dir
        )
        all_results['chemical_similarity'] = sim_results
    else:
        print("\nAnalysis 1: SKIPPED")
        all_results['chemical_similarity'] = {}

    # Analysis 2: L1 Embedding Analysis
    emb_results = analysis_embedding(state_dict)
    all_results['embedding_analysis'] = emb_results

    # Analysis 3: Prediction Distribution Analysis
    pred_results = analysis_prediction_distributions(
        model, device, analysis_targets, args.data_dir, args.batch_size
    )
    all_results['prediction_distributions'] = pred_results

    # Analysis 4: Training Data Statistics
    stat_results = analysis_training_stats(analysis_targets, args.training_data)
    all_results['training_stats'] = stat_results

    # ==========================================================================
    # Summary: Compare BACE1 vs Controls
    # ==========================================================================
    print("\n" + "="*70)
    print("SUMMARY: BACE1 vs Control Targets")
    print("="*70)

    bace1_emb = emb_results.get('targets', {}).get('bace1', {})
    if bace1_emb:
        print(f"\n  Embedding:")
        print(f"    BACE1 norm z-score:     {bace1_emb.get('norm_zscore', 'N/A')}")
        print(f"    BACE1 centroid z-score: {bace1_emb.get('dist_zscore', 'N/A')}")
        print(f"    BACE1 mean cosine:      {bace1_emb.get('mean_cosine_to_dude_targets', 'N/A')}")

    bace1_pred = pred_results.get('bace1', {})
    if bace1_pred:
        print(f"\n  Prediction Distribution:")
        print(f"    BACE1 AUC delta (correct-generic): {bace1_pred.get('auc_delta', 'N/A')}")
        print(f"    Active score shift:  {bace1_pred.get('score_shift', {}).get('mean_shift_actives', 'N/A')}")
        print(f"    Decoy score shift:   {bace1_pred.get('score_shift', {}).get('mean_shift_decoys', 'N/A')}")

    bace1_sim = all_results.get('chemical_similarity', {}).get('bace1', {})
    if bace1_sim:
        print(f"\n  Chemical Similarity (train↔DUD-E):")
        print(f"    BACE1 mean Tanimoto: {bace1_sim.get('mean_tanimoto_train_vs_dude', 'N/A')}")
        for ctrl in CONTROL_TARGETS:
            ctrl_sim = all_results.get('chemical_similarity', {}).get(ctrl, {})
            if ctrl_sim:
                print(f"    {ctrl:<5s} mean Tanimoto: {ctrl_sim.get('mean_tanimoto_train_vs_dude', 'N/A')}")

    bace1_stats = stat_results.get('bace1', {})
    if bace1_stats:
        print(f"\n  Training Data:")
        print(f"    BACE1: {bace1_stats.get('n_compounds', 'N/A')} compounds, "
              f"{bace1_stats.get('n_unique_smiles', 'N/A')} unique, "
              f"active%={bace1_stats.get('active_fraction', 0)*100:.1f}%")
        for ctrl in CONTROL_TARGETS:
            ctrl_stats = stat_results.get(ctrl, {})
            if ctrl_stats:
                print(f"    {ctrl:<5s}: {ctrl_stats.get('n_compounds', 'N/A')} compounds, "
                      f"{ctrl_stats.get('n_unique_smiles', 'N/A')} unique, "
                      f"active%={ctrl_stats.get('active_fraction', 0)*100:.1f}%")

    # Save results
    output_file = output_dir / 'bace1_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
