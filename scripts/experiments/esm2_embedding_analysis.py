#!/usr/bin/env python3
"""
ESM-2 Protein Embedding Analysis for NEST-DRUG

Addresses reviewer critique E5/W9: "Why not use pretrained protein
language model embeddings (ESM-2) as L1 initialization?"

Analyses:
1. Compute ESM-2 embeddings for 10 DUD-E target proteins
2. Compare ESM-2 similarity with learned L1 similarity (correlation)
3. Zero-shot L1 prediction: use ESM-2 similarity to transfer L1 embeddings
4. Evaluate ESM-2-derived L1 on DUD-E benchmark

Requirements: fair-esm (pip install fair-esm), torch, rdkit
Run with: conda activate nest

Usage:
    python scripts/experiments/esm2_embedding_analysis.py \
        --checkpoint results/v3/best_model.pt \
        --output results/experiments/esm2_analysis \
        --gpu 0
"""

import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph
from torch.cuda.amp import autocast


# DUD-E target → program_id mapping (from V3 model)
DUDE_TO_PROGRAM_ID = {
    'egfr': 1606, 'drd2': 1448, 'adrb2': 580, 'bace1': 516,
    'esr1': 1628, 'hdac2': 2177, 'jak2': 4780, 'pparg': 3307,
    'cyp3a4': 810, 'fxa': 1103,
}

DUDE_TARGETS = ['egfr', 'drd2', 'adrb2', 'bace1', 'esr1', 'hdac2', 'jak2', 'pparg', 'cyp3a4', 'fxa']

# UniProt accessions for 10 DUD-E target proteins
UNIPROT_ACCESSIONS = {
    'egfr': 'P00533',   # Epidermal growth factor receptor
    'drd2': 'P14416',   # Dopamine D2 receptor
    'adrb2': 'P07550',  # Beta-2 adrenergic receptor
    'bace1': 'P56817',  # Beta-secretase 1
    'esr1': 'P03372',   # Estrogen receptor alpha
    'hdac2': 'Q92769',  # Histone deacetylase 2
    'jak2': 'O60674',   # Tyrosine-protein kinase JAK2
    'pparg': 'P37231',  # Peroxisome proliferator-activated receptor gamma
    'cyp3a4': 'P08684', # Cytochrome P450 3A4
    'fxa': 'P00742',    # Coagulation factor X (Factor Xa)
}


def fetch_uniprot_sequence(accession):
    """Fetch protein sequence from UniProt REST API."""
    import urllib.request
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            fasta = resp.read().decode('utf-8')
        # Parse FASTA: skip header line, join sequence lines
        lines = fasta.strip().split('\n')
        seq = ''.join(l.strip() for l in lines[1:])
        return seq
    except Exception as e:
        print(f"    Error fetching {accession}: {e}")
        return None


def compute_esm2_embeddings(sequences, device, model_name='esm2_t33_650M_UR50D'):
    """Compute ESM-2 protein embeddings for a dict of sequences."""
    import esm

    print(f"\n  Loading ESM-2 model ({model_name})...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()

    # Get the number of layers for mean representation
    num_layers = model.num_layers

    embeddings = {}
    for target, seq in sequences.items():
        # Truncate to 1022 tokens if needed (ESM-2 max position)
        if len(seq) > 1022:
            print(f"    {target}: truncating from {len(seq)} to 1022 residues")
            seq = seq[:1022]

        data = [(target, seq)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[num_layers], return_contacts=False)

        # Mean pooling over sequence (exclude BOS/EOS tokens)
        token_representations = results["representations"][num_layers]
        seq_len = len(seq)
        # Tokens: [BOS, aa1, aa2, ..., aaN, EOS]
        mean_repr = token_representations[0, 1:seq_len+1].mean(dim=0)
        embeddings[target] = mean_repr.cpu()
        print(f"    {target}: {len(seq)} aa → {mean_repr.shape[0]}-dim embedding")

    return embeddings


def load_model(checkpoint_path, device):
    """Load NEST-DRUG model from checkpoint."""
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

    return model, state_dict


def get_learned_l1_embeddings(state_dict):
    """Extract learned L1 (program) embeddings for DUD-E targets."""
    emb_weight = state_dict['context_module.program_embeddings.embeddings.weight']
    embeddings = {}
    for target in DUDE_TARGETS:
        pid = DUDE_TO_PROGRAM_ID[target]
        if pid < emb_weight.shape[0]:
            embeddings[target] = emb_weight[pid].cpu()
    return embeddings


def cosine_similarity(a, b):
    """Compute cosine similarity between two tensors."""
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def compute_similarity_matrices(embeddings_dict):
    """Compute pairwise cosine similarity matrix for a dict of embeddings."""
    targets = list(embeddings_dict.keys())
    n = len(targets)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = cosine_similarity(
                embeddings_dict[targets[i]],
                embeddings_dict[targets[j]]
            )
    return sim_matrix, targets


def similarity_correlation_analysis(esm_embeddings, l1_embeddings):
    """Analysis 1: Do ESM-2 similarities predict learned L1 similarities?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: ESM-2 vs Learned L1 Similarity Correlation")
    print("=" * 70)

    esm_sim, targets = compute_similarity_matrices(esm_embeddings)
    l1_sim, _ = compute_similarity_matrices(l1_embeddings)

    # Extract upper triangle (excluding diagonal)
    n = len(targets)
    esm_upper = []
    l1_upper = []
    pair_labels = []
    for i in range(n):
        for j in range(i+1, n):
            esm_upper.append(esm_sim[i, j])
            l1_upper.append(l1_sim[i, j])
            pair_labels.append(f"{targets[i]}-{targets[j]}")

    esm_upper = np.array(esm_upper)
    l1_upper = np.array(l1_upper)

    # Pearson and Spearman correlation
    r, p_pearson = stats.pearsonr(esm_upper, l1_upper)
    rho, p_spearman = stats.spearmanr(esm_upper, l1_upper)

    print(f"\n  Pairwise similarities ({len(pair_labels)} pairs):")
    print(f"    Pearson r  = {r:.4f} (p = {p_pearson:.4f})")
    print(f"    Spearman ρ = {rho:.4f} (p = {p_spearman:.4f})")

    if r > 0.5 and p_pearson < 0.05:
        print(f"    → STRONG correlation: Learned L1 captures protein similarity structure")
    elif r > 0.3:
        print(f"    → MODERATE correlation: L1 partially reflects protein similarity")
    else:
        print(f"    → WEAK correlation: L1 captures dataset-specific patterns beyond protein similarity")

    # Print pairwise detail
    print(f"\n  {'Pair':<20} {'ESM-2 sim':>10} {'L1 sim':>10}")
    print(f"  {'-'*40}")
    sorted_idx = np.argsort(esm_upper)[::-1]
    for idx in sorted_idx[:10]:
        print(f"  {pair_labels[idx]:<20} {esm_upper[idx]:>10.4f} {l1_upper[idx]:>10.4f}")

    return {
        'pearson_r': float(r),
        'pearson_p': float(p_pearson),
        'spearman_rho': float(rho),
        'spearman_p': float(p_spearman),
        'esm_similarities': {pair_labels[i]: float(esm_upper[i]) for i in range(len(pair_labels))},
        'l1_similarities': {pair_labels[i]: float(l1_upper[i]) for i in range(len(pair_labels))},
        'esm_sim_matrix': esm_sim.tolist(),
        'l1_sim_matrix': l1_sim.tolist(),
        'targets': targets,
    }


def load_dude_target(target, data_dir='data/external/dude'):
    """Load DUD-E actives and decoys for a target."""
    target_dir = Path(data_dir) / target
    actives = []
    with open(target_dir / 'actives_final.smi') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                actives.append(parts[0])
    decoys = []
    with open(target_dir / 'decoys_final.smi') as f:
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


def score_with_embedding(model, smiles_list, device, custom_embedding, batch_size=256):
    """
    Score compounds using a custom L1 embedding (replacing the learned one).
    Monkey-patches the program embedding lookup.
    """
    model.eval()
    all_scores = []
    all_valid = []

    # Save original forward
    original_forward = model.context_module.forward

    # Create patched forward that uses custom embedding
    def patched_forward(h_mol, program_ids, assay_ids, round_ids):
        batch_size_actual = h_mol.shape[0]
        # Use custom embedding for all molecules in batch
        z_program = custom_embedding.unsqueeze(0).expand(batch_size_actual, -1).to(h_mol.device)
        z_assay = model.context_module.assay_embeddings(assay_ids)
        z_round = model.context_module.round_embeddings(round_ids)
        context = torch.cat([z_program, z_assay, z_round], dim=-1)
        context = model.context_module.context_interaction(context)
        h_mod = model.context_module.film(h_mol, context)
        return h_mod

    model.context_module.forward = patched_forward

    try:
        with torch.no_grad():
            for i in range(0, len(smiles_list), batch_size):
                batch_smiles = smiles_list[i:i+batch_size]
                batch_data, valid_indices = prepare_batch(batch_smiles, device)
                if batch_data is None:
                    continue

                n_mols = batch_data['n_mols']
                program_ids = torch.zeros(n_mols, dtype=torch.long, device=device)
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


def score_with_program_id(model, smiles_list, device, program_id, batch_size=256):
    """Score compounds using the model's learned program_id embedding."""
    model.eval()
    all_scores = []
    all_valid = []

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

    return all_scores, all_valid


def esm2_zeroshot_l1_evaluation(model, esm_embeddings, l1_embeddings, device):
    """
    Analysis 2: Zero-shot L1 prediction from ESM-2 similarity.

    For each target (leave-one-out):
    1. Compute ESM-2 cosine similarity to other 9 targets
    2. Create similarity-weighted average of their learned L1 embeddings
    3. Evaluate this predicted L1 on DUD-E
    4. Compare to: correct L1, generic L1 (program_id=0)
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Zero-Shot L1 Prediction via ESM-2 Similarity")
    print("=" * 70)
    print("  For each target: predict L1 from ESM-2 similarity to other targets")
    print("  Then evaluate on DUD-E\n")

    results = {}

    for target in DUDE_TARGETS:
        print(f"\n  --- {target.upper()} ---")

        # Load DUD-E data
        actives, decoys = load_dude_target(target)
        all_smiles = actives + decoys
        labels = np.concatenate([np.ones(len(actives)), np.zeros(len(decoys))])

        # 1. ESM-2 similarity-weighted L1 (leave-one-out)
        other_targets = [t for t in DUDE_TARGETS if t != target]
        sims = []
        for ot in other_targets:
            sim = cosine_similarity(esm_embeddings[target], esm_embeddings[ot])
            sims.append(sim)
        sims = np.array(sims)

        # Softmax weights with temperature
        temperature = 0.5
        weights = np.exp(sims / temperature)
        weights /= weights.sum()

        # Weighted average of other targets' learned L1 embeddings
        predicted_l1 = torch.zeros_like(l1_embeddings[other_targets[0]])
        nearest_target = other_targets[np.argmax(sims)]
        nearest_sim = float(sims.max())

        for ot, w in zip(other_targets, weights):
            predicted_l1 += w * l1_embeddings[ot]

        print(f"    Nearest ESM-2 neighbor: {nearest_target} (sim={nearest_sim:.4f})")
        print(f"    Predicted L1 cosine to correct L1: {cosine_similarity(predicted_l1, l1_embeddings[target]):.4f}")

        # 2. Nearest-neighbor L1 (just use closest target's L1)
        nn_l1 = l1_embeddings[nearest_target]

        # 3. Score with different L1 conditions
        # a) Correct L1
        pid = DUDE_TO_PROGRAM_ID[target]
        scores_correct, valid_correct = score_with_program_id(model, all_smiles, device, pid)
        labels_correct = labels[valid_correct]
        auc_correct = roc_auc_score(labels_correct, scores_correct)

        # b) Generic L1 (program_id=0)
        scores_generic, valid_generic = score_with_program_id(model, all_smiles, device, 0)
        labels_generic = labels[valid_generic]
        auc_generic = roc_auc_score(labels_generic, scores_generic)

        # c) ESM-2 weighted L1
        scores_esm_weighted, valid_esm_w = score_with_embedding(model, all_smiles, device, predicted_l1)
        labels_esm_w = labels[valid_esm_w]
        auc_esm_weighted = roc_auc_score(labels_esm_w, scores_esm_weighted)

        # d) ESM-2 nearest-neighbor L1
        scores_esm_nn, valid_esm_nn = score_with_embedding(model, all_smiles, device, nn_l1)
        labels_esm_nn = labels[valid_esm_nn]
        auc_esm_nn = roc_auc_score(labels_esm_nn, scores_esm_nn)

        print(f"    Generic L1 (pid=0):      AUC = {auc_generic:.4f}")
        print(f"    ESM-2 Nearest-Neighbor:   AUC = {auc_esm_nn:.4f}  (using {nearest_target}'s L1)")
        print(f"    ESM-2 Weighted Average:   AUC = {auc_esm_weighted:.4f}")
        print(f"    Correct L1 (pid={pid}):  AUC = {auc_correct:.4f}")

        results[target] = {
            'auc_generic': float(auc_generic),
            'auc_esm_nn': float(auc_esm_nn),
            'auc_esm_weighted': float(auc_esm_weighted),
            'auc_correct': float(auc_correct),
            'nearest_esm_target': nearest_target,
            'nearest_esm_similarity': float(nearest_sim),
            'predicted_l1_cosine_to_correct': float(cosine_similarity(predicted_l1, l1_embeddings[target])),
            'esm_weights': {ot: float(w) for ot, w in zip(other_targets, weights)},
        }

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: ESM-2 Zero-Shot L1 vs Baselines")
    print("=" * 70)
    print(f"\n  {'Target':<10} {'Generic':>9} {'ESM-2 NN':>9} {'ESM-2 Wt':>9} {'Correct':>9} {'ESM-2 NN':>10}")
    print(f"  {'':10} {'(pid=0)':>9} {'':>9} {'':>9} {'(learn)':>9} {'neighbor':>10}")
    print(f"  {'-'*56}")

    generic_aucs, esm_nn_aucs, esm_wt_aucs, correct_aucs = [], [], [], []
    for target in DUDE_TARGETS:
        r = results[target]
        print(f"  {target:<10} {r['auc_generic']:>9.4f} {r['auc_esm_nn']:>9.4f} "
              f"{r['auc_esm_weighted']:>9.4f} {r['auc_correct']:>9.4f} "
              f"{r['nearest_esm_target']:>10}")
        generic_aucs.append(r['auc_generic'])
        esm_nn_aucs.append(r['auc_esm_nn'])
        esm_wt_aucs.append(r['auc_esm_weighted'])
        correct_aucs.append(r['auc_correct'])

    print(f"  {'-'*56}")
    print(f"  {'Mean':<10} {np.mean(generic_aucs):>9.4f} {np.mean(esm_nn_aucs):>9.4f} "
          f"{np.mean(esm_wt_aucs):>9.4f} {np.mean(correct_aucs):>9.4f}")

    # Key finding
    print(f"\n  KEY FINDING:")
    esm_wt_mean = np.mean(esm_wt_aucs)
    generic_mean = np.mean(generic_aucs)
    correct_mean = np.mean(correct_aucs)
    if esm_wt_mean > generic_mean:
        print(f"    ESM-2 weighted ({esm_wt_mean:.4f}) > Generic L1 ({generic_mean:.4f})")
        print(f"    → Protein sequence provides useful context signal")
        gap_closed = (esm_wt_mean - generic_mean) / (correct_mean - generic_mean) * 100
        print(f"    → Closes {gap_closed:.1f}% of the gap to correct L1 ({correct_mean:.4f})")
    else:
        print(f"    Generic L1 ({generic_mean:.4f}) ≥ ESM-2 weighted ({esm_wt_mean:.4f})")
        print(f"    → Learned L1 captures dataset-specific patterns beyond protein similarity")
        print(f"    → This justifies learning L1 embeddings during training")

    return results


def main():
    parser = argparse.ArgumentParser(description='ESM-2 Protein Embedding Analysis')
    parser.add_argument('--checkpoint', type=str, default='results/v3/best_model.pt')
    parser.add_argument('--output', type=str, default='results/experiments/esm2_analysis')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--esm-model', type=str, default='esm2_t33_650M_UR50D',
                        help='ESM-2 model name (default: 650M)')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("=" * 70)
    print("ESM-2 PROTEIN EMBEDDING ANALYSIS FOR NEST-DRUG")
    print("Addresses: E5 (protein language model embeddings as L1)")
    print("=" * 70)

    # Step 1: Fetch protein sequences
    print("\nStep 1: Fetching protein sequences from UniProt...")
    sequences = {}
    for target in DUDE_TARGETS:
        acc = UNIPROT_ACCESSIONS[target]
        print(f"  {target} ({acc})...", end=" ")
        seq = fetch_uniprot_sequence(acc)
        if seq:
            sequences[target] = seq
            print(f"{len(seq)} aa")
        else:
            print("FAILED")

    if len(sequences) < 8:
        print(f"\nERROR: Only {len(sequences)}/10 sequences fetched. Need at least 8.")
        print("Check internet connectivity or provide sequences manually.")
        return

    # Step 2: Compute ESM-2 embeddings
    print(f"\nStep 2: Computing ESM-2 embeddings ({args.esm_model})...")
    esm_embeddings = compute_esm2_embeddings(sequences, device, args.esm_model)

    # Step 3: Load NEST-DRUG model and extract learned L1 embeddings
    print(f"\nStep 3: Loading NEST-DRUG checkpoint ({args.checkpoint})...")
    model, state_dict = load_model(args.checkpoint, device)
    l1_embeddings = get_learned_l1_embeddings(state_dict)
    print(f"  Extracted L1 embeddings for {len(l1_embeddings)} targets")

    # Step 4: Similarity correlation analysis
    # Only use targets present in both ESM and L1
    common_targets = [t for t in DUDE_TARGETS if t in esm_embeddings and t in l1_embeddings]
    esm_common = {t: esm_embeddings[t] for t in common_targets}
    l1_common = {t: l1_embeddings[t] for t in common_targets}

    correlation_results = similarity_correlation_analysis(esm_common, l1_common)

    # Step 5: Zero-shot L1 evaluation on DUD-E
    zeroshot_results = esm2_zeroshot_l1_evaluation(model, esm_common, l1_common, device)

    # Save all results
    # Convert torch tensors to lists for JSON
    esm_emb_data = {t: emb.tolist() for t, emb in esm_embeddings.items()}
    l1_emb_data = {t: emb.tolist() for t, emb in l1_embeddings.items()}

    output_data = {
        'timestamp': datetime.now().isoformat(),
        'description': 'ESM-2 protein embedding analysis for NEST-DRUG',
        'note': 'Addresses reviewer critique E5: protein language model embeddings as L1',
        'config': {
            'esm_model': args.esm_model,
            'checkpoint': args.checkpoint,
            'n_targets': len(common_targets),
            'targets': common_targets,
        },
        'sequences': {t: {'accession': UNIPROT_ACCESSIONS[t], 'length': len(seq)}
                      for t, seq in sequences.items()},
        'similarity_correlation': correlation_results,
        'zeroshot_evaluation': zeroshot_results,
        # Don't save full embeddings (large), save norms and key stats
        'embedding_stats': {
            target: {
                'esm_norm': float(torch.norm(esm_embeddings[target]).item()) if target in esm_embeddings else None,
                'l1_norm': float(torch.norm(l1_embeddings[target]).item()) if target in l1_embeddings else None,
                'esm_dim': int(esm_embeddings[target].shape[0]) if target in esm_embeddings else None,
                'l1_dim': int(l1_embeddings[target].shape[0]) if target in l1_embeddings else None,
            }
            for target in DUDE_TARGETS
        },
    }

    output_file = output_dir / 'esm2_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
