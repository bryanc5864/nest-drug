#!/usr/bin/env python3
"""
Experiment 2B: Context-Conditional Attribution (FIXED)

How does the SAME molecule get different attributions for different targets?
Uses manual integrated gradients (Captum fails on GNNs).

Usage:
    python scripts/experiments/context_attribution_fixed.py --checkpoint results/v3/best_model.pt --gpu 0
"""

import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# V3 target-specific program IDs
TARGET_PROGRAM_IDS = {
    'EGFR': 1606,
    'DRD2': 1448,
    'BACE1': 516,
    'ESR1': 1628,
    'HDAC2': 2177,
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
        num_programs=num_programs,
        num_assays=num_assays,
        num_rounds=num_rounds,
        endpoints=endpoints,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, {'num_programs': num_programs, 'num_assays': num_assays, 'num_rounds': num_rounds}


def integrated_gradients_manual(model, node_features, edge_index, edge_features, batch,
                                 program_ids, assay_ids, round_ids, n_steps=50):
    """Manual integrated gradients implementation."""
    baseline = torch.zeros_like(node_features)
    integrated_grads = torch.zeros_like(node_features)

    for step in range(n_steps):
        alpha = step / n_steps
        interpolated = baseline + alpha * (node_features - baseline)
        interpolated = interpolated.clone().requires_grad_(True)

        predictions = model(interpolated, edge_index, edge_features, batch,
                           program_ids, assay_ids, round_ids)
        pred = list(predictions.values())[0].squeeze()

        if pred.dim() == 0:
            pred.backward(retain_graph=True)
        else:
            pred.sum().backward(retain_graph=True)

        if interpolated.grad is not None:
            integrated_grads += interpolated.grad.detach()

        model.zero_grad()

    integrated_grads = integrated_grads * (node_features - baseline) / n_steps
    return integrated_grads


def get_atom_importance(model, smiles, program_id, device, n_steps=50):
    """Compute per-atom importance using manual Integrated Gradients."""
    graph = smiles_to_graph(smiles)
    if graph is None:
        return None

    node_features = graph['node_features'].to(device)
    edge_index = graph['edge_index'].to(device)
    edge_features = graph['edge_features'].to(device)
    batch = torch.zeros(graph['num_atoms'], dtype=torch.long, device=device)

    program_ids = torch.tensor([program_id], dtype=torch.long, device=device)
    assay_ids = torch.zeros(1, dtype=torch.long, device=device)
    round_ids = torch.zeros(1, dtype=torch.long, device=device)

    attributions = integrated_gradients_manual(
        model, node_features, edge_index, edge_features, batch,
        program_ids, assay_ids, round_ids, n_steps=n_steps
    )

    atom_importance = attributions.abs().sum(dim=-1).cpu().numpy()
    return atom_importance


def compute_divergence(imp1, imp2):
    """Compute divergence between two attribution vectors."""
    imp1 = np.array(imp1)
    imp2 = np.array(imp2)

    # Normalize to distributions
    if imp1.sum() > 0:
        imp1_norm = imp1 / imp1.sum()
    else:
        imp1_norm = imp1
    if imp2.sum() > 0:
        imp2_norm = imp2 / imp2.sum()
    else:
        imp2_norm = imp2

    # KL divergence (symmetrized)
    eps = 1e-10
    kl1 = np.sum(imp1_norm * np.log((imp1_norm + eps) / (imp2_norm + eps)))
    kl2 = np.sum(imp2_norm * np.log((imp2_norm + eps) / (imp1_norm + eps)))
    kl_sym = (kl1 + kl2) / 2

    # Cosine similarity
    cos_sim = np.dot(imp1, imp2) / (np.linalg.norm(imp1) * np.linalg.norm(imp2) + eps)

    # L1 distance
    l1_dist = np.abs(imp1_norm - imp2_norm).sum()

    return {
        'kl_divergence': float(kl_sym),
        'cosine_similarity': float(cos_sim),
        'l1_distance': float(l1_dist),
    }


def visualize_comparison(smiles, importances_dict, output_path):
    """Visualize attributions for multiple contexts."""
    if not HAS_RDKIT:
        return

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return

    AllChem.Compute2DCoords(mol)

    n_contexts = len(importances_dict)
    fig, axes = plt.subplots(1, n_contexts, figsize=(4*n_contexts, 4))
    if n_contexts == 1:
        axes = [axes]

    for ax, (context_name, importance) in zip(axes, importances_dict.items()):
        importance = np.array(importance)
        if importance.max() > importance.min():
            norm_imp = (importance - importance.min()) / (importance.max() - importance.min())
        else:
            norm_imp = np.zeros_like(importance)

        atom_colors = {i: (float(imp), 0.2, float(1-imp)) for i, imp in enumerate(norm_imp)}

        drawer = Draw.MolDraw2DCairo(300, 300)
        drawer.DrawMolecule(mol, highlightAtoms=list(range(len(importance))),
                           highlightAtomColors=atom_colors)
        drawer.FinishDrawing()

        import io
        from PIL import Image
        img_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_data))

        ax.imshow(img)
        ax.set_title(f"{context_name}\nmax={importance.max():.3f}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Context-Conditional Attribution (Fixed)")
    parser.add_argument('--checkpoint', type=str, default='results/v3/best_model.pt')
    parser.add_argument('--output', type=str, default='results/experiments/context_attribution_fixed')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n-steps', type=int, default=50)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLoading model: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    print(f"Config: {config}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test molecules (drugs with known targets)
    molecules = [
        ("Celecoxib", "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"),  # COX-2 inhibitor
        ("Erlotinib", "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC"),  # EGFR inhibitor
        ("Donepezil", "COC1=C(C=C2C(=C1)CC(C2=O)CC3CCN(CC3)CC4=CC=CC=C4)OC"),  # AChE inhibitor
    ]

    # Select target program IDs based on model size
    if config['num_programs'] >= 5123:
        # V3 model - use actual target IDs
        targets = TARGET_PROGRAM_IDS
    else:
        # V1/V2 model - use generic IDs
        targets = {f"P{i}": i for i in range(min(5, config['num_programs']))}

    print(f"Comparing targets: {list(targets.keys())}")

    all_results = []

    for mol_name, smiles in tqdm(molecules, desc="Processing"):
        print(f"\n{mol_name}")

        importances = {}
        for target_name, pid in targets.items():
            try:
                imp = get_atom_importance(model, smiles, pid, device, args.n_steps)
                if imp is not None:
                    importances[target_name] = imp
                    print(f"  {target_name}: mean={imp.mean():.4f}, max={imp.max():.4f}")
            except Exception as e:
                print(f"  {target_name}: ERROR - {e}")

        if len(importances) < 2:
            print(f"  Skipping - need at least 2 successful attributions")
            continue

        # Compute pairwise divergences
        divergences = []
        target_names = list(importances.keys())
        for i in range(len(target_names)):
            for j in range(i+1, len(target_names)):
                t1, t2 = target_names[i], target_names[j]
                div = compute_divergence(importances[t1], importances[t2])
                divergences.append({
                    'target1': t1,
                    'target2': t2,
                    **div
                })
                print(f"  {t1} vs {t2}: KL={div['kl_divergence']:.4f}, cos={div['cosine_similarity']:.4f}")

        # Save visualization
        viz_path = output_dir / f"{mol_name}_comparison.png"
        visualize_comparison(smiles, importances, viz_path)

        all_results.append({
            'name': mol_name,
            'smiles': smiles,
            'attributions': {k: {'importance': v.tolist(), 'mean': float(v.mean()), 'max': float(v.max())}
                           for k, v in importances.items()},
            'divergences': divergences,
        })

    # Summary
    all_kl = [d['kl_divergence'] for r in all_results for d in r['divergences']]
    all_cos = [d['cosine_similarity'] for r in all_results for d in r['divergences']]

    summary = {
        'mean_kl_divergence': float(np.mean(all_kl)) if all_kl else 0,
        'mean_cosine_similarity': float(np.mean(all_cos)) if all_cos else 0,
        'n_molecules': len(all_results),
        'n_comparisons': len(all_kl),
    }

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Molecules processed: {summary['n_molecules']}")
    print(f"Pairwise comparisons: {summary['n_comparisons']}")
    print(f"Mean KL Divergence: {summary['mean_kl_divergence']:.4f}")
    print(f"Mean Cosine Similarity: {summary['mean_cosine_similarity']:.4f}")

    # Save results
    results_path = output_dir / 'context_attribution_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'config': config,
            'targets': {k: v for k, v in targets.items()},
            'summary': summary,
            'results': all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
