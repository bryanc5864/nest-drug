#!/usr/bin/env python3
"""
Experiment 2B: Context-Conditional Attribution

How does the SAME molecule get different attributions for different targets?
Proves FiLM modulation is meaningful.

Usage:
    python scripts/experiments/context_conditional_attribution.py \
        --checkpoint checkpoints/pretrain/best_model.pt \
        --output results/experiments/context_attribution \
        --gpu 0
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph

try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    config = checkpoint.get('config', {})
    if config:
        num_programs = config.get('n_programs', config.get('num_programs', 5))
        num_assays = config.get('n_assays', config.get('num_assays', 50))
        num_rounds = config.get('n_rounds', config.get('num_rounds', 150))
    else:
        num_programs = state_dict['context_module.program_embeddings.embeddings.weight'].shape[0]
        num_assays = state_dict['context_module.assay_embeddings.embeddings.weight'].shape[0]
        num_rounds = state_dict['context_module.round_embeddings.embeddings.weight'].shape[0]

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

    program_mapping = checkpoint.get('program_mapping', {})

    return model, {'num_programs': num_programs, 'num_assays': num_assays,
                   'num_rounds': num_rounds}, program_mapping


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, program_id=0, assay_id=0, round_id=0):
        super().__init__()
        self.model = model
        self.program_id = program_id
        self.assay_id = assay_id
        self.round_id = round_id

    def forward(self, node_features, edge_index, edge_features, batch):
        batch_size = batch.max().item() + 1
        program_ids = torch.full((batch_size,), self.program_id, dtype=torch.long, device=node_features.device)
        assay_ids = torch.full((batch_size,), self.assay_id, dtype=torch.long, device=node_features.device)
        round_ids = torch.full((batch_size,), self.round_id, dtype=torch.long, device=node_features.device)

        predictions = self.model(node_features, edge_index, edge_features, batch,
                                  program_ids, assay_ids, round_ids)
        pred = list(predictions.values())[0]
        return pred.squeeze()


def get_atom_importance(model, smiles, program_id, device, n_steps=50):
    """Compute per-atom importance using Integrated Gradients."""
    if not HAS_CAPTUM:
        raise ImportError("captum required")

    graph = smiles_to_graph(smiles)
    if graph is None:
        return None

    node_features = graph['node_features'].to(device).requires_grad_(True)
    edge_index = graph['edge_index'].to(device)
    edge_features = graph['edge_features'].to(device)
    batch = torch.zeros(graph['num_atoms'], dtype=torch.long, device=device)

    wrapper = ModelWrapper(model, program_id=program_id)
    wrapper.eval()

    baseline = torch.zeros_like(node_features)
    ig = IntegratedGradients(lambda x: wrapper(x, edge_index, edge_features, batch))
    attributions = ig.attribute(node_features, baselines=baseline, n_steps=n_steps)
    atom_importance = attributions.abs().sum(dim=-1).detach().cpu().numpy()

    return atom_importance


def visualize_multi_context(smiles, importances_dict, output_path):
    """Visualize attributions for multiple contexts side by side."""
    if not HAS_RDKIT:
        return

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return

    AllChem.Compute2DCoords(mol)

    n_contexts = len(importances_dict)
    fig, axes = plt.subplots(1, n_contexts, figsize=(5*n_contexts, 5))
    if n_contexts == 1:
        axes = [axes]

    for ax, (context_name, importance) in zip(axes, importances_dict.items()):
        # Normalize
        importance = np.array(importance)
        if importance.max() > importance.min():
            norm_imp = (importance - importance.min()) / (importance.max() - importance.min())
        else:
            norm_imp = np.zeros_like(importance)

        # Create image
        atom_colors = {i: (imp, 0.2, 1-imp) for i, imp in enumerate(norm_imp)}

        drawer = Draw.MolDraw2DCairo(400, 400)
        drawer.DrawMolecule(mol, highlightAtoms=list(range(len(importance))),
                            highlightAtomColors=atom_colors)
        drawer.FinishDrawing()

        # Convert to numpy array for matplotlib
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
    print(f"Saved: {output_path}")


def compute_attribution_divergence(imp1, imp2):
    """Compute divergence between two attribution vectors."""
    # Normalize
    imp1 = np.array(imp1)
    imp2 = np.array(imp2)

    if imp1.sum() > 0:
        imp1 = imp1 / imp1.sum()
    if imp2.sum() > 0:
        imp2 = imp2 / imp2.sum()

    # KL divergence (symmetrized)
    eps = 1e-10
    kl1 = np.sum(imp1 * np.log((imp1 + eps) / (imp2 + eps)))
    kl2 = np.sum(imp2 * np.log((imp2 + eps) / (imp1 + eps)))
    kl_sym = (kl1 + kl2) / 2

    # Cosine similarity
    cos_sim = np.dot(imp1, imp2) / (np.linalg.norm(imp1) * np.linalg.norm(imp2) + eps)

    return {'kl_divergence': float(kl_sym), 'cosine_similarity': float(cos_sim)}


def main():
    parser = argparse.ArgumentParser(description="Context-Conditional Attribution")
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--output', type=str, default='results/experiments/context_attribution',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--n-steps', type=int, default=50, help='IG steps')
    parser.add_argument('--program-ids', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='Program IDs to compare')
    parser.add_argument('--smiles', type=str, nargs='+', default=None,
                        help='SMILES to analyze')
    args = parser.parse_args()

    if not HAS_CAPTUM:
        print("ERROR: captum required. Install with: pip install captum")
        sys.exit(1)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLoading model: {args.checkpoint}")
    model, config, program_mapping = load_model(args.checkpoint, device)
    print(f"Config: {config}")
    print(f"Program mapping: {len(program_mapping)} entries")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default molecules if not specified
    if args.smiles is None:
        smiles_list = [
            ("Celecoxib", "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"),
            ("Imatinib", "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"),
            ("Gefitinib", "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"),
        ]
    else:
        smiles_list = [(s[:20], s) for s in args.smiles]

    # Limit program IDs to available
    program_ids = [p for p in args.program_ids if p < config['num_programs']]
    print(f"Comparing program IDs: {program_ids}")

    all_results = []

    for name, smiles in tqdm(smiles_list, desc="Processing molecules"):
        mol_results = {
            'name': name,
            'smiles': smiles,
            'attributions': {},
            'divergences': [],
        }

        importances = {}
        for pid in program_ids:
            try:
                imp = get_atom_importance(model, smiles, pid, device, args.n_steps)
                if imp is not None:
                    importances[f"P{pid}"] = imp.tolist()
                    mol_results['attributions'][f"P{pid}"] = {
                        'importance': imp.tolist(),
                        'mean': float(imp.mean()),
                        'max': float(imp.max()),
                        'std': float(imp.std()),
                    }
            except Exception as e:
                print(f"Error for {name}, P{pid}: {e}")

        # Compute pairwise divergences
        pids = list(importances.keys())
        for i in range(len(pids)):
            for j in range(i+1, len(pids)):
                div = compute_attribution_divergence(
                    importances[pids[i]], importances[pids[j]])
                mol_results['divergences'].append({
                    'context1': pids[i],
                    'context2': pids[j],
                    **div
                })

        # Visualize
        if importances:
            viz_path = output_dir / f"{name.replace(' ', '_')}_multi_context.png"
            visualize_multi_context(smiles, importances, viz_path)

        all_results.append(mol_results)

    # Summary statistics
    summary = {
        'mean_kl_divergence': [],
        'mean_cosine_similarity': [],
    }
    for mol_res in all_results:
        for div in mol_res['divergences']:
            summary['mean_kl_divergence'].append(div['kl_divergence'])
            summary['mean_cosine_similarity'].append(div['cosine_similarity'])

    summary_stats = {
        'mean_kl_divergence': float(np.mean(summary['mean_kl_divergence'])) if summary['mean_kl_divergence'] else 0,
        'mean_cosine_similarity': float(np.mean(summary['mean_cosine_similarity'])) if summary['mean_cosine_similarity'] else 0,
    }

    # Save results
    results_path = output_dir / 'context_attribution_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'config': config,
            'program_ids': program_ids,
            'summary': summary_stats,
            'results': all_results,
        }, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Mean KL Divergence: {summary_stats['mean_kl_divergence']:.4f}")
    print(f"Mean Cosine Similarity: {summary_stats['mean_cosine_similarity']:.4f}")
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
