#!/usr/bin/env python3
"""
Experiment 2A: Integrated Gradients Attribution

Per-atom importance scores showing what molecular features drive predictions.

Usage:
    python scripts/experiments/integrated_gradients.py \
        --checkpoint checkpoints/pretrain/best_model.pt \
        --output results/experiments/integrated_gradients \
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph

try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    print("Warning: captum not installed. Run: pip install captum")

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

    # Infer config from weights
    config = checkpoint.get('config', {})
    if config:
        num_programs = config.get('n_programs', config.get('num_programs', 5))
        num_assays = config.get('n_assays', config.get('num_assays', 50))
        num_rounds = config.get('n_rounds', config.get('num_rounds', 150))
    else:
        num_programs = state_dict['context_module.program_embeddings.embeddings.weight'].shape[0]
        num_assays = state_dict['context_module.assay_embeddings.embeddings.weight'].shape[0]
        num_rounds = state_dict['context_module.round_embeddings.embeddings.weight'].shape[0]

    # Get endpoints
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


class ModelWrapper(torch.nn.Module):
    """Wrapper for captum compatibility."""

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


def get_atom_importance(model, smiles, program_id=0, device='cuda', n_steps=50):
    """Compute per-atom importance using Integrated Gradients."""
    if not HAS_CAPTUM:
        raise ImportError("captum required for integrated gradients")

    graph = smiles_to_graph(smiles)
    if graph is None:
        return None

    node_features = graph['node_features'].to(device).requires_grad_(True)
    edge_index = graph['edge_index'].to(device)
    edge_features = graph['edge_features'].to(device)
    batch = torch.zeros(graph['num_atoms'], dtype=torch.long, device=device)

    wrapper = ModelWrapper(model, program_id=program_id)
    wrapper.eval()

    # Baseline: zero features
    baseline = torch.zeros_like(node_features)

    ig = IntegratedGradients(lambda x: wrapper(x, edge_index, edge_features, batch))

    attributions = ig.attribute(node_features, baselines=baseline, n_steps=n_steps)

    # Sum over features to get per-atom importance
    atom_importance = attributions.abs().sum(dim=-1).detach().cpu().numpy()

    return atom_importance


def visualize_importance(smiles, importance, output_path, title=""):
    """Visualize atom importance on molecule."""
    if not HAS_RDKIT:
        print("RDKit required for visualization")
        return

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return

    # Normalize importance to [0, 1]
    importance = np.array(importance)
    if importance.max() > importance.min():
        importance = (importance - importance.min()) / (importance.max() - importance.min())

    # Create color map (blue to red)
    atom_colors = {}
    for i, imp in enumerate(importance):
        atom_colors[i] = (imp, 0, 1-imp)  # RGB

    # Generate 2D coords
    AllChem.Compute2DCoords(mol)

    # Draw
    drawer = Draw.MolDraw2DCairo(500, 500)
    drawer.DrawMolecule(mol, highlightAtoms=list(range(len(importance))),
                        highlightAtomColors=atom_colors)
    drawer.FinishDrawing()

    with open(output_path, 'wb') as f:
        f.write(drawer.GetDrawingText())

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Integrated Gradients Attribution")
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--output', type=str, default='results/experiments/integrated_gradients',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--n-steps', type=int, default=50, help='IG steps')
    parser.add_argument('--program-id', type=int, default=0, help='Program context ID')
    parser.add_argument('--smiles-file', type=str, default=None,
                        help='File with SMILES to analyze (one per line)')
    parser.add_argument('--example-smiles', type=str, nargs='+', default=None,
                        help='Example SMILES to analyze')
    args = parser.parse_args()

    if not HAS_CAPTUM:
        print("ERROR: captum required. Install with: pip install captum")
        sys.exit(1)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    print(f"Config: {config}")

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get SMILES to analyze
    smiles_list = []
    if args.smiles_file:
        with open(args.smiles_file) as f:
            smiles_list = [line.strip() for line in f if line.strip()]
    elif args.example_smiles:
        smiles_list = args.example_smiles
    else:
        # Default examples
        smiles_list = [
            ("Celecoxib", "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"),
            ("Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
            ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ]

    results = []

    for item in tqdm(smiles_list, desc="Computing attributions"):
        if isinstance(item, tuple):
            name, smiles = item
        else:
            name = item[:20]
            smiles = item

        try:
            importance = get_atom_importance(model, smiles, program_id=args.program_id,
                                             device=device, n_steps=args.n_steps)
            if importance is not None:
                results.append({
                    'name': name,
                    'smiles': smiles,
                    'importance': importance.tolist(),
                    'mean_importance': float(importance.mean()),
                    'max_importance': float(importance.max()),
                })

                # Visualize
                viz_path = output_dir / f"{name.replace(' ', '_')}_attribution.png"
                visualize_importance(smiles, importance, viz_path, title=name)
        except Exception as e:
            print(f"Error processing {name}: {e}")

    # Save results
    results_path = output_dir / 'attribution_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'config': config,
            'program_id': args.program_id,
            'n_steps': args.n_steps,
            'results': results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
