#!/usr/bin/env python3
"""
Data Utilities for NEST-DRUG Training

Provides:
- MoleculeDataset: PyTorch dataset for molecular graphs
- Collate functions for batching variable-size graphs
- Data loaders for portfolio and program data
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm import tqdm

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# =============================================================================
# Atom and Bond Featurization
# =============================================================================

ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si',
             'B', 'Na', 'K', 'Ca', 'Mg', 'Fe', 'Zn', 'Cu', 'Mn',
             'H', 'Se', 'As', 'Li', 'Al', 'Ga', 'Ge', 'Sn', 'Pb',
             'Bi', 'Sb', 'Te', 'Ag', 'Au', 'Pt', 'Pd', 'Rh', 'Ru',
             'Re', 'Os', 'Ir', 'Hg', 'Cd', 'Ti', 'V', 'Other']


def atom_features(atom) -> List[float]:
    """Extract features for an atom."""
    features = []

    # Element type (one-hot, 44 + 1 for 'Other')
    symbol = atom.GetSymbol()
    if symbol in ATOM_LIST:
        idx = ATOM_LIST.index(symbol)
    else:
        idx = len(ATOM_LIST) - 1  # 'Other'
    elem_vec = [0.0] * len(ATOM_LIST)
    elem_vec[idx] = 1.0
    features.extend(elem_vec)

    # Degree (0-6)
    degree = min(atom.GetDegree(), 6)
    degree_vec = [0.0] * 7
    degree_vec[degree] = 1.0
    features.extend(degree_vec)

    # Formal charge (-2 to +2)
    charge = max(-2, min(2, atom.GetFormalCharge()))
    charge_vec = [0.0] * 5
    charge_vec[charge + 2] = 1.0
    features.extend(charge_vec)

    # Hybridization
    hyb = atom.GetHybridization()
    hyb_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    hyb_vec = [1.0 if hyb == h else 0.0 for h in hyb_types]
    features.extend(hyb_vec)

    # Aromaticity
    features.append(1.0 if atom.GetIsAromatic() else 0.0)

    # Ring membership
    features.append(1.0 if atom.IsInRing() else 0.0)

    # Hydrogen count (0-4)
    h_count = min(atom.GetTotalNumHs(), 4)
    h_vec = [0.0] * 5
    h_vec[h_count] = 1.0
    features.extend(h_vec)

    return features  # Total: 45 + 7 + 5 + 5 + 1 + 1 + 5 = 69 features


def bond_features(bond) -> List[float]:
    """Extract features for a bond."""
    features = []

    # Bond type
    bt = bond.GetBondType()
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    bt_vec = [1.0 if bt == t else 0.0 for t in bond_types]
    features.extend(bt_vec)

    # Conjugation
    features.append(1.0 if bond.GetIsConjugated() else 0.0)

    # Ring membership
    features.append(1.0 if bond.IsInRing() else 0.0)

    # Stereochemistry
    stereo = bond.GetStereo()
    stereo_types = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]
    stereo_vec = [1.0 if stereo == s else 0.0 for s in stereo_types]
    features.extend(stereo_vec)

    return features  # Total: 4 + 1 + 1 + 3 = 9 features


def smiles_to_graph(smiles: str) -> Optional[Dict[str, torch.Tensor]]:
    """
    Convert SMILES to molecular graph tensors.

    Returns dict with node_features, edge_index, edge_features, num_atoms.
    """
    if not RDKIT_AVAILABLE:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Get atom features
        node_feats = []
        for atom in mol.GetAtoms():
            node_feats.append(atom_features(atom))
        node_feats = torch.tensor(node_feats, dtype=torch.float32)

        # Get bond features and edge indices
        edge_index = []
        edge_feats = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            bf = bond_features(bond)

            # Add both directions
            edge_index.extend([[i, j], [j, i]])
            edge_feats.extend([bf, bf])

        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_feats = torch.tensor(edge_feats, dtype=torch.float32)
        else:
            # Handle molecules with no bonds (single atoms)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_feats = torch.zeros((0, 9), dtype=torch.float32)

        return {
            'node_features': node_feats,
            'edge_index': edge_index,
            'edge_features': edge_feats,
            'num_atoms': mol.GetNumAtoms(),
        }

    except Exception as e:
        return None


# =============================================================================
# Datasets
# =============================================================================

class MoleculeDataset(Dataset):
    """
    Dataset for molecular property prediction.

    Loads molecules from CSV/Parquet with SMILES and endpoint columns.
    Converts SMILES to graphs on-the-fly or uses cached graphs.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        smiles_col: str = 'smiles',
        endpoint_cols: Optional[List[str]] = None,
        program_col: Optional[str] = None,
        assay_col: Optional[str] = None,
        round_col: Optional[str] = None,
        cache_graphs: bool = True,
        max_samples: Optional[int] = None,
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.smiles_col = smiles_col
        self.cache_graphs = cache_graphs
        self._graph_cache: Dict[str, Dict] = {}

        # Load data
        if self.data_path.suffix == '.parquet':
            self.data = pd.read_parquet(self.data_path)
        else:
            self.data = pd.read_csv(self.data_path)

        if max_samples and len(self.data) > max_samples:
            self.data = self.data.sample(n=max_samples, random_state=42)

        # Determine endpoint columns
        if endpoint_cols is None:
            # Auto-detect numeric columns
            exclude = [smiles_col, program_col, assay_col, round_col]
            exclude = [c for c in exclude if c is not None]
            self.endpoint_cols = [c for c in self.data.columns
                                  if c not in exclude and self.data[c].dtype in ['float64', 'float32', 'int64']]
        else:
            self.endpoint_cols = [c for c in endpoint_cols if c in self.data.columns]

        # Context columns
        self.program_col = program_col
        self.assay_col = assay_col
        self.round_col = round_col

        # Create ID mappings for contexts
        self.program_map = {}
        self.assay_map = {}
        self.round_map = {}

        if program_col and program_col in self.data.columns:
            programs = self.data[program_col].unique()
            self.program_map = {p: i for i, p in enumerate(programs)}

        if assay_col and assay_col in self.data.columns:
            assays = self.data[assay_col].unique()
            self.assay_map = {a: i for i, a in enumerate(assays)}

        if round_col and round_col in self.data.columns:
            rounds = sorted(self.data[round_col].unique())
            self.round_map = {r: i for i, r in enumerate(rounds)}

        print(f"Loaded {len(self)} samples with {len(self.endpoint_cols)} endpoints")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]
        smiles = row[self.smiles_col]

        # Get or compute graph
        if self.cache_graphs and smiles in self._graph_cache:
            graph = self._graph_cache[smiles]
        else:
            graph = smiles_to_graph(smiles)
            if self.cache_graphs and graph is not None:
                self._graph_cache[smiles] = graph

        if graph is None:
            # Return dummy graph for invalid SMILES (69 atom features, 9 bond features)
            graph = {
                'node_features': torch.zeros((1, 69), dtype=torch.float32),
                'edge_index': torch.zeros((2, 0), dtype=torch.long),
                'edge_features': torch.zeros((0, 9), dtype=torch.float32),
                'num_atoms': 1,
            }

        # Get endpoint values and masks
        endpoints = {}
        masks = {}

        for col in self.endpoint_cols:
            val = row[col]
            if pd.notna(val):
                endpoints[col] = torch.tensor([float(val)], dtype=torch.float32)
                masks[col] = torch.tensor([1.0], dtype=torch.float32)
            else:
                endpoints[col] = torch.tensor([0.0], dtype=torch.float32)
                masks[col] = torch.tensor([0.0], dtype=torch.float32)

        # Get context IDs
        program_id = 0
        assay_id = 0
        round_id = 0

        if self.program_col and self.program_col in row:
            program_id = self.program_map.get(row[self.program_col], 0)
        if self.assay_col and self.assay_col in row:
            assay_id = self.assay_map.get(row[self.assay_col], 0)
        if self.round_col and self.round_col in row:
            round_id = self.round_map.get(row[self.round_col], 0)

        return {
            'smiles': smiles,
            'graph': graph,
            'endpoints': endpoints,
            'masks': masks,
            'program_id': program_id,
            'assay_id': assay_id,
            'round_id': round_id,
        }


def collate_molecules(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for batching molecular graphs.

    Combines variable-size graphs into a single batched graph
    using PyTorch Geometric-style batch indices.
    """
    # Filter out None entries
    batch = [b for b in batch if b is not None and b['graph'] is not None]

    if len(batch) == 0:
        return None

    # Combine graphs
    node_features_list = []
    edge_index_list = []
    edge_features_list = []
    batch_indices = []

    node_offset = 0

    for i, sample in enumerate(batch):
        graph = sample['graph']
        num_atoms = graph['num_atoms']

        node_features_list.append(graph['node_features'])
        edge_index_list.append(graph['edge_index'] + node_offset)
        edge_features_list.append(graph['edge_features'])
        batch_indices.extend([i] * num_atoms)

        node_offset += num_atoms

    # Stack into tensors
    batched = {
        'node_features': torch.cat(node_features_list, dim=0),
        'edge_index': torch.cat(edge_index_list, dim=1),
        'edge_features': torch.cat(edge_features_list, dim=0),
        'batch': torch.tensor(batch_indices, dtype=torch.long),
        'smiles': [b['smiles'] for b in batch],
    }

    # Stack context IDs
    batched['program_ids'] = torch.tensor([b['program_id'] for b in batch], dtype=torch.long)
    batched['assay_ids'] = torch.tensor([b['assay_id'] for b in batch], dtype=torch.long)
    batched['round_ids'] = torch.tensor([b['round_id'] for b in batch], dtype=torch.long)

    # Stack endpoints and masks
    endpoint_names = list(batch[0]['endpoints'].keys())

    batched['endpoints'] = {}
    batched['masks'] = {}

    for name in endpoint_names:
        batched['endpoints'][name] = torch.stack([b['endpoints'][name] for b in batch])
        batched['masks'][name] = torch.stack([b['masks'][name] for b in batch])

    return batched


# =============================================================================
# Data Loaders
# =============================================================================

def PortfolioDataLoader(
    data_path: Union[str, Path],
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    smiles_col: Optional[str] = None,
    **kwargs,
) -> DataLoader:
    """
    Create data loader for portfolio pretraining.

    Args:
        data_path: Path to portfolio data (parquet/csv)
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        max_samples: Maximum samples to load (for debugging)
        smiles_col: Name of SMILES column (auto-detected if None)
    """
    # Auto-detect SMILES column if not specified
    if smiles_col is None:
        path = Path(data_path)
        if path.suffix == '.parquet':
            import pandas as pd
            cols = pd.read_parquet(path, columns=None).columns.tolist()
        else:
            import pandas as pd
            cols = pd.read_csv(path, nrows=1).columns.tolist()

        for candidate in ['smiles', 'canonical_smiles', 'SMILES', 'Smiles']:
            if candidate in cols:
                smiles_col = candidate
                break
        if smiles_col is None:
            smiles_col = 'smiles'  # fallback

    dataset = MoleculeDataset(
        data_path=data_path,
        smiles_col=smiles_col,
        cache_graphs=True,
        max_samples=max_samples,
    )

    # Disable pin_memory if CUDA not available
    import torch
    pin_mem = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_molecules,
        pin_memory=pin_mem,
        **kwargs,
    )


def ProgramDataLoader(
    data_path: Union[str, Path],
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 2,
    round_filter: Optional[List[int]] = None,
    **kwargs,
) -> DataLoader:
    """
    Create data loader for program-specific training.

    Args:
        data_path: Path to program data
        batch_size: Batch size
        shuffle: Whether to shuffle
        round_filter: Only include specific rounds (for temporal splits)
    """
    dataset = MoleculeDataset(
        data_path=data_path,
        smiles_col='smiles',
        program_col='program_id',
        assay_col='assay_id',
        round_col='round_id',
        cache_graphs=True,
    )

    # Filter by rounds if specified
    if round_filter is not None:
        mask = dataset.data['round_id'].isin(round_filter)
        dataset.data = dataset.data[mask].reset_index(drop=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_molecules,
        pin_memory=True,
        **kwargs,
    )


if __name__ == '__main__':
    # Test data utilities
    print("Testing Data Utilities...")

    # Test SMILES to graph
    test_smiles = "CCO"  # Ethanol
    graph = smiles_to_graph(test_smiles)

    if graph:
        print(f"\nSMILES: {test_smiles}")
        print(f"  Atoms: {graph['num_atoms']}")
        print(f"  Node features: {graph['node_features'].shape}")
        print(f"  Edge index: {graph['edge_index'].shape}")
        print(f"  Edge features: {graph['edge_features'].shape}")
