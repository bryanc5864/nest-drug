#!/usr/bin/env python3
"""
NEST-DRUG Dataset Classes

Implements PyTorch datasets for:
- PortfolioDataset: Large-scale pretraining on ChEMBL/BindingDB/TDC
- ProgramDataset: Program-specific data with temporal structure
- DMTAReplayDataset: Simulation of DMTA rounds for retrospective evaluation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
import warnings

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from .standardize import (
    canonicalize_smiles,
    standardize_units,
    aggregate_replicates,
    assign_rounds,
    ENDPOINT_DEFINITIONS,
)


# =============================================================================
# Molecular Graph Features
# =============================================================================

# Atom feature dimensions
ATOM_FEATURES = {
    'element': 44,           # Common elements
    'formal_charge': 5,      # -2 to +2
    'hybridization': 5,      # sp, sp2, sp3, sp3d, sp3d2
    'aromaticity': 1,        # Binary
    'ring_membership': 7,    # 0-6 rings
    'hydrogen_count': 5,     # 0-4
    'chirality': 3,          # R, S, unspecified
}
ATOM_FEATURE_DIM = sum(ATOM_FEATURES.values())  # 70

# Bond feature dimensions
BOND_FEATURES = {
    'bond_type': 4,          # Single, double, triple, aromatic
    'conjugation': 1,        # Binary
    'ring_membership': 1,    # Binary
    'stereochemistry': 3,    # E, Z, none
}
BOND_FEATURE_DIM = sum(BOND_FEATURES.values())  # 9


def smiles_to_graph(smiles: str) -> Optional[Dict[str, torch.Tensor]]:
    """
    Convert SMILES to molecular graph representation.

    Parameters
    ----------
    smiles : str
        Input SMILES string

    Returns
    -------
    dict or None
        Dictionary with:
        - 'node_features': Tensor of shape (num_atoms, atom_feature_dim)
        - 'edge_index': Tensor of shape (2, num_edges)
        - 'edge_features': Tensor of shape (num_edges, bond_feature_dim)
        - 'num_atoms': int
    """
    if not RDKIT_AVAILABLE:
        warnings.warn("RDKit required for graph conversion")
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Add hydrogens for more accurate features
        mol = Chem.AddHs(mol)

        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return None

        # Extract atom features
        node_features = []
        for atom in mol.GetAtoms():
            features = get_atom_features(atom)
            node_features.append(features)
        node_features = torch.tensor(node_features, dtype=torch.float32)

        # Extract bond features and adjacency
        edge_index = []
        edge_features = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # Add both directions
            edge_index.append([i, j])
            edge_index.append([j, i])

            features = get_bond_features(bond)
            edge_features.append(features)
            edge_features.append(features)  # Same for both directions

        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_features = torch.tensor(edge_features, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_features = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float32)

        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'num_atoms': num_atoms,
        }

    except Exception as e:
        return None


def get_atom_features(atom) -> List[float]:
    """Extract features for a single atom."""
    features = []

    # Element type (one-hot, 44 elements)
    element_list = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si',
                   'B', 'Na', 'K', 'Ca', 'Mg', 'Fe', 'Zn', 'Cu', 'Mn',
                   'H', 'Se', 'As', 'Li', 'Al', 'Ga', 'Ge', 'Sn', 'Pb',
                   'Bi', 'Sb', 'Te', 'Ag', 'Au', 'Pt', 'Pd', 'Rh', 'Ru',
                   'Re', 'Os', 'Ir', 'Hg', 'Cd', 'Ti', 'V']
    symbol = atom.GetSymbol()
    element_vec = [1.0 if symbol == e else 0.0 for e in element_list]
    features.extend(element_vec)

    # Formal charge (-2 to +2, one-hot)
    charge = atom.GetFormalCharge()
    charge_vec = [0.0] * 5
    charge_idx = max(0, min(4, charge + 2))
    charge_vec[charge_idx] = 1.0
    features.extend(charge_vec)

    # Hybridization
    hyb = atom.GetHybridization()
    hyb_types = [Chem.rdchem.HybridizationType.SP,
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3,
                 Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2]
    hyb_vec = [1.0 if hyb == h else 0.0 for h in hyb_types]
    features.extend(hyb_vec)

    # Aromaticity
    features.append(1.0 if atom.GetIsAromatic() else 0.0)

    # Ring membership (0-6 rings)
    ring_info = atom.GetOwningMol().GetRingInfo()
    num_rings = min(6, ring_info.NumAtomRings(atom.GetIdx()))
    ring_vec = [0.0] * 7
    ring_vec[num_rings] = 1.0
    features.extend(ring_vec)

    # Hydrogen count (0-4)
    h_count = min(4, atom.GetTotalNumHs())
    h_vec = [0.0] * 5
    h_vec[h_count] = 1.0
    features.extend(h_vec)

    # Chirality
    chirality = atom.GetChiralTag()
    chiral_types = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                    Chem.rdchem.ChiralType.CHI_UNSPECIFIED]
    chiral_vec = [1.0 if chirality == c else 0.0 for c in chiral_types]
    features.extend(chiral_vec)

    return features


def get_bond_features(bond) -> List[float]:
    """Extract features for a single bond."""
    features = []

    # Bond type
    bt = bond.GetBondType()
    bond_types = [Chem.rdchem.BondType.SINGLE,
                  Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE,
                  Chem.rdchem.BondType.AROMATIC]
    bt_vec = [1.0 if bt == t else 0.0 for t in bond_types]
    features.extend(bt_vec)

    # Conjugation
    features.append(1.0 if bond.GetIsConjugated() else 0.0)

    # Ring membership
    features.append(1.0 if bond.IsInRing() else 0.0)

    # Stereochemistry
    stereo = bond.GetStereo()
    stereo_types = [Chem.rdchem.BondStereo.STEREOE,
                    Chem.rdchem.BondStereo.STEREOZ,
                    Chem.rdchem.BondStereo.STEREONONE]
    stereo_vec = [1.0 if stereo == s else 0.0 for s in stereo_types]
    features.extend(stereo_vec)

    return features


# =============================================================================
# Portfolio Dataset (Pretraining)
# =============================================================================

class PortfolioDataset(Dataset):
    """
    Large-scale dataset for pretraining on pooled bioactivity data.

    Combines data from ChEMBL, BindingDB, and TDC.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        endpoints: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        cache_graphs: bool = True,
    ):
        """
        Parameters
        ----------
        data_dir : str or Path
            Path to processed data directory
        endpoints : list of str, optional
            Endpoints to include (default: all available)
        max_samples : int, optional
            Maximum number of samples to load
        cache_graphs : bool
            Whether to cache molecular graphs in memory
        """
        self.data_dir = Path(data_dir)
        self.endpoints = endpoints
        self.cache_graphs = cache_graphs
        self._graph_cache = {}

        # Load data
        self.data = self._load_data(max_samples)
        self.smiles_list = self.data['smiles'].tolist()

        print(f"PortfolioDataset initialized with {len(self)} samples")
        print(f"Endpoints: {list(self.data.columns[1:])}")

    def _load_data(self, max_samples: Optional[int]) -> pd.DataFrame:
        """Load and combine data from all sources."""
        dfs = []

        # Load TDC ADMET data
        tdc_dir = self.data_dir / 'tdc'
        if tdc_dir.exists():
            for csv_file in tdc_dir.glob('*.csv'):
                try:
                    df = pd.read_csv(csv_file)
                    if 'Drug' in df.columns:
                        df = df.rename(columns={'Drug': 'smiles', 'Y': csv_file.stem})
                    elif 'smiles' not in df.columns and len(df.columns) >= 2:
                        df.columns = ['smiles', csv_file.stem] + list(df.columns[2:])
                    dfs.append(df[['smiles', csv_file.stem]])
                except Exception as e:
                    print(f"Warning: Could not load {csv_file}: {e}")

        # Load ChEMBL data if available
        chembl_dir = self.data_dir / 'chembl'
        # Implementation for ChEMBL loading would go here

        # Combine all dataframes
        if len(dfs) == 0:
            raise ValueError(f"No data files found in {self.data_dir}")

        # Merge on SMILES
        combined = dfs[0]
        for df in dfs[1:]:
            combined = pd.merge(combined, df, on='smiles', how='outer')

        # Filter endpoints if specified
        if self.endpoints:
            cols_to_keep = ['smiles'] + [e for e in self.endpoints if e in combined.columns]
            combined = combined[cols_to_keep]

        # Limit samples
        if max_samples and len(combined) > max_samples:
            combined = combined.sample(n=max_samples, random_state=42)

        return combined.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        row = self.data.iloc[idx]
        smiles = row['smiles']

        # Get or compute molecular graph
        if self.cache_graphs and smiles in self._graph_cache:
            graph = self._graph_cache[smiles]
        else:
            graph = smiles_to_graph(smiles)
            if self.cache_graphs and graph is not None:
                self._graph_cache[smiles] = graph

        # Get endpoint values
        endpoints = {}
        masks = {}
        for col in self.data.columns[1:]:
            val = row[col]
            if pd.notna(val):
                endpoints[col] = torch.tensor([val], dtype=torch.float32)
                masks[col] = torch.tensor([1.0], dtype=torch.float32)
            else:
                endpoints[col] = torch.tensor([0.0], dtype=torch.float32)
                masks[col] = torch.tensor([0.0], dtype=torch.float32)

        return {
            'smiles': smiles,
            'graph': graph,
            'endpoints': endpoints,
            'masks': masks,
        }


# =============================================================================
# Program Dataset (DMTA Replay)
# =============================================================================

class ProgramDataset(Dataset):
    """
    Dataset for a single drug discovery program with temporal structure.

    Supports:
    - Round-based data organization
    - Context embeddings (L1 program, L2 assay, L3 round)
    - Temporal train/test splits
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        program_id: str,
        assay_id_column: str = 'assay_id',
        date_column: str = 'test_date',
        smiles_column: str = 'smiles',
        endpoint_columns: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        data_path : str or Path
            Path to program data CSV
        program_id : str
            Identifier for this program
        assay_id_column : str
            Column containing assay identifiers
        date_column : str
            Column containing test dates
        smiles_column : str
            Column containing SMILES strings
        endpoint_columns : list of str, optional
            Columns containing endpoint measurements
        """
        self.data_path = Path(data_path)
        self.program_id = program_id
        self.assay_id_column = assay_id_column
        self.date_column = date_column
        self.smiles_column = smiles_column

        # Load and process data
        self.data = self._load_and_process(endpoint_columns)
        self._graph_cache = {}

        # Extract metadata
        self.num_rounds = self.data['round_id'].nunique()
        self.num_assays = self.data[assay_id_column].nunique() if assay_id_column in self.data.columns else 1
        self.endpoint_columns = [c for c in self.data.columns
                                if c not in [smiles_column, date_column, assay_id_column, 'round_id']]

        print(f"ProgramDataset '{program_id}' initialized:")
        print(f"  Compounds: {len(self)}")
        print(f"  Rounds: {self.num_rounds}")
        print(f"  Assays: {self.num_assays}")
        print(f"  Endpoints: {self.endpoint_columns}")

    def _load_and_process(self, endpoint_columns: Optional[List[str]]) -> pd.DataFrame:
        """Load and preprocess program data."""
        df = pd.read_csv(self.data_path)

        # Assign rounds if date column exists
        if self.date_column in df.columns:
            df = assign_rounds(df, self.date_column)
        else:
            df['round_id'] = 0

        # Filter endpoint columns if specified
        if endpoint_columns:
            keep_cols = [self.smiles_column, self.date_column, self.assay_id_column, 'round_id']
            keep_cols = [c for c in keep_cols if c in df.columns]
            keep_cols.extend([c for c in endpoint_columns if c in df.columns])
            df = df[keep_cols]

        return df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample with context information."""
        row = self.data.iloc[idx]
        smiles = row[self.smiles_column]

        # Get molecular graph
        if smiles in self._graph_cache:
            graph = self._graph_cache[smiles]
        else:
            graph = smiles_to_graph(smiles)
            self._graph_cache[smiles] = graph

        # Get context IDs
        round_id = row.get('round_id', 0)
        assay_id = row.get(self.assay_id_column, 0)

        # Get endpoint values
        endpoints = {}
        masks = {}
        for col in self.endpoint_columns:
            val = row[col]
            if pd.notna(val):
                endpoints[col] = torch.tensor([val], dtype=torch.float32)
                masks[col] = torch.tensor([1.0], dtype=torch.float32)
            else:
                endpoints[col] = torch.tensor([0.0], dtype=torch.float32)
                masks[col] = torch.tensor([0.0], dtype=torch.float32)

        return {
            'smiles': smiles,
            'graph': graph,
            'program_id': self.program_id,
            'assay_id': assay_id,
            'round_id': round_id,
            'endpoints': endpoints,
            'masks': masks,
        }

    def get_round_data(self, round_id: int) -> pd.DataFrame:
        """Get all data for a specific round."""
        return self.data[self.data['round_id'] == round_id]

    def get_temporal_split(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Create temporal train/val/test split based on rounds.

        Returns indices for each split.
        """
        rounds = sorted(self.data['round_id'].unique())
        n_rounds = len(rounds)

        train_rounds = rounds[:int(n_rounds * train_ratio)]
        val_rounds = rounds[int(n_rounds * train_ratio):int(n_rounds * (train_ratio + val_ratio))]
        test_rounds = rounds[int(n_rounds * (train_ratio + val_ratio)):]

        train_idx = self.data[self.data['round_id'].isin(train_rounds)].index.tolist()
        val_idx = self.data[self.data['round_id'].isin(val_rounds)].index.tolist()
        test_idx = self.data[self.data['round_id'].isin(test_rounds)].index.tolist()

        return train_idx, val_idx, test_idx


# =============================================================================
# DMTA Replay Dataset
# =============================================================================

class DMTAReplayDataset:
    """
    Dataset for simulating DMTA replay experiments.

    Implements the replay engine from the research plan:
    - Timeline reconstruction
    - Candidate pool definition
    - Budget constraints
    """

    def __init__(
        self,
        program_dataset: ProgramDataset,
        seed_rounds: int = 2,
        budget_per_round: Optional[int] = None,
        budget_ratio: float = 0.5,
    ):
        """
        Parameters
        ----------
        program_dataset : ProgramDataset
            The underlying program data
        seed_rounds : int
            Number of initial rounds for model initialization
        budget_per_round : int, optional
            Fixed budget per round (overrides budget_ratio)
        budget_ratio : float
            Fraction of historical round size as budget
        """
        self.program = program_dataset
        self.seed_rounds = seed_rounds
        self.budget_per_round = budget_per_round
        self.budget_ratio = budget_ratio

        # Get round information
        self.rounds = sorted(self.program.data['round_id'].unique())
        self.num_rounds = len(self.rounds)

        # Seed window
        self.seed_window = self.rounds[:seed_rounds]
        self.replay_rounds = self.rounds[seed_rounds:]

        print(f"DMTAReplayDataset initialized:")
        print(f"  Total rounds: {self.num_rounds}")
        print(f"  Seed rounds: {len(self.seed_window)}")
        print(f"  Replay rounds: {len(self.replay_rounds)}")

    def get_seed_data(self) -> pd.DataFrame:
        """Get data for model initialization."""
        return self.program.data[self.program.data['round_id'].isin(self.seed_window)]

    def get_candidate_pool(self, current_round: int) -> pd.DataFrame:
        """
        Get candidate pool for a round.

        Pool includes all compounds from current and future rounds.
        """
        future_rounds = [r for r in self.rounds if r >= current_round]
        return self.program.data[self.program.data['round_id'].isin(future_rounds)]

    def get_round_budget(self, round_id: int) -> int:
        """Calculate budget for a specific round."""
        if self.budget_per_round:
            return self.budget_per_round

        # Calculate based on historical round size
        historical_size = len(self.program.data[self.program.data['round_id'] == round_id])
        budget = int(historical_size * self.budget_ratio)

        # Enforce bounds
        return max(10, min(50, budget))

    def simulate_round(
        self,
        round_id: int,
        selection_fn,
        budget: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Simulate a single DMTA round.

        Parameters
        ----------
        round_id : int
            Round to simulate
        selection_fn : callable
            Function that takes candidate pool and returns selected indices
        budget : int, optional
            Override budget for this round

        Returns
        -------
        dict
            Results including selected compounds and ground truth
        """
        if budget is None:
            budget = self.get_round_budget(round_id)

        # Get candidate pool
        pool = self.get_candidate_pool(round_id)

        # Apply selection function
        selected_idx = selection_fn(pool, budget)

        # Get selected compounds
        selected = pool.iloc[selected_idx]

        return {
            'round_id': round_id,
            'budget': budget,
            'pool_size': len(pool),
            'selected': selected,
            'selected_idx': selected_idx,
        }


# =============================================================================
# Data Loading Utilities
# =============================================================================

def collate_graphs(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for batching molecular graphs.

    Handles variable-size graphs by creating batch indices.
    """
    smiles = [b['smiles'] for b in batch]
    graphs = [b['graph'] for b in batch]

    # Filter out None graphs
    valid_idx = [i for i, g in enumerate(graphs) if g is not None]
    if len(valid_idx) == 0:
        return None

    valid_graphs = [graphs[i] for i in valid_idx]

    # Combine graphs into batch
    node_features = []
    edge_index = []
    edge_features = []
    batch_idx = []

    node_offset = 0
    for i, g in enumerate(valid_graphs):
        num_atoms = g['num_atoms']
        node_features.append(g['node_features'])
        edge_index.append(g['edge_index'] + node_offset)
        edge_features.append(g['edge_features'])
        batch_idx.extend([i] * num_atoms)
        node_offset += num_atoms

    batched = {
        'smiles': [smiles[i] for i in valid_idx],
        'node_features': torch.cat(node_features, dim=0),
        'edge_index': torch.cat(edge_index, dim=1),
        'edge_features': torch.cat(edge_features, dim=0),
        'batch': torch.tensor(batch_idx, dtype=torch.long),
    }

    # Batch endpoint values
    if 'endpoints' in batch[0]:
        endpoints = {}
        masks = {}
        for key in batch[0]['endpoints'].keys():
            endpoints[key] = torch.stack([batch[i]['endpoints'][key] for i in valid_idx])
            masks[key] = torch.stack([batch[i]['masks'][key] for i in valid_idx])
        batched['endpoints'] = endpoints
        batched['masks'] = masks

    return batched


if __name__ == '__main__':
    # Test graph conversion
    print("Testing SMILES to graph conversion...")
    test_smiles = "CCO"  # Ethanol
    graph = smiles_to_graph(test_smiles)
    if graph:
        print(f"  SMILES: {test_smiles}")
        print(f"  Atoms: {graph['num_atoms']}")
        print(f"  Node features shape: {graph['node_features'].shape}")
        print(f"  Edge index shape: {graph['edge_index'].shape}")
        print(f"  Edge features shape: {graph['edge_features'].shape}")
