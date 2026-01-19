#!/usr/bin/env python3
"""
NEST-DRUG: Nested-Learning Platform for Drug Discovery

Complete model integrating:
- MPNN backbone for molecular encoding (L0)
- Nested context modules (L1-L3) for hierarchical adaptation
- FiLM conditioning for context modulation
- Multi-task prediction heads for potency and ADMET
- Deep ensemble for uncertainty quantification

Architecture Overview:
    Molecule (Graph) → MPNN → h_mol (512-dim)
                                ↓
    Contexts (L1,L2,L3) → FiLM → h_mod (512-dim)
                                ↓
                        Multi-Task Heads → Predictions
                                ↓
                            D-Score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

from .mpnn import MPNN
from .context import NestedContextModule, ContextRegularizer
from .heads import MultiTaskHead, DScoreHead


class NESTDRUG(nn.Module):
    """
    Complete NEST-DRUG model.

    Combines molecular encoding with hierarchical context adaptation
    for multi-task prediction of drug-like properties.
    """

    def __init__(
        self,
        # MPNN configuration
        node_input_dim: int = 70,
        edge_input_dim: int = 12,
        hidden_dim: int = 256,
        num_mpnn_layers: int = 6,
        # Context configuration
        program_dim: int = 128,
        assay_dim: int = 64,
        round_dim: int = 32,
        num_programs: int = 10,
        num_assays: int = 100,
        num_rounds: int = 200,
        # Prediction heads
        endpoints: Optional[Dict[str, Dict]] = None,
        head_hidden_dims: List[int] = [256, 128],
        # Regularization
        dropout: float = 0.1,
        # D-score
        dscore_config: Optional[Dict] = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.feature_dim = 2 * hidden_dim  # MPNN output after pooling

        # L0: MPNN Backbone (shared molecular encoder)
        self.mpnn = MPNN(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mpnn_layers,
            dropout=dropout,
        )

        # L1-L3: Nested Context Modules
        self.context_module = NestedContextModule(
            program_dim=program_dim,
            assay_dim=assay_dim,
            round_dim=round_dim,
            num_programs=num_programs,
            num_assays=num_assays,
            num_rounds=num_rounds,
            feature_dim=self.feature_dim,
        )

        # Multi-Task Prediction Heads
        self.prediction_heads = MultiTaskHead(
            input_dim=self.feature_dim,
            endpoints=endpoints,
            hidden_dims=head_hidden_dims,
            dropout=dropout,
        )

        # D-Score computation
        self.dscore_head = DScoreHead(endpoint_configs=dscore_config)

        # Context regularizer for continual learning
        self.context_regularizer = ContextRegularizer()

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch: torch.Tensor,
        program_ids: torch.Tensor,
        assay_ids: torch.Tensor,
        round_ids: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.

        Args:
            node_features: Atom features [num_atoms, node_input_dim]
            edge_index: Bond indices [2, num_edges]
            edge_features: Bond features [num_edges, edge_input_dim]
            batch: Batch assignment for atoms [num_atoms]
            program_ids: Program context IDs [batch_size]
            assay_ids: Assay context IDs [batch_size]
            round_ids: Round context IDs [batch_size]
            return_embeddings: If True, include intermediate embeddings

        Returns:
            Dictionary with predictions per endpoint
        """
        # L0: Molecular encoding
        h_mol = self.mpnn(node_features, edge_index, edge_features, batch)

        # L1-L3: Context modulation
        h_mod = self.context_module(h_mol, program_ids, assay_ids, round_ids)

        # Multi-task prediction
        predictions = self.prediction_heads(h_mod)

        if return_embeddings:
            predictions['h_mol'] = h_mol
            predictions['h_mod'] = h_mod

        return predictions

    def forward_with_dscore(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch: torch.Tensor,
        program_ids: torch.Tensor,
        assay_ids: torch.Tensor,
        round_ids: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass with D-score computation.

        Returns:
            predictions: Endpoint predictions
            d_score: Overall desirability score
            gates_pass: Boolean mask for hard gate satisfaction
        """
        predictions = self.forward(
            node_features, edge_index, edge_features, batch,
            program_ids, assay_ids, round_ids,
        )

        d_score, _ = self.dscore_head(predictions)
        gates_pass = self.dscore_head.check_hard_gates(predictions)

        return predictions, d_score, gates_pass

    def compute_loss(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch: torch.Tensor,
        program_ids: torch.Tensor,
        assay_ids: torch.Tensor,
        round_ids: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
        include_drift_penalty: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute training loss.

        Args:
            ... (same as forward)
            targets: Ground truth values per endpoint
            masks: Valid label masks per endpoint
            include_drift_penalty: Add context drift regularization

        Returns:
            total_loss: Combined loss value
            endpoint_losses: Individual losses per endpoint
        """
        predictions = self.forward(
            node_features, edge_index, edge_features, batch,
            program_ids, assay_ids, round_ids,
        )

        total_loss, endpoint_losses = self.prediction_heads.compute_loss(
            predictions, targets, masks
        )

        # Add context drift penalty if enabled
        if include_drift_penalty:
            drift_loss = self.context_regularizer.compute_drift_loss(self.context_module)
            total_loss = total_loss + drift_loss
            endpoint_losses['drift'] = drift_loss

        return total_loss, endpoint_losses

    def encode_molecules(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get molecular embeddings without context modulation.

        Useful for pretraining or analysis.
        """
        return self.mpnn(node_features, edge_index, edge_features, batch)

    def predict_with_context(
        self,
        h_mol: torch.Tensor,
        program_ids: torch.Tensor,
        assay_ids: torch.Tensor,
        round_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict from pre-computed molecular embeddings.

        Useful for efficient scoring of large compound libraries.
        """
        h_mod = self.context_module(h_mol, program_ids, assay_ids, round_ids)
        return self.prediction_heads(h_mod)

    # ===== Context Management =====

    def add_program(self, num_new: int = 1) -> None:
        """Add new program context(s)."""
        self.context_module.add_program(num_new)

    def add_assay(self, num_new: int = 1) -> None:
        """Add new assay context(s)."""
        self.context_module.add_assay(num_new)

    def add_round(self, num_new: int = 1, clone_from: Optional[int] = None) -> None:
        """Add new round context(s), optionally cloning from previous."""
        self.context_module.add_round(num_new, clone_from)

    def freeze_backbone(self) -> None:
        """Freeze MPNN backbone (L0) parameters."""
        for param in self.mpnn.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze MPNN backbone parameters."""
        for param in self.mpnn.parameters():
            param.requires_grad = True

    def freeze_context_level(self, level: str) -> None:
        """Freeze specific context level (L1/L2/L3)."""
        self.context_module.freeze_level(level)

    def unfreeze_context_level(self, level: str) -> None:
        """Unfreeze specific context level."""
        self.context_module.unfreeze_level(level)

    def store_context_snapshot(self) -> None:
        """Store current context values for drift regularization."""
        self.context_regularizer.store_contexts(self.context_module)

    # ===== Parameter Groups for Different Learning Rates =====

    def get_parameter_groups(
        self,
        backbone_lr: float = 1e-5,
        context_lr: float = 1e-3,
        head_lr: float = 1e-4,
    ) -> List[Dict[str, Any]]:
        """
        Get parameter groups with different learning rates.

        Useful for multi-timescale training:
        - Backbone (L0): Slow updates to preserve pretraining
        - Contexts (L1-L3): Faster adaptation to new data
        - Heads: Moderate updates
        """
        param_groups = [
            {'params': self.mpnn.parameters(), 'lr': backbone_lr, 'name': 'backbone'},
            {'params': self.context_module.parameters(), 'lr': context_lr, 'name': 'context'},
            {'params': self.prediction_heads.parameters(), 'lr': head_lr, 'name': 'heads'},
        ]
        return param_groups

    def get_context_parameter_groups(
        self,
        l1_lr: float = 5e-4,
        l2_lr: float = 5e-4,
        l3_lr: float = 1e-3,
    ) -> List[Dict[str, Any]]:
        """
        Get parameter groups for context levels with different learning rates.

        L3 (round) typically updates faster than L1 (program).
        """
        param_groups = [
            {
                'params': [self.context_module.program_embeddings.embeddings.weight],
                'lr': l1_lr,
                'name': 'L1_program',
            },
            {
                'params': [self.context_module.assay_embeddings.embeddings.weight],
                'lr': l2_lr,
                'name': 'L2_assay',
            },
            {
                'params': [self.context_module.round_embeddings.embeddings.weight],
                'lr': l3_lr,
                'name': 'L3_round',
            },
        ]
        return param_groups


def create_nest_drug(
    num_programs: int = 5,
    num_assays: int = 50,
    num_rounds: int = 150,
    endpoints: Optional[Dict] = None,
) -> NESTDRUG:
    """
    Factory function to create NEST-DRUG with default configuration.

    Args:
        num_programs: Expected number of programs
        num_assays: Expected number of assay contexts
        num_rounds: Expected number of round contexts
        endpoints: Endpoint configuration (default: standard ADMET panel)

    Returns:
        Configured NESTDRUG model
    """
    if endpoints is None:
        endpoints = {
            # Primary potency
            'pActivity': {'type': 'regression', 'weight': 3.0},
            # ADMET regression
            'solubility': {'type': 'regression', 'weight': 1.0},
            'lipophilicity': {'type': 'regression', 'weight': 1.0},
            'clearance_hepatocyte': {'type': 'regression', 'weight': 1.0},
            'clearance_microsome': {'type': 'regression', 'weight': 1.0},
            'caco2': {'type': 'regression', 'weight': 1.0},
            'ppbr': {'type': 'regression', 'weight': 1.0},
            # ADMET classification
            'herg': {'type': 'classification', 'weight': 1.5},
            'ames': {'type': 'classification', 'weight': 1.0},
            'bbb': {'type': 'classification', 'weight': 1.0},
        }

    model = NESTDRUG(
        # MPNN - atom_features() returns 69 dims, bond_features() returns 9 dims
        node_input_dim=69,
        edge_input_dim=9,
        hidden_dim=256,
        num_mpnn_layers=6,
        # Context
        program_dim=128,
        assay_dim=64,
        round_dim=32,
        num_programs=num_programs,
        num_assays=num_assays,
        num_rounds=num_rounds,
        # Prediction
        endpoints=endpoints,
        head_hidden_dims=[256, 128],
        dropout=0.1,
    )

    return model


if __name__ == '__main__':
    # Test complete model
    print("Testing NEST-DRUG Model...")
    print("=" * 60)

    # Create model
    model = create_nest_drug(
        num_programs=5,
        num_assays=50,
        num_rounds=150,
    )

    # Print architecture summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Architecture:")
    print(f"  MPNN layers: 6")
    print(f"  Hidden dim: 256")
    print(f"  Feature dim: 512")
    print(f"  Context dims: L1={128}, L2={64}, L3={32}")
    print(f"  Endpoints: {list(model.prediction_heads.endpoint_names)}")
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Component breakdown
    mpnn_params = sum(p.numel() for p in model.mpnn.parameters())
    context_params = sum(p.numel() for p in model.context_module.parameters())
    head_params = sum(p.numel() for p in model.prediction_heads.parameters())

    print(f"\nParameter Distribution:")
    print(f"  MPNN (L0): {mpnn_params:,} ({mpnn_params/total_params*100:.1f}%)")
    print(f"  Context (L1-L3): {context_params:,} ({context_params/total_params*100:.1f}%)")
    print(f"  Heads: {head_params:,} ({head_params/total_params*100:.1f}%)")

    # Test forward pass
    print("\nTesting Forward Pass...")

    batch_size = 8
    num_atoms = 40
    num_edges = 80

    # Create dummy molecular graph (69 atom features, 9 bond features)
    node_features = torch.randn(num_atoms, 69)
    edge_index = torch.randint(0, num_atoms, (2, num_edges))
    edge_features = torch.randn(num_edges, 9)
    batch = torch.tensor([0]*5 + [1]*5 + [2]*5 + [3]*5 + [4]*5 + [5]*5 + [6]*5 + [7]*5)

    # Context IDs
    program_ids = torch.randint(0, 5, (batch_size,))
    assay_ids = torch.randint(0, 50, (batch_size,))
    round_ids = torch.randint(0, 150, (batch_size,))

    # Forward
    predictions = model(
        node_features, edge_index, edge_features, batch,
        program_ids, assay_ids, round_ids,
        return_embeddings=True,
    )

    print(f"  Input: {num_atoms} atoms, {num_edges} edges, {batch_size} molecules")
    print(f"  h_mol: {predictions['h_mol'].shape}")
    print(f"  h_mod: {predictions['h_mod'].shape}")
    print(f"  Predictions:")
    for name, pred in predictions.items():
        if name not in ['h_mol', 'h_mod']:
            print(f"    {name}: {pred.shape}")

    # Test with D-score
    print("\nTesting D-Score Computation...")
    predictions, d_score, gates_pass = model.forward_with_dscore(
        node_features, edge_index, edge_features, batch,
        program_ids, assay_ids, round_ids,
    )

    print(f"  D-score: {d_score.shape}, range [{d_score.min():.3f}, {d_score.max():.3f}]")
    print(f"  Gates pass: {gates_pass.sum()}/{len(gates_pass)}")

    print("\n" + "=" * 60)
    print("NEST-DRUG Model Test Complete!")
