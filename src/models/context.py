#!/usr/bin/env python3
"""
Nested Context Modules for NEST-DRUG

Implements the hierarchical context adaptation system:
- L1: Program/Target context (128-dim)
- L2: Assay/Platform context (64-dim)
- L3: Round/Batch context (32-dim)

Context modulation via Feature-wise Linear Modulation (FiLM):
    h_mod = γ ⊙ h_mol + β

References:
- Perez et al. (2018). FiLM: Visual Reasoning with a General Conditioning Layer
- Behrouz et al. (2025). Nested Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class ContextEmbedding(nn.Module):
    """
    Learnable context embedding for a single hierarchy level.

    Maintains a lookup table of embeddings indexed by context ID.
    Supports dynamic addition of new contexts during continual learning.
    """

    def __init__(
        self,
        num_contexts: int,
        embedding_dim: int,
        init_std: float = 0.01,
    ):
        super().__init__()

        self.num_contexts = num_contexts
        self.embedding_dim = embedding_dim
        self.init_std = init_std

        # Embedding table
        self.embeddings = nn.Embedding(num_contexts, embedding_dim)

        # Initialize with small random values
        nn.init.normal_(self.embeddings.weight, mean=0, std=init_std)

    def forward(self, context_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup context embeddings.

        Args:
            context_ids: Context indices [batch_size]

        Returns:
            Context embeddings [batch_size, embedding_dim]
        """
        return self.embeddings(context_ids)

    def add_contexts(self, num_new: int) -> None:
        """
        Add new context embeddings (for continual learning).

        New embeddings are initialized from the mean of existing embeddings
        plus small random noise.
        """
        old_weight = self.embeddings.weight.data
        old_num = old_weight.shape[0]
        new_num = old_num + num_new

        # Create new embedding table
        new_embeddings = nn.Embedding(new_num, self.embedding_dim)

        # Copy old weights
        new_embeddings.weight.data[:old_num] = old_weight

        # Initialize new weights from mean + noise
        mean_emb = old_weight.mean(dim=0)
        for i in range(num_new):
            noise = torch.randn_like(mean_emb) * self.init_std
            new_embeddings.weight.data[old_num + i] = mean_emb + noise

        self.embeddings = new_embeddings
        self.num_contexts = new_num

    def clone_context(self, source_id: int, target_id: int, noise_std: float = 0.001) -> None:
        """
        Clone a context embedding with small perturbation.

        Useful for initializing new round contexts from previous round.
        """
        with torch.no_grad():
            source_emb = self.embeddings.weight[source_id]
            noise = torch.randn_like(source_emb) * noise_std
            self.embeddings.weight[target_id] = source_emb + noise


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    Computes affine transformation parameters (γ, β) from context
    and applies element-wise modulation to input features:

        output = γ ⊙ input + β

    This allows context to modulate the molecular representation
    without changing the underlying feature extraction.
    """

    def __init__(
        self,
        context_dim: int,
        feature_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()

        self.context_dim = context_dim
        self.feature_dim = feature_dim

        if hidden_dim is None:
            hidden_dim = max(context_dim, feature_dim // 2)

        # MLP to compute γ (scale)
        self.gamma_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # MLP to compute β (shift)
        self.beta_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Initialize to identity transformation (γ=1, β=0)
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.ones_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply FiLM modulation.

        Args:
            x: Input features [batch_size, feature_dim]
            context: Context vector [batch_size, context_dim]

        Returns:
            Modulated features [batch_size, feature_dim]
        """
        gamma = self.gamma_net(context)
        beta = self.beta_net(context)

        return gamma * x + beta

    def get_modulation_params(
        self,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get modulation parameters without applying them.

        Useful for analysis/visualization.
        """
        gamma = self.gamma_net(context)
        beta = self.beta_net(context)
        return gamma, beta


class NestedContextModule(nn.Module):
    """
    Complete nested context system for NEST-DRUG.

    Manages three levels of context:
    - L1 (Program): Target biology, medicinal chemistry strategy
    - L2 (Assay): Lab-specific noise, calibration, platform effects
    - L3 (Round): Current SAR regime, local distribution shifts

    Concatenates all contexts and applies FiLM modulation to
    molecular embeddings from the MPNN backbone.
    """

    def __init__(
        self,
        # Context dimensions
        program_dim: int = 128,
        assay_dim: int = 64,
        round_dim: int = 32,
        # Number of contexts (can grow dynamically)
        num_programs: int = 10,
        num_assays: int = 100,
        num_rounds: int = 200,
        # Feature dimension (from MPNN)
        feature_dim: int = 512,
        # Initialization
        init_std: float = 0.01,
    ):
        super().__init__()

        self.program_dim = program_dim
        self.assay_dim = assay_dim
        self.round_dim = round_dim
        self.feature_dim = feature_dim

        # Total context dimension
        self.total_context_dim = program_dim + assay_dim + round_dim

        # Context embeddings
        self.program_embeddings = ContextEmbedding(num_programs, program_dim, init_std)
        self.assay_embeddings = ContextEmbedding(num_assays, assay_dim, init_std)
        self.round_embeddings = ContextEmbedding(num_rounds, round_dim, init_std)

        # FiLM layer
        self.film = FiLMLayer(
            context_dim=self.total_context_dim,
            feature_dim=feature_dim,
        )

        # Optional: Context interaction layer
        self.context_interaction = nn.Sequential(
            nn.Linear(self.total_context_dim, self.total_context_dim),
            nn.LayerNorm(self.total_context_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        h_mol: torch.Tensor,
        program_ids: torch.Tensor,
        assay_ids: torch.Tensor,
        round_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply nested context modulation to molecular embeddings.

        Args:
            h_mol: Molecular embeddings from MPNN [batch_size, feature_dim]
            program_ids: Program context indices [batch_size]
            assay_ids: Assay context indices [batch_size]
            round_ids: Round context indices [batch_size]

        Returns:
            Context-modulated embeddings [batch_size, feature_dim]
        """
        # Get context embeddings
        z_program = self.program_embeddings(program_ids)
        z_assay = self.assay_embeddings(assay_ids)
        z_round = self.round_embeddings(round_ids)

        # Concatenate all contexts
        context = torch.cat([z_program, z_assay, z_round], dim=-1)

        # Optional context interaction
        context = self.context_interaction(context)

        # Apply FiLM modulation
        h_mod = self.film(h_mol, context)

        return h_mod

    def get_context_vector(
        self,
        program_ids: torch.Tensor,
        assay_ids: torch.Tensor,
        round_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get concatenated context vector without modulation.
        """
        z_program = self.program_embeddings(program_ids)
        z_assay = self.assay_embeddings(assay_ids)
        z_round = self.round_embeddings(round_ids)

        context = torch.cat([z_program, z_assay, z_round], dim=-1)
        return self.context_interaction(context)

    def add_program(self, num_new: int = 1) -> None:
        """Add new program context(s)."""
        self.program_embeddings.add_contexts(num_new)

    def add_assay(self, num_new: int = 1) -> None:
        """Add new assay context(s)."""
        self.assay_embeddings.add_contexts(num_new)

    def add_round(self, num_new: int = 1, clone_from: Optional[int] = None) -> None:
        """
        Add new round context(s).

        If clone_from is specified, initialize from that round's embedding.
        """
        old_num = self.round_embeddings.num_contexts
        self.round_embeddings.add_contexts(num_new)

        if clone_from is not None:
            for i in range(num_new):
                self.round_embeddings.clone_context(clone_from, old_num + i)

    def get_context_parameters(self, level: str) -> nn.Parameter:
        """Get parameters for a specific context level (for targeted updates)."""
        if level == 'program' or level == 'L1':
            return self.program_embeddings.embeddings.weight
        elif level == 'assay' or level == 'L2':
            return self.assay_embeddings.embeddings.weight
        elif level == 'round' or level == 'L3':
            return self.round_embeddings.embeddings.weight
        else:
            raise ValueError(f"Unknown context level: {level}")

    def freeze_level(self, level: str) -> None:
        """Freeze parameters for a context level."""
        params = self.get_context_parameters(level)
        params.requires_grad = False

    def unfreeze_level(self, level: str) -> None:
        """Unfreeze parameters for a context level."""
        params = self.get_context_parameters(level)
        params.requires_grad = True


class ContextRegularizer(nn.Module):
    """
    Regularization for context embeddings during continual learning.

    Implements:
    - L2 penalty on context drift from previous values
    - EWC-style importance weighting (optional)
    """

    def __init__(self, drift_weight: float = 0.1):
        super().__init__()
        self.drift_weight = drift_weight
        self.previous_contexts: Dict[str, torch.Tensor] = {}

    def store_contexts(self, context_module: NestedContextModule) -> None:
        """Store current context values as reference."""
        self.previous_contexts['program'] = context_module.program_embeddings.embeddings.weight.data.clone()
        self.previous_contexts['assay'] = context_module.assay_embeddings.embeddings.weight.data.clone()
        self.previous_contexts['round'] = context_module.round_embeddings.embeddings.weight.data.clone()

    def compute_drift_loss(self, context_module: NestedContextModule) -> torch.Tensor:
        """Compute L2 drift penalty from stored contexts."""
        if not self.previous_contexts:
            return torch.tensor(0.0)

        loss = 0.0

        # Program drift
        if 'program' in self.previous_contexts:
            current = context_module.program_embeddings.embeddings.weight
            previous = self.previous_contexts['program'][:current.shape[0]]
            loss += F.mse_loss(current[:previous.shape[0]], previous)

        # Assay drift
        if 'assay' in self.previous_contexts:
            current = context_module.assay_embeddings.embeddings.weight
            previous = self.previous_contexts['assay'][:current.shape[0]]
            loss += F.mse_loss(current[:previous.shape[0]], previous)

        # Round drift (typically only penalize recent rounds)
        if 'round' in self.previous_contexts:
            current = context_module.round_embeddings.embeddings.weight
            previous = self.previous_contexts['round'][:current.shape[0]]
            loss += F.mse_loss(current[:previous.shape[0]], previous)

        return self.drift_weight * loss


if __name__ == '__main__':
    # Test context modules
    print("Testing Nested Context Module...")

    batch_size = 8
    feature_dim = 512

    # Create module
    context_module = NestedContextModule(
        program_dim=128,
        assay_dim=64,
        round_dim=32,
        num_programs=5,
        num_assays=50,
        num_rounds=100,
        feature_dim=feature_dim,
    )

    # Create dummy inputs
    h_mol = torch.randn(batch_size, feature_dim)
    program_ids = torch.randint(0, 5, (batch_size,))
    assay_ids = torch.randint(0, 50, (batch_size,))
    round_ids = torch.randint(0, 100, (batch_size,))

    # Forward pass
    h_mod = context_module(h_mol, program_ids, assay_ids, round_ids)

    print(f"  Input h_mol: {h_mol.shape}")
    print(f"  Output h_mod: {h_mod.shape}")
    print(f"  Context dim: {context_module.total_context_dim}")
    print(f"  Parameters: {sum(p.numel() for p in context_module.parameters()):,}")

    # Test dynamic context addition
    print("\nTesting dynamic context addition...")
    context_module.add_program(2)
    context_module.add_round(5, clone_from=99)
    print(f"  Programs: {context_module.program_embeddings.num_contexts}")
    print(f"  Rounds: {context_module.round_embeddings.num_contexts}")
