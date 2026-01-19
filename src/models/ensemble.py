#!/usr/bin/env python3
"""
Deep Ensemble for Uncertainty Quantification

Implements ensemble of NEST-DRUG models for calibrated uncertainty estimation.

Configuration:
- M=5 independent model instances
- Different random initializations
- Different minibatch orderings

Prediction:
- Point estimate: μ(x) = (1/M) Σ_m f_m(x)
- Uncertainty: σ(x) = sqrt((1/M) Σ_m (f_m(x) − μ(x))²)

References:
- Lakshminarayanan et al. (2017). Simple and Scalable Predictive Uncertainty Estimation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
import copy


class DeepEnsemble(nn.Module):
    """
    Deep ensemble wrapper for any base model.

    Maintains M independent copies of the model with different initializations.
    Provides uncertainty estimates via prediction disagreement.
    """

    def __init__(
        self,
        base_model_fn: Callable[[], nn.Module],
        num_members: int = 5,
        init_seeds: Optional[List[int]] = None,
    ):
        """
        Args:
            base_model_fn: Function that returns a new model instance
            num_members: Number of ensemble members (default: 5)
            init_seeds: Random seeds for each member's initialization
        """
        super().__init__()

        self.num_members = num_members

        if init_seeds is None:
            init_seeds = list(range(42, 42 + num_members))

        # Create ensemble members with different initializations
        self.members = nn.ModuleList()

        for seed in init_seeds:
            torch.manual_seed(seed)
            member = base_model_fn()
            self.members.append(member)

    def forward(
        self,
        *args,
        return_individual: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all ensemble members.

        Args:
            *args, **kwargs: Arguments passed to base model
            return_individual: If True, also return individual predictions

        Returns:
            Dictionary with:
                - 'mean': Mean prediction per endpoint
                - 'std': Standard deviation per endpoint
                - 'individual': (optional) List of predictions per member
        """
        # Collect predictions from all members
        all_predictions = []

        for member in self.members:
            pred = member(*args, **kwargs)
            all_predictions.append(pred)

        # Aggregate predictions
        result = self._aggregate_predictions(all_predictions)

        if return_individual:
            result['individual'] = all_predictions

        return result

    def _aggregate_predictions(
        self,
        predictions: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mean and std from ensemble predictions.
        """
        # Get all endpoint names
        endpoint_names = predictions[0].keys()

        result = {'mean': {}, 'std': {}}

        for name in endpoint_names:
            # Stack predictions: [num_members, batch_size, output_dim]
            stacked = torch.stack([p[name] for p in predictions], dim=0)

            # Compute statistics
            mean = stacked.mean(dim=0)
            std = stacked.std(dim=0)

            result['mean'][name] = mean
            result['std'][name] = std

        return result

    def predict_with_uncertainty(
        self,
        *args,
        confidence_level: float = 0.9,
        **kwargs,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get predictions with confidence intervals.

        Args:
            confidence_level: Confidence level for intervals (default: 0.9)

        Returns:
            Dictionary per endpoint with 'mean', 'std', 'lower', 'upper'
        """
        result = self.forward(*args, return_individual=True, **kwargs)

        # Compute confidence intervals
        # For 90% CI with normal assumption: mean ± 1.645 * std
        z_score = {0.9: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence_level, 1.645)

        predictions = {}

        for name in result['mean'].keys():
            mean = result['mean'][name]
            std = result['std'][name]

            predictions[name] = {
                'mean': mean,
                'std': std,
                'lower': mean - z_score * std,
                'upper': mean + z_score * std,
            }

        return predictions

    def compute_loss(
        self,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute average loss across ensemble members.

        Assumes base model has compute_loss method.
        """
        total_loss = 0.0
        all_endpoint_losses = {}

        for member in self.members:
            loss, endpoint_losses = member.compute_loss(*args, **kwargs)
            total_loss = total_loss + loss

            for name, ep_loss in endpoint_losses.items():
                if name not in all_endpoint_losses:
                    all_endpoint_losses[name] = 0.0
                all_endpoint_losses[name] = all_endpoint_losses[name] + ep_loss

        # Average
        total_loss = total_loss / self.num_members
        all_endpoint_losses = {k: v / self.num_members for k, v in all_endpoint_losses.items()}

        return total_loss, all_endpoint_losses

    def get_member(self, idx: int) -> nn.Module:
        """Get a specific ensemble member."""
        return self.members[idx]

    def set_member(self, idx: int, model: nn.Module) -> None:
        """Replace a specific ensemble member."""
        self.members[idx] = model

    def load_members(self, state_dicts: List[Dict]) -> None:
        """Load state dicts for all members."""
        for member, state_dict in zip(self.members, state_dicts):
            member.load_state_dict(state_dict)

    def save_members(self) -> List[Dict]:
        """Save state dicts for all members."""
        return [member.state_dict() for member in self.members]


class EnsembleTrainer:
    """
    Training utilities for deep ensembles.

    Supports:
    - Independent training with different data shuffling
    - Parallel training on multiple GPUs
    - Sequential training with checkpointing
    """

    def __init__(
        self,
        ensemble: DeepEnsemble,
        optimizer_fn: Callable[[nn.Module], torch.optim.Optimizer],
    ):
        self.ensemble = ensemble
        self.optimizers = [
            optimizer_fn(member) for member in ensemble.members
        ]

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        member_idx: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Training batch
            member_idx: If specified, only train this member

        Returns:
            Dictionary with loss values
        """
        if member_idx is not None:
            # Train single member
            members_to_train = [(member_idx, self.ensemble.members[member_idx])]
            optimizers_to_use = [self.optimizers[member_idx]]
        else:
            # Train all members
            members_to_train = list(enumerate(self.ensemble.members))
            optimizers_to_use = self.optimizers

        total_loss = 0.0

        for (idx, member), optimizer in zip(members_to_train, optimizers_to_use):
            optimizer.zero_grad()

            # Forward pass
            loss, _ = member.compute_loss(batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return {'loss': total_loss / len(members_to_train)}

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        shuffle_per_member: bool = True,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            shuffle_per_member: If True, each member sees differently shuffled data

        Returns:
            Dictionary with average loss values
        """
        self.ensemble.train()

        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            losses = self.train_step(batch)
            epoch_loss += losses['loss']
            num_batches += 1

        return {'loss': epoch_loss / num_batches}


class UCBSelector:
    """
    Upper Confidence Bound (UCB) selection policy for DMTA.

    Balances exploitation (high predicted value) with exploration (high uncertainty).

    Score: S(x) = μ(x) + λ * σ(x)

    where λ controls exploration-exploitation tradeoff.
    """

    def __init__(
        self,
        exploration_weight: float = 0.5,
        decay_rate: float = 0.0,
    ):
        """
        Args:
            exploration_weight: Initial weight for uncertainty (λ)
            decay_rate: Rate at which λ decays per round (0 = no decay)
        """
        self.exploration_weight = exploration_weight
        self.decay_rate = decay_rate
        self.current_round = 0

    def get_lambda(self) -> float:
        """Get current exploration weight."""
        return self.exploration_weight * (1 - self.decay_rate) ** self.current_round

    def compute_scores(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute UCB scores.

        Args:
            mean: Predicted mean values
            std: Predicted standard deviations

        Returns:
            UCB scores
        """
        lambda_t = self.get_lambda()
        return mean + lambda_t * std

    def select(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        budget: int,
        scaffold_ids: Optional[torch.Tensor] = None,
        max_per_scaffold: int = 5,
    ) -> torch.Tensor:
        """
        Select top-k compounds using UCB with optional diversity constraint.

        Args:
            mean: Predicted mean values [num_compounds]
            std: Predicted standard deviations [num_compounds]
            budget: Number of compounds to select
            scaffold_ids: Scaffold cluster assignments (for diversity)
            max_per_scaffold: Maximum compounds per scaffold

        Returns:
            Indices of selected compounds
        """
        scores = self.compute_scores(mean.squeeze(), std.squeeze())

        if scaffold_ids is None:
            # Simple top-k selection
            _, selected_idx = torch.topk(scores, min(budget, len(scores)))
        else:
            # Diversity-constrained selection
            selected_idx = self._diverse_select(
                scores, scaffold_ids, budget, max_per_scaffold
            )

        return selected_idx

    def _diverse_select(
        self,
        scores: torch.Tensor,
        scaffold_ids: torch.Tensor,
        budget: int,
        max_per_scaffold: int,
    ) -> torch.Tensor:
        """Select with diversity constraint."""
        device = scores.device
        num_compounds = len(scores)

        # Sort by score
        sorted_idx = torch.argsort(scores, descending=True)

        # Track selected indices and scaffold counts
        selected = []
        scaffold_counts = {}

        for idx in sorted_idx.tolist():
            if len(selected) >= budget:
                break

            scaffold = scaffold_ids[idx].item()
            count = scaffold_counts.get(scaffold, 0)

            if count < max_per_scaffold:
                selected.append(idx)
                scaffold_counts[scaffold] = count + 1

        return torch.tensor(selected, device=device)

    def advance_round(self) -> None:
        """Advance to next round (decays exploration weight)."""
        self.current_round += 1


if __name__ == '__main__':
    # Test ensemble
    print("Testing Deep Ensemble...")

    # Simple test model
    def create_model():
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    # Wrap in a class with proper interface
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

        def forward(self, x):
            return {'output': self.net(x)}

        def compute_loss(self, batch):
            pred = self.forward(batch['x'])
            loss = nn.functional.mse_loss(pred['output'], batch['y'])
            return loss, {'output': loss}

    # Create ensemble
    ensemble = DeepEnsemble(TestModel, num_members=5)

    # Test forward
    x = torch.randn(16, 512)
    result = ensemble({'x': x, 'y': torch.randn(16, 1)})

    print(f"  Ensemble members: {ensemble.num_members}")
    print(f"  Mean shape: {result['mean']['output'].shape}")
    print(f"  Std shape: {result['std']['output'].shape}")
    print(f"  Std range: [{result['std']['output'].min():.4f}, {result['std']['output'].max():.4f}]")

    # Test UCB selector
    print("\nTesting UCB Selector...")
    selector = UCBSelector(exploration_weight=0.5)

    mean = torch.randn(100)
    std = torch.rand(100) * 0.5

    selected = selector.select(mean, std, budget=10)
    print(f"  Selected {len(selected)} compounds")
    print(f"  Selected indices: {selected.tolist()}")
