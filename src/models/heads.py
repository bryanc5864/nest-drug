#!/usr/bin/env python3
"""
Multi-Task Prediction Heads for NEST-DRUG

Implements prediction heads for potency and ADMET endpoints:
- Regression heads: potency (pKi/pIC50), solubility, LogD, clearance
- Classification heads: hERG, AMES, DILI, BBB

Architecture per head: 512 → 256 → 128 → 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class PredictionHead(nn.Module):
    """
    Single prediction head for one endpoint.

    Architecture: input_dim → 256 → 128 → 1
    Supports both regression and classification tasks.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 1,
        task_type: str = 'regression',  # 'regression' or 'classification'
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Output layer (no activation for regression, sigmoid for classification)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Predictions [batch_size, output_dim]
        """
        logits = self.mlp(x)

        if self.task_type == 'classification':
            return torch.sigmoid(logits)
        else:
            return logits


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction module with separate heads for each endpoint.

    Supports:
    - Multiple regression endpoints (potency, solubility, LogD, clearance)
    - Multiple classification endpoints (hERG, AMES, DILI, BBB)
    - Masked loss computation for missing labels
    - Uncertainty estimation via ensemble disagreement
    """

    # Default endpoint configurations
    DEFAULT_ENDPOINTS = {
        # Regression endpoints
        'pActivity': {'type': 'regression', 'weight': 3.0},
        'solubility': {'type': 'regression', 'weight': 1.0},
        'lipophilicity': {'type': 'regression', 'weight': 1.0},
        'clearance_hepatocyte': {'type': 'regression', 'weight': 1.0},
        'clearance_microsome': {'type': 'regression', 'weight': 1.0},
        'caco2': {'type': 'regression', 'weight': 1.0},
        'ppbr': {'type': 'regression', 'weight': 1.0},
        # Classification endpoints
        'herg': {'type': 'classification', 'weight': 1.5},
        'ames': {'type': 'classification', 'weight': 1.0},
        'bbb': {'type': 'classification', 'weight': 1.0},
    }

    def __init__(
        self,
        input_dim: int = 512,
        endpoints: Optional[Dict[str, Dict]] = None,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim

        if endpoints is None:
            endpoints = self.DEFAULT_ENDPOINTS

        self.endpoints = endpoints
        self.endpoint_names = list(endpoints.keys())

        # Create head for each endpoint
        self.heads = nn.ModuleDict()
        self.task_types = {}
        self.task_weights = {}

        for name, config in endpoints.items():
            task_type = config.get('type', 'regression')
            weight = config.get('weight', 1.0)

            self.heads[name] = PredictionHead(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=1,
                task_type=task_type,
                dropout=dropout,
            )

            self.task_types[name] = task_type
            self.task_weights[name] = weight

    def forward(
        self,
        x: torch.Tensor,
        endpoints: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict all endpoints.

        Args:
            x: Input features [batch_size, input_dim]
            endpoints: Optional list of endpoints to predict (default: all)

        Returns:
            Dictionary mapping endpoint names to predictions
        """
        if endpoints is None:
            endpoints = self.endpoint_names

        predictions = {}
        for name in endpoints:
            if name in self.heads:
                predictions[name] = self.heads[name](x)

        return predictions

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted multi-task loss with optional masking.

        Args:
            predictions: Predicted values per endpoint
            targets: Ground truth values per endpoint
            masks: Binary masks indicating valid labels (1=valid, 0=missing)

        Returns:
            total_loss: Weighted sum of endpoint losses (always a tensor)
            endpoint_losses: Individual losses per endpoint
        """
        endpoint_losses = {}
        total_loss = None  # Will be tensor after first loss
        device = None

        for name in predictions.keys():
            if name not in targets:
                continue

            pred = predictions[name]
            target = targets[name]
            device = pred.device

            # Get mask (default to all valid)
            if masks is not None and name in masks:
                mask = masks[name]
            else:
                mask = torch.ones_like(target)

            # Compute loss based on task type
            if self.task_types[name] == 'regression':
                loss = F.mse_loss(pred, target, reduction='none')
            else:  # classification
                loss = F.binary_cross_entropy(pred, target, reduction='none')

            # Apply mask
            masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)

            # Weight and accumulate
            weight = self.task_weights.get(name, 1.0)
            endpoint_losses[name] = masked_loss

            if total_loss is None:
                total_loss = weight * masked_loss
            else:
                total_loss = total_loss + weight * masked_loss

        # Ensure total_loss is always a tensor (even if no matching endpoints)
        if total_loss is None:
            if device is None:
                # Get device from any prediction
                for pred in predictions.values():
                    device = pred.device
                    break
            if device is None:
                device = torch.device('cpu')
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss, endpoint_losses

    def compute_loss_with_censoring(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        censoring: Optional[Dict[str, torch.Tensor]] = None,
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss with censored data handling.

        Censoring codes:
            0: Exact measurement (normal loss)
            1: Left-censored "<X" (loss only if pred > X)
            -1: Right-censored ">X" (loss only if pred < X)

        Args:
            predictions: Predicted values
            targets: Target values (threshold for censored)
            censoring: Censoring codes per endpoint
            masks: Valid label masks
        """
        endpoint_losses = {}
        total_loss = 0.0

        for name in predictions.keys():
            if name not in targets:
                continue

            pred = predictions[name]
            target = targets[name]

            # Get censoring (default to exact)
            if censoring is not None and name in censoring:
                censor = censoring[name]
            else:
                censor = torch.zeros_like(target)

            # Get mask
            if masks is not None and name in masks:
                mask = masks[name]
            else:
                mask = torch.ones_like(target)

            # Only for regression tasks
            if self.task_types[name] == 'regression':
                # Compute base loss
                loss = F.mse_loss(pred, target, reduction='none')

                # Apply censoring logic
                # Left-censored: only penalize if pred > target
                left_mask = (censor == 1) & (pred <= target)
                loss = loss * (~left_mask).float()

                # Right-censored: only penalize if pred < target
                right_mask = (censor == -1) & (pred >= target)
                loss = loss * (~right_mask).float()
            else:
                loss = F.binary_cross_entropy(pred, target, reduction='none')

            # Apply mask
            masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)

            weight = self.task_weights.get(name, 1.0)
            endpoint_losses[name] = masked_loss
            total_loss = total_loss + weight * masked_loss

        return total_loss, endpoint_losses


class DScoreHead(nn.Module):
    """
    Compute D-score (desirability score) from endpoint predictions.

    D-score aggregates multiple endpoints into a single quality metric
    using weighted geometric mean of desirability functions.

    D(x) = ∏_i d_i(x)^{w_i}

    where d_i ∈ [0, 1] is the desirability for endpoint i.
    """

    def __init__(
        self,
        endpoint_configs: Optional[Dict] = None,
    ):
        super().__init__()

        # Default desirability configurations
        if endpoint_configs is None:
            endpoint_configs = {
                'pActivity': {
                    'type': 'higher_is_better',
                    'lower_bound': 5.0,
                    'target': 7.0,
                    'weight': 3.0,
                },
                'solubility': {
                    'type': 'higher_is_better',
                    'lower_bound': 10,  # µM
                    'target': 100,
                    'weight': 1.0,
                },
                'lipophilicity': {
                    'type': 'range_optimal',
                    'lower': 1.0,
                    'upper': 3.0,
                    'margin': 1.0,
                    'weight': 1.0,
                },
                'clearance_hepatocyte': {
                    'type': 'lower_is_better',
                    'target': 10,  # mL/min/kg
                    'upper_bound': 50,
                    'weight': 1.0,
                },
                'herg': {
                    'type': 'hard_gate',
                    'threshold': 0.5,  # Must be < 0.5 (not a blocker)
                    'weight': 1.5,
                },
            }

        self.endpoint_configs = endpoint_configs

    def compute_desirability(
        self,
        value: torch.Tensor,
        config: Dict,
    ) -> torch.Tensor:
        """
        Compute desirability d ∈ [0, 1] for a single endpoint.
        """
        d_type = config['type']

        if d_type == 'higher_is_better':
            lower = config['lower_bound']
            target = config['target']

            d = torch.clamp((value - lower) / (target - lower + 1e-8), 0, 1)

        elif d_type == 'lower_is_better':
            target = config['target']
            upper = config['upper_bound']

            d = torch.clamp((upper - value) / (upper - target + 1e-8), 0, 1)

        elif d_type == 'range_optimal':
            lower = config['lower']
            upper = config['upper']
            margin = config.get('margin', 0.5)

            # 1 inside range, linear decay outside
            in_range = (value >= lower) & (value <= upper)
            below = value < lower
            above = value > upper

            d = torch.zeros_like(value)
            d[in_range] = 1.0
            d[below] = torch.clamp((value[below] - (lower - margin)) / margin, 0, 1)
            d[above] = torch.clamp(((upper + margin) - value[above]) / margin, 0, 1)

        elif d_type == 'hard_gate':
            threshold = config['threshold']
            d = (value < threshold).float()

        else:
            d = torch.ones_like(value)

        return d

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute D-score from endpoint predictions.

        Args:
            predictions: Predicted values per endpoint

        Returns:
            d_score: Overall desirability score [batch_size, 1]
            d_individual: Desirability per endpoint
        """
        d_individual = {}
        log_d_sum = 0.0
        total_weight = 0.0

        for name, config in self.endpoint_configs.items():
            if name not in predictions:
                continue

            value = predictions[name]
            d = self.compute_desirability(value, config)
            d_individual[name] = d

            weight = config.get('weight', 1.0)
            # Use log for numerical stability: log(∏d^w) = Σ w*log(d)
            log_d_sum = log_d_sum + weight * torch.log(d + 1e-8)
            total_weight += weight

        # Geometric mean: exp(Σ w*log(d) / Σ w)
        d_score = torch.exp(log_d_sum / (total_weight + 1e-8))

        return d_score, d_individual

    def check_hard_gates(
        self,
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Check if all hard gate constraints are satisfied.

        Returns:
            Boolean mask [batch_size] where True = all gates pass
        """
        batch_size = None
        all_pass = None

        for name, config in self.endpoint_configs.items():
            if config['type'] != 'hard_gate':
                continue
            if name not in predictions:
                continue

            value = predictions[name]
            if batch_size is None:
                batch_size = value.shape[0]
                all_pass = torch.ones(batch_size, dtype=torch.bool, device=value.device)

            threshold = config['threshold']
            passes = (value.squeeze() < threshold)
            all_pass = all_pass & passes

        if all_pass is None:
            return torch.ones(1, dtype=torch.bool)

        return all_pass


if __name__ == '__main__':
    # Test prediction heads
    print("Testing Multi-Task Heads...")

    batch_size = 16
    input_dim = 512

    # Create module
    mtl_head = MultiTaskHead(input_dim=input_dim)

    # Forward pass
    x = torch.randn(batch_size, input_dim)
    predictions = mtl_head(x)

    print(f"  Input: {x.shape}")
    print(f"  Endpoints: {list(predictions.keys())}")
    for name, pred in predictions.items():
        print(f"    {name}: {pred.shape}")

    # Test loss computation
    targets = {name: torch.randn(batch_size, 1) for name in predictions.keys()}
    targets['herg'] = torch.sigmoid(targets['herg'])  # Binary
    targets['ames'] = torch.sigmoid(targets['ames'])
    targets['bbb'] = torch.sigmoid(targets['bbb'])

    loss, endpoint_losses = mtl_head.compute_loss(predictions, targets)
    print(f"\n  Total loss: {loss.item():.4f}")

    # Test D-score
    print("\nTesting D-Score computation...")
    dscore_head = DScoreHead()
    d_score, d_individual = dscore_head(predictions)
    print(f"  D-score shape: {d_score.shape}")
    print(f"  D-score range: [{d_score.min():.3f}, {d_score.max():.3f}]")
