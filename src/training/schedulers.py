#!/usr/bin/env python3
"""
Learning Rate Schedulers for NEST-DRUG Training

Implements:
- WarmupCosineScheduler: Warmup + cosine annealing
- MultiTimescaleScheduler: Different decay rates for backbone/context/heads
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Optional


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.

    Schedule:
    - Warmup: lr = base_lr * (step / warmup_steps)
    - Cosine: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(1.0, progress)

            return [
                self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class MultiTimescaleScheduler:
    """
    Scheduler for multi-timescale training with different learning rates
    for backbone, context, and head parameters.

    Supports:
    - Independent warmup/decay for each parameter group
    - Different base learning rates
    - Coordinated stepping
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        # Learning rate multipliers for each group
        backbone_lr_scale: float = 0.1,  # Slow updates
        context_lr_scale: float = 1.0,   # Normal updates
        head_lr_scale: float = 0.5,      # Moderate updates
        min_lr: float = 1e-7,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        # Store base learning rates and scales
        self.base_lrs = []
        self.lr_scales = []

        for group in optimizer.param_groups:
            self.base_lrs.append(group['lr'])

            # Determine scale based on group name
            name = group.get('name', '')
            if 'backbone' in name.lower():
                self.lr_scales.append(backbone_lr_scale)
            elif 'context' in name.lower() or name.startswith('L'):
                self.lr_scales.append(context_lr_scale)
            elif 'head' in name.lower():
                self.lr_scales.append(head_lr_scale)
            else:
                self.lr_scales.append(1.0)

        self.current_step = 0

    def step(self) -> None:
        """Update learning rates for all parameter groups."""
        self.current_step += 1

        for i, (group, base_lr, scale) in enumerate(
            zip(self.optimizer.param_groups, self.base_lrs, self.lr_scales)
        ):
            effective_base_lr = base_lr * scale

            if self.current_step < self.warmup_steps:
                # Linear warmup
                warmup_factor = self.current_step / self.warmup_steps
                lr = effective_base_lr * warmup_factor
            else:
                # Cosine annealing
                progress = (self.current_step - self.warmup_steps) / max(
                    1, self.total_steps - self.warmup_steps
                )
                progress = min(1.0, progress)
                lr = self.min_lr + 0.5 * (effective_base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress)
                )

            group['lr'] = lr

    def get_last_lr(self) -> List[float]:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        """Get scheduler state."""
        return {
            'current_step': self.current_step,
            'base_lrs': self.base_lrs,
            'lr_scales': self.lr_scales,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state."""
        self.current_step = state_dict['current_step']
        self.base_lrs = state_dict['base_lrs']
        self.lr_scales = state_dict['lr_scales']


class ContinualLearningScheduler:
    """
    Scheduler for continual learning during DMTA rounds.

    Features:
    - Reset warmup at start of each round
    - Gradual unfreezing of backbone layers
    - Adaptive learning rate based on data size
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float = 1e-4,
        warmup_ratio: float = 0.1,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_ratio = warmup_ratio
        self.min_lr = min_lr

        self.round_step = 0
        self.round_total_steps = 0
        self.warmup_steps = 0

    def start_round(self, num_steps: int, lr_scale: float = 1.0) -> None:
        """
        Start a new DMTA round.

        Args:
            num_steps: Total steps in this round
            lr_scale: Scale factor for learning rate (can decay over rounds)
        """
        self.round_step = 0
        self.round_total_steps = num_steps
        self.warmup_steps = int(num_steps * self.warmup_ratio)

        # Update base learning rate with scale
        effective_lr = self.base_lr * lr_scale
        for group in self.optimizer.param_groups:
            group['lr'] = effective_lr

    def step(self) -> None:
        """Update learning rate within current round."""
        self.round_step += 1

        if self.round_step < self.warmup_steps:
            # Warmup
            factor = self.round_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.round_step - self.warmup_steps) / max(
                1, self.round_total_steps - self.warmup_steps
            )
            factor = 0.5 * (1 + math.cos(math.pi * progress))

        for group in self.optimizer.param_groups:
            base = group.get('initial_lr', self.base_lr)
            group['lr'] = max(self.min_lr, base * factor)

    def get_last_lr(self) -> List[float]:
        return [group['lr'] for group in self.optimizer.param_groups]


if __name__ == '__main__':
    # Test schedulers
    print("Testing Learning Rate Schedulers...")

    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Test WarmupCosineScheduler
    print("\nWarmupCosineScheduler:")
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=100,
        total_steps=1000,
    )

    lrs = []
    for step in range(1000):
        scheduler.step()
        lrs.append(scheduler.get_lr()[0])

    print(f"  Warmup phase (step 50): {lrs[49]:.6f}")
    print(f"  Peak (step 100): {lrs[99]:.6f}")
    print(f"  Mid decay (step 500): {lrs[499]:.6f}")
    print(f"  End (step 999): {lrs[998]:.6f}")

    # Test MultiTimescaleScheduler
    print("\nMultiTimescaleScheduler:")

    # Create optimizer with named groups
    param_groups = [
        {'params': [torch.nn.Parameter(torch.randn(10))], 'lr': 1e-4, 'name': 'backbone'},
        {'params': [torch.nn.Parameter(torch.randn(10))], 'lr': 1e-3, 'name': 'context'},
        {'params': [torch.nn.Parameter(torch.randn(10))], 'lr': 5e-4, 'name': 'heads'},
    ]
    optimizer = torch.optim.Adam(param_groups)

    scheduler = MultiTimescaleScheduler(
        optimizer,
        warmup_steps=100,
        total_steps=1000,
        backbone_lr_scale=0.1,
        context_lr_scale=1.0,
        head_lr_scale=0.5,
    )

    for step in range(100):
        scheduler.step()

    lrs = scheduler.get_last_lr()
    print(f"  After warmup (step 100):")
    print(f"    Backbone: {lrs[0]:.6f}")
    print(f"    Context: {lrs[1]:.6f}")
    print(f"    Heads: {lrs[2]:.6f}")

    print("\nScheduler tests complete!")
