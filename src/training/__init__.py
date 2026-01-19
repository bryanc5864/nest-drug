"""
NEST-DRUG Training Module

Implements the three-phase training protocol:
- Phase 1: Global pretraining on portfolio data (L0 backbone)
- Phase 2: Program-specific initialization (seed window)
- Phase 3: Continual nested updates during DMTA replay
"""

from .trainer import (
    NESTDRUGTrainer,
    PretrainingConfig,
    ProgramConfig,
    ContinualConfig,
)

from .data_utils import (
    MoleculeDataset,
    PortfolioDataLoader,
    ProgramDataLoader,
    collate_molecules,
)

from .schedulers import (
    WarmupCosineScheduler,
    MultiTimescaleScheduler,
)

__all__ = [
    'NESTDRUGTrainer',
    'PretrainingConfig',
    'ProgramConfig',
    'ContinualConfig',
    'MoleculeDataset',
    'PortfolioDataLoader',
    'ProgramDataLoader',
    'collate_molecules',
    'WarmupCosineScheduler',
    'MultiTimescaleScheduler',
]
