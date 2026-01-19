"""
NEST-DRUG Evaluation Module

Provides:
- DMTA replay evaluation engine
- Metrics for temporal performance assessment
- Compound selection strategies
"""

from .dmta_replay import (
    DMTAReplayEngine,
    ReplayConfig,
    RoundResult,
)

from .metrics import (
    compute_enrichment_factor,
    compute_auc,
    compute_hit_rate,
    compute_temporal_metrics,
)

__all__ = [
    'DMTAReplayEngine',
    'ReplayConfig',
    'RoundResult',
    'compute_enrichment_factor',
    'compute_auc',
    'compute_hit_rate',
    'compute_temporal_metrics',
]
