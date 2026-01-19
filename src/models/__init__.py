"""
NEST-DRUG Model Architecture

Hierarchical prediction framework with:
- MPNN backbone for molecular encoding
- Nested context modules (L0-L3) for hierarchical adaptation
- FiLM conditioning for context modulation
- Multi-task prediction heads for potency and ADMET
- Deep ensemble for uncertainty quantification
"""

from .mpnn import MPNN, MPNNLayer
from .context import (
    ContextEmbedding,
    NestedContextModule,
    FiLMLayer,
)
from .heads import (
    PredictionHead,
    MultiTaskHead,
)
from .ensemble import DeepEnsemble
from .nest_drug import NESTDRUG

__all__ = [
    'MPNN',
    'MPNNLayer',
    'ContextEmbedding',
    'NestedContextModule',
    'FiLMLayer',
    'PredictionHead',
    'MultiTaskHead',
    'DeepEnsemble',
    'NESTDRUG',
]
