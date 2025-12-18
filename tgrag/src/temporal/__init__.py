"""Temporal normalization, decomposition, and time-aware operations."""

# Import temporal operations (Phase 4)
from . import operations

# Import centralized normalizer (Phase 3)
from .normalization import TemporalNormalizer, get_temporal_normalizer

# Import enhanced normalizer classes (migrated from OLD)
from .normalizer import (
    EnhancedTemporalNormalizer,
    TemporalGranularity,
    TemporalRange,
    TemporalNormalizationResult,
)

__all__ = [
    "operations",
    "TemporalNormalizer",
    "get_temporal_normalizer",
    "EnhancedTemporalNormalizer",
    "TemporalGranularity",
    "TemporalRange",
    "TemporalNormalizationResult",
    # TODO: Export QueryDecomposer once moved in Phase 5
]


