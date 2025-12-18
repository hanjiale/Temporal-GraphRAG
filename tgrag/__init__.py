"""Temporal-GraphRAG: Time-Sensitive Modeling and Retrieval for Evolving Knowledge."""

# Package version
__version__ = "0.1.0"

# Main exports
from .src.temporal_graphrag import TemporalGraphRAG
from .src.core import QueryParam
from .src.build import create_temporal_graphrag_from_config

__all__ = [
    "__version__",
    "TemporalGraphRAG",
    "QueryParam",
    "create_temporal_graphrag_from_config",
]


