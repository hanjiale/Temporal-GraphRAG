"""Storage backends for graph and vector databases."""

from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
)

# Graph storage implementations
from .graph_networkx import NetworkXStorage

# Neo4j storage - optional dependency
try:
    from .graph_neo4j import Neo4jStorage
except ImportError:
    Neo4jStorage = None  # type: ignore

# Vector storage implementations
from .vector_hnswlib import HNSWVectorStorage
from .vector_nanovectordb import NanoVectorDBStorage

# Key-value storage implementations
from .kv_json import JsonKVStorage

__all__ = [
    # Base classes
    "BaseGraphStorage",
    "BaseKVStorage",
    "BaseVectorStorage",
    "StorageNameSpace",
    # Graph storage
    "NetworkXStorage",
    # Vector storage
    "HNSWVectorStorage",
    "NanoVectorDBStorage",
    # Key-value storage
    "JsonKVStorage",
]

# Conditionally add Neo4jStorage if available
if Neo4jStorage is not None:
    __all__.append("Neo4jStorage")


