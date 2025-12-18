"""
Base storage interfaces for Temporal GraphRAG.

This module defines the abstract base classes for all storage backends:
- BaseVectorStorage: Vector database storage
- BaseKVStorage: Key-value storage
- BaseGraphStorage: Graph database storage
- StorageNameSpace: Base namespace for all storage types
"""

from dataclasses import dataclass, field
from typing import Generic, TypeVar, Union, TYPE_CHECKING
import numpy as np

# Forward references for types
if TYPE_CHECKING:
    from ..core.types import SingleCommunitySchema, SingleTemporalSchema
    from ..utils.types import EmbeddingFunc
else:
    # Runtime type stubs
    from typing import Any
    EmbeddingFunc = Any
    SingleCommunitySchema = Any
    SingleTemporalSchema = Any


T = TypeVar("T")


@dataclass
class StorageNameSpace:
    """Base namespace class for all storage backends.
    
    Provides a common interface for storage initialization and lifecycle callbacks.
    All storage implementations should inherit from this class.
    
    Attributes:
        namespace: Unique namespace identifier for this storage instance
        global_config: Global configuration dictionary passed to storage
    """
    namespace: str
    global_config: dict

    async def index_start_callback(self):
        """Called at the start of indexing operations.
        
        Use this to initialize transactions, prepare storage, etc.
        """
        pass

    async def index_done_callback(self):
        """Called when indexing operations are complete.
        
        Use this to commit transactions, finalize writes, etc.
        """
        pass

    async def query_done_callback(self):
        """Called when querying operations are complete.
        
        Use this to commit any query-related changes, close connections, etc.
        """
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    """Abstract base class for vector database storage.
    
    Vector storage is used for storing and querying embeddings of entities,
    relations, and text chunks.
    
    Attributes:
        embedding_func: Function to generate embeddings for content
        meta_fields: Set of metadata field names to store alongside embeddings
        entity_only: If True, only store entity embeddings (no relation embeddings)
    """
    embedding_func: 'EmbeddingFunc'
    meta_fields: set = field(default_factory=set)
    entity_only: bool = False

    async def query(self, query: str, top_k: int) -> list[dict]:
        """Query the vector database for similar embeddings."""
        raise NotImplementedError

    async def temporal_query(self, query: str, sub_graph_entities: list[str], top_k: int) -> list[dict]:
        """Query with temporal filtering based on sub-graph entities."""
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Upsert embeddings into the vector database.
        
        Uses 'content' field from value for embedding, uses key as id.
        If embedding_func is None, uses 'embedding' field from value.
        
        Args:
            data: Dictionary mapping IDs to data dictionaries containing:
                - 'content': Text content to embed (if embedding_func provided)
                - 'embedding': Pre-computed embedding (if embedding_func is None)
                - Additional metadata fields as specified in meta_fields
                
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    """Abstract base class for key-value storage.
    
    Key-value storage is used for storing documents, chunks, community reports,
    and LLM response cache.
    
    Type Parameters:
        T: Type of values stored in this key-value store
        
    Attributes:
        namespace: Unique namespace identifier
        global_config: Global configuration dictionary
    """
    
    async def all_keys(self) -> list[str]:
        """Get all keys in the storage.
        
        Returns:
            List of all keys currently stored
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        """Get a value by its ID.
        
        Args:
            id: Key to retrieve
            
        Returns:
            The value associated with the key, or None if not found
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        """Get multiple values by their IDs."""
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """Filter out keys that already exist in storage."""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        """Upsert key-value pairs into storage."""
        raise NotImplementedError

    async def drop(self):
        """Drop all data from storage."""
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    """Abstract base class for graph database storage."""
    
    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph.
        
        Args:
            node_id: Node identifier to check
            
        Returns:
            True if node exists, False otherwise
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes.
        
        Args:
            source_node_id: Source node identifier
            target_node_id: Target node identifier
            
        Returns:
            True if edge exists, False otherwise
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of connections) of a node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Number of edges connected to this node
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the number of edges between two nodes.
        
        Args:
            src_id: Source node identifier
            tgt_id: Target node identifier
            
        Returns:
            Number of edges between the two nodes
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        """Get node data by ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Dictionary containing node data, or None if not found
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        """Get edge data between two nodes.
        
        Args:
            source_node_id: Source node identifier
            target_node_id: Target node identifier
            
        Returns:
            Dictionary containing edge data, or None if not found
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        """Get all edges connected to a node.
        
        Args:
            source_node_id: Node identifier
            
        Returns:
            List of (target_node_id, edge_data) tuples, or None if node not found
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    async def get_temporal_edges(
        self, source_node_id: str, timestamps: list[str], top_k: int = 20
    ) -> list[dict]:
        """Get temporal edges for a given node and timestamps.
        
        Args:
            source_node_id: The source node ID
            timestamps: List of timestamps to filter by
            top_k: Maximum number of edges to return
            
        Returns:
            List of edge dictionaries with temporal information
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        """Upsert a node into the graph.
        
        Args:
            node_id: Node identifier
            node_data: Dictionary of node attributes
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        """Upsert an edge into the graph.
        
        Args:
            source_node_id: Source node identifier
            target_node_id: Target node identifier
            edge_data: Dictionary of edge attributes
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    async def clustering(self, algorithm: str):
        """Perform graph clustering using the specified algorithm.
        
        Args:
            algorithm: Clustering algorithm name (e.g., 'leiden', 'louvain')
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    async def community_schema(self) -> 'dict[str, SingleCommunitySchema]':
        """Return the community representation with report and nodes.
        
        Returns:
            Dictionary mapping community IDs to SingleCommunitySchema objects
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    async def temporal_hierarchy(self, entity_relation_graph_inst: 'BaseGraphStorage') -> 'dict[str, SingleTemporalSchema]':
        """Return the temporal hierarchy representation with temporal edges and nodes.
        
        Args:
            entity_relation_graph_inst: The entity-relation graph instance
            
        Returns:
            Dictionary mapping temporal hierarchy node IDs to SingleTemporalSchema objects
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

