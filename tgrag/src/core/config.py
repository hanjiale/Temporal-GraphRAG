"""
Configuration classes for Temporal GraphRAG.

This module contains the refactored configuration structure, splitting the
large TemporalGraphRAG dataclass into focused sub-configuration classes.
"""

from dataclasses import dataclass, field
from typing import Callable, Type, Optional
import tiktoken

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..storage.base import BaseVectorStorage, BaseKVStorage, BaseGraphStorage
    from ..utils.types import EmbeddingFunc
else:
    from typing import Any
    BaseVectorStorage = Any
    BaseKVStorage = Any
    BaseGraphStorage = Any
    EmbeddingFunc = Any


@dataclass
class CoreConfig:
    """Core configuration settings for Temporal GraphRAG.
    
    Attributes:
        working_dir: Working directory for storing graph artifacts
        enable_local: Enable local query mode
        enable_naive_rag: Enable naive RAG mode
        always_create_working_dir: Automatically create working directory if it doesn't exist
    """
    working_dir: str = field(
        default_factory=lambda: f"./temporal_graphrag_cache_{__import__('datetime').datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    enable_local: bool = True
    enable_naive_rag: bool = False
    always_create_working_dir: bool = True


@dataclass
class ChunkingConfig:
    """Configuration for text chunking.
    
    Attributes:
        chunk_func: Function to perform chunking
        chunk_token_size: Maximum token size per chunk
        chunk_overlap_token_size: Overlap token size between consecutive chunks
        tiktoken_model_name: Model name for tiktoken encoding
    """
    chunk_func: Callable[
        [
            list[list[int]],
            list[str],
            tiktoken.Encoding,
            Optional[int],
            Optional[int],
        ],
        list[dict[str, str | int]],
    ] = None  # Will be set to chunking_by_token_size in __post_init__
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"


@dataclass
class LLMConfig:
    """Configuration for LLM providers and models.
    
    Attributes:
        using_azure_openai: Use Azure OpenAI instead of OpenAI
        using_amazon_bedrock: Use Amazon Bedrock
        best_model_id: Model ID for best quality (Bedrock)
        cheap_model_id: Model ID for cheaper operations (Bedrock)
        best_model_func: Function for best quality model calls
        best_model_max_token_size: Maximum tokens for best model
        best_model_max_async: Maximum concurrent calls for best model
        cheap_model_func: Function for cheaper model calls
        cheap_model_max_token_size: Maximum tokens for cheap model
        cheap_model_max_async: Maximum concurrent calls for cheap model
        enable_llm_cache: Enable caching of LLM responses
    """
    using_azure_openai: bool = False
    using_amazon_bedrock: bool = False
    best_model_id: str = "us.anthropic.claude-3-sonnet-20240229-v1:0"
    cheap_model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0"
    best_model_func: Callable = None  # Will be set in __post_init__
    best_model_max_token_size: int = 65536
    best_model_max_async: int = 32
    cheap_model_func: Callable = None  # Will be set in __post_init__
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 32
    enable_llm_cache: bool = True


@dataclass
class StorageConfig:
    """Configuration for storage backends.
    
    Attributes:
        key_string_value_json_storage_cls: Class for key-value storage
        vector_db_storage_cls: Class for vector database storage
        vector_db_storage_cls_kwargs: Additional kwargs for vector storage
        graph_storage_cls: Class for graph database storage
    """
    key_string_value_json_storage_cls: Type[BaseKVStorage] = None  # Will be set in __post_init__
    vector_db_storage_cls: Type[BaseVectorStorage] = None  # Will be set in __post_init__
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = None  # Will be set in __post_init__


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings.
    
    Attributes:
        embedding_func: Function to generate embeddings
        embedding_batch_num: Batch size for embedding operations
        embedding_func_max_async: Maximum concurrent embedding calls
        query_better_than_threshold: Threshold for query quality
        enable_entity_retrieval: Enable entity-based retrieval
    """
    embedding_func: EmbeddingFunc = None  # Will be set in __post_init__
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    query_better_than_threshold: float = 0.2
    enable_entity_retrieval: bool = False


@dataclass
class EntityConfig:
    """Configuration for entity extraction.
    
    Attributes:
        entity_extract_max_gleaning: Maximum gleaning iterations for entity extraction
        entity_summary_to_max_tokens: Maximum tokens for entity summaries
        disable_entity_summarization: Disable entity summarization
        entity_extraction_func: Function to perform entity extraction
    """
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    disable_entity_summarization: bool = False
    entity_extraction_func: Callable = None  # Will be set in __post_init__


@dataclass
class GraphConfig:
    """Configuration for graph operations.
    
    Attributes:
        max_graph_cluster_size: Maximum size for graph clusters
        graph_cluster_seed: Random seed for graph clustering
        special_community_report_llm_kwargs: LLM kwargs for community reports
        enable_community_summary: Enable community summary generation
        building_temporal_hierarchy_func: Function to build temporal hierarchy
        enable_incremental: Enable incremental update mode
        preserve_communities: Preserve existing communities during incremental updates
    """
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    enable_community_summary: bool = True
    building_temporal_hierarchy_func: Callable = None  # Will be set in __post_init__
    enable_incremental: bool = False
    preserve_communities: bool = False


@dataclass
class ExtensionConfig:
    """Configuration for extensions and addons.
    
    Attributes:
        addon_params: Additional parameters for extensions
        convert_response_to_json_func: Function to convert LLM responses to JSON
    """
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: Callable = None  # Will be set in __post_init__

