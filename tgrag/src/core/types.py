"""
Core type definitions and schemas for Temporal GraphRAG.

This module defines:
- QueryParam: Query parameters dataclass
- TemporalQuadruple: Temporal quadruple data structure (v₁, v₂, e, τ) from the paper
- TypedDict schemas for data structures (TextChunkSchema, CommunitySchema, etc.)
"""

from dataclasses import dataclass, field
from typing import TypedDict, Literal, Optional


@dataclass
class TemporalQuadruple:
    """Represents a temporal quadruple (v₁, v₂, e, τ) as described in the paper.
    
    A temporal quadruple captures a relationship between two entities at a specific time:
    - v₁: First entity (source, non-temporal)
    - v₂: Second entity (target, non-temporal)
    - e: Relation/edge description
    - τ (tau): Normalized timestamp when this relationship is active
    
    Attributes:
        v1: Source entity name
        v2: Target entity name
        e: Relation description/text
        tau: Normalized timestamp (e.g., "2024-Q1", "2024-01-15")
        source_id: Chunk ID where this quadruple was extracted from
        raw_timestamp: Original timestamp string before normalization (optional)
        
    Example:
        >>> quad = TemporalQuadruple(
        ...     v1="Apple Inc",
        ...     v2="iPhone",
        ...     e="launched",
        ...     tau="2024-Q1",
        ...     source_id="chunk-123",
        ...     raw_timestamp="Q1 2024"
        ... )
    """
    v1: str  # Source entity
    v2: str  # Target entity
    e: str   # Relation description
    tau: str  # Normalized timestamp
    source_id: str  # Chunk ID where extracted
    raw_timestamp: Optional[str] = None  # Original timestamp before normalization


@dataclass
class QueryParam:
    """Query parameters for Temporal GraphRAG queries.
    
    Attributes:
        mode: Query mode - "local", "global", or "naive"
        only_need_context: If True, return only context without LLM response
        response_type: Type of response format (e.g., "Multiple Paragraphs")
        level: Hierarchy level for querying
        top_k: Number of top results to retrieve
        temporal_granularity: Temporal granularity filter (None for auto)
        seed_node_method: Method for seed node retrieval - "entities" or "relations"
        
        # Naive search parameters
        naive_max_token_for_text_unit: Maximum tokens for naive RAG text units
        
        # Local search parameters
        local_max_token_for_text_unit: Maximum tokens for local text units
        local_max_token_for_local_context: Maximum tokens for local context
        local_max_token_for_community_report: Maximum tokens for community reports
        local_community_single_one: If True, use single community for local queries
        sub_graph: If True, enable subgraph retrieval
        mix_relation: If True, enable mixed relationship retrieval
        
        # Global search parameters
        global_min_community_rating: Minimum community rating threshold
        global_max_consider_community: Maximum number of communities to consider
        global_max_token_for_community_report: Maximum tokens for community reports
        global_special_community_map_llm_kwargs: LLM kwargs for special community mapping
        
        retrieval_details: Optional dictionary for storing retrieval details
    """
    mode: Literal["local", "global", "naive"] = "global"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    level: int = 2
    top_k: int = 20
    temporal_granularity: None = None
    seed_node_method: Literal["entities", "relations"] = "entities"
    
    # Naive search parameters
    naive_max_token_for_text_unit: int = 12000
    
    # Local search parameters
    local_max_token_for_text_unit: int = 4000
    local_max_token_for_local_context: int = 6000
    local_max_token_for_community_report: int = 2000
    local_community_single_one: bool = False
    sub_graph: bool = False
    mix_relation: bool = False
    
    # Global search parameters
    global_min_community_rating: float = 0
    global_max_consider_community: float = 512
    global_max_token_for_community_report: int = 16384
    global_special_community_map_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    retrieval_details: dict = None


# TypedDict schemas for data structures

class TextChunkSchema(TypedDict):
    """Schema for text chunks.
    
    Attributes:
        tokens: Number of tokens in the chunk
        content: Text content of the chunk
        full_doc_id: ID of the parent document
        chunk_order_index: Order index of this chunk within the document
    """
    tokens: int
    content: str
    full_doc_id: str
    chunk_order_index: int


class SingleCommunitySchema(TypedDict):
    """Schema for a single community in the graph.
    
    Attributes:
        level: Hierarchy level of the community
        title: Title/name of the community
        edges: List of edges as [source, target] pairs
        nodes: List of node IDs in this community
        chunk_ids: List of chunk IDs associated with this community
        occurrence: Occurrence frequency/score
        sub_communities: List of sub-community IDs
    """
    level: int
    title: str
    edges: list[list[str, str]]
    nodes: list[str]
    chunk_ids: list[str]
    occurrence: float
    sub_communities: list[str]


class SingleTemporalSchema(TypedDict):
    """Schema for a single temporal hierarchy node.
    
    Attributes:
        level: Hierarchy level of the temporal node
        title: Title/name of the temporal node
        temporal_edges: List of temporal edges as [source, target, timestamp] tuples
        nodes: List of node IDs in this temporal node
        chunk_ids: List of chunk IDs associated with this temporal node
        sub_communities: List of sub-community IDs
        all_sub_communities: List of all sub-community IDs (including nested)
    """
    level: int
    title: str
    temporal_edges: list[list[str, str, str]]
    nodes: list[str]
    chunk_ids: list[str]
    sub_communities: list[str]
    all_sub_communities: list[str]


class CommunitySchema(SingleCommunitySchema):
    """Extended community schema with report information.
    
    Attributes:
        report_string: String representation of the community report
        report_json: JSON representation of the community report
    """
    report_string: str
    report_json: dict


class TemporalSchema(SingleTemporalSchema):
    """Extended temporal schema with report information.
    
    Attributes:
        report_string: String representation of the temporal report
        report_json: JSON representation of the temporal report
    """
    report_string: str
    report_json: dict

