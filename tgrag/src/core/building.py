"""Building operations for temporal knowledge graphs."""

from __future__ import annotations

import re
import json
import asyncio
from typing import Union, List, Optional, Dict, Any
from collections import Counter, defaultdict
import logging

# Import from new modules
from ..config.prompts import get_prompt_manager, GRAPH_FIELD_SEP
from ..temporal.operations import (
    enhanced_infer_timestamp_level,
    enhanced_normalize_timestamp,
)
from ..utils.hashing import compute_mdhash_id
from ..utils.helpers import (
    logger,
    clean_str,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    get_parent_timestamp_name,
    complete_timestamp_range_by_level,
    sort_timestamp_by_datetime,
    convert_timestamp_to_datetime,
)

# Import storage base classes
from ..storage.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
)

# Import schema types from core.types
from ..core.types import (
    SingleCommunitySchema,
    SingleTemporalSchema,
    CommunitySchema,
    TemporalSchema,
    TextChunkSchema,
    QueryParam,
    TemporalQuadruple,
)

# Import centralized temporal normalizer
from ..temporal.normalization import get_temporal_normalizer

# Get prompt manager instance
_prompt_manager = None

def _get_prompts():
    """Get prompts dict from PromptManager."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = get_prompt_manager()
    return _prompt_manager.prompts

# Create PROMPTS alias for backward compatibility during migration
PROMPTS = property(lambda self: _get_prompts())
# Use a class to make PROMPTS work like a dict
class _PromptsProxy:
    def __getitem__(self, key):
        return _get_prompts()[key]
    def get(self, key, default=None):
        return _get_prompts().get(key, default)
    def __contains__(self, key):
        return key in _get_prompts()
    def keys(self):
        return _get_prompts().keys()
    def values(self):
        return _get_prompts().values()
    def items(self):
        return _get_prompts().items()

PROMPTS = _PromptsProxy()

# Helper function: sanitize attribute
def _sanitize_attribute(attr: str) -> str:
    """Removes leading/trailing whitespace and quotes from a string."""
    return attr.strip().strip('"').strip("'")


# Helper function: handle entity/relation summary
async def _handle_entity_relation_summary(
        entity_or_relation_name: str,
        description: str,
        global_config: dict,
) -> str:
    # Check if summarization is disabled
    if global_config.get("disable_entity_summarization", False):
        return description
    
    use_llm_func: callable = global_config["cheap_model_func"]
    llm_max_tokens = global_config["cheap_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    try:
        summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
        if summary is None or len(summary) == 0:
            logger.warning(f"LLM returned None or empty result for entity {entity_or_relation_name}, using original description")
            return description
        return summary[0] if isinstance(summary, (list, tuple)) else str(summary)
    except Exception as e:
        logger.warning(f"An error occurred during entity summary for {entity_or_relation_name}: {e}")
        return description


# Helper function: handle single entity extraction
async def _handle_single_entity_extraction(
        record_attributes: list[str],
        chunk_key: str,
):
    if len(record_attributes) < 4:
        logger.debug(f"Entity extraction failed: insufficient attributes ({len(record_attributes)} < 4) for {record_attributes}")
        return None
    
    # Loosened condition to handle cases with or without quotes
    first_attr = _sanitize_attribute(record_attributes[0]).lower()
    if "entity" not in first_attr:
        logger.debug(f"Entity extraction failed: first attribute '{first_attr}' does not contain 'entity'")
        return None
    
    # add this record as a node in the G
    entity_name = _sanitize_attribute(record_attributes[1].upper())
    if not entity_name.strip():
        logger.debug(f"Entity extraction failed: empty entity name for chunk {chunk_key}, attributes: {record_attributes}")
        return None
    entity_type = _sanitize_attribute(record_attributes[2].upper())
    entity_description = _sanitize_attribute(record_attributes[3])
    entity_source_id = chunk_key
    
    logger.debug(f"Entity extraction processing: name='{entity_name}', type='{entity_type}', description='{entity_description[:50]}...'")

    # Enhanced timestamp processing with enhanced temporal normalizer
    if entity_type.lower() in PROMPTS['DEFAULT_TEMPORAL_HIERARCHY']:
        try:
            # Use centralized temporal normalizer for consistent timestamp normalization
            normalizer = get_temporal_normalizer()
            normalized_result = normalizer.normalize_temporal_expression(entity_name)
            
            if normalized_result and normalized_result.normalized_forms:
                # Create one entity with the original name, but store normalized forms as metadata
                # This avoids source_id conflicts while preserving temporal alignment information
                logger.info(f"Enhanced normalized timestamp entity: {entity_name} -> {normalized_result.normalized_forms} (confidence: {normalized_result.confidence}, type: {normalized_result.normalization_type})")
                
                # Use enhanced normalizer result's granularity directly
                type_ = normalized_result.granularity.value
                
                # Create entity with original name but include normalized forms as metadata
                return dict(
                    entity_name=entity_name,  # Keep original name
                    entity_type=type_.upper(),
                    description=entity_description,
                    source_id=entity_source_id,
                    is_temporal=True,  # Mark as temporal entity
                    normalized_forms=normalized_result.normalized_forms,  # Store normalized forms as metadata
                    normalization_confidence=normalized_result.confidence,
                    normalization_type=normalized_result.normalization_type
                )
            else:
                logger.warning(f"Failed to normalize timestamp {entity_name} with enhanced normalizer, falling back to basic normalization")
                # Fall back to basic inference
                type_ = enhanced_infer_timestamp_level(entity_name)
                return dict(
                    entity_name=entity_name,
                    entity_type=type_.upper(),
                    description=entity_description,
                    source_id=entity_source_id,
                    is_temporal=True,  # Mark as temporal entity
                )
        except Exception as e:
            logger.warning(f"Failed to infer timestamp level for {entity_name}: {e}")
            return None
    result = dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )
    logger.debug(f"Entity extraction success: {result}")
    return result


# Helper function: handle single timestamp extraction
async def _handle_single_timestamp_extraction(
        record_attributes: list[str],
):
    if len(record_attributes) < 2:
        return None
    temporal_logic = _sanitize_attribute(record_attributes[1].lower())
    
    if temporal_logic == 'between':
        if len(record_attributes) < 5:
            return None
        entity_name_start = _sanitize_attribute(record_attributes[2].upper())
        entity_name_end = _sanitize_attribute(record_attributes[3].upper())
        
        # Enhanced normalization for timestamp range
        try:
            normalizer = get_temporal_normalizer()
            
            # Normalize start timestamp
            start_normalized = normalizer.normalize_temporal_expression(entity_name_start)
            if start_normalized.normalized_forms:
                entity_name_start = start_normalized.normalized_forms[0]
                logger.info(f"Enhanced normalized start timestamp: {record_attributes[2]} -> {entity_name_start}")
            
            # Normalize end timestamp
            end_normalized = normalizer.normalize_temporal_expression(entity_name_end)
            if end_normalized and end_normalized.normalized_forms:
                entity_name_end = end_normalized.normalized_forms[0]
                logger.info(f"Enhanced normalized end timestamp: {record_attributes[3]} -> {entity_name_end}")
            
            entity_name_start, entity_name_end = sort_timestamp_by_datetime([entity_name_start, entity_name_end])
            
            # Get granularity with proper null checking
            start_result = normalizer.normalize_temporal_expression(entity_name_start)
            end_result = normalizer.normalize_temporal_expression(entity_name_end)
            
            if not start_result or not end_result or not start_result.granularity or not end_result.granularity:
                logger.warning(f"Failed to infer timestamp level for {entity_name_start} and {entity_name_end}: Missing granularity")
                return None
                
            type_start = start_result.granularity.value
            type_end = end_result.granularity.value
            
            # For 'between' logic, use the more specific granularity (higher number = more specific)
            # If granularities differ, use the more specific one
            if type_start != type_end:
                granularity_hierarchy = {"year": 1, "quarter": 2, "month": 3, "week": 4, "date": 5}
                if granularity_hierarchy.get(type_start, 0) < granularity_hierarchy.get(type_end, 0):
                    type_start = type_end  # Use the more specific granularity
                else:
                    type_end = type_start  # Use the more specific granularity
                logger.info(f"Adjusted granularity mismatch: using '{type_start}' for both timestamps")

        except Exception as e:
            logger.warning(f"Failed to infer timestamp level for {entity_name_start} and {entity_name_end}: {e}")
            return None

        return dict(
            temporal_logic='between',
            timestamp=[entity_name_start, entity_name_end],  # list
            entity_type=type_start.upper(),
        )

    elif temporal_logic in ['at', 'before', 'after']:
        if len(record_attributes) < 4:
            return None
        entity_name = _sanitize_attribute(record_attributes[2].upper())
        entity_type = _sanitize_attribute(record_attributes[3].upper())
        
        # Enhanced normalization for single timestamp
        try:
            normalizer = get_temporal_normalizer()
            normalized_result = normalizer.normalize_temporal_expression(entity_name)
            
            if normalized_result.normalized_forms:
                entity_name = normalized_result.normalized_forms[0]
                logger.info(f"Enhanced normalized single timestamp: {record_attributes[2]} -> {entity_name}")
                # Use the granularity from the enhanced normalizer result with null checking
                if normalized_result.granularity:
                    type_ = normalized_result.granularity.value
                else:
                    logger.warning(f"Missing granularity for normalized timestamp: {entity_name}")
                    type_ = enhanced_infer_timestamp_level(entity_name)
            else:
                # Fallback if no normalized forms
                type_ = enhanced_infer_timestamp_level(entity_name)
        except Exception as e:
            logger.warning(f"Failed to infer timestamp level for {entity_name}: {e}")
            return None
        return dict(
            temporal_logic=temporal_logic,
            timestamp=entity_name,  # Remove quotes to match temporal hierarchy keys
            entity_type=type_.upper(),
        )

    else:
        return None


# Helper function: handle single temporal relationship extraction
async def _handle_single_temporal_relationship_extraction(
        record_attributes: list[str],
        chunk_key: str,
):
    if len(record_attributes) < 5 or "relationship" not in _sanitize_attribute(record_attributes[0]).lower():
        return None
    description, edge_source_id, temporal_level = dict(), dict(), dict()
    # add this record as edge
    timestamp = _sanitize_attribute(record_attributes[1].upper())
    try:
        normalizer = get_temporal_normalizer()
        normalized_result = normalizer.normalize_temporal_expression(timestamp)
        if normalized_result is None or not normalized_result.granularity:
            logger.warning(f"Failed to infer temporal relationship for {timestamp}: no granularity")
            return None
        type_ = normalized_result.granularity.value
    except Exception as e:
        logger.warning(f"Failed to infer temporal relationship for {timestamp}: {e}")
        return None

    source = _sanitize_attribute(record_attributes[2].upper())
    target = _sanitize_attribute(record_attributes[3].upper())
    edge_description = _sanitize_attribute(record_attributes[4])
    description[timestamp] = edge_description
    edge_source_id[timestamp] = chunk_key
    # Handle unknown entity types by defaulting to UNKNOWN
    if type_ not in PROMPTS['DEFAULT_TEMPORAL_HIERARCHY_LEVEL']:
        type_ = "UNKNOWN"
    temporal_level[timestamp] = PROMPTS['DEFAULT_TEMPORAL_HIERARCHY_LEVEL'][type_]

    return dict(
        timestamp=timestamp,
        temporal_level=temporal_level,
        src_id=source,
        tgt_id=target,
        description=description,
        source_id=edge_source_id,
    )


# Helper function: handle flexible relationship extraction
async def _handle_flexible_relationship_extraction(
        record_attributes: list[str],
        chunk_key: str,
):
    """
    More flexible relationship extraction that can handle various formats
    """
    if len(record_attributes) < 3:
        logger.debug(f"Relationship extraction failed: insufficient attributes ({len(record_attributes)} < 3) for {record_attributes}")
        return None
    
    # Try to identify relationship patterns
    first_attr = _sanitize_attribute(record_attributes[0]).lower()
    if "relationship" in first_attr:
        # Standard format: ("relationship", timestamp, source, target, description)
        if len(record_attributes) >= 5:
            return await _handle_single_temporal_relationship_extraction(record_attributes, chunk_key)
    
    # Alternative format: ("relationship", source, target, description) - no timestamp
    if "relationship" in _sanitize_attribute(record_attributes[0]).lower() and len(record_attributes) >= 4:
        source = _sanitize_attribute(record_attributes[1].upper())
        target = _sanitize_attribute(record_attributes[2].upper())
        edge_description = _sanitize_attribute(record_attributes[3])
        
        # Try to extract timestamp from description or use a default
        timestamp = "UNKNOWN_TIME"
        try:
            # Look for timestamp patterns in description
            timestamp_patterns = [
                r'\b(20\d{2})[-\s]?(Q[1-4])\b',  # 2023-Q2
                r'\b(20\d{2})[-\s]?(\d{1,2})\b',  # 2023-06
                r'\b(Q[1-4])\s+(20\d{2})\b',      # Q2 2023
            ]
            
            for pattern in timestamp_patterns:
                match = re.search(pattern, edge_description, re.IGNORECASE)
                if match:
                    if 'Q' in match.group():
                        timestamp = f"{match.group(1)}-{match.group(2)}"
                    else:
                        timestamp = f"{match.group(1)}-{match.group(2).zfill(2)}"
                    break
        except:
            pass
        
        description = {timestamp: edge_description}
        edge_source_id = {timestamp: chunk_key}
        temporal_level = {timestamp: 1}  # Default level
        
        return dict(
            timestamp=timestamp,
            temporal_level=temporal_level,
            src_id=source,
            tgt_id=target,
            description=description,
            source_id=edge_source_id,
        )
    
    # Entity-entity relationship without explicit relationship marker
    if len(record_attributes) >= 3 and "entity" not in _sanitize_attribute(record_attributes[0]).lower():
        # Might be a relationship in disguise
        source = _sanitize_attribute(record_attributes[0].upper())
        target = _sanitize_attribute(record_attributes[1].upper())
        edge_description = _sanitize_attribute(record_attributes[2])
        
        timestamp = "UNKNOWN_TIME"
        description = {timestamp: edge_description}
        edge_source_id = {timestamp: chunk_key}
        temporal_level = {timestamp: 1}
        
        return dict(
            timestamp=timestamp,
            temporal_level=temporal_level,
            src_id=source,
            tgt_id=target,
            description=description,
            source_id=edge_source_id,
        )
    
    return None


# Merge functions
async def _merge_nodes_then_upsert(
        entity_name: str,
        nodes_data: list[dict],
        knwoledge_graph_inst: BaseGraphStorage,
        global_config: dict,
):
    # issue existing node info can be {}
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knwoledge_graph_inst.get_node(entity_name)
    if already_node is not None and already_node:
        already_entitiy_types.append(already_node["source_id"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    # Ensure description is never None
    if description is None:
        logger.warning(f"Description is None for entity {entity_name}, using empty string")
        description = ""
    
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knwoledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
        src_id: str,
        tgt_id: str,
        edges_data: list[dict],
        knwoledge_graph_inst: BaseGraphStorage,
        global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        # Ensure we get a valid integer for order, defaulting to 1 if None or missing
        order_val = already_edge.get("order")
        if order_val is None:
            order_val = 1
        already_order.append(order_val)

    # Filter out None values from already_order to prevent concatenation errors
    valid_already_order = [order_val for order_val in already_order if order_val is not None]
    order = min([dp.get("order", 1) for dp in edges_data] + valid_already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    # Ensure description is never None
    if description is None:
        logger.warning(f"Description is None for edge {src_id} -> {tgt_id}, using empty string")
        description = ""
    
    await knwoledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight, description=description, source_id=source_id, order=order
        ),
    )


async def _merge_temporal_edges_then_upsert(
        timestamp_id: str,
        src_id: str,
        tgt_id: str,
        edges_data: list[dict],
        knwoledge_graph_inst: BaseGraphStorage,
        global_config: dict,
):
    already_source_ids = dict()
    already_description = dict()
    already_temporal_level = dict()
    already_order = []

    logger.info(f"maybe edge {timestamp_id}, {src_id}, {tgt_id}")
    # no placeholder
    if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
        src_data = await knwoledge_graph_inst.get_node(src_id)
        tgt_data = await knwoledge_graph_inst.get_node(tgt_id)
        if src_data.get('entity_type').lower() in PROMPTS['DEFAULT_TEMPORAL_HIERARCHY'] or tgt_data.get(
                'entity_type').lower() in PROMPTS['DEFAULT_TEMPORAL_HIERARCHY']:
            logger.info(f"Skipping temporal edge {src_id} -> {tgt_id} (temporal entities: {src_data.get('entity_type')}, {tgt_data.get('entity_type')})")
            return
        already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
        already_source_ids = already_edge['source_id']
        already_description = already_edge['description']
        # Ensure we get a valid integer for order, defaulting to 1 if None or missing
        order_val = already_edge.get("order")
        if order_val is None:
            order_val = 1
        already_order.append(order_val)

    # Filter out None values from already_order to prevent concatenation errors
    valid_already_order = [order_val for order_val in already_order if order_val is not None]
    order = min([dp.get("order", 1) for dp in edges_data] + valid_already_order)
    
    for dp in edges_data:
        already_description.update(dp['description'])
        already_source_ids.update(dp['source_id'])

    for need_insert_id in [src_id, tgt_id]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            description = GRAPH_FIELD_SEP.join(list(already_description.values()))
            source_id = GRAPH_FIELD_SEP.join(
                list(already_source_ids.values())
            )
            description = await _handle_entity_relation_summary(
                need_insert_id, description, global_config
            )
            # Ensure description is never None
            if description is None:
                logger.warning(f"Description is None for temporal node {need_insert_id}, using empty string")
                description = ""
            
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,  # not dict
                    "description": description,  # not dict
                    "entity_type": '"UNKNOWN"',
                },
            )
    
    logger.info(f"upsert {src_id} and {tgt_id} edge")
    await knwoledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            description=already_description, source_id=already_source_ids, order=order
        ),
    )
    logger.info(f"upsert {timestamp_id} and {tgt_id} edge")
    logger.info(f"upsert {timestamp_id} and {src_id} edge")

    # add timestamp and entity edge
    await knwoledge_graph_inst.upsert_edge(
        timestamp_id,
        tgt_id,
        edge_data=dict(
            description=dict(), source_id=already_source_ids, order=order
        ),
    )
    await knwoledge_graph_inst.upsert_edge(
        timestamp_id,
        src_id,
        edge_data=dict(
            description=dict(), source_id=already_source_ids, order=order
        ),
    )
    edge_data = dict(
            description=already_description, source_id=already_source_ids, order=order
        )
    edge_data['src_id'] = src_id
    edge_data['tgt_id'] = tgt_id

    return edge_data


# Helper function: convert extracted relationships to temporal quadruples
def _convert_to_temporal_quadruples(
    maybe_edges: Dict[tuple, List[dict]],
    maybe_nodes: Dict[str, List[dict]],
    normalizer=None
) -> List[TemporalQuadruple]:
    """
    Convert extracted relationships to explicit temporal quadruples (v₁, v₂, e, τ).
    
    This function transforms the extracted relationship data into the explicit
    TemporalQuadruple structure as described in the paper, where:
    - v₁ and v₂ are non-temporal entities
    - e is the relation description
    - τ is the normalized timestamp
    
    Args:
        maybe_edges: Dictionary mapping (timestamp, src_id, tgt_id) to list of edge data
        maybe_nodes: Dictionary mapping entity names to list of node data
        normalizer: Optional temporal normalizer instance (uses global if None)
        
    Returns:
        List of TemporalQuadruple objects
    """
    from ..temporal.normalization import get_temporal_normalizer
    
    if normalizer is None:
        normalizer = get_temporal_normalizer()
    
    quadruples = []
    
    for (timestamp, src_id, tgt_id), edge_data_list in maybe_edges.items():
        # Skip if either entity is temporal (quadruples only contain non-temporal entities)
        src_is_temporal = False
        tgt_is_temporal = False
        
        if src_id in maybe_nodes:
            src_data = maybe_nodes[src_id][0]
            src_is_temporal = src_data.get('is_temporal', False) or \
                            src_data.get('entity_type', '').lower() in PROMPTS['DEFAULT_TEMPORAL_HIERARCHY']
        
        if tgt_id in maybe_nodes:
            tgt_data = maybe_nodes[tgt_id][0]
            tgt_is_temporal = tgt_data.get('is_temporal', False) or \
                            tgt_data.get('entity_type', '').lower() in PROMPTS['DEFAULT_TEMPORAL_HIERARCHY']
        
        # Only create quadruples for non-temporal entity relationships
        # Temporal entities are handled separately in the time hierarchy
        if src_is_temporal or tgt_is_temporal:
            continue
        
        # Clean entity names (remove embedded delimiters and quotes)
        clean_src_id = _sanitize_attribute(src_id).strip()
        clean_tgt_id = _sanitize_attribute(tgt_id).strip()
        
        # Skip if entities are empty or contain only delimiters
        if not clean_src_id or not clean_tgt_id:
            continue
        
        # Additional filtering: Skip if entity looks like a financial metric
        # (starts with $ or is a percentage/metric)
        financial_pattern = re.compile(r'^[\$%]|^\d+[\.\d]*\s*(MILLION|BILLION|THOUSAND|PAIRS|%|PERCENT)', re.IGNORECASE)
        if financial_pattern.match(clean_src_id) or financial_pattern.match(clean_tgt_id):
            continue
        
        # Validate and normalize timestamp
        raw_timestamp = timestamp
        # Skip if timestamp looks like a financial amount or other non-temporal value
        if raw_timestamp.startswith('$') or financial_pattern.match(raw_timestamp):
            logger.debug(f"Skipping quadruple with non-temporal timestamp: {raw_timestamp}")
            continue
        
        # Try to normalize timestamp using the enhanced normalizer
        try:
            normalized_result = normalizer.normalize_temporal_expression(raw_timestamp)
            if normalized_result and normalized_result.normalized_forms and normalized_result.granularity:
                # Valid temporal expression
                normalized_timestamp = normalized_result.normalized_forms[0]
            else:
                # Invalid temporal expression - skip this quadruple
                logger.debug(f"Skipping quadruple with invalid timestamp: {raw_timestamp}")
                continue
        except Exception as e:
            logger.debug(f"Error normalizing timestamp '{raw_timestamp}': {e}, skipping quadruple")
            continue
        
        # Extract relation description from edge data
        for edge_data in edge_data_list:
            # Get description (can be dict or string)
            description = edge_data.get('description', '')
            if isinstance(description, dict):
                # If description is a dict keyed by timestamp, get the value
                description = description.get(timestamp, description.get(raw_timestamp, ''))
                if isinstance(description, dict):
                    # If still a dict, get first value or convert to string
                    description = str(list(description.values())[0]) if description else ''
            
            # Clean description
            if description:
                description = _sanitize_attribute(str(description)).strip()
            
            source_id = edge_data.get('source_id', '')
            if isinstance(source_id, dict):
                source_id = source_id.get(timestamp, source_id.get(raw_timestamp, ''))
                if isinstance(source_id, dict):
                    source_id = list(source_id.values())[0] if source_id else ''
            
            # Skip if description is empty or just whitespace
            if not description:
                continue
            
            quadruple = TemporalQuadruple(
                v1=clean_src_id,
                v2=clean_tgt_id,
                e=description,
                tau=normalized_timestamp,
                source_id=str(source_id) if source_id else '',
                raw_timestamp=raw_timestamp
            )
            quadruples.append(quadruple)
    
    logger.info(f"Converted {len(maybe_edges)} relationship groups to {len(quadruples)} temporal quadruples")
    return quadruples


# Main function: extract_entities
async def extract_entities(
        chunks: dict[str, TextChunkSchema],
        knwoledge_graph_inst: BaseGraphStorage,
        entity_vdb: BaseVectorStorage,
        relation_vdb: BaseVectorStorage,
        entity_vdb_new: BaseVectorStorage,
        global_config: dict,
        using_amazon_bedrock: bool = False,
) -> tuple[BaseGraphStorage, list[str], List[TemporalQuadruple]]:
    use_llm_func: callable = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    entity_extract_prompt = PROMPTS["temporal_entity_extraction_new"]

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        timestamp_format=json.dumps(PROMPTS["DEFAULT_TIMESTAMP_FORMAT"]),
        timestamp_types=",".join(PROMPTS["DEFAULT_TEMPORAL_HIERARCHY"])
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        
        # Retry logic for extraction
        max_retries = 3
        for attempt in range(max_retries):
            logger.info(f"Processing chunk {chunk_key}: Attempt {attempt + 1}/{max_retries}")
            try:
                # Try different prompts based on attempt number
                if attempt == 0:
                    # Standard prompt
                    hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
                elif attempt == 1:
                    # Simplified prompt for retry
                    simplified_prompt = PROMPTS.get("temporal_entity_extraction_old", entity_extract_prompt)
                    hint_prompt = simplified_prompt.format(**context_base, input_text=content)
                else:
                    # Most basic prompt for final attempt
                    basic_prompt = """Extract entities and relationships from the following text. Focus on companies, financial metrics, and temporal relationships.

Text: {input_text}

Extract in this format:
("entity", "entity_name", "entity_type", "description")
("relationship", "timestamp", "source", "target", "description")

Output:"""
                    hint_prompt = basic_prompt.format(input_text=content)
                
                raw_llm_result = await use_llm_func(hint_prompt)
                logger.info(f"Raw use_llm_func result for chunk {chunk_key}: type={type(raw_llm_result)}, content={repr(raw_llm_result)}")
                
                # Handle the response based on its type
                if isinstance(raw_llm_result, tuple) and len(raw_llm_result) >= 2:
                    final_result = raw_llm_result[0]
                    logger.info(f"Extracted response from tuple for chunk {chunk_key}: {repr(final_result)}")
                elif isinstance(raw_llm_result, list) and len(raw_llm_result) > 0:
                    first_element = raw_llm_result[0]
                    if isinstance(first_element, tuple) and len(first_element) >= 2:
                        final_result = first_element[0]
                        logger.info(f"Extracted response from tuple in list for chunk {chunk_key}: {repr(final_result)}")
                    elif isinstance(first_element, str):
                        final_result = first_element
                        logger.info(f"Extracted string from list for chunk {chunk_key}: {repr(final_result)}")
                    else:
                        final_result = str(first_element)
                        logger.info(f"Converted list element to string for chunk {chunk_key}: {repr(final_result)}")
                elif isinstance(raw_llm_result, str):
                    final_result = raw_llm_result
                    logger.info(f"Using direct string response for chunk {chunk_key}: {repr(final_result)}")
                else:
                    final_result = str(raw_llm_result)
                    logger.warning(f"Fallback string conversion for chunk {chunk_key}: {repr(final_result)}")
                
                # Ensure final_result is a string
                if not isinstance(final_result, str):
                    final_result = str(final_result)
                    logger.warning(f"Final conversion to string for chunk {chunk_key}: {repr(final_result)}")

                history = pack_user_ass_to_openai_messages(hint_prompt, final_result, using_amazon_bedrock)
                for now_glean_index in range(entity_extract_max_gleaning):
                    try:
                        raw_glean_result = await use_llm_func(continue_prompt, history_messages=history)
                        # Handle the same response format issues
                        if isinstance(raw_glean_result, tuple) and len(raw_glean_result) >= 2:
                            glean_result = raw_glean_result[0]
                        elif isinstance(raw_glean_result, list) and len(raw_glean_result) > 0:
                            first_element = raw_glean_result[0]
                            if isinstance(first_element, tuple) and len(first_element) >= 2:
                                glean_result = first_element[0]
                            else:
                                glean_result = str(first_element)
                        elif isinstance(raw_glean_result, str):
                            glean_result = raw_glean_result
                        else:
                            glean_result = str(raw_glean_result)
                    except Exception as e:
                        logger.info(f"An error occurred during gleaning: {e}")
                        glean_result = ''

                    history += pack_user_ass_to_openai_messages(continue_prompt, glean_result, using_amazon_bedrock)
                    final_result += glean_result
                    if now_glean_index == entity_extract_max_gleaning - 1:
                        break

                    try:
                        raw_loop_result = await use_llm_func(
                            if_loop_prompt, history_messages=history
                        )
                        # Handle the same response format issues
                        if isinstance(raw_loop_result, tuple) and len(raw_loop_result) >= 2:
                            if_loop_result = raw_loop_result[0]
                        elif isinstance(raw_loop_result, list) and len(raw_loop_result) > 0:
                            first_element = raw_loop_result[0]
                            if isinstance(first_element, tuple) and len(first_element) >= 2:
                                if_loop_result = first_element[0]
                            else:
                                if_loop_result = str(first_element)
                        elif isinstance(raw_loop_result, str):
                            if_loop_result = raw_loop_result
                        else:
                            if_loop_result = str(raw_loop_result)
                    except Exception as e:
                        logger.info(f"An error occurred during loop check: {e}")
                        if_loop_result = ''
                    if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
                    if if_loop_result != "yes":
                        break

                # Debug: Log the raw LLM response
                logger.info(f"Processing chunk {chunk_key}: Raw LLM response length: {len(final_result)}")
                logger.info(f"Raw LLM response for chunk {chunk_key}: {repr(final_result)}")
                
                # Check if response is too short or malformed
                if len(final_result) < 10:
                    logger.error(f"LLM response is too short for chunk {chunk_key}: length={len(final_result)}, content={repr(final_result)}")
                    logger.error(f"Original use_llm_func result type: {type(final_result)}")
                    logger.warning(f"Skipping chunk {chunk_key} due to malformed LLM response")
                    continue
                
                records = split_string_by_multi_markers(
                    final_result,
                    [context_base["record_delimiter"], context_base["completion_delimiter"]],
                )
                
                # Debug: Log the initial split results
                logger.info(f"Initial split for chunk {chunk_key}: {len(records)} records")
                for i, record in enumerate(records):
                    logger.info(f"  Record {i}: {record[:200]}...")

                # Handle case where LLM returns a single string that needs to be split
                if len(records) == 1 and context_base["record_delimiter"] in records[0]:
                    logger.info(f"Re-splitting single string for chunk {chunk_key}")
                    records = split_string_by_multi_markers(
                        records[0],
                        [context_base["record_delimiter"], context_base["completion_delimiter"]],
                    )
                    logger.info(f"After re-split for chunk {chunk_key}: {len(records)} records")
                    for i, record in enumerate(records):
                        logger.info(f"  Re-split record {i}: {record[:200]}...")

                maybe_nodes = defaultdict(list)
                maybe_edges = defaultdict(list)
                valid_records = 0
                total_records = len(records)
                rejected_records = []
                
                for record_idx, record in enumerate(records):
                    logger.info(f"Processing record {record_idx} for chunk {chunk_key}")
                    
                    # Skip empty records
                    if not record.strip():
                        logger.info(f"Skipping empty record {record_idx} for chunk {chunk_key}")
                        continue
                    
                    # Clean the record of any extra whitespace or quotes
                    original_record = record
                    record = record.strip().strip('"').strip("'")
                    logger.info(f"Cleaned record {record_idx} for chunk {chunk_key}: {record}")
                    
                    # Try to extract content between parentheses
                    record_match = re.search(r"\((.*)\)", record)
                    if record_match is None:
                        rejected_records.append(("no_parens", record))
                        logger.warning(f"Record {record_idx} rejected (no parentheses) for chunk {chunk_key}: {record[:100]}")
                        continue
                    
                    record_content = record_match.group(1)
                    logger.info(f"Extracted content for record {record_idx} chunk {chunk_key}: {record_content}")
                    
                    record_attributes = split_string_by_multi_markers(
                        record_content, [context_base["tuple_delimiter"]]
                    )
                    logger.info(f"Split attributes for record {record_idx} chunk {chunk_key}: {record_attributes}")
                    
                    # Clean each attribute
                    record_attributes = [attr.strip().strip('"').strip("'") for attr in record_attributes]
                    logger.info(f"Cleaned attributes for record {record_idx} chunk {chunk_key}: {record_attributes}")
                    
                    # Skip if we don't have enough attributes
                    if len(record_attributes) < 3:
                        rejected_records.append(("insufficient_attributes", record_attributes))
                        logger.warning(f"Record {record_idx} rejected (insufficient attributes: {len(record_attributes)} < 3) for chunk {chunk_key}: {record_attributes}")
                        continue
                    
                    # Try entity extraction first
                    logger.info(f"Attempting entity extraction for record {record_idx} chunk {chunk_key}")
                    if_entities = await _handle_single_entity_extraction(
                        record_attributes, chunk_key
                    )
                    if if_entities is not None:
                        logger.info(f"✓ Successfully extracted entity for record {record_idx} chunk {chunk_key}: {if_entities['entity_name']} ({if_entities['entity_type']})")
                        maybe_nodes[if_entities["entity_name"]].append(if_entities)
                        valid_records += 1
                        continue

                    # Try relationship extraction
                    logger.info(f"Attempting relationship extraction for record {record_idx} chunk {chunk_key}")
                    if_relation = await _handle_flexible_relationship_extraction(
                        record_attributes, chunk_key
                    )
                    if if_relation is not None:
                        logger.info(f"✓ Successfully extracted relationship for record {record_idx} chunk {chunk_key}: {if_relation['src_id']} -> {if_relation['tgt_id']}")
                        maybe_edges[(if_relation["timestamp"], if_relation["src_id"], if_relation["tgt_id"])].append(
                            if_relation
                        )
                        valid_records += 1
                    else:
                        rejected_records.append(("invalid_format", record_attributes))
                        logger.warning(f"✗ Record {record_idx} rejected (invalid format) for chunk {chunk_key}: {record_attributes}")
                
                # Log extraction statistics
                if valid_records == 0 and total_records > 0:
                    logger.warning(f"Chunk {chunk_key}: {total_records} records processed, {valid_records} valid, {len(rejected_records)} rejected")
                    if len(rejected_records) > 0:
                        logger.debug(f"Sample rejected records: {rejected_records[:3]}")
                    logger.debug(f"Raw records for chunk {chunk_key}: {records[:3]}")
                    logger.debug(f"Final result from LLM: {final_result[:500]}...")
                
                # Check if extraction was successful (at least some valid records)
                if valid_records > 0 or total_records == 0:
                    logger.info(f"Chunk {chunk_key}: Successfully extracted {valid_records} valid records on attempt {attempt + 1}")
                    break
                else:
                    logger.warning(f"Attempt {attempt + 1}: No valid records extracted from {total_records} records for chunk {chunk_key}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        logger.error(f"Failed to extract valid records after {max_retries} attempts for chunk {chunk_key}")
                        return dict(), dict()
                        
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for chunk {chunk_key}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    logger.error(f"Failed to process chunk {chunk_key} after {max_retries} attempts: {e}")
                    return dict(), dict()
        
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
            ]
        total_chunks = len(ordered_chunks)
        if total_chunks > 0:
            percentage = min(100, (already_processed * 100) // total_chunks)
        else:
            percentage = 100
        print(
            f"{now_ticks} Processed {already_processed}({percentage}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar
    logger.info("Entity extraction completed. Processing extracted data...")
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            # it's undirected graph
            sorted_k = (k[0],) + tuple(sorted(k[1:]))
            maybe_edges[sorted_k].extend(v)

    # Separate temporal and non-temporal entities
    temporal_entities = {}
    non_temporal_entities = {}
    
    for name, data in maybe_nodes.items():
        if data[0].get('is_temporal', False) or data[0]['entity_type'].lower() in PROMPTS['DEFAULT_TEMPORAL_HIERARCHY']:
            temporal_entities[name] = data
        else:
            non_temporal_entities[name] = data
    
    # Get all temporal entity names for hierarchy building
    maybe_hierarchy_node_names = list(temporal_entities.keys())
    
    # Determine which temporal entities should be included in entity relation graph
    temporal_entities_for_entity_graph = set()
    
    for (timestamp, src_id, tgt_id), edge_data in maybe_edges.items():
        # Check if this edge involves a temporal entity and a non-temporal entity
        if timestamp in temporal_entities:
            if src_id in non_temporal_entities or tgt_id in non_temporal_entities:
                temporal_entities_for_entity_graph.add(timestamp)
        elif src_id in temporal_entities:
            if timestamp in non_temporal_entities or tgt_id in non_temporal_entities:
                temporal_entities_for_entity_graph.add(src_id)
        elif tgt_id in temporal_entities:
            if timestamp in non_temporal_entities or src_id in non_temporal_entities:
                temporal_entities_for_entity_graph.add(tgt_id)
    
    # Combine entities for entity relation graph
    entities_for_entity_graph = {**non_temporal_entities}
    for temp_entity in temporal_entities_for_entity_graph:
        if temp_entity in temporal_entities:
            entities_for_entity_graph[temp_entity] = temporal_entities[temp_entity]
    
    logger.info(f"Temporal entities: {len(temporal_entities)} total, {len(temporal_entities_for_entity_graph)} with relationships")
    logger.info(f"Non-temporal entities: {len(non_temporal_entities)}")
    logger.info(f"Entities for entity relation graph: {len(entities_for_entity_graph)}")
    logger.info(f"Found {len(maybe_nodes)} unique entities and {len(maybe_edges)} unique relations")
    
    # Convert relationships to explicit temporal quadruples (v₁, v₂, e, τ)
    # This aligns with the paper's methodology
    temporal_quadruples = _convert_to_temporal_quadruples(maybe_edges, maybe_nodes)
    logger.info(f"Extracted {len(temporal_quadruples)} temporal quadruples from relationships")
    
    logger.info(f"Starting entity merging and graph upsert for {len(entities_for_entity_graph)} entities (filtered for entity relation graph)...")
    
    # Process entities in batches
    batch_size = 100
    all_entities_data = []
    
    entity_items = list(entities_for_entity_graph.items())
    total_batches = (len(entity_items) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(entity_items))
        batch_items = entity_items[start_idx:end_idx]
        
        logger.info(f"Processing entity batch {batch_idx + 1}/{total_batches} ({len(batch_items)} entities)")
        
        batch_results = await asyncio.gather(
            *[
                _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config, )
                for k, v in batch_items
            ],
            return_exceptions=True
        )
        
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Exception in entity processing batch {batch_idx + 1}, item {i}: {result}")
                batch_results[i] = None
        
        valid_results = [r for r in batch_results if r is not None]
        all_entities_data.extend(valid_results)
        
        logger.info(f"Completed entity batch {batch_idx + 1}/{total_batches} ({len(valid_results)} valid entities)")
    
    logger.info(f"Entity merging completed. Starting relation processing for {len(maybe_edges)} relations...")

    # Process relations in batches
    relation_batch_size = 50
    all_relations_data = []
    
    relation_items = list(maybe_edges.items())
    total_relation_batches = (len(relation_items) + relation_batch_size - 1) // relation_batch_size
    
    for batch_idx in range(total_relation_batches):
        start_idx = batch_idx * relation_batch_size
        end_idx = min(start_idx + relation_batch_size, len(relation_items))
        batch_items = relation_items[start_idx:end_idx]
        
        logger.info(f"Processing relation batch {batch_idx + 1}/{total_relation_batches} ({len(batch_items)} relations)")
        
        batch_results = await asyncio.gather(
            *[
                _merge_temporal_edges_then_upsert(k[0], k[1], k[2], v, knwoledge_graph_inst, global_config)
                for k, v in batch_items
            ],
            return_exceptions=True
        )
        
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Exception in relation processing batch {batch_idx + 1}, item {i}: {result}")
                batch_results[i] = None
        
        valid_results = [r for r in batch_results if r is not None]
        all_relations_data.extend(valid_results)
        
        logger.info(f"Completed relation batch {batch_idx + 1}/{total_relation_batches} ({len(valid_results)} valid relations)")

    total_relations = len(all_relations_data)
    none_relations = sum(1 for dp in all_relations_data if dp is None)
    valid_relations = total_relations - none_relations
    logger.info(f"Relations processing: {total_relations} total, {valid_relations} valid, {none_relations} None values")

    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return knwoledge_graph_inst, [], []
    
    logger.info("Starting vector database upserts...")
    if entity_vdb is not None and entity_vdb_new is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + " " + dp.get("description", ""),
                "entity_name": dp["entity_name"],
                "description": dp.get("description", ""),
                "entity_type": dp.get("entity_type", ""),
            }
            for dp in all_entities_data
        }
        data_for_vdb_new = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + " " + dp.get("description", ""),
                "entity_name": dp["entity_name"],
                "description": dp.get("description", ""),
                "entity_type": dp.get("entity_type", ""),
            }
            for dp in all_entities_data
        }
        
        logger.info(f"Upserting {len(data_for_vdb)} entities to entity_vdb...")
        await entity_vdb.upsert(data_for_vdb)
        
        logger.info(f"Upserting {len(data_for_vdb_new)} new entities to entity_vdb_new...")
        await entity_vdb_new.upsert(data_for_vdb_new)
    if relation_vdb is not None:
        valid_relations_data = [dp for dp in all_relations_data if dp is not None]
        data_for_vdb_relation = {
            compute_mdhash_id(dp["src_id"]+'_'+dp["tgt_id"]+'_'+timestamp, prefix="rel-"): {
                "content": des,
                "entity_name": dp["src_id"]+'_'+dp["tgt_id"]+'_'+timestamp,
            }
            for dp in valid_relations_data for timestamp, des in dp.get('description', {}).items()
        }
        await relation_vdb.upsert(data_for_vdb_relation)

    logger.info("Entity extraction and graph building completed successfully!")
    return knwoledge_graph_inst, maybe_hierarchy_node_names, temporal_quadruples


# Helper function: find timestamp in hierarchy
def _find_timestamp_in_hierarchy(timestamp_value: str, temporal_hierarchy: dict[str, SingleTemporalSchema]) -> tuple[bool, str]:
    """
    Helper function to find a timestamp in the temporal hierarchy, handling both quoted and unquoted formats.
    
    Args:
        timestamp_value: The timestamp to look for
        temporal_hierarchy: The temporal hierarchy dictionary
        
    Returns:
        Tuple of (found, matched_key) where found is a boolean and matched_key is the actual key found
    """
    # Try exact match first
    if temporal_hierarchy.get(timestamp_value):
        return True, timestamp_value
    
    # Try with quotes added
    quoted_timestamp = f'"{timestamp_value}"'
    if temporal_hierarchy.get(quoted_timestamp):
        return True, quoted_timestamp
    
    # Try with quotes removed
    unquoted_timestamp = timestamp_value.strip('"').strip("'")
    if temporal_hierarchy.get(unquoted_timestamp):
        return True, unquoted_timestamp
    
    # Try with quotes removed and then added
    quoted_unquoted_timestamp = f'"{unquoted_timestamp}"'
    if temporal_hierarchy.get(quoted_unquoted_timestamp):
        return True, quoted_unquoted_timestamp
    
    # Try HTML entity encoded version (for GraphML compatibility)
    html_quoted_timestamp = f'&quot;{timestamp_value}&quot;'
    if temporal_hierarchy.get(html_quoted_timestamp):
        return True, html_quoted_timestamp
    
    # Try HTML entity encoded version of unquoted timestamp
    html_quoted_unquoted_timestamp = f'&quot;{unquoted_timestamp}&quot;'
    if temporal_hierarchy.get(html_quoted_unquoted_timestamp):
        return True, html_quoted_unquoted_timestamp
    
    return False, timestamp_value


# Helper function: pack single community by sub communities
def _pack_single_community_by_sub_communities(
        community: SingleCommunitySchema,
        max_token_size: int,
        already_reports: dict[str, CommunitySchema],
) -> tuple[str, int]:
    all_sub_communities = [
        already_reports[k] for k in community["sub_communities"] if k in already_reports
    ]
    all_sub_communities = sorted(
        all_sub_communities, key=lambda x: x["occurrence"], reverse=True
    )
    may_trun_all_sub_communities = truncate_list_by_token_size(
        all_sub_communities,
        key=lambda x: x["report_string"],
        max_token_size=max_token_size,
    )
    sub_fields = ["id", "report", "rating", "importance"]
    sub_communities_describe = list_of_list_to_csv(
        [sub_fields]
        + [
            [
                i,
                c["report_string"],
                c["report_json"].get("rating", -1),
                c["occurrence"],
            ]
            for i, c in enumerate(may_trun_all_sub_communities)
        ]
    )
    already_nodes = []
    already_edges = []
    for c in may_trun_all_sub_communities:
        already_nodes.extend(c["nodes"])
        already_edges.extend([tuple(e) for e in c["temporal_edges"]])
    return (
        sub_communities_describe,
        len(encode_string_by_tiktoken(sub_communities_describe)),
        set(already_nodes),
        set(already_edges),
    )


# Helper function: pack single community describe
async def _pack_single_community_describe(
        knwoledge_graph_inst: BaseGraphStorage,
        community: SingleCommunitySchema,
        max_token_size: int = 12000,
        already_reports: dict[str, CommunitySchema] = {},
        global_config: dict = {},
) -> str:
    nodes_in_order = sorted(community["nodes"])
    edges_in_order = sorted(community["edges"], key=lambda x: x[0] + x[1])

    nodes_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_node(n) for n in nodes_in_order]
    )
    edges_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_edge(src, tgt) for src, tgt in edges_in_order]
    )
    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]
    nodes_list_data = [
        [
            i,
            node_name,
            node_data.get("entity_type", "UNKNOWN"),
            node_data.get("description", "UNKNOWN"),
            await knwoledge_graph_inst.node_degree(node_name),
        ]
        for i, (node_name, node_data) in enumerate(zip(nodes_in_order, nodes_data))
    ]
    nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)
    nodes_may_truncate_list_data = truncate_list_by_token_size(
        nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )
    edges_list_data = [
        [
            i,
            edge_name[0],
            edge_name[1],
            edge_data.get("description", "UNKNOWN"),
            await knwoledge_graph_inst.edge_degree(*edge_name),
        ]
        for i, (edge_name, edge_data) in enumerate(zip(edges_in_order, edges_data))
    ]
    edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)
    edges_may_truncate_list_data = truncate_list_by_token_size(
        edges_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )

    truncated = len(nodes_list_data) > len(nodes_may_truncate_list_data) or len(
        edges_list_data
    ) > len(edges_may_truncate_list_data)

    report_describe = ""
    need_to_use_sub_communities = (
            truncated and len(community["sub_communities"]) and len(already_reports)
    )
    force_to_use_sub_communities = global_config["addon_params"].get(
        "force_to_use_sub_communities", False
    )
    if need_to_use_sub_communities or force_to_use_sub_communities:
        logger.debug(
            f"Community {community['title']} exceeds the limit or you set force_to_use_sub_communities to True, using its sub-communities"
        )
        report_describe, report_size, contain_nodes, contain_edges = (
            _pack_single_community_by_sub_communities(
                community, max_token_size, already_reports
            )
        )
        report_exclude_nodes_list_data = [
            n for n in nodes_list_data if n[1] not in contain_nodes
        ]
        report_include_nodes_list_data = [
            n for n in nodes_list_data if n[1] in contain_nodes
        ]
        report_exclude_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) not in contain_edges
        ]
        report_include_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) in contain_edges
        ]
        nodes_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_nodes_list_data + report_include_nodes_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
        edges_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_edges_list_data + report_include_edges_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
    nodes_describe = list_of_list_to_csv([node_fields] + nodes_may_truncate_list_data)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_may_truncate_list_data)
    return f"""-----Reports-----
```csv
{report_describe}
```
-----Entities-----
```csv
{nodes_describe}
```
-----Relationships-----
```csv
{edges_describe}
```"""


# Helper function: pack single timestamp describe
async def _pack_single_timestamp_describe(
        knowledge_graph_inst: BaseGraphStorage,
        community: SingleCommunitySchema,
        max_token_size: int = 12000,
        already_reports: dict[str, CommunitySchema] = {},
        global_config: dict = {},
) -> str:
    nodes_in_order = sorted(community["nodes"])
    temopral_edges_in_order = sorted(community["temporal_edges"], key=lambda x: x[0] + x[1] + x[2])

    nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(n) for n in nodes_in_order]
    )
    temporal_edges_data = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(src, tgt) for timestamp, src, tgt in temopral_edges_in_order]
    )
    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "timestamp", "source", "target", "description", "rank"]
    nodes_list_data = [
        [
            i,
            node_name,
            node_data.get("entity_type", "UNKNOWN"),
            node_data.get("description", "UNKNOWN"),
            await knowledge_graph_inst.node_degree(node_name),
        ]
        for i, (node_name, node_data) in enumerate(zip(nodes_in_order, nodes_data))
        if node_data.get("description", "UNKNOWN") is not None and node_data.get("entity_type", "UNKNOWN") is not None
    ]
    nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)
    nodes_may_truncate_list_data = truncate_list_by_token_size(
        nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )
    edges_list_data = []

    for i, (edge_name, edge_data) in enumerate(zip(temopral_edges_in_order, temporal_edges_data)):
        try:
            if isinstance(edge_data.get("description") or {}, dict):
                desc = (edge_data.get("description") or {}).get(edge_name[0], None)
            else:
                desc = json.loads(edge_data.get("description") or {}).get(edge_name[0], None)

            if desc is None:
                continue
            degree = await knowledge_graph_inst.edge_degree(*edge_name[1:])
            edges_list_data.append([
                i,
                edge_name[0],
                edge_name[1],
                edge_name[2],
                desc,
                degree,
            ])
        except Exception as e:
            logger.error(
                f"Failed to process edge {i} with edge_name={edge_name}: {e}， edge_data={edge_data}",
                exc_info=True
            )

    edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)
    edges_may_truncate_list_data = truncate_list_by_token_size(
        edges_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )

    truncated = len(nodes_list_data) > len(nodes_may_truncate_list_data) or len(
        edges_list_data
    ) > len(edges_may_truncate_list_data)

    report_describe = ""
    need_to_use_sub_communities = (
            truncated and len(community["sub_communities"]) and len(already_reports)
    )
    force_to_use_sub_communities = global_config["addon_params"].get(
        "force_to_use_sub_communities", False
    )
    if need_to_use_sub_communities or force_to_use_sub_communities:
        logger.debug(
            f"Community {community['title']} exceeds the limit or you set force_to_use_sub_communities to True, using its sub-communities"
        )
        report_describe, report_size, contain_nodes, contain_edges = (
            _pack_single_community_by_sub_communities(
                community, max_token_size, already_reports
            )
        )
        report_exclude_nodes_list_data = [
            n for n in nodes_list_data if n[1] not in contain_nodes
        ]
        report_include_nodes_list_data = [
            n for n in nodes_list_data if n[1] in contain_nodes
        ]
        report_exclude_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) not in contain_edges
        ]
        report_include_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) in contain_edges
        ]
        nodes_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_nodes_list_data + report_include_nodes_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
        edges_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_edges_list_data + report_include_edges_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
    nodes_describe = list_of_list_to_csv([node_fields] + nodes_may_truncate_list_data)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_may_truncate_list_data)
    return f"""-----Reports-----
```csv
{report_describe}
```
-----Entities-----
```csv
{nodes_describe}
```
-----Relationships-----
```csv
{edges_describe}
```"""


# Helper function: community report json to str
def _community_report_json_to_str(parsed_output: dict) -> str:
    """refer official graphrag: index/graph/extractors/community_reports"""
    title = parsed_output.get("title", "Report")
    summary = parsed_output.get("summary", "")
    findings = parsed_output.get("findings", [])

    def finding_summary(finding: dict):
        if isinstance(finding, str):
            return finding
        return finding.get("summary")

    def finding_explanation(finding: dict):
        if isinstance(finding, str):
            return ""
        return finding.get("explanation")

    report_sections = "\n\n".join(
        f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
    )
    return f"# {title}\n\n{summary}\n\n{report_sections}"


# Main function: building_temporal_hierarchy
async def building_temporal_hierarchy(
        timestamps: List[str],
        temporal_hierarchy_graph_inst: BaseGraphStorage,
        knowledge_graph_inst: BaseGraphStorage
) -> Union[BaseGraphStorage, None]:
    async def _initialize_parent_timestamp_node(node,
                                                hierarchy: List[str]):
        nonlocal temporal_hierarchy_graph_inst
        node_index = hierarchy.index(node['entity_type'].lower()) if node['entity_type'].lower() in hierarchy else -1
        new_parent_nodes = []
        if node_index >= 0:
            parent_hierarchies = hierarchy[:node_index]
            for hierarchy in parent_hierarchies:
                entity_name = get_parent_timestamp_name(node['entity_name'], hierarchy)
                entity_type = enhanced_infer_timestamp_level(entity_name)

                entity_name = clean_str(entity_name.upper())
                entity_type = clean_str(entity_type.upper())
                already_node = await temporal_hierarchy_graph_inst.get_node(entity_name)
                if already_node is None:
                    new_parent_nodes.append(dict(
                        entity_name=f'"{entity_name}"',
                        entity_type=entity_type,
                        instantiation=False,
                    ))
                else:
                    new_parent_nodes.append(dict(
                        entity_name=f'"{entity_name}"',
                        entity_type=already_node['entity_type'],
                        instantiation=already_node['instantiation'],
                    ))

        return new_parent_nodes

    for timestamp in timestamps:
        # Check if temporal entity exists in entity relation graph
        already_timestamp_node = await knowledge_graph_inst.get_node(timestamp)
        if already_timestamp_node:
            # Temporal entity exists in entity relation graph (has relationships)
            normalized_forms = already_timestamp_node.get('normalized_forms', [])
            if normalized_forms:
                logger.info(f"Temporal entity {timestamp} has normalized forms: {normalized_forms}")
                normalized_entity_name = normalized_forms[0]
                node = dict(
                    entity_name=f'"{normalized_entity_name}"',
                    entity_type=already_timestamp_node['entity_type'],
                    instantiation=True,
                )
            else:
                try:
                    normalizer = get_temporal_normalizer()
                    result = normalizer.normalize_temporal_expression(timestamp)
                    if result.normalized_forms:
                        logger.info(f"Temporal entity {timestamp} normalized to: {result.normalized_forms}")
                        normalized_entity_name = result.normalized_forms[0]
                        node = dict(
                            entity_name=f'"{normalized_entity_name}"',
                            entity_type=already_timestamp_node['entity_type'],
                            instantiation=True,
                        )
                    else:
                        node = dict(
                            entity_name=f'"{timestamp}"',
                            entity_type=already_timestamp_node['entity_type'],
                            instantiation=True,
                        )
                except Exception as e:
                    logger.warning(f"Failed to normalize {timestamp}: {e}")
                    node = dict(
                        entity_name=f'"{timestamp}"',
                        entity_type=already_timestamp_node['entity_type'],
                        instantiation=True,
                    )
        else:
            # Temporal entity doesn't exist in entity relation graph
            try:
                normalizer = get_temporal_normalizer()
                result = normalizer.normalize_temporal_expression(timestamp)
                if result and result.normalized_forms:
                    normalized_entity_name = result.normalized_forms[0]
                    entity_type = result.granularity.value
                    node = dict(
                        entity_name=f'"{normalized_entity_name}"',
                        entity_type=entity_type.upper(),
                        instantiation=False,
                    )
                    normalized_forms = result.normalized_forms
                    logger.info(f"Created temporal entity {timestamp} -> {normalized_entity_name} in hierarchy (no relationships in entity graph)")
                else:
                    entity_type = enhanced_infer_timestamp_level(timestamp)
                    node = dict(
                        entity_name=f'"{timestamp}"',
                        entity_type=entity_type.upper(),
                        instantiation=False,
                    )
                    normalized_forms = []
                    logger.info(f"Created temporal entity {timestamp} in hierarchy using fallback (no relationships in entity graph)")
            except Exception as e:
                logger.warning(f"Failed to normalize temporal entity {timestamp}: {e}")
                try:
                    entity_type = enhanced_infer_timestamp_level(timestamp)
                    node = dict(
                        entity_name=f'"{timestamp}"',
                        entity_type=entity_type.upper(),
                        instantiation=False,
                    )
                    normalized_forms = []
                    logger.info(f"Created temporal entity {timestamp} in hierarchy using basic fallback")
                except Exception as fallback_e:
                    logger.warning(f"Failed to infer entity type for {timestamp}: {fallback_e}")
                    continue
        
        # Create nodes for the main entity and its additional normalized forms
        all_nodes_to_create = [node]
        
        # Add additional normalized forms
        if len(normalized_forms) > 1:
            for normalized_form in normalized_forms[1:]:
                try:
                    normalizer = get_temporal_normalizer()
                    result = normalizer.normalize_temporal_expression(normalized_form)
                    if result.normalized_forms:
                        normalized_entity_type = result.granularity.value
                        final_normalized_form = result.normalized_forms[0]
                        normalized_node = dict(
                            entity_name=f'"{final_normalized_form}"',
                            entity_type=normalized_entity_type.upper(),
                            instantiation=False,
                        )
                        all_nodes_to_create.append(normalized_node)
                        logger.info(f"Added additional normalized form {normalized_form} -> {final_normalized_form} to temporal hierarchy")
                    else:
                        normalized_entity_type = enhanced_infer_timestamp_level(normalized_form)
                        normalized_node = dict(
                            entity_name=f'"{normalized_form}"',
                            entity_type=normalized_entity_type.upper(),
                            instantiation=False,
                        )
                        all_nodes_to_create.append(normalized_node)
                        logger.info(f"Added additional normalized form {normalized_form} to temporal hierarchy using fallback")
                except Exception as e:
                    logger.warning(f"Failed to create additional normalized form {normalized_form}: {e}")
                    try:
                        fallback_normalized = enhanced_normalize_timestamp(normalized_form)
                        fallback_entity_type = enhanced_infer_timestamp_level(fallback_normalized)
                        fallback_node = dict(
                            entity_name=f'"{fallback_normalized}"',
                            entity_type=fallback_entity_type.upper(),
                            instantiation=False,
                        )
                        all_nodes_to_create.append(fallback_node)
                        logger.info(f"Added fallback normalized form {fallback_normalized} to temporal hierarchy")
                    except Exception as fallback_e:
                        logger.warning(f"Failed to create fallback normalized form {normalized_form}: {fallback_e}")
                        continue
        
        # Process each node (main entity + normalized forms)
        for current_node in all_nodes_to_create:
            new_parent_nodes = await _initialize_parent_timestamp_node(current_node, PROMPTS['DEFAULT_TEMPORAL_HIERARCHY'])
            
            nodes = new_parent_nodes + [current_node]
            edges = [(p_n, n) for p_n, n in zip(nodes[:-1], nodes[1:])]

            for n in nodes:
                node_data = dict(entity_type=n['entity_type'],
                               instantiation=n['instantiation'])
                
                if n['instantiation']:
                    entity_name = n['entity_name']
                    entity_node = await knowledge_graph_inst.get_node(entity_name)
                    
                    if not entity_node and entity_name.startswith('"') and entity_name.endswith('"'):
                        unquoted_name = entity_name[1:-1]
                        entity_node = await knowledge_graph_inst.get_node(unquoted_name)
                        if entity_node:
                            logger.info(f"Found entity {unquoted_name} in entity relation graph for quoted node {entity_name}")
                    
                    if entity_node and 'source_id' in entity_node:
                        node_data['source_id'] = entity_node['source_id']
                    else:
                        node_data['source_id'] = ""
                        logger.warning(f"Node {n['entity_name']} marked as instantiation=True but not found in entity relation graph")
                else:
                    node_data['source_id'] = ""
                
                await temporal_hierarchy_graph_inst.upsert_node(
                    n['entity_name'],
                    node_data=node_data
                )

            for e in edges:
                await temporal_hierarchy_graph_inst.upsert_edge(
                    e[0]['entity_name'],
                    e[1]['entity_name'],
                    edge_data=dict(description=f"{e[0]['entity_type']}->{e[1]['entity_type']}")
                )

    return temporal_hierarchy_graph_inst


# Main function: generate_community_report
async def generate_community_report(
        community_report_kv: BaseKVStorage[CommunitySchema],
        knwoledge_graph_inst: BaseGraphStorage,
        global_config: dict,
):
    llm_extra_kwargs = global_config["special_community_report_llm_kwargs"]
    use_llm_func: callable = global_config["best_model_func"]
    use_string_json_convert_func: callable = global_config[
        "convert_response_to_json_func"
    ]

    community_report_prompt = PROMPTS["temporal_community_report"]

    communities_schema = await knwoledge_graph_inst.community_schema()
    community_keys, community_values = list(communities_schema.keys()), list(
        communities_schema.values()
    )
    already_processed = 0

    async def _form_single_community_report(
            community: SingleCommunitySchema, already_reports: dict[str, CommunitySchema]
    ):
        nonlocal already_processed
        describe = await _pack_single_community_describe(
            knwoledge_graph_inst,
            community,
            max_token_size=global_config["best_model_max_token_size"],
            already_reports=already_reports,
            global_config=global_config,
        )
        logger.info(f"[DEBUG - _form_single_community_report] describe: {describe}")
        prompt = community_report_prompt.format(input_text=describe)
        
        response = await use_llm_func(prompt, **llm_extra_kwargs)
        logger.info(f"[DEBUG - _form_single_community_report] response: {response}")
        response = response[0]
        data = use_string_json_convert_func(response)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
            ]
        print(
            f"{now_ticks} Processed {already_processed} communities\r",
            end="",
            flush=True,
        )
        return data

    levels = sorted(set([c["level"] for c in community_values]), reverse=True)
    logger.info(f"Generating by levels: {levels}")
    community_datas = {}
    for level in levels:
        this_level_community_keys, this_level_community_values = zip(
            *[
                (k, v)
                for k, v in zip(community_keys, community_values)
                if v["level"] == level
            ]
        )
        this_level_communities_reports = await asyncio.gather(
            *[
                _form_single_community_report(c, community_datas)
                for c in this_level_community_values
            ]
        )
        community_datas.update(
            {
                k: {
                    "report_string": _community_report_json_to_str(r),
                    "report_json": r,
                    **v,
                }
                for k, r, v in zip(
                this_level_community_keys,
                this_level_communities_reports,
                this_level_community_values,
            )
            }
        )
    
    await community_report_kv.upsert(community_datas)


# Main function: generate_temporal_report
async def generate_temporal_report(
        community_report_kv: BaseKVStorage[CommunitySchema],
        temporal_hierarchy_graph_inst: BaseGraphStorage,
        knowledge_graph_inst: BaseGraphStorage,
        global_config: dict,
):
    llm_extra_kwargs = global_config["special_community_report_llm_kwargs"]
    use_llm_func: callable = global_config["best_model_func"]
    use_string_json_convert_func: callable = global_config[
        "convert_response_to_json_func"
    ]

    community_report_prompt = PROMPTS["community_report"]

    communities_schema = await temporal_hierarchy_graph_inst.temporal_hierarchy(
        entity_relation_graph_inst=knowledge_graph_inst)
    community_keys, community_values = list(communities_schema.keys()), list(
        communities_schema.values()
    )
    already_processed = 0

    async def _form_single_timestamp_report(
            community: SingleTemporalSchema, already_reports: dict[str, CommunitySchema]
    ):
        nonlocal already_processed
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                describe = await _pack_single_timestamp_describe(
                    knowledge_graph_inst,
                    community,
                    max_token_size=global_config["best_model_max_token_size"],
                    already_reports=already_reports,
                    global_config=global_config,
                )
                prompt = community_report_prompt.format(input_text=describe)
                
                response = await use_llm_func(prompt, response_format={'type': 'json_object'})
                
                if isinstance(response, (list, tuple)) and len(response) > 0:
                    response = response[0]
                elif not isinstance(response, str):
                    response = str(response)
                
                if not response or not response.strip():
                    raise ValueError("Empty response from LLM")
                
                data = use_string_json_convert_func(response)
                
                if not isinstance(data, dict):
                    raise ValueError("Parsed data is not a dictionary")
                
                required_fields = ["title", "summary", "rating", "rating_explanation", "findings"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    logger.warning(f"Missing required fields in community report: {missing_fields}")
                    if "title" not in data:
                        data["title"] = f"Community Report for {community.get('name', 'Unknown')}"
                    if "summary" not in data:
                        data["summary"] = "Summary not available"
                    if "rating" not in data:
                        data["rating"] = 0.0
                    if "rating_explanation" not in data:
                        data["rating_explanation"] = "Rating not available"
                    if "findings" not in data:
                        data["findings"] = []
                
                if not isinstance(data.get("rating"), (int, float)):
                    data["rating"] = 0.0
                
                if not isinstance(data.get("findings"), list):
                    data["findings"] = []
                
                already_processed += 1
                now_ticks = PROMPTS["process_tickers"][
                    already_processed % len(PROMPTS["process_tickers"])
                    ]
                print(
                    f"{now_ticks} Processed {already_processed} communities\r",
                    end="",
                    flush=True,
                )
                return data
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"Attempt {retry_count} failed for community report: {str(e)}")
                
                if retry_count >= max_retries:
                    logger.error(f"Failed to generate community report after {max_retries} attempts")
                    return {
                        "title": f"Error Report for {community.get('name', 'Unknown')}",
                        "summary": f"Failed to generate report: {str(e)}",
                        "rating": 0.0,
                        "rating_explanation": "Report generation failed",
                        "findings": []
                    }
                
                await asyncio.sleep(1)

    levels = sorted(set([c["level"] for c in community_values]), reverse=True)
    logger.info(f"Generating by levels: {levels}")
    community_datas = {}
    for level in levels:
        this_level_community_keys, this_level_community_values = zip(
            *[
                (k, v)
                for k, v in zip(community_keys, community_values)
                if v["level"] == level
            ]
        )
        this_level_communities_reports = await asyncio.gather(
            *[
                _form_single_timestamp_report(c, community_datas)
                for c in this_level_community_values
            ]
        )
        logger.info(f"[DEBUG - generate_temporal_report] this_level_communities_reports: {this_level_communities_reports}")
        community_datas.update(
            {
                k: {
                    "report_string": _community_report_json_to_str(r),
                    "report_json": r,
                    **v,
                }
                for k, r, v in zip(
                this_level_community_keys,
                this_level_communities_reports,
                this_level_community_values,
            )
            }
        )
    print()  # clear the progress bar
    await community_report_kv.upsert(community_datas)


# Export aliases for backward compatibility
merge_nodes_then_upsert = _merge_nodes_then_upsert
merge_edges_then_upsert = _merge_edges_then_upsert
merge_temporal_edges_then_upsert = _merge_temporal_edges_then_upsert
handle_entity_relation_summary = _handle_entity_relation_summary
handle_single_entity_extraction = _handle_single_entity_extraction
handle_single_timestamp_extraction = _handle_single_timestamp_extraction
handle_single_temporal_relationship_extraction = _handle_single_temporal_relationship_extraction
handle_flexible_relationship_extraction = _handle_flexible_relationship_extraction
find_timestamp_in_hierarchy = _find_timestamp_in_hierarchy
pack_single_community_by_sub_communities = _pack_single_community_by_sub_communities
pack_single_community_describe = _pack_single_community_describe
pack_single_timestamp_describe = _pack_single_timestamp_describe
community_report_json_to_str = _community_report_json_to_str

__all__ = [
    "extract_entities",
    "building_temporal_hierarchy",
    "generate_community_report",
    "generate_temporal_report",
    "merge_nodes_then_upsert",
    "merge_edges_then_upsert",
    "merge_temporal_edges_then_upsert",
    "handle_entity_relation_summary",
    "handle_single_entity_extraction",
    "handle_single_timestamp_extraction",
    "handle_single_temporal_relationship_extraction",
    "handle_flexible_relationship_extraction",
    "find_timestamp_in_hierarchy",
    "pack_single_community_by_sub_communities",
    "pack_single_community_describe",
    "pack_single_timestamp_describe",
    "community_report_json_to_str",
    "GRAPH_FIELD_SEP",
    "PROMPTS",
]
