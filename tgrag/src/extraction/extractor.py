"""
DSPy-based entity extraction (UNUSED by default).

NOTE: This module is NOT used by the default extraction pipeline. The default
entity extraction uses the LLM-based `extract_entities` function from
`tgrag.src.core.building`, not the DSPy-based approach implemented here.

This module provides:
- `extract_entities_dspy`: DSPy-based entity extraction alternative
- `generate_dataset`: Utility for generating DSPy training datasets

To use these functions, you would need to set:
  entity_extraction_func = extract_entities_dspy

However, this is not recommended as the DSPy-based approach does not return
temporal hierarchy node names, which are required for building temporal graphs.
"""

from typing import Union
import pickle
import asyncio
from openai import BadRequestError
from collections import defaultdict
import dspy

from ..storage.base import (
    BaseGraphStorage,
    BaseVectorStorage,
)
from ..core.types import TextChunkSchema
from ..config.prompts import get_prompt_manager
from ..utils.helpers import logger
from ..utils.hashing import compute_mdhash_id
from .module import TypedEntityRelationshipExtractor
from ..core.building import _merge_edges_then_upsert, _merge_nodes_then_upsert

# Get prompts for process_tickers
_prompt_manager = None
def _get_prompts():
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = get_prompt_manager()
    return _prompt_manager.prompts

PROMPTS = _get_prompts()  # Get the actual prompts dict


async def generate_dataset(
    chunks: dict[str, TextChunkSchema],
    filepath: str,
    save_dataset: bool = True,
    global_config: dict = {},
) -> list[dspy.Example]:
    """
    Generate a DSPy training dataset from chunks.
    
    NOTE: This function is unused by the default pipeline.
    
    Args:
        chunks: Dictionary of text chunks to process
        filepath: Path to save the dataset (pickle format)
        save_dataset: Whether to save the dataset to disk
        global_config: Configuration dictionary
        
    Returns:
        List of dspy.Example objects
    """
    entity_extractor = TypedEntityRelationshipExtractor(num_refine_turns=1, self_refine=True)

    if global_config.get("use_compiled_dspy_entity_relationship", False):
        entity_extractor.load(global_config["entity_relationship_module_path"])

    ordered_chunks = list(chunks.items())
    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(
        chunk_key_dp: tuple[str, TextChunkSchema]
    ) -> dspy.Example:
        nonlocal already_processed, already_entities, already_relations
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        try:
            prediction = await asyncio.to_thread(entity_extractor, input_text=content)
            entities, relationships = prediction.entities, prediction.relationships
        except BadRequestError as e:
            logger.error(f"Error in TypedEntityRelationshipExtractor: {e}")
            entities, relationships = [], []
        example = dspy.Example(
            input_text=content, entities=entities, relationships=relationships
        ).with_inputs("input_text")
        already_entities += len(entities)
        already_relations += len(relationships)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return example

    examples = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    filtered_examples = [
        example
        for example in examples
        if len(example.entities) > 0 and len(example.relationships) > 0
    ]
    num_filtered_examples = len(examples) - len(filtered_examples)
    if save_dataset:
        with open(filepath, "wb") as f:
            pickle.dump(filtered_examples, f)
            logger.info(
                f"Saved {len(filtered_examples)} examples with keys: {filtered_examples[0].keys()}, filtered {num_filtered_examples} examples"
            )

    return filtered_examples


async def extract_entities_dspy(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    """
    Extract entities using DSPy framework (UNUSED by default).
    
    NOTE: This function is NOT used by the default extraction pipeline. The default
    entity extraction uses the LLM-based `extract_entities` function from
    `tgrag.src.core.building`, which handles temporal relationships and returns
    hierarchy node names needed for building temporal graphs.
    
    This DSPy-based alternative does not return temporal hierarchy node names,
    so it cannot be used with the default temporal graph building pipeline.
    
    To use this function, you would need to:
    1. Set entity_extraction_func = extract_entities_dspy
    2. Modify the pipeline to handle the missing hierarchy node names
    3. Ensure DSPy is properly configured
    
    Args:
        chunks: Dictionary of text chunks to process
        knwoledge_graph_inst: Graph storage instance
        entity_vdb: Vector database for entities
        global_config: Configuration dictionary
        
    Returns:
        Updated graph storage instance, or None if no entities extracted
    """
    entity_extractor = TypedEntityRelationshipExtractor(num_refine_turns=1, self_refine=True)

    if global_config.get("use_compiled_dspy_entity_relationship", False):
        entity_extractor.load(global_config["entity_relationship_module_path"])

    ordered_chunks = list(chunks.items())
    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        try:
            prediction = await asyncio.to_thread(entity_extractor, input_text=content)
            entities, relationships = prediction.entities, prediction.relationships
        except BadRequestError as e:
            logger.error(f"Error in TypedEntityRelationshipExtractor: {e}")
            entities, relationships = [], []

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        for entity in entities:
            entity["source_id"] = chunk_key
            maybe_nodes[entity["entity_name"]].append(entity)
            already_entities += 1

        for relationship in relationships:
            relationship["source_id"] = chunk_key
            maybe_edges[(relationship["src_id"], relationship["tgt_id"])].append(
                relationship
            )
            already_relations += 1

        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[k].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + " " + dp.get("description", ""),
                "entity_name": dp["entity_name"],
                "description": dp.get("description", ""),
                "entity_type": dp.get("entity_type", ""),
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    return knwoledge_graph_inst

__all__ = ["generate_dataset", "extract_entities_dspy"]

