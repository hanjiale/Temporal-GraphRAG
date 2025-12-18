"""
Entity extraction modules (DSPy-based - UNUSED by default).

NOTE: This module contains DSPy-based entity extraction implementations that are
NOT used by the default extraction pipeline. The default entity extraction uses
the LLM-based `extract_entities` function from `tgrag.src.core.building`.

These DSPy-based functions are provided as alternatives but are not recommended
for use with the default temporal graph building pipeline because they do not
return temporal hierarchy node names.

To use these functions, you would need to:
1. Set entity_extraction_func = extract_entities_dspy
2. Modify the pipeline to handle missing hierarchy node names
3. Ensure DSPy is properly configured
"""

from .extractor import generate_dataset, extract_entities_dspy
from .module import (
    Entity,
    Relationship,
    TypedEntityRelationshipExtractor,
    ENTITY_TYPES,
)
from .metric import relationships_similarity_metric, entity_recall_metric

__all__ = [
    # Extractors (unused by default)
    "generate_dataset",
    "extract_entities_dspy",
    # Models
    "Entity",
    "Relationship",
    "TypedEntityRelationshipExtractor",
    "ENTITY_TYPES",
    # Metrics (unused by default)
    "relationships_similarity_metric",
    "entity_recall_metric",
]
