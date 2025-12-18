"""
DSPy metrics for evaluating entity extraction (UNUSED by default).

NOTE: This module is NOT used by the default extraction pipeline. The default
entity extraction uses the LLM-based approach in `tgrag.src.core.building`,
not the DSPy-based approach that these metrics are designed to evaluate.

This module provides:
- `relationships_similarity_metric`: Evaluates similarity between gold and predicted relationships
- `entity_recall_metric`: Calculates entity recall score
"""

import dspy
from .module import Relationship


class AssessRelationships(dspy.Signature):
    """
    Assess the similarity between gold and predicted relationships:
    1. Match relationships based on src_id and tgt_id pairs, allowing for slight variations in entity names.
    2. For matched pairs, compare:
       a) Description similarity (semantic meaning)
       b) Weight similarity
       c) Order similarity
    3. Consider unmatched relationships as penalties.
    4. Aggregate scores, accounting for precision and recall.
    5. Return a final similarity score between 0 (no similarity) and 1 (perfect match).

    Key considerations:
    - Prioritize matching based on entity pairs over exact string matches.
    - Use semantic similarity for descriptions rather than exact matches.
    - Weight the importance of different aspects (e.g., entity matching, description, weight, order).
    - Balance the impact of matched and unmatched relationships in the final score.
    """

    gold_relationships: list[Relationship] = dspy.InputField(
        desc="The gold-standard relationships to compare against."
    )
    predicted_relationships: list[Relationship] = dspy.InputField(
        desc="The predicted relationships to compare against the gold-standard relationships."
    )
    similarity_score: float = dspy.OutputField(
        desc="Similarity score between 0 and 1, with 1 being the highest similarity."
    )


def relationships_similarity_metric(
    gold: dspy.Example, pred: dspy.Prediction, trace=None
) -> float:
    """
    Calculate similarity metric between gold and predicted relationships.
    
    NOTE: This function is unused by the default extraction pipeline.
    """
    model = dspy.TypedChainOfThought(AssessRelationships)
    gold_relationships = [Relationship(**item) for item in gold["relationships"]]
    predicted_relationships = [Relationship(**item) for item in pred["relationships"]]
    similarity_score = float(
        model(
            gold_relationships=gold_relationships,
            predicted_relationships=predicted_relationships,
        ).similarity_score
    )
    return similarity_score


def entity_recall_metric(
    gold: dspy.Example, pred: dspy.Prediction, trace=None
) -> float:
    """
    Calculate entity recall metric.
    
    NOTE: This function is unused by the default extraction pipeline.
    
    Args:
        gold: Gold-standard example with entities
        pred: Predicted extraction with entities
        trace: Optional trace (unused)
        
    Returns:
        Recall score between 0 and 1
    """
    true_set = set(item["entity_name"] for item in gold["entities"])
    pred_set = set(item["entity_name"] for item in pred["entities"])
    true_positives = len(pred_set.intersection(true_set))
    false_negatives = len(true_set - pred_set)
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    return recall

__all__ = ["relationships_similarity_metric", "entity_recall_metric", "AssessRelationships"]

