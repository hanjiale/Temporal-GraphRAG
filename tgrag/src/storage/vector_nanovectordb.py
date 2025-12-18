"""
NanoVectorDB-based vector storage implementation.

This module provides a vector database storage backend that keeps
embeddings as numeric arrays for better performance compared to base64 encoding.
"""

import asyncio
import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any

# Import logger from new utils module
from ..utils.logging import get_logger

logger = get_logger(__name__)
from .base import BaseVectorStorage


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    """NanoVectorDB-based vector storage that keeps embeddings as numeric arrays.
    
    This implementation stores embeddings as numeric arrays instead of base64 strings
    for better performance. It's a drop-in replacement for the original NanoVectorDB
    implementation but with improved efficiency.
    """
    cosine_better_than_threshold: float = 0.2  # Keep at 0.2 for quality
    
    def __post_init__(self):
        if self.global_config.get('enable_entity_retrieval', False):
            self._storage_file = os.path.join(
                self.global_config["working_dir"], f"vdb_{self.namespace}_new.json"
            )
        else:
            self._storage_file = os.path.join(
                self.global_config["working_dir"], f"vdb_{self.namespace}.json"
            )
        
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._data: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        
        # Load existing data if file exists
        if os.path.exists(self._storage_file):
            with open(self._storage_file, 'r') as f:
                data = json.load(f)
                self._data = data.get('data', {})
                # Convert stored embeddings back to numpy arrays
                embeddings_data = data.get('embeddings', {})
                for key, embedding_list in embeddings_data.items():
                    self._embeddings[key] = np.array(embedding_list, dtype=np.float32)
        
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", self.cosine_better_than_threshold
        )
    
    def _enhance_query_for_semantic_matching(self, query: str) -> str:
        """
        Enhance query with additional context for better semantic matching.
        """
        enhanced_parts = [query]
        
        # Add financial context if query mentions financial terms
        financial_terms = ['earnings', 'revenue', 'profit', 'loss', 'growth', 'decline', 'quarter', 'annual']
        if any(term in query.lower() for term in financial_terms):
            enhanced_parts.append("financial performance metrics business data")
        
        # Add temporal context if query mentions time
        temporal_terms = ['when', 'time', 'period', 'year', 'quarter', 'month', 'date']
        if any(term in query.lower() for term in temporal_terms):
            enhanced_parts.append("temporal information time period historical data")
        
        # Add company context if query mentions companies
        company_indicators = ['company', 'corporation', 'inc', 'ltd', 'corp', 'ceo', 'executive']
        if any(indicator in query.lower() for indicator in company_indicators):
            enhanced_parts.append("company organization business entity corporate information")
        
        # Add event context if query mentions events
        event_terms = ['event', 'announcement', 'acquisition', 'merger', 'bankruptcy', 'crisis']
        if any(term in query.lower() for term in event_terms):
            enhanced_parts.append("business events announcements corporate developments")
        
        return " ".join(enhanced_parts)

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        
        logger.info(f"Processing {len(data)} items in batches of {self._max_batch_size}...")
        
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        
        logger.info(f"Generating embeddings for {len(batches)} batches...")
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        logger.info(f"Completed embedding generation for {len(embeddings_list)} batches")
        embeddings = np.concatenate(embeddings_list)
        
        # Store data and embeddings separately
        for i, d in enumerate(list_data):
            doc_id = d["__id__"]
            self._data[doc_id] = d
            self._embeddings[doc_id] = embeddings[i]
        
        return list_data

    async def query(self, query: str, top_k=5):
        if not self._embeddings:
            return []
        
        # Enhance query with context for better semantic matching
        enhanced_query = self._enhance_query_for_semantic_matching(query)
        
        query_embedding = await self.embedding_func([enhanced_query])
        query_embedding = query_embedding[0]
        
        # Calculate cosine similarities for ALL documents
        similarities = []
        for doc_id, embedding in self._embeddings.items():
            # Normalize vectors for cosine similarity
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norm = embedding / np.linalg.norm(embedding)
            similarity = np.dot(query_norm, doc_norm)
            
            # Store all similarities for better ranking
            similarities.append((doc_id, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Apply threshold after sorting to get best matches
        filtered_similarities = [
            (doc_id, sim) for doc_id, sim in similarities 
            if sim >= self.cosine_better_than_threshold
        ]
        
        # If threshold is too restrictive, include some lower similarity results
        if len(filtered_similarities) < 2 and len(similarities) > 0:
            # Include top results even if below threshold, but mark them
            additional_results = similarities[:min(3, len(similarities))]
            for doc_id, sim in additional_results:
                if sim not in [s[1] for s in filtered_similarities]:
                    filtered_similarities.append((doc_id, sim))
        
        # Get top_k results
        top_results = filtered_similarities[:top_k]
        
        results = []
        for doc_id, similarity in top_results:
            doc_data = self._data.get(doc_id, {})
            # Mark if result was below threshold
            below_threshold = similarity < self.cosine_better_than_threshold
            results.append({
                **doc_data,
                "id": doc_id,
                "distance": 1 - similarity,  # Convert similarity to distance
                "similarity": similarity,
                "below_threshold": below_threshold
            })
        
        return results

    async def temporal_query(self, query: str, sub_graph_entities: list[str], top_k=5):
        """
        Query with temporal filtering based on temporal context.
        This method filters entities based on their temporal relationships and source context,
        specifically checking if the entity's temporal information matches the query timestamp.
        
        Args:
            query: The query string
            sub_graph_entities: List of entity names that are temporally relevant (contains timestamps)
            top_k: Maximum number of results to return
            
        Returns:
            List of filtered results with temporal context
        """
        
        # First try temporal query
        temporal_results = await self._temporal_query_impl(query, sub_graph_entities, top_k)
        
        # If temporal query doesn't return enough results, fall back to general query
        if len(temporal_results) < min(2, top_k):
            logger.info(f"Temporal query returned only {len(temporal_results)} results, falling back to general query")
            general_results = await self.query(query, top_k)
            
            # Combine and deduplicate results
            combined_results = temporal_results.copy()
            seen_entities = {r.get('entity_name', '') for r in temporal_results}
            
            for result in general_results:
                if result.get('entity_name', '') not in seen_entities:
                    result['retrieval_method'] = 'general_fallback'
                    combined_results.append(result)
                    seen_entities.add(result.get('entity_name', ''))
                    if len(combined_results) >= top_k:
                        break
            
            return combined_results[:top_k]
        
        return temporal_results

    async def _temporal_query_impl(self, query: str, sub_graph_entities: list[str], top_k=5):
        """
        Internal implementation of temporal query.
        """
        if not self._embeddings:
            return []
        
        # Enhance query with context for better semantic matching
        enhanced_query = self._enhance_query_for_semantic_matching(query)
        
        query_embedding = await self.embedding_func([enhanced_query])
        query_embedding = query_embedding[0]
        
        # Extract timestamps from sub_graph_entities (these should be the temporal context)
        target_timestamps = []
        for entity in sub_graph_entities:
            # Check if this entity is a timestamp (YEAR, QUARTER, MONTH, DATE)
            if any(temporal_indicator in entity.upper() for temporal_indicator in 
                   ['2020', '2021', '2022', '2023', '2024', 'Q1', 'Q2', 'Q3', 'Q4']):
                target_timestamps.append(entity)
        
        # If no explicit timestamps found, be more lenient and include all entities
        if not target_timestamps:
            logger.info("No explicit timestamps found in sub_graph_entities, including all entities for temporal query")
            target_timestamps = sub_graph_entities
        else:
            # Also include entities that might be temporally related
            logger.info(f"Found explicit timestamps: {target_timestamps}")
            # Add entities that might be temporally related based on context
            for entity in sub_graph_entities:
                if entity not in target_timestamps:
                    # Check if entity might be temporally related
                    if any(ts.lower() in entity.lower() for ts in ['year', 'quarter', 'month', 'date', 'time']):
                        target_timestamps.append(entity)
                        logger.debug(f"Enhanced temporal entity: {entity}")
        
        logger.info(f"Temporal query: Looking for entities with timestamps: {target_timestamps}")
        logger.info(f"Enhanced query: {self._enhance_query_for_semantic_matching(query)}")
        
        # FIRST: Calculate similarities for all documents
        similarities = []
        for doc_id, embedding in self._embeddings.items():
            doc_data = self._data.get(doc_id, {})
            entity_name = doc_data.get('entity_name', '')
            description = doc_data.get('description', '')
            source_id = doc_data.get('source_id', '')
            
            # Check if this entity is temporally relevant
            is_temporally_relevant = False
            
            # Method 1: Check if entity name matches any target timestamp
            if entity_name in target_timestamps:
                is_temporally_relevant = True
                logger.debug(f"Entity {entity_name} matches target timestamp")
            
            # Method 2: Check if entity is in the temporal subgraph (meaning it's related to the target timestamps)
            elif entity_name in sub_graph_entities:
                is_temporally_relevant = True
                logger.debug(f"Entity {entity_name} is in temporal subgraph")
            
            # Method 3: Check if description contains temporal information that matches target timestamps
            elif description and target_timestamps:
                description_upper = description.upper()
                for timestamp in target_timestamps:
                    if timestamp.upper() in description_upper:
                        is_temporally_relevant = True
                        logger.debug(f"Entity {entity_name} description contains timestamp {timestamp}")
                        break
            
            # Method 4: Check if source_id contains temporal information
            elif source_id and target_timestamps:
                source_id_upper = source_id.upper()
                for timestamp in target_timestamps:
                    if timestamp.upper() in source_id_upper:
                        is_temporally_relevant = True
                        logger.debug(f"Entity {entity_name} source_id contains timestamp {timestamp}")
                        break
            
            # Method 5: Check for semantic temporal relevance (e.g., financial terms, events)
            elif description:
                description_lower = description.lower()
                temporal_indicators = [
                    'earnings', 'revenue', 'quarter', 'annual', 'monthly', 'yearly',
                    'financial', 'performance', 'growth', 'decline', 'increase', 'decrease',
                    'report', 'announcement', 'acquisition', 'merger', 'bankruptcy',
                    'crisis', 'recovery', 'expansion', 'contraction'
                ]
                if any(indicator in description_lower for indicator in temporal_indicators):
                    is_temporally_relevant = True
                    logger.debug(f"Entity {entity_name} has semantic temporal relevance")
            
            # Method 6: Check entity type for temporal relevance
            entity_type = doc_data.get('entity_type', '')
            if entity_type and entity_type.lower() in ['date', 'time', 'quarter', 'month', 'year', 'event', 'metric']:
                is_temporally_relevant = True
                logger.debug(f"Entity {entity_name} has temporal entity type: {entity_type}")
            
            # Only process temporally relevant entities
            if is_temporally_relevant:
                # Normalize vectors for cosine similarity
                query_norm = query_embedding / np.linalg.norm(query_embedding)
                doc_norm = embedding / np.linalg.norm(embedding)
                similarity = np.dot(query_norm, doc_norm)
                
                if similarity >= self.cosine_better_than_threshold:
                    similarities.append((doc_id, similarity))
        
        # SECOND: Sort by similarity and get top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # THIRD: Build results with temporal context
        results = []
        for doc_id, similarity in top_results:
            doc_data = self._data.get(doc_id, {})
            results.append({
                **doc_data,
                "id": doc_id,
                "distance": 1 - similarity,
                "similarity": similarity,
                "retrieval_method": "temporal"
            })
        
        logger.info(f"Temporal query: Found {len(similarities)} temporally relevant entities, returned {len(results)} results")
        return results

    async def index_done_callback(self):
        # Save data and embeddings as numeric arrays
        save_data = {
            'data': self._data,
            'embeddings': {k: v.tolist() for k, v in self._embeddings.items()}
        }
        
        with open(self._storage_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Saved {len(self._embeddings)} embeddings to {self._storage_file}") 