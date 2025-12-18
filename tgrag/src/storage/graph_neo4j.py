"""
Neo4j-based graph storage implementation with connection pooling and retry logic.

This module provides a graph database storage backend using Neo4j,
suitable for large-scale graph operations and production deployments.
"""

import json
import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Union, Callable, Any

# Optional Neo4j dependency - handle gracefully if not installed
try:
    from neo4j import AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    AsyncGraphDatabase = None  # type: ignore

from .base import BaseGraphStorage
from ..core.types import SingleCommunitySchema
# Import logger from new utils module
from ..utils.logging import get_logger
# Import prompts from new config module
from ..config.prompts import PromptManager

logger = get_logger(__name__)
_prompt_manager = PromptManager()
GRAPH_FIELD_SEP = _prompt_manager.prompts.get('GRAPH_FIELD_SEP', '<SEP>')

neo4j_lock = asyncio.Lock()


def make_path_idable(path):
    return path.replace(".", "_").replace("/", "__").replace("-", "_")


@dataclass
class Neo4jStorage(BaseGraphStorage):
    """Neo4j graph storage with connection pooling and retry logic.
    
    Attributes:
        max_retries: Maximum number of retry attempts for failed operations (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 1.0)
        retry_backoff: Exponential backoff multiplier (default: 2.0)
        max_connection_pool_size: Maximum connection pool size (default: 50)
        connection_timeout: Connection timeout in seconds (default: 30.0)
    """
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    max_connection_pool_size: int = 50
    connection_timeout: float = 30.0
    
    def __post_init__(self):
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "Neo4j driver is not installed. Install it with: pip install neo4j"
            )
        
        self.neo4j_url = self.global_config["addon_params"].get("neo4j_url", None)
        self.neo4j_auth = self.global_config["addon_params"].get("neo4j_auth", None)
        self.namespace = (
            f"{make_path_idable(self.global_config['working_dir'])}__{self.namespace}"
        )
        logger.info(f"Using the label {self.namespace} for Neo4j as identifier")
        if self.neo4j_url is None or self.neo4j_auth is None:
            raise ValueError("Missing neo4j_url or neo4j_auth in addon_params")
        
        # Configure connection pooling
        self.async_driver = AsyncGraphDatabase.driver(
            self.neo4j_url,
            auth=self.neo4j_auth,
            max_connection_pool_size=self.max_connection_pool_size,
            connection_timeout=self.connection_timeout,
        )
    
    async def _execute_with_retry(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute a database operation with exponential backoff retry logic.
        
        Args:
            operation: Async function to execute
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Neo4j operation failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                    delay *= self.retry_backoff
                else:
                    logger.error(f"Neo4j operation failed after {self.max_retries} attempts: {e}")
        
        raise last_exception

    # async def create_database(self):
    #     async with self.async_driver.session() as session:
    #         try:
    #             constraints = await session.run("SHOW CONSTRAINTS")
    #             # TODO I don't know why CREATE CONSTRAINT IF NOT EXISTS still trigger error
    #             # so have to check if the constrain exists
    #             constrain_exists = False

    #             async for record in constraints:
    #                 if (
    #                     self.namespace in record["labelsOrTypes"]
    #                     and "id" in record["properties"]
    #                     and record["type"] == "UNIQUENESS"
    #                 ):
    #                     constrain_exists = True
    #                     break
    #             if not constrain_exists:
    #                 await session.run(
    #                     f"CREATE CONSTRAINT FOR (n:{self.namespace}) REQUIRE n.id IS UNIQUE"
    #                 )
    #                 logger.info(f"Add constraint for namespace: {self.namespace}")

    #         except Exception as e:
    #             logger.error(f"Error accessing or setting up the database: {str(e)}")
    #             raise

    async def _init_workspace(self):
        await self.async_driver.verify_authentication()
        await self.async_driver.verify_connectivity()
        # TODOLater: create database if not exists always cause an error when async
        # await self.create_database()

    async def index_start_callback(self):
        logger.info("Init Neo4j workspace")
        await self._init_workspace()

    async def has_node(self, node_id: str) -> bool:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:{self.namespace}) WHERE n.id = $node_id RETURN COUNT(n) > 0 AS exists",
                node_id=node_id,
            )
            record = await result.single()
            return record["exists"] if record else False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]->(t:{self.namespace}) "
                "WHERE s.id = $source_id AND t.id = $target_id "
                "RETURN COUNT(r) > 0 AS exists",
                source_id=source_node_id,
                target_id=target_node_id,
            )
            record = await result.single()
            return record["exists"] if record else False

    async def node_degree(self, node_id: str) -> int:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:{self.namespace}) WHERE n.id = $node_id "
                f"RETURN COUNT {{(n)-[]-(:{self.namespace})}} AS degree",
                node_id=node_id,
            )
            record = await result.single()
            return record["degree"] if record else 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace}), (t:{self.namespace}) "
                "WHERE s.id = $src_id AND t.id = $tgt_id "
                f"RETURN COUNT {{(s)-[]-(:{self.namespace})}} + COUNT {{(t)-[]-(:{self.namespace})}} AS degree",
                src_id=src_id,
                tgt_id=tgt_id,
            )
            record = await result.single()
            return record["degree"] if record else 0

    async def get_node(self, node_id: str) -> Union[dict, None]:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:{self.namespace}) WHERE n.id = $node_id RETURN properties(n) AS node_data",
                node_id=node_id,
            )
            record = await result.single()
            raw_node_data = record["node_data"] if record else None
        if raw_node_data is None:
            return None
        raw_node_data["clusters"] = json.dumps(
            [
                {
                    "level": index,
                    "cluster": cluster_id,
                }
                for index, cluster_id in enumerate(
                    raw_node_data.get("communityIds", [])
                )
            ]
        )
        return raw_node_data

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]->(t:{self.namespace}) "
                "WHERE s.id = $source_id AND t.id = $target_id "
                "RETURN properties(r) AS edge_data",
                source_id=source_node_id,
                target_id=target_node_id,
            )
            record = await result.single()
            return record["edge_data"] if record else None

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]->(t:{self.namespace}) WHERE s.id = $source_id "
                "RETURN s.id AS source, t.id AS target",
                source_id=source_node_id,
            )
            edges = []
            async for record in result:
                edges.append((record["source"], record["target"]))
            return edges

    async def get_temporal_edges(self, source_node_id: str, timestamps: list[str], top_k: int = 20):
        """
        Get temporal edges for a given node and timestamps.
        This method filters edges based on the provided timestamps.
        
        Args:
            source_node_id: The source node ID
            timestamps: List of timestamps to filter by
            top_k: Maximum number of edges to return
            
        Returns:
            List of edge dictionaries with temporal information
        """
        if not timestamps:
            return []
            
        async with self.async_driver.session() as session:
            temporal_edges = []
            
            # Method 1: Check edges with temporal descriptions
            timestamp_conditions = []
            for timestamp in timestamps:
                # Check if timestamp exists in description (as dict or string)
                timestamp_conditions.append(f"ANY(ts IN $timestamps WHERE ts IN keys(r.description) OR r.description CONTAINS ts)")
            
            if timestamp_conditions:
                timestamp_filter = " OR ".join(timestamp_conditions)
                result = await session.run(
                    f"""
                    MATCH (s:{self.namespace})-[r]->(t:{self.namespace}) 
                    WHERE s.id = $source_id AND ({timestamp_filter})
                    RETURN s.id AS source, t.id AS target, properties(r) AS edge_data
                    ORDER BY r.weight DESC, r.order ASC
                    LIMIT $top_k
                    """,
                    source_id=source_node_id,
                    timestamps=timestamps,
                    top_k=top_k
                )
                
                async for record in result:
                    edge_data = record["edge_data"]
                    temporal_edges.append({
                        "src_tgt": (record["source"], record["target"]),
                        "description": edge_data.get("description", {}),
                        "source_id": edge_data.get("source_id", {}),
                        "weight": edge_data.get("weight", 1),
                        "order": edge_data.get("order", 1)
                    })
            
            # Method 2: Check edges connected to timestamp nodes
            for timestamp in timestamps:
                # Check if source node is connected to this timestamp
                result = await session.run(
                    f"""
                    MATCH (s:{self.namespace})-[r]->(t:{self.namespace})
                    WHERE s.id = $source_id AND t.id = $timestamp
                    RETURN s.id AS source, t.id AS target, properties(r) AS edge_data
                    """,
                    source_id=source_node_id,
                    timestamp=timestamp
                )
                
                async for record in result:
                    edge_data = record["edge_data"]
                    temporal_edges.append({
                        "src_tgt": (record["source"], record["target"]),
                        "description": edge_data.get("description", {}),
                        "source_id": edge_data.get("source_id", {}),
                        "weight": edge_data.get("weight", 1),
                        "order": edge_data.get("order", 1)
                    })
                
                # Check if timestamp is connected to source node
                result = await session.run(
                    f"""
                    MATCH (s:{self.namespace})-[r]->(t:{self.namespace})
                    WHERE s.id = $timestamp AND t.id = $source_id
                    RETURN s.id AS source, t.id AS target, properties(r) AS edge_data
                    """,
                    source_id=source_node_id,
                    timestamp=timestamp
                )
                
                async for record in result:
                    edge_data = record["edge_data"]
                    temporal_edges.append({
                        "src_tgt": (record["source"], record["target"]),
                        "description": edge_data.get("description", {}),
                        "source_id": edge_data.get("source_id", {}),
                        "weight": edge_data.get("weight", 1),
                        "order": edge_data.get("order", 1)
                    })
            
            # Method 3: Check edges that connect to temporal entities
            result = await session.run(
                f"""
                MATCH (s:{self.namespace})-[r]->(t:{self.namespace})
                WHERE s.id = $source_id AND t.entity_type IN ['YEAR', 'QUARTER', 'MONTH', 'DATE']
                RETURN s.id AS source, t.id AS target, properties(r) AS edge_data, t.entity_type AS target_type
                ORDER BY r.weight DESC, r.order ASC
                LIMIT $top_k
                """,
                source_id=source_node_id,
                top_k=top_k
            )
            
            async for record in result:
                target = record["target"]
                # Check if this temporal node matches any of our timestamps
                for timestamp in timestamps:
                    if timestamp == target or timestamp in target:
                        edge_data = record["edge_data"]
                        temporal_edges.append({
                            "src_tgt": (record["source"], record["target"]),
                            "description": edge_data.get("description", {}),
                            "source_id": edge_data.get("source_id", {}),
                            "weight": edge_data.get("weight", 1),
                            "order": edge_data.get("order", 1)
                        })
                        break
            
            # Remove duplicates based on src_tgt pairs
            seen_edges = set()
            unique_temporal_edges = []
            for edge in temporal_edges:
                edge_key = tuple(sorted(edge["src_tgt"]))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    unique_temporal_edges.append(edge)
            
            # Sort by weight and order, then limit to top_k
            unique_temporal_edges.sort(key=lambda x: (x.get("weight", 0), x.get("order", 1)), reverse=True)
            return unique_temporal_edges[:top_k]

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        node_type = node_data.get("entity_type", "UNKNOWN").strip('"')
        async with self.async_driver.session() as session:
            await session.run(
                f"MERGE (n:{self.namespace}:{node_type} {{id: $node_id}}) "
                "SET n += $node_data",
                node_id=node_id,
                node_data=node_data,
            )

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        edge_data.setdefault("weight", 0.0)
        async with self.async_driver.session() as session:
            await session.run(
                f"MATCH (s:{self.namespace}), (t:{self.namespace}) "
                "WHERE s.id = $source_id AND t.id = $target_id "
                "MERGE (s)-[r:RELATED]->(t) "  # Added relationship type 'RELATED'
                "SET r += $edge_data",
                source_id=source_node_id,
                target_id=target_node_id,
                edge_data=edge_data,
            )

    async def clustering(self, algorithm: str):
        if algorithm != "leiden":
            raise ValueError(
                f"Clustering algorithm {algorithm} not supported in Neo4j implementation"
            )

        random_seed = self.global_config["graph_cluster_seed"]
        max_level = self.global_config["max_graph_cluster_size"]
        async with self.async_driver.session() as session:
            try:
                # Project the graph with undirected relationships
                await session.run(
                    f"""
                    CALL gds.graph.project(
                        'graph_{self.namespace}',
                        ['{self.namespace}'],
                        {{
                            RELATED: {{
                                orientation: 'UNDIRECTED',
                                properties: ['weight']
                            }}
                        }}
                    )
                    """
                )

                # Run Leiden algorithm
                result = await session.run(
                    f"""
                    CALL gds.leiden.write(
                        'graph_{self.namespace}',
                        {{
                            writeProperty: 'communityIds',
                            includeIntermediateCommunities: True,
                            relationshipWeightProperty: "weight",
                            maxLevels: {max_level},
                            tolerance: 0.0001,
                            gamma: 1.0,
                            theta: 0.01,
                            randomSeed: {random_seed}
                        }}
                    )
                    YIELD communityCount, modularities;
                    """
                )
                result = await result.single()
                community_count: int = result["communityCount"]
                modularities = result["modularities"]
                logger.info(
                    f"Performed graph clustering with {community_count} communities and modularities {modularities}"
                )
            finally:
                # Drop the projected graph
                await session.run(f"CALL gds.graph.drop('graph_{self.namespace}')")

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        results = defaultdict(
            lambda: dict(
                level=None,
                title=None,
                edges=set(),
                nodes=set(),
                chunk_ids=set(),
                occurrence=0.0,
                sub_communities=[],
            )
        )

        async with self.async_driver.session() as session:
            # Fetch community data
            result = await session.run(
                f"""
                MATCH (n:{self.namespace})
                WITH n, n.communityIds AS communityIds, [(n)-[]-(m:{self.namespace}) | m.id] AS connected_nodes
                RETURN n.id AS node_id, n.source_id AS source_id, 
                       communityIds AS cluster_key,
                       connected_nodes
                """
            )

            # records = await result.fetch()

            max_num_ids = 0
            async for record in result:
                for index, c_id in enumerate(record["cluster_key"]):
                    node_id = str(record["node_id"])
                    source_id = record["source_id"]
                    level = index
                    cluster_key = str(c_id)
                    connected_nodes = record["connected_nodes"]

                    results[cluster_key]["level"] = level
                    results[cluster_key]["title"] = f"Cluster {cluster_key}"
                    results[cluster_key]["nodes"].add(node_id)
                    results[cluster_key]["edges"].update(
                        [
                            tuple(sorted([node_id, str(connected)]))
                            for connected in connected_nodes
                            if connected != node_id
                        ]
                    )
                    chunk_ids = source_id.split(GRAPH_FIELD_SEP)
                    results[cluster_key]["chunk_ids"].update(chunk_ids)
                    max_num_ids = max(
                        max_num_ids, len(results[cluster_key]["chunk_ids"])
                    )

            # Process results
            for k, v in results.items():
                v["edges"] = [list(e) for e in v["edges"]]
                v["nodes"] = list(v["nodes"])
                v["chunk_ids"] = list(v["chunk_ids"])
                v["occurrence"] = len(v["chunk_ids"]) / max_num_ids

            # Compute sub-communities (this is a simplified approach)
            for cluster in results.values():
                cluster["sub_communities"] = [
                    sub_key
                    for sub_key, sub_cluster in results.items()
                    if sub_cluster["level"] > cluster["level"]
                    and set(sub_cluster["nodes"]).issubset(set(cluster["nodes"]))
                ]

        return dict(results)

    async def index_done_callback(self):
        await self.async_driver.close()

    async def _debug_delete_all_node_edges(self):
        async with self.async_driver.session() as session:
            try:
                # Delete all relationships in the namespace
                await session.run(f"MATCH (n:{self.namespace})-[r]-() DELETE r")

                # Delete all nodes in the namespace
                await session.run(f"MATCH (n:{self.namespace}) DELETE n")

                logger.info(
                    f"All nodes and edges in namespace '{self.namespace}' have been deleted."
                )
            except Exception as e:
                logger.error(f"Error deleting nodes and edges: {str(e)}")
                raise
