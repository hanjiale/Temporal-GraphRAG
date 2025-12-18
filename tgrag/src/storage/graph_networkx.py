import html
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Union, cast, List
import networkx as nx
import numpy as np

# Import logger from new utils module
from ..utils.logging import get_logger
from .base import BaseGraphStorage
# Import prompts from new config module
from ..config.prompts import PromptManager

logger = get_logger(__name__)
_prompt_manager = PromptManager()
PROMPTS = _prompt_manager.prompts
GRAPH_FIELD_SEP = _prompt_manager.prompts.get('GRAPH_FIELD_SEP', '<SEP>')

from ..core.types import SingleCommunitySchema, SingleTemporalSchema

@dataclass
class NetworkXStorage(BaseGraphStorage):
    is_directed: bool=False
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        import json
        import networkx as nx

        def restore_graphml_attributes(G: nx.Graph):
            for s, t, data in G.edges(data=True):
                for k, v in data.items():
                    if isinstance(v, str):
                        try:
                            parsed = json.loads(v)
                            if isinstance(parsed, dict):
                                G.edges[s, t][k] = parsed
                        except json.JSONDecodeError:
                            pass
            return G

        if os.path.exists(file_name):
            graph = nx.read_graphml(file_name)
            restore_graphml_attributes(graph)
            return graph
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )

        def clean_graphml_attributes(graph: nx.Graph) -> nx.Graph:
            for node_id, data in graph.nodes(data=True):
                for k, v in list(data.items()):
                    if v is None:
                        graph.nodes[node_id][k] = ""  # or delete: del graph.nodes[node_id][k]
                    elif isinstance(v, dict):
                        graph.nodes[node_id][k] = json.dumps(v)
                    elif not isinstance(v, (str, int, float, bool)):
                        graph.nodes[node_id][k] = str(v)

            for s, t, data in graph.edges(data=True):
                for k, v in list(data.items()):
                    if v is None:
                        graph.edges[s, t][k] = ""
                    elif isinstance(v, dict):
                        graph.edges[s, t][k] = json.dumps(v)
                    elif not isinstance(v, (str, int, float, bool)):
                        graph.edges[s, t][k] = str(v)

            return graph
        graph = clean_graphml_attributes(graph)
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        # self._graph = preloaded_graph or nx.Graph()
        self._graph = preloaded_graph or (nx.DiGraph() if self.is_directed else nx.Graph())
        self._clustering_algorithms = {
            "leiden": self._leiden_clustering,
        }

    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        # [numberchiffre]: node_id not part of graph returns `DegreeView({})` instead of 0
        if not self._graph.has_node(node_id):
            return 0
        degree = self._graph.degree(node_id)
        # Handle case where degree() returns DegreeView instead of int
        if hasattr(degree, '__len__'):
            return len(degree)
        return degree

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_degree = self._graph.degree(src_id) if self._graph.has_node(src_id) else 0
        tgt_degree = self._graph.degree(tgt_id) if self._graph.has_node(tgt_id) else 0
        
        # Handle case where degree() returns DegreeView instead of int
        if hasattr(src_degree, '__len__'):
            src_degree = len(src_degree)
        if hasattr(tgt_degree, '__len__'):
            tgt_degree = len(tgt_degree)
            
        return src_degree + tgt_degree

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

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
        if not self._graph.has_node(source_node_id):
            return []
        
        temporal_edges = []
        
        # Method 1: Check edges with temporal descriptions
        all_edges = list(self._graph.edges(source_node_id, data=True))
        for source, target, edge_data in all_edges:
            if edge_data and isinstance(edge_data.get("description"), dict):
                # Edge has temporal description dict
                for timestamp in timestamps:
                    if timestamp in edge_data["description"]:
                        temporal_edges.append({
                            "src_tgt": (source, target),
                            "description": edge_data["description"],
                            "source_id": edge_data.get("source_id", {}),
                            "weight": edge_data.get("weight", 1),
                            "order": edge_data.get("order", 1)
                        })
                        break  # Found a matching timestamp, move to next edge
            elif edge_data and isinstance(edge_data.get("description"), str):
                # Edge has string description, check if any timestamp is mentioned
                description = edge_data.get("description", "").lower()
                for timestamp in timestamps:
                    if timestamp.lower() in description:
                        temporal_edges.append({
                            "src_tgt": (source, target),
                            "description": edge_data["description"],
                            "source_id": edge_data.get("source_id", ""),
                            "weight": edge_data.get("weight", 1),
                            "order": edge_data.get("order", 1)
                        })
                        break  # Found a matching timestamp, move to next edge
        
        # Method 2: Check edges connected to timestamp nodes
        for timestamp in timestamps:
            if self._graph.has_node(timestamp):
                # Check if source node is connected to this timestamp
                if self._graph.has_edge(source_node_id, timestamp):
                    edge_data = self._graph.edges[source_node_id, timestamp]
                    temporal_edges.append({
                        "src_tgt": (source_node_id, timestamp),
                        "description": edge_data.get("description", {}),
                        "source_id": edge_data.get("source_id", {}),
                        "weight": edge_data.get("weight", 1),
                        "order": edge_data.get("order", 1)
                    })
                
                # Check if timestamp is connected to source node
                if self._graph.has_edge(timestamp, source_node_id):
                    edge_data = self._graph.edges[timestamp, source_node_id]
                    temporal_edges.append({
                        "src_tgt": (timestamp, source_node_id),
                        "description": edge_data.get("description", {}),
                        "source_id": edge_data.get("source_id", {}),
                        "weight": edge_data.get("weight", 1),
                        "order": edge_data.get("order", 1)
                    })
        
        # Method 3: Check edges that connect to entities mentioned in temporal context
        # This is a fallback for when temporal information is stored in community reports
        # but not directly in edge descriptions
        for source, target, edge_data in all_edges:
            # Check if the target node is a timestamp
            target_data = self._graph.nodes[target]
            if target_data.get("entity_type", "").upper() in ["YEAR", "QUARTER", "MONTH", "DATE"]:
                # This is a temporal edge, include it if it matches any of our timestamps
                for timestamp in timestamps:
                    if timestamp == target or timestamp in target:
                        temporal_edges.append({
                            "src_tgt": (source, target),
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
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def node_in_degree(self, node_id: str) -> int:
        if not self._graph.has_node(node_id):
            return 0
        return self._graph.in_degree(node_id)

    async def node_out_degree(self, node_id: str) -> int:
        if not self._graph.has_node(node_id):
            return 0
        return self._graph.out_degree(node_id)

    async def clustering(self, algorithm: str):
        if algorithm not in self._clustering_algorithms:
            raise ValueError(f"Clustering algorithm {algorithm} not supported")
        await self._clustering_algorithms[algorithm]()

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
        max_num_ids = 0
        levels = defaultdict(set)
        for node_id, node_data in self._graph.nodes(data=True):
            if "clusters" not in node_data:
                continue
            clusters = json.loads(node_data["clusters"])
            this_node_edges = self._graph.edges(node_id)

            for cluster in clusters:
                level = cluster["level"]
                cluster_key = str(cluster["cluster"])
                levels[level].add(cluster_key)
                results[cluster_key]["level"] = level
                results[cluster_key]["title"] = f"Cluster {cluster_key}"
                results[cluster_key]["nodes"].add(node_id)
                results[cluster_key]["edges"].update(
                    [tuple(sorted(e)) for e in this_node_edges]
                )
                results[cluster_key]["chunk_ids"].update(
                    node_data["source_id"].split(GRAPH_FIELD_SEP)
                )
                max_num_ids = max(max_num_ids, len(results[cluster_key]["chunk_ids"]))

        ordered_levels = sorted(levels.keys())
        for i, curr_level in enumerate(ordered_levels[:-1]):
            next_level = ordered_levels[i + 1]
            this_level_comms = levels[curr_level]
            next_level_comms = levels[next_level]
            # compute the sub-communities by nodes intersection
            for comm in this_level_comms:
                results[comm]["sub_communities"] = [
                    c
                    for c in next_level_comms
                    if results[c]["nodes"].issubset(results[comm]["nodes"])
                ]

        for k, v in results.items():
            v["edges"] = list(v["edges"])
            v["edges"] = [list(e) for e in v["edges"]]
            v["nodes"] = list(v["nodes"])
            v["chunk_ids"] = list(v["chunk_ids"])
            v["occurrence"] = len(v["chunk_ids"]) / max_num_ids
        return dict(results)

    async def temporal_hierarchy(self,
                                 # highest_hierarchy: str, lowest_hierarchy: str,
                                 entity_relation_graph_inst: BaseGraphStorage) -> dict[str, SingleTemporalSchema]:
        results = defaultdict(
            lambda: dict(
                level=None,
                title=None,
                temporal_edges=set(),
                nodes=set(),
                chunk_ids=set(),
                sub_communities=[],
                occurrence=1.0
            )
        )
        async def build_temporal_entity_dict(temporal_graph: BaseGraphStorage, entity_relation_graph: BaseGraphStorage) -> dict:
            entity_results = {}

            for node_id, node_data in temporal_graph.nodes(data=True):
                entity_type = node_data.get("entity_type", "UNKNOWN").lower()
                # Handle unknown entity types by defaulting to UNKNOWN
                if entity_type not in PROMPTS['DEFAULT_TEMPORAL_HIERARCHY_LEVEL']:
                    entity_type = "UNKNOWN"
                level = PROMPTS['DEFAULT_TEMPORAL_HIERARCHY_LEVEL'][entity_type]
                children = list(temporal_graph.successors(node_id))
                entities,edges, chunk_ids = [],[], []
                if node_data.get('instantiation'):
                    node_in_relation_entity_graph = await entity_relation_graph.get_node(node_id)
                    
                    # If not found, try unquoted version (for backward compatibility)
                    if not node_in_relation_entity_graph and node_id.startswith('"') and node_id.endswith('"'):
                        unquoted_name = node_id[1:-1]  # Remove quotes
                        node_in_relation_entity_graph = await entity_relation_graph.get_node(unquoted_name)
                        if node_in_relation_entity_graph:
                            logger.debug(f"Found entity {unquoted_name} in entity relation graph for quoted node {node_id}")
                    
                    if node_in_relation_entity_graph and "source_id" in node_in_relation_entity_graph:
                        chunk_ids = node_in_relation_entity_graph["source_id"].split(GRAPH_FIELD_SEP)
                        # Use the correct entity name for getting neighbors (unquoted if we found it that way)
                        entity_name_for_neighbors = unquoted_name if 'unquoted_name' in locals() and node_in_relation_entity_graph else node_id
                        entities = [
                            n for n in entity_relation_graph._graph.neighbors(entity_name_for_neighbors)
                        ]
                        edges = []
                        entities = list(set(entities))
                        for e1 in entities:
                            edge_list = await entity_relation_graph.get_node_edges(e1)
                            for edge in edge_list:
                                edges.append([entity_name_for_neighbors]+sorted(edge))
                        edges = set([tuple(e) for e in edges])
                    else:
                        # Node doesn't exist in entity relation graph or doesn't have source_id
                        # This can happen with temporal entities that don't have relationships
                        chunk_ids = []
                        entities = []
                        edges = []
                        logger.debug(f"Temporal entity {node_id} not found in entity relation graph or missing source_id")

                entity_results[node_id] = {
                    "level": level,
                    "children": children,
                    "entities": entities,
                    "temporal_edges": edges,
                    "chunk_ids": chunk_ids
                }

            return entity_results

        def dfs_helper(node_id: str):
            nonlocal temporal_entity_tree_dict
            node = temporal_entity_tree_dict[node_id]
            total_entities = set(node.get("entities", []))
            total_edges = set(node.get("temporal_edges", []))
            total_chunk_ids = set(node.get("chunk_ids", []))
            total_children = set(node.get("children", []))
            for child_id in node.get("children", []):
                r = dfs_helper(child_id)
                total_entities.update(r[0])
                total_edges.update(r[1])
                total_chunk_ids.update(r[2])
                total_children.update(r[3])


            temporal_entity_tree_dict[node_id]["all_entities"] = list(total_entities)
            temporal_entity_tree_dict[node_id]["all_temporal_edges"] = list(total_edges)
            temporal_entity_tree_dict[node_id]["all_chunk_ids"] = list(total_chunk_ids)
            temporal_entity_tree_dict[node_id]["all_timestamp_children"] = list(total_children)


            return total_entities, total_edges, total_chunk_ids, total_children

        temporal_entity_tree_dict = await build_temporal_entity_dict(self._graph, entity_relation_graph_inst)

        for node_id, node_dict in temporal_entity_tree_dict.items():
            dfs_helper(node_id)
            results[node_id]['level'] = node_dict['level']
            results[node_id]['title'] = node_id
            results[node_id]['temporal_edges'] = [list(edge) for edge in node_dict['all_temporal_edges']]
            results[node_id]['nodes'] = node_dict['all_entities']
            results[node_id]['sub_communities'] = node_dict['children']
            results[node_id]['chunk_ids'] = node_dict['all_chunk_ids']
            results[node_id]['all_sub_communities'] = node_dict['all_timestamp_children']



        return dict(results)

    def _cluster_data_to_subgraphs(self, cluster_data: dict[str, list[dict[str, str]]]):
        for node_id, clusters in cluster_data.items():
            self._graph.nodes[node_id]["clusters"] = json.dumps(clusters)

    async def _leiden_clustering(self):
        from graspologic.partition import hierarchical_leiden

        graph = NetworkXStorage.stable_largest_connected_component(self._graph)
        community_mapping = hierarchical_leiden(
            graph,
            max_cluster_size=self.global_config["max_graph_cluster_size"],
            random_seed=self.global_config["graph_cluster_seed"],
        )

        node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
        __levels = defaultdict(set)
        for partition in community_mapping:
            level_key = partition.level
            cluster_id = partition.cluster
            node_communities[partition.node].append(
                {"level": level_key, "cluster": cluster_id}
            )
            __levels[level_key].add(cluster_id)
        node_communities = dict(node_communities)
        __levels = {k: len(v) for k, v in __levels.items()}
        logger.info(f"Each level has communities: {dict(__levels)}")
        self._cluster_data_to_subgraphs(node_communities)

    async def personalized_pagerank(self, 
                                   personalization_nodes: list[str] = None,
                                   personalization_weights: dict[str, float] = None,
                                   alpha: float = 0.85,
                                   max_iter: int = 100,
                                   tol: float = 1e-06,
                                   weight: str = 'weight') -> dict[str, float]:
        """
        Compute personalized PageRank for the graph.
        
        Args:
            personalization_nodes: List of node IDs to personalize towards. If None, uses all nodes equally.
            personalization_weights: Dict mapping node IDs to personalization weights. 
                                   If None, equal weights are assigned to personalization_nodes.
            alpha: Damping parameter for PageRank (default: 0.85)
            max_iter: Maximum number of iterations (default: 100)
            tol: Error tolerance for convergence (default: 1e-06)
            weight: Edge attribute to use as weight (default: 'weight')
            
        Returns:
            Dict mapping node IDs to their personalized PageRank scores
            
        Example:
            # Personalize towards specific nodes
            ppr_scores = await storage.personalized_pagerank(
                personalization_nodes=['node1', 'node2'],
                alpha=0.9
            )
            
            # Use custom weights
            ppr_scores = await storage.personalized_pagerank(
                personalization_weights={'node1': 0.7, 'node2': 0.3}
            )
        """
        if not self._graph.number_of_nodes():
            logger.warning("Graph is empty, returning empty PageRank scores")
            return {}
        
        # Create personalization vector
        personalization = None
        if personalization_nodes is not None:
            if personalization_weights is not None:
                # Use provided weights
                personalization = personalization_weights.copy()
                # Normalize weights to sum to 1
                total_weight = sum(personalization.values())
                if total_weight > 0:
                    personalization = {k: v/total_weight for k, v in personalization.items()}
                else:
                    logger.warning("All personalization weights are zero, using uniform distribution")
                    personalization = None
            else:
                # Equal weights for personalization nodes
                if personalization_nodes:
                    weight_per_node = 1.0 / len(personalization_nodes)
                    personalization = {node: weight_per_node for node in personalization_nodes}
                else:
                    personalization = None
        
        # Ensure all personalization nodes exist in the graph
        if personalization is not None:
            missing_nodes = [node for node in personalization.keys() if not self._graph.has_node(node)]
            if missing_nodes:
                logger.warning(f"Personalization nodes not found in graph: {missing_nodes}")
                # Remove missing nodes from personalization
                personalization = {k: v for k, v in personalization.items() if k not in missing_nodes}
                # Renormalize if any nodes were removed
                if personalization:
                    total_weight = sum(personalization.values())
                    personalization = {k: v/total_weight for k, v in personalization.items()}
                else:
                    personalization = None
        
        try:
            # Compute personalized PageRank
            ppr_scores = nx.pagerank(
                self._graph,
                alpha=alpha,
                personalization=personalization,
                max_iter=max_iter,
                tol=tol,
                weight=weight
            )
            
            logger.info(f"Computed personalized PageRank for {len(ppr_scores)} nodes")
            if personalization:
                logger.info(f"Personalized towards {len(personalization)} nodes: {list(personalization.keys())}")
            
            return ppr_scores
            
        except Exception as e:
            logger.error(f"Error computing personalized PageRank: {e}")
            return {}

    async def get_top_pagerank_nodes(self, 
                                   personalization_nodes: list[str] = None,
                                   personalization_weights: dict[str, float] = None,
                                   top_k: int = 10,
                                   alpha: float = 0.85) -> list[tuple[str, float]]:
        """
        Get top-k nodes by personalized PageRank score.
        
        Args:
            personalization_nodes: List of node IDs to personalize towards
            personalization_weights: Dict mapping node IDs to personalization weights
            top_k: Number of top nodes to return
            alpha: Damping parameter for PageRank
            
        Returns:
            List of tuples (node_id, pagerank_score) sorted by score (descending)
        """
        ppr_scores = await self.personalized_pagerank(
            personalization_nodes=personalization_nodes,
            personalization_weights=personalization_weights,
            alpha=alpha
        )
        
        # Sort by score and return top-k
        sorted_nodes = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_k]