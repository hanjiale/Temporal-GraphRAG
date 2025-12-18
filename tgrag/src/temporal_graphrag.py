"""Temporal GraphRAG: Main class for building and querying temporal knowledge graphs."""

import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast

import tiktoken

# Import LLM functions from new structure (Phase 3 migration complete)
from .llm import (
    amazon_bedrock_embedding,
    create_amazon_bedrock_complete_function,
    gpt_4o_complete,
    gpt_4o_mini_complete,
    openai_embedding,
    azure_gpt_4o_complete,
    azure_openai_embedding,
    azure_gpt_4o_mini_complete,
)
# Import from new Phase 4 modules
from .core.chunking import (
    chunking_by_token_size,
    chunking_by_separators,
    get_chunks,
)
from .core.building import (
    extract_entities,
    building_temporal_hierarchy,
    generate_temporal_report,
    generate_community_report,
)
from .core.querying import (
    local_query,
    global_query,
    naive_query,
)
# Import storage classes from new structure
from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
# Import utilities from new structure
from .utils.hashing import compute_mdhash_id
from .utils.async_utils import limit_async_func_call, always_get_an_event_loop
from .utils.json_utils import convert_response_to_json
from .utils import logger

# Import from new structure
from .core.types import QueryParam
from .storage.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
)
from .utils.types import EmbeddingFunc


@dataclass
class TemporalGraphRAG:
    """Main class for Temporal GraphRAG operations.
    
    This class provides methods for:
    - Indexing documents and building temporal knowledge graphs
    - Querying graphs in local, global, or naive RAG modes
    
    Attributes:
        working_dir: Working directory for storing graph artifacts
        enable_local: Enable local query mode
        enable_naive_rag: Enable naive RAG mode
        chunk_func: Function to perform text chunking
        chunk_token_size: Maximum token size per chunk
        chunk_overlap_token_size: Overlap token size between consecutive chunks
        tiktoken_model_name: Model name for tiktoken encoding
        entity_extract_max_gleaning: Maximum gleaning iterations for entity extraction
        entity_summary_to_max_tokens: Maximum tokens for entity summaries
        disable_entity_summarization: Disable entity summarization
        max_graph_cluster_size: Maximum size for graph clusters
        graph_cluster_seed: Random seed for graph clustering
        special_community_report_llm_kwargs: LLM kwargs for community reports
        enable_community_summary: Enable community summary generation
        enable_incremental: Enable incremental update mode
        preserve_communities: Preserve existing communities during incremental updates
        embedding_func: Function to generate embeddings
        embedding_batch_num: Batch size for embedding operations
        embedding_func_max_async: Maximum concurrent embedding calls
        query_better_than_threshold: Threshold for query quality
        enable_entity_retrieval: Enable entity-based retrieval
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
        entity_extraction_func: Function to perform entity extraction
        building_temporal_hierarchy_func: Function to build temporal hierarchy
        key_string_value_json_storage_cls: Class for key-value storage
        vector_db_storage_cls: Class for vector database storage
        vector_db_storage_cls_kwargs: Additional kwargs for vector storage
        graph_storage_cls: Class for graph database storage
        enable_llm_cache: Enable caching of LLM responses
        always_create_working_dir: Automatically create working directory if it doesn't exist
        addon_params: Additional parameters for extensions
        convert_response_to_json_func: Function to convert LLM responses to JSON
    """
    working_dir: str = field(
        default_factory=lambda: f"./temporal_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = False
    # text chunking
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_token_size
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    # entity extraction
    entity_extract_max_gleaning: int = 1  
    entity_summary_to_max_tokens: int = 500
    disable_entity_summarization: bool = False

    # graph clustering
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF

    # community reports
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    enable_community_summary: bool = True

    # incremental update support
    enable_incremental: bool = False
    preserve_communities: bool = False

    # text embedding
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    query_better_than_threshold: float = 0.2
    enable_entity_retrieval: bool = False

    # LLM
    using_azure_openai: bool = False
    using_amazon_bedrock: bool = False
    best_model_id: str = "us.anthropic.claude-3-sonnet-20240229-v1:0"
    cheap_model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0"
    best_model_func: callable = gpt_4o_mini_complete
    best_model_max_token_size: int = 65536
    best_model_max_async: int = 32
    cheap_model_func: callable = gpt_4o_mini_complete
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 32

    # entity extraction
    entity_extraction_func: callable = extract_entities

    # temporal hierarchy
    building_temporal_hierarchy_func: callable = building_temporal_hierarchy

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # extension
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        """Initialize storage instances and configure LLM providers."""
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"TemporalGraphRAG init with param:\n\n  {_print_config}\n")

        if self.using_azure_openai:
            # If there's no OpenAI API key, use Azure OpenAI
            if self.best_model_func == gpt_4o_complete:
                self.best_model_func = azure_gpt_4o_complete
            if self.cheap_model_func == gpt_4o_mini_complete:
                self.cheap_model_func = azure_gpt_4o_mini_complete
            if self.embedding_func == openai_embedding:
                self.embedding_func = azure_openai_embedding
            logger.info(
                "Switched the default openai funcs to Azure OpenAI if you didn't set any of it"
            )

        if self.using_amazon_bedrock:
            self.best_model_func = create_amazon_bedrock_complete_function(self.best_model_id)
            self.cheap_model_func = create_amazon_bedrock_complete_function(self.cheap_model_id)
            self.embedding_func = amazon_bedrock_embedding
            logger.info(
                "Switched the default openai funcs to Amazon Bedrock"
            )

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=asdict(self)
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self), is_directed=False
        )
        self.temporal_hierarchy_graph = self.graph_storage_cls(
            namespace="temporal_hierarchy", global_config=asdict(self), is_directed=True
        )

        self._temporal_hierarchy = None

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        self.entities_vdb_new = (
            self.vector_db_storage_cls(
                namespace="entities_new",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        self.relations_vdb = (
            self.vector_db_storage_cls(
                namespace="relations",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name", "content"},
            )
            if self.enable_local
            else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )

        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
        )

    def insert(self, dict_or_dicts):
        """Insert documents into the knowledge graph.
        
        Args:
            dict_or_dicts: Document(s) in format {"title": "", "doc": ""}
                Can be a single dict or list of dicts
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(dict_or_dicts))

    async def get_temporal_hierarchy(self):
        """Get the temporal hierarchy representation."""
        if self._temporal_hierarchy is None:
            self._temporal_hierarchy = await self.temporal_hierarchy_graph.temporal_hierarchy(
                self.chunk_entity_relation_graph
            )
        return self._temporal_hierarchy

    def query(self, query: str, param: QueryParam = QueryParam()):
        """Query the knowledge graph synchronously.
        
        Args:
            query: Query string
            param: Query parameters
            
        Returns:
            Query response (and retrieval details for local mode)
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        """Query the knowledge graph asynchronously.
        
        Args:
            query: Query string
            param: Query parameters
            
        Returns:
            Query response (and retrieval details for local mode)
        """
        if param.mode == "local" and not self.enable_local:
            raise ValueError("enable_local is False, cannot query in local mode")
        if param.mode == "naive" and not self.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in naive mode")
        temporal_hierarchy = await self.get_temporal_hierarchy()
        if param.mode == "local":
            global_config_dict = asdict(self)
            # Add full_docs to global_config so it can be accessed in local_query
            global_config_dict["full_docs"] = self.full_docs
            response, retrieval_detail = await local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relations_vdb,
                self.community_reports,
                self.text_chunks,
                temporal_hierarchy,
                param,
                global_config_dict,
            )
            await self._query_done()
            return response, retrieval_detail
        elif param.mode == "global":
            response = await global_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relations_vdb,
                self.community_reports,
                self.text_chunks,
                temporal_hierarchy,
                param,
                asdict(self),
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def ainsert(self, dict_or_dicts):
        """Insert documents asynchronously.
        
        Args:
            dict_or_dicts: Document(s) in format {"title": "", "doc": ""}
        """
        await self._insert_start()
        try:
            if isinstance(dict_or_dicts, dict):
                dict_or_dicts = [dict_or_dicts]
            # ---------- new docs
            new_doc_dicts = {
                compute_mdhash_id(c['doc'].strip(), prefix="doc-"): {"doc": c['doc'].strip(), "title": c['title'].strip()}
                for c in dict_or_dicts
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_doc_dicts.keys()))
            new_doc_dicts = {k: v for k, v in new_doc_dicts.items() if k in _add_doc_keys}
            if not len(new_doc_dicts):
                logger.warning(f"All docs are already in the storage")
                return
            logger.info(f"[New Docs] inserting {len(new_doc_dicts)} docs")

            # ---------- chunking
            inserting_chunks = get_chunks(
                new_doc_dicts=new_doc_dicts,
                chunk_func=self.chunk_func,
                overlap_token_size=self.chunk_overlap_token_size,
                max_token_size=self.chunk_token_size,
            )

            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            # Initialize variable for community preservation
            existing_communities = []
            
            # Incremental update: preserve existing community summaries if enabled
            if self.enable_incremental and self.preserve_communities:
                logger.info("[Incremental Mode] Preserving existing community summaries...")
                existing_communities = await self.community_reports.all_keys()
                if existing_communities:
                    logger.info(f"Found {len(existing_communities)} existing community summaries to preserve")
                else:
                    logger.info("No existing community summaries found")
            else:
                logger.info("[Standard Mode] Dropping all existing community summaries")
                await self.community_reports.drop()

            # ---------- extract/summary entity and upsert to graph
            logger.info("[Entity Extraction]...")

            try:
                maybe_new_kg, maybe_new_hierarchy_node_names, _ = await asyncio.wait_for(
                    self.entity_extraction_func(
                        inserting_chunks,
                        knwoledge_graph_inst=self.chunk_entity_relation_graph,
                        entity_vdb=self.entities_vdb,
                        entity_vdb_new=self.entities_vdb_new,
                        relation_vdb=self.relations_vdb,
                        global_config=asdict(self),
                        using_amazon_bedrock=self.using_amazon_bedrock,
                    ),
                    timeout=21600  
                )
            except asyncio.TimeoutError:
                logger.error("Entity extraction timed out after 6 hours. This may indicate an issue with the LLM or network connectivity, or the dataset is too large.")
                raise
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg
            
            # ---------- update clusterings of graph
            logger.info("[Building Temporal Hierarchy]...")
            logger.info(f"Found {len(maybe_new_hierarchy_node_names)} new hierarchy node names")

            await self.building_temporal_hierarchy_func(
                maybe_new_hierarchy_node_names,
                temporal_hierarchy_graph_inst=self.temporal_hierarchy_graph,
                knowledge_graph_inst=self.chunk_entity_relation_graph
            )

            # Generate community reports only if enabled
            if self.enable_community_summary:
                logger.info("[Generating Community Reports]...")
                await generate_temporal_report(
                    self.community_reports,
                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                    temporal_hierarchy_graph_inst=self.temporal_hierarchy_graph,
                    global_config=asdict(self)
                )
                
                # Incremental update: restore preserved community summaries
                if self.enable_incremental and self.preserve_communities:
                    logger.info("[Incremental Mode] Restoring preserved community summaries...")
                    logger.info(f"Would restore {len(existing_communities)} preserved community summaries")
            else:
                logger.info("[Community Reports] Skipped (disabled in configuration)")

            logger.info("[Finalizing Storage Operations]...")
            # ---------- commit upsertings and indexing
            await self.full_docs.upsert(new_doc_dicts)
            await self.text_chunks.upsert(inserting_chunks)

        finally:
            # Call _insert_done only once at the end to persist all changes
            await self._insert_done()

    async def _insert_start(self):
        """Call index_start_callback on all storage instances."""
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
            self.temporal_hierarchy_graph
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        """Call index_done_callback on all storage instances."""
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.community_reports,
            self.entities_vdb,
            self.entities_vdb_new,
            self.relations_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
            self.temporal_hierarchy_graph
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        """Call query_done_callback on query-related storage instances."""
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

