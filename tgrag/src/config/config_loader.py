"""Configuration loader utility for Temporal GraphRAG."""

import os
import yaml
import argparse
from typing import Dict, Any, Optional, Literal
from datetime import datetime
import re

# Import config classes for type-safe configuration
from ..core.config import (
    CoreConfig,
    ChunkingConfig,
    LLMConfig,
    StorageConfig,
    EmbeddingConfig,
    EntityConfig,
    GraphConfig,
    ExtensionConfig,
)


class ConfigLoader:
    """Utility class to load and manage configuration for Temporal GraphRAG"""
    
    def __init__(self, config_path: str = "tgrag/configs/config.yaml"):
        """
        Initialize the config loader
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing config file {self.config_path}: {e}")
            return {}
    
    def get_config(self, config_type: Literal["building", "querying"], override_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get configuration based on task type as a dictionary."""
        # Get config by type
        config = self.config.get(config_type, {}).copy()

        # Override with provided arguments
        if override_args:
            config.update({k: v for k, v in override_args.items() if v is not None})
        
        if not config:
            raise ValueError(f"No configuration found for type: {config_type}")

        # Handle provider inference and backward compatibility
        model = config.get('model', '')
        use_google = config.get('use_google')
        provider = config.get('provider')
        
        # If provider not set, infer from model or use_google
        if provider is None:
            provider = infer_provider_from_model(model, use_google)
            config['provider'] = provider
        
        # Remove deprecated use_google if provider is set
        if 'use_google' in config and provider:
            # Keep use_google for backward compatibility but prefer provider
            pass

        return config
    
    def get_structured_configs(self, config_type: Literal["building", "querying"], override_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get configuration as structured config objects.
        
        Maps YAML config to the structured config classes from core.config.
        
        Args:
            config_type: Type of configuration to load ("building" or "querying")
            override_args: Dictionary of arguments to override config values
            
        Returns:
            Dictionary containing structured config objects:
            - core: CoreConfig
            - chunking: ChunkingConfig
            - llm: LLMConfig
            - storage: StorageConfig
            - embedding: EmbeddingConfig
            - entity: EntityConfig
            - graph: GraphConfig
            - extension: ExtensionConfig
        """
        # Get raw config dict
        raw_config = self.get_config(config_type, override_args)
        
        # Map YAML config to structured config classes
        structured = {}
        
        # CoreConfig
        from datetime import datetime
        default_working_dir = f"./temporal_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        structured['core'] = CoreConfig(
            working_dir=raw_config.get('working_dir') or default_working_dir,
            enable_local=raw_config.get('enable_local', True),
            enable_naive_rag=raw_config.get('enable_naive_rag', False),
            always_create_working_dir=raw_config.get('always_create_working_dir', True),
        )
        
        # ChunkingConfig
        structured['chunking'] = ChunkingConfig(
            chunk_token_size=raw_config.get('chunk_size', 1200),
            chunk_overlap_token_size=raw_config.get('chunk_overlap', 100),
            tiktoken_model_name=raw_config.get('tiktoken_model_name', 'gpt-4o-mini'),
        )
        
        # LLMConfig (extract from raw config)
        # Use default values from LLMConfig class definition
        structured['llm'] = LLMConfig(
            using_azure_openai=raw_config.get('using_azure_openai', False),
            using_amazon_bedrock=raw_config.get('using_amazon_bedrock', False),
            best_model_id=raw_config.get('best_model_id', "us.anthropic.claude-3-sonnet-20240229-v1:0"),
            cheap_model_id=raw_config.get('cheap_model_id', "us.anthropic.claude-3-haiku-20240307-v1:0"),
            best_model_max_token_size=raw_config.get('best_model_max_token_size', 65536),
            best_model_max_async=raw_config.get('best_model_max_async', 32),
            cheap_model_max_token_size=raw_config.get('cheap_model_max_token_size', 32768),
            cheap_model_max_async=raw_config.get('cheap_model_max_async', 32),
            enable_llm_cache=raw_config.get('enable_llm_cache', True),
        )
        
        # StorageConfig (defaults, will be set by TemporalGraphRAG)
        structured['storage'] = StorageConfig()
        
        # EmbeddingConfig
        structured['embedding'] = EmbeddingConfig(
            embedding_batch_num=raw_config.get('embedding_batch_num', 32),
            embedding_func_max_async=raw_config.get('embedding_func_max_async', 16),
            query_better_than_threshold=raw_config.get('query_better_than_threshold', 0.2),
            enable_entity_retrieval=raw_config.get('enable_entity_retrieval', False),
        )
        
        # EntityConfig
        structured['entity'] = EntityConfig(
            entity_extract_max_gleaning=raw_config.get('entity_extract_max_gleaning', 1),
            entity_summary_to_max_tokens=raw_config.get('entity_summary_to_max_tokens', 500),
            disable_entity_summarization=raw_config.get('disable_entity_summarization', False),
        )
        
        # GraphConfig
        structured['graph'] = GraphConfig(
            max_graph_cluster_size=raw_config.get('max_graph_cluster_size', 10),
            graph_cluster_seed=raw_config.get('graph_cluster_seed', 0xDEADBEEF),
            enable_community_summary=raw_config.get('enable_community_summary', True),
            enable_incremental=raw_config.get('enable_incremental', False),
            preserve_communities=raw_config.get('preserve_communities', False),
        )
        
        
        # ExtensionConfig
        structured['extension'] = ExtensionConfig(
            addon_params=raw_config.get('addon_params', {}),
        )
        
        return structured    
    
    @staticmethod
    def create_building_parser() -> argparse.ArgumentParser:
        """Create argument parser for building graph with config file support"""
        parser = argparse.ArgumentParser(
            description="Build Temporal GraphRAG knowledge graph",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Config file
        parser.add_argument(
            '--config', type=str, default='tgrag/configs/config.yaml',
            help='Path to configuration file'
        )
        
        # Core arguments that can override config
        parser.add_argument(
            '--corpus_path', type=str, default=None,
            help='Path to your corpus directory used for building the graph'
        )
        parser.add_argument(
            '--baseline', type=str, choices=['temporalrag'], default=None,
            help='Baseline to build the graph (only temporalrag is supported)'
        )
        parser.add_argument(
            '--model', type=str, default=None,
            help='LLM used for extraction and report generation (model id)'
        )
        parser.add_argument(
            '--provider', type=str, choices=['openai', 'azure', 'bedrock', 'gemini', 'ollama'], default=None,
            help='LLM provider: openai, azure, bedrock, gemini, or ollama (auto-detected from model if not specified)'
        )
        parser.add_argument(
            '--working_dir', type=str, default=None,
            help='Subdirectory under ./graph_storage where graph artifacts are saved'
        )
        parser.add_argument(
            '--chunk_size', type=int, default=None,
            help='Max token size per chunk'
        )
        parser.add_argument(
            '--use_google', type=bool, default=None,
            help='Whether to use Gemini for LLM calls (True) or OpenAI-compatible API (False)'
        )
        parser.add_argument(
            '--chunk_overlap', type=int, default=None,
            help='Overlap token size between consecutive chunks'
        )
        parser.add_argument(
            '--enable_seasonal_matching', action='store_true',
            help='Enable seasonal matching in temporal normalization'
        )
        parser.add_argument(
            '--enable_community_summary', type=bool, default=None,
            help='Enable/disable community summary generation (True/False)'
        )
        
        return parser
    
    @staticmethod
    def create_querying_parser() -> argparse.ArgumentParser:
        """Create argument parser for querying graph with config file support"""
        parser = argparse.ArgumentParser(
            description="Query Temporal GraphRAG knowledge graph",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Config file
        parser.add_argument(
            '--config', type=str, default='tgrag/configs/config.yaml',
            help='Path to configuration file'
        )
        
        # Core arguments that can override config
        parser.add_argument(
            '--corpus_path', type=str, default=None,
            help='Your corpus to be used for building graph'
        )
        parser.add_argument(
            '--baseline', type=str, choices=['temporalrag', 'groundtruth'], default=None,
            help='Baseline to be used for building graph'
        )
        parser.add_argument(
            '--model', type=str, default=None,
            help='LLM to be used for extracting the relations and entities'
        )
        parser.add_argument(
            '--provider', type=str, choices=['openai', 'azure', 'bedrock', 'gemini', 'ollama'], default=None,
            help='LLM provider: openai, azure, bedrock, gemini, or ollama (auto-detected from model if not specified)'
        )
        parser.add_argument(
            '--working_dir', type=str, default=None,
            help='Path of graph to be saved'
        )
        parser.add_argument(
            '--evaluation_mode', type=str, choices=['local', 'global'], default=None,
            help='Select local or global query evaluation set'
        )
        parser.add_argument(
            '--top_k', type=int, default=None,
            help='Retrieve top_k entities'
        )
        parser.add_argument(
            '--truncated_context_length', type=int, default=None,
            help='Context length constraint for in-context learning'
        )
        parser.add_argument(
            '--output_file', type=str, default=None,
            help='Output file name'
        )
        parser.add_argument(
            '--enable_entity_retrieval', type=bool, default=None,
            help='Entity embedding retrieval or entity+description embedding retrieval'
        )
        parser.add_argument(
            '--enable_subgraph', type=bool, default=None,
            help='Enable subgraph retrieval'
        )
        parser.add_argument(
            '--enable_mixed_relationship', type=bool, default=None,
            help='Enable mixed relationship retrieval'
        )
        parser.add_argument(
            '--seed_node_method', type=str, choices=['entities', 'relations'], default=None,
            help='Method for seed node retrieval: entities or relations'
        )
        parser.add_argument(
            '--num_questions', type=int, default=None,
            help='Number of questions to test (if not specified, uses all questions)'
        )
        parser.add_argument(
            '--use_enhanced_retrieval', type=bool, default=None,
            help='Enable enhanced temporal retrieval system'
        )
        parser.add_argument(
            '--enhanced_output_suffix', type=str, default=None,
            help='Suffix to add to output file when using enhanced retrieval'
        )
        
        return parser


def _sanitize_for_path(name: str) -> str:
    """Sanitize a string for use in file paths"""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
    safe = re.sub(r"-+", "-", safe).strip("-")
    return safe or "model"


def infer_provider_from_model(model: str, use_google: Optional[bool] = None) -> str:
    """Infer LLM provider from model name"""
    # If use_google is explicitly set, use it for backward compatibility
    if use_google is not None:
        return "gemini" if use_google else "openai"
    
    # Infer from model name
    model_lower = model.lower()
    
    if "gemini" in model_lower or model_lower.startswith("google/"):
        return "gemini"
    elif "claude" in model_lower or "anthropic" in model_lower or model_lower.startswith("us."):
        return "bedrock"
    elif "llama" in model_lower or model_lower.startswith("ollama/"):
        return "ollama"
    elif "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        return "openai"
    else:
        # Default to openai if cannot be determined
        return "openai"


def build_default_working_dir(corpus_path: str, model: str) -> str:
    """Build default working directory name from corpus path, model, and timestamp"""
    dataset = os.path.basename(os.path.normpath(corpus_path)) or "dataset"
    model_safe = _sanitize_for_path(model)
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{dataset}__{model_safe}__{date_str}"

