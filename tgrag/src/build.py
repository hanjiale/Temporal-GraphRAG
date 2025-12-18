"""Convenience functions for building temporal knowledge graphs."""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path

from .temporal_graphrag import TemporalGraphRAG
from .config.config_loader import ConfigLoader, build_default_working_dir
from .llm.completion import create_provider_complete_function
from .llm.embedding import (
    openai_embedding,
    azure_openai_embedding,
    amazon_bedrock_embedding,
)
from .utils.types import EmbeddingFunc
import numpy as np


def get_api_key_for_provider(provider: str) -> Optional[str]:
    """Get API key from environment variable for the provider."""
    provider_lower = provider.lower()
    
    if provider_lower == "gemini":
        # Check both GOOGLE_API_KEY (preferred) and GEMINI_API_KEY (legacy)
        return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    elif provider_lower == "openai":
        return os.getenv("OPENAI_API_KEY")
    elif provider_lower == "azure":
        return os.getenv("AZURE_OPENAI_API_KEY")
    elif provider_lower == "bedrock":
        return os.getenv("AWS_ACCESS_KEY_ID")  # Bedrock uses AWS credentials
    elif provider_lower == "ollama":
        return None  # Ollama doesn't need API key
    else:
        return None


def create_llm_function(provider: str, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Create LLM completion function based on provider."""
    kwargs = {}
    if api_key:
        kwargs['api_key'] = api_key
    if base_url:
        kwargs['base_url'] = base_url
    
    return create_provider_complete_function(
        provider=provider,
        model=model,
        **kwargs
    )


def create_embedding_function(embedding_provider: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Create embedding function based on embedding provider."""
    # Get base_url from env if not provided and provider supports it
    if not base_url and embedding_provider in ("openai", "azure", "gemini"):
        base_url = os.getenv('OPENAI_BASE_URL')
    elif not base_url and embedding_provider == "bedrock":
        base_url = os.getenv('BEDROCK_BASE_URL')
    
    async def embedding_wrapper(texts: List[str]) -> np.ndarray:
        kwargs = {}
        if api_key:
            kwargs['api_key'] = api_key
        if base_url:
            kwargs['base_url'] = base_url
        
        if embedding_provider == "openai":
            return await openai_embedding(texts, **kwargs)
        elif embedding_provider == "azure":
            return await azure_openai_embedding(texts)
        elif embedding_provider == "bedrock":
            return await amazon_bedrock_embedding(texts)
        else:
            # Default to OpenAI
            return await openai_embedding(texts, **kwargs)
    
    # Return appropriate EmbeddingFunc based on embedding provider
    if embedding_provider in ("openai", "azure"):
        return EmbeddingFunc(
            embedding_dim=1536,
            func=embedding_wrapper,
            max_token_size=8192
        )
    elif embedding_provider == "bedrock":
        return EmbeddingFunc(
            embedding_dim=1024,  # Typical for Bedrock embeddings
            func=embedding_wrapper,
            max_token_size=8192
        )
    else:
        # Default to OpenAI (1536-dim)
        return EmbeddingFunc(
            embedding_dim=1536,
            func=embedding_wrapper,
            max_token_size=8192
        )


def create_temporal_graphrag_from_config(
    config_path: str = "tgrag/configs/config.yaml",
    config_type: str = "building",
    override_config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> TemporalGraphRAG:
    """
    Create a TemporalGraphRAG instance from a configuration file.
    
    This is a convenience function that handles:
    - Loading configuration from YAML
    - Creating LLM completion functions based on provider/model
    - Creating embedding functions based on provider
    - Setting up all TemporalGraphRAG parameters
    
    Args:
        config_path: Path to the configuration YAML file
        config_type: Type of config to load ("building" or "querying")
        override_config: Dictionary of config values to override
        api_key: API key (optional, will use env var if not provided)
        base_url: Base URL (optional, for OpenAI/Azure custom endpoints, 
                 can also use OPENAI_BASE_URL env var)
        
    Returns:
        Fully configured TemporalGraphRAG instance
        
    Note:
        The OPENAI_BASE_URL environment variable will be automatically used
        for OpenAI and Azure providers if base_url is not explicitly provided.
        
    Example:
        >>> # Simple usage with default config
        >>> graph = create_temporal_graphrag_from_config()
        >>> graph.insert(documents)
        
        >>> # With custom config and overrides
        >>> graph = create_temporal_graphrag_from_config(
        ...     config_path="my_config.yaml",
        ...     override_config={"chunk_size": 1000}
        ... )
    """
    # Load configuration
    config_loader = ConfigLoader(config_path=config_path)
    config = config_loader.get_config(config_type, override_args=override_config)
    
    # Get provider and model
    provider = config.get('provider', 'openai')
    model = config.get('model', 'gpt-4o-mini')
    
    # Get API key (use provided or from env)
    if not api_key:
        api_key = get_api_key_for_provider(provider)
        if not api_key and provider != "ollama":
            # Provide helpful error messages for each provider
            if provider == "gemini":
                error_msg = (
                    f"API key not found for provider '{provider}'. "
                    f"Please set either GOOGLE_API_KEY or GEMINI_API_KEY environment variable, "
                    f"or pass api_key parameter."
                )
            else:
                env_var_map = {
                    "openai": "OPENAI_API_KEY",
                    "azure": "AZURE_OPENAI_API_KEY",
                    "bedrock": "AWS_ACCESS_KEY_ID",
                }
                env_var = env_var_map.get(provider, "API key")
                error_msg = (
                    f"API key not found for provider '{provider}'. "
                    f"Please set the {env_var} environment variable or pass api_key parameter."
                )
            raise ValueError(error_msg)
    
    # Get base URL (use provided or from env)
    # OpenAI and Azure can use OPENAI_BASE_URL
    if not base_url:
        if provider in ("openai", "azure"):
            base_url = os.getenv('OPENAI_BASE_URL')
        # For other providers, you might want to add provider-specific env vars here
    
    # Determine working directory
    working_dir = config.get('working_dir')
    if not working_dir:
        # Build default working directory
        corpus_path = config.get('corpus_path', './data')
        working_dir = build_default_working_dir(corpus_path, model)
    
    # Get embedding provider (defaults to LLM provider, but can be overridden)
    embedding_provider = config.get('embedding_provider', provider)
    # If embedding provider is gemini, default to openai (gemini embeddings not supported)
    if embedding_provider == "gemini":
        embedding_provider = "openai"
    
    embedding_base_url = base_url
    
    if embedding_provider == "openai":
        # Always use OpenAI API key for OpenAI embeddings
        embedding_api_key = get_api_key_for_provider("openai")
        if not embedding_api_key:
            raise ValueError(
                "OpenAI API key not found for embeddings. "
                "Please set OPENAI_API_KEY environment variable."
            )
        if not embedding_base_url:
            embedding_base_url = os.getenv('OPENAI_BASE_URL')
    elif embedding_provider == "azure":
        # Always use Azure API key for Azure embeddings
        embedding_api_key = get_api_key_for_provider("azure")
        if not embedding_api_key:
            raise ValueError(
                "Azure OpenAI API key not found for embeddings. "
                "Please set AZURE_OPENAI_API_KEY environment variable."
            )
        if not embedding_base_url:
            embedding_base_url = os.getenv('OPENAI_BASE_URL')  # Azure also uses OPENAI_BASE_URL
    elif embedding_provider == "bedrock":
        # Always use Bedrock credentials for Bedrock embeddings
        embedding_api_key = get_api_key_for_provider("bedrock")
        if not embedding_api_key:
            raise ValueError(
                "AWS credentials not found for Bedrock embeddings. "
                "Please set AWS_ACCESS_KEY_ID environment variable."
            )
        # Bedrock doesn't use base_url, uses AWS region instead
    
    # Create LLM and embedding functions
    llm_func = create_llm_function(provider, model, api_key=api_key, base_url=base_url)
    embedding_func = create_embedding_function(
        embedding_provider=embedding_provider, 
        api_key=embedding_api_key, 
        base_url=embedding_base_url
    )
    
    # Create TemporalGraphRAG instance
    graph_rag = TemporalGraphRAG(
        working_dir=working_dir,
        enable_local=config.get('enable_local', True),
        enable_naive_rag=config.get('enable_naive_rag', False),
        chunk_token_size=config.get('chunk_size', 1200),
        chunk_overlap_token_size=config.get('chunk_overlap', 100),
        disable_entity_summarization=config.get('disable_entity_summarization', False),
        embedding_func=embedding_func,
        best_model_func=llm_func,
        cheap_model_func=llm_func,  # Use same model for both
        enable_llm_cache=config.get('enable_llm_cache', True),
        always_create_working_dir=True,
        enable_community_summary=config.get('enable_community_summary', True),
        enable_incremental=config.get('enable_incremental', False),
        preserve_communities=config.get('preserve_communities', False),
    )
    
    return graph_rag


__all__ = [
    "create_temporal_graphrag_from_config",
    "create_llm_function",
    "create_embedding_function",
    "get_api_key_for_provider",
]

