"""LLM client management and completion handling."""

from .client import (
    LLMClientManager,
    get_client_manager,
    generate_request_id,
    ollama_complete,
    gemini_complete,
)
from .completion import (
    openai_complete_if_cache,
    azure_openai_complete_if_cache,
    amazon_bedrock_complete_if_cache,
    ollama_complete_if_cache,
    gemini_complete_if_cache,
    gpt_4o_complete,
    gpt_4o_mini_complete,
    azure_gpt_4o_complete,
    azure_gpt_4o_mini_complete,
    create_amazon_bedrock_complete_function,
    create_provider_complete_function,
)
from .embedding import (
    openai_embedding,
    azure_openai_embedding,
    amazon_bedrock_embedding,
    ollama_embedding,
    gemini_embedding,
)
from .config import LLMConfig, LLMProviderConfig

__all__ = [
    # Client management
    "LLMClientManager",
    "get_client_manager",
    "generate_request_id",
    # Direct client functions
    "ollama_complete",
    "gemini_complete",
    # Completion functions
    "openai_complete_if_cache",
    "azure_openai_complete_if_cache",
    "amazon_bedrock_complete_if_cache",
    "ollama_complete_if_cache",
    "gemini_complete_if_cache",
    "gpt_4o_complete",
    "gpt_4o_mini_complete",
    "azure_gpt_4o_complete",
    "azure_gpt_4o_mini_complete",
    "create_amazon_bedrock_complete_function",
    "create_provider_complete_function",
    # Embedding functions
    "openai_embedding",
    "azure_openai_embedding",
    "amazon_bedrock_embedding",
    "ollama_embedding",
    "gemini_embedding",
    # Config
    "LLMConfig",
    "LLMProviderConfig",
]


