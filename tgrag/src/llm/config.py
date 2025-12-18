"""LLM configuration classes and provider settings."""

from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any


@dataclass
class LLMProviderConfig:
    """Configuration for a specific LLM provider.
    
    Attributes:
        provider: Provider name (openai, azure, bedrock, gemini, ollama)
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens for completion
        base_url: Base URL for API (optional, for custom endpoints)
        api_key: API key (optional, can be set via environment)
        timeout: Request timeout in seconds
        extra_kwargs: Additional provider-specific kwargs
    """
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: float = 120.0
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """Configuration for LLM providers and models.
    
    Attributes:
        indexing_provider: Provider config for indexing operations
        qa_provider: Provider config for QA operations
        using_azure_openai: Use Azure OpenAI instead of OpenAI (legacy)
        using_amazon_bedrock: Use Amazon Bedrock (legacy)
        best_model_id: Model ID for best quality (Bedrock legacy)
        cheap_model_id: Model ID for cheaper operations (Bedrock legacy)
        best_model_func: Function for best quality model calls (legacy)
        best_model_max_token_size: Maximum tokens for best model
        best_model_max_async: Maximum concurrent calls for best model
        cheap_model_func: Function for cheaper model calls (legacy)
        cheap_model_max_token_size: Maximum tokens for cheap model
        cheap_model_max_async: Maximum concurrent calls for cheap model
        enable_llm_cache: Enable caching of LLM responses
    """
    indexing_provider: Optional[LLMProviderConfig] = None
    qa_provider: Optional[LLMProviderConfig] = None
    
    # Legacy fields for backward compatibility
    using_azure_openai: bool = False
    using_amazon_bedrock: bool = False
    best_model_id: str = "us.anthropic.claude-3-sonnet-20240229-v1:0"
    cheap_model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0"
    best_model_func: Optional[Callable] = None
    best_model_max_token_size: int = 65536
    best_model_max_async: int = 32
    cheap_model_func: Optional[Callable] = None
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 32
    enable_llm_cache: bool = True
    
    def get_provider_config(self, operation: str = "qa") -> Optional[LLMProviderConfig]:
        """Get provider config for a specific operation.
        
        Args:
            operation: Operation type ("indexing" or "qa")
            
        Returns:
            Provider config or None
        """
        if operation == "indexing":
            return self.indexing_provider
        return self.qa_provider

