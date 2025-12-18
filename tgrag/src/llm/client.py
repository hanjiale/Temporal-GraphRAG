"""LLM client implementations for various providers."""

import os
import json
import logging
from typing import Optional, Dict, Any, List
import uuid

import aioboto3
from openai import AsyncOpenAI, AsyncAzureOpenAI, RateLimitError

logger = logging.getLogger("temporal-graphrag.llm")


class LLMClientManager:
    """Manages LLM client instances with dependency injection.
    
    Replaces global singleton clients with a proper manager pattern.
    """
    
    def __init__(self):
        self._openai_client: Optional[AsyncOpenAI] = None
        self._openai_client_config: Dict[str, Optional[str]] = {}  # Track api_key and base_url
        self._azure_client: Optional[AsyncAzureOpenAI] = None
        self._bedrock_session: Optional[Any] = None
        self._gemini_client: Optional[Any] = None
        self._ollama_base_url: Optional[str] = None
        
    def get_openai_client(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> AsyncOpenAI:
        """Get or create OpenAI client instance."""
        # Get defaults from environment
        env_api_key = os.getenv("OPENAI_API_KEY_TEMPORALRAG") or os.getenv("OPENAI_API_KEY")
        env_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        # Use provided values or fall back to environment
        final_api_key = api_key or env_api_key
        final_base_url = base_url or env_base_url
        
        # Check if we need to recreate the client
        # Recreate if client doesn't exist or config changed
        config_changed = (
            self._openai_client is None or
            self._openai_client_config.get("api_key") != final_api_key or
            self._openai_client_config.get("base_url") != final_base_url
        )
        
        if config_changed:
            self._openai_client = AsyncOpenAI(api_key=final_api_key, base_url=final_base_url)
            self._openai_client_config = {
                "api_key": final_api_key,
                "base_url": final_base_url
            }
        
        return self._openai_client
    
    def get_azure_client(self) -> AsyncAzureOpenAI:
        """Get or create Azure OpenAI client instance."""
        if self._azure_client is None:
            self._azure_client = AsyncAzureOpenAI()
        return self._azure_client
    
    def get_bedrock_session(self) -> Any:
        """Get or create Amazon Bedrock session."""
        if self._bedrock_session is None:
            self._bedrock_session = aioboto3.Session()
        return self._bedrock_session
    
    def get_gemini_client(self) -> Any:
        """Get or create Google Gemini client instance."""
        if self._gemini_client is None:
            try:
                # Use the new google.genai.Client pattern (matches OLD/building_graph.py)
                from google import genai
                # Check both GOOGLE_API_KEY (preferred) and GEMINI_API_KEY (legacy)
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "Google API key not found. Please set either GOOGLE_API_KEY "
                        "or GEMINI_API_KEY environment variable"
                    )
                self._gemini_client = genai.Client(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "google-generativeai package not installed. "
                    "Install with: pip install google-generativeai"
                )
        return self._gemini_client
    
    def get_ollama_base_url(self, base_url: Optional[str] = None) -> str:
        """Get Ollama base URL."""
        if base_url:
            self._ollama_base_url = base_url
        elif self._ollama_base_url is None:
            self._ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return self._ollama_base_url
    
    async def close_clients(self):
        """Properly close all HTTP client sessions."""
        if self._openai_client is not None:
            await self._openai_client.close()
            self._openai_client = None
            self._openai_client_config = {}
        
        if self._azure_client is not None:
            await self._azure_client.close()
            self._azure_client = None
        
        # Bedrock and Gemini clients don't need explicit closing
        self._bedrock_session = None
        self._gemini_client = None
        self._ollama_base_url = None
    
    def reset_clients(self):
        """Reset all client instances (useful for testing)."""
        self._openai_client = None
        self._openai_client_config = {}
        self._azure_client = None
        self._bedrock_session = None
        self._gemini_client = None
        self._ollama_base_url = None


# Global client manager instance (can be replaced with dependency injection)
_global_client_manager = LLMClientManager()


def get_client_manager() -> LLMClientManager:
    """Get the global client manager instance.
    
    Returns:
        LLMClientManager instance
    """
    return _global_client_manager


def generate_request_id() -> str:
    """Generate a unique request ID for logging.
    
    Returns:
        UUID string
    """
    return str(uuid.uuid4())


async def ollama_complete(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, Any]] = [],
    base_url: Optional[str] = None,
    timeout: float = 120.0,
    **kwargs
) -> str:
    """Complete a prompt using Ollama."""
    import aiohttp
    
    client_manager = get_client_manager()
    ollama_url = client_manager.get_ollama_base_url(base_url)
    
    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    # Prepare request payload
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    
    # Add optional parameters
    if "temperature" in kwargs:
        payload["options"] = {"temperature": kwargs["temperature"]}
    if "max_tokens" in kwargs or "max_completion_tokens" in kwargs:
        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
        if "options" not in payload:
            payload["options"] = {}
        payload["options"]["num_predict"] = max_tokens
    
    request_id = generate_request_id()
    logger.debug(f"[{request_id}] Ollama request: model={model}, messages={len(messages)}")
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.post(
                f"{ollama_url}/api/chat",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
                
                result = await response.json()
                response_text = result.get("message", {}).get("content", "")
                
                logger.debug(f"[{request_id}] Ollama response: {len(response_text)} chars")
                return response_text
    except aiohttp.ClientError as e:
        logger.error(f"[{request_id}] Ollama request failed: {e}")
        raise Exception(f"Ollama request failed: {e}")


async def gemini_complete(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, Any]] = [],
    timeout: float = 45.0,
    **kwargs
) -> str:
    """Complete a prompt using Google Gemini."""
    import asyncio
    
    client_manager = get_client_manager()
    google_async_client = client_manager.get_gemini_client()
    
    request_id = generate_request_id()
    logger.debug(f"[{request_id}] Gemini request: model={model}")
    
    # Build the full prompt (Gemini doesn't support system prompts the same way)
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"
    
    # Configure generation parameters (matches OLD/building_graph.py pattern)
    from google.genai import types
    thinking_budget = kwargs.get("thinking_budget", 1000)
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        temperature=kwargs.get("temperature", 0.0),
        max_output_tokens=kwargs.get("max_tokens", kwargs.get("max_output_tokens", 4096))
    )
    
    # Retry logic for Gemini API calls (matches OLD/building_graph.py pattern)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use google_async_client.aio.models.generate_content pattern (matches OLD code)
            if config:
                response = await asyncio.wait_for(
                    google_async_client.aio.models.generate_content(
                        model=model,
                        contents=full_prompt,
                        config=config,
                    ),
                    timeout=timeout
                )
            else:
                # Fallback: no config
                response = await asyncio.wait_for(
                    google_async_client.aio.models.generate_content(
                        model=model,
                        contents=full_prompt,
                    ),
                    timeout=timeout
                )
            response_text = response.text
            logger.debug(f"[{request_id}] Gemini response: {len(response_text)} chars")
            return response_text
        except asyncio.TimeoutError:
            if attempt == max_retries - 1:
                raise Exception(f"Gemini API call timed out after {timeout} seconds")
            await asyncio.sleep(1 * (attempt + 1))
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Gemini API error: {str(e)}")
            await asyncio.sleep(1 * (attempt + 1))

