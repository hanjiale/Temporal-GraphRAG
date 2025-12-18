"""LLM completion functions with caching and retry logic."""

import logging
from typing import Optional, List, Any, Dict, Callable

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from openai import RateLimitError

from ..storage.base import BaseKVStorage
from ..utils import compute_args_hash
from .client import (
    get_client_manager,
    generate_request_id,
    ollama_complete,
    gemini_complete,
)

logger = logging.getLogger("temporal-graphrag.llm")


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError,)),
)
async def openai_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, Any]] = [],
    hashing_kv: Optional[BaseKVStorage] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> str:
    """Complete a prompt using OpenAI with caching support."""
    client_manager = get_client_manager()
    openai_client = client_manager.get_openai_client(api_key=api_key, base_url=base_url)
    
    request_id = generate_request_id()
    logger.debug(f"[{request_id}] OpenAI request: model={model}, base_url={base_url or 'default'}")
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    # Check cache
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        cached_response = await hashing_kv.get_by_id(args_hash)
        if cached_response is not None:
            logger.debug(f"[{request_id}] Cache hit")
            return cached_response["return"]
        logger.debug(f"[{request_id}] Cache miss")
    
    # Make API call
    response = await openai_client.chat.completions.create(
        model=model, messages=messages, timeout=120.0, **kwargs
    )
    
    response_text = response.choices[0].message.content
    
    # Store in cache
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response_text, "model": model}}
        )
        await hashing_kv.index_done_callback()
    
    logger.debug(f"[{request_id}] OpenAI response: {len(response_text)} chars")
    return response_text


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError,)),
)
async def azure_openai_complete_if_cache(
    deployment_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, Any]] = [],
    hashing_kv: Optional[BaseKVStorage] = None,
    **kwargs
) -> str:
    """Complete a prompt using Azure OpenAI with caching support.
    
    Args:
        deployment_name: Azure deployment name
        prompt: User prompt
        system_prompt: Optional system prompt
        history_messages: Conversation history
        hashing_kv: Optional cache storage
        **kwargs: Additional parameters for Azure OpenAI API
        
    Returns:
        Generated text response
    """
    client_manager = get_client_manager()
    azure_client = client_manager.get_azure_client()
    
    request_id = generate_request_id()
    logger.debug(f"[{request_id}] Azure OpenAI request: deployment={deployment_name}")
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    # Check cache
    if hashing_kv is not None:
        args_hash = compute_args_hash(deployment_name, messages)
        cached_response = await hashing_kv.get_by_id(args_hash)
        if cached_response is not None:
            logger.debug(f"[{request_id}] Cache hit")
            return cached_response["return"]
        logger.debug(f"[{request_id}] Cache miss")
    
    # Make API call
    response = await azure_client.chat.completions.create(
        model=deployment_name, messages=messages, timeout=120.0, **kwargs
    )
    
    response_text = response.choices[0].message.content
    
    # Store in cache
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": response_text,
                    "model": deployment_name,
                }
            }
        )
        await hashing_kv.index_done_callback()
    
    logger.debug(f"[{request_id}] Azure OpenAI response: {len(response_text)} chars")
    return response_text


async def amazon_bedrock_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, Any]] = [],
    hashing_kv: Optional[BaseKVStorage] = None,
    **kwargs
) -> str:
    """Complete a prompt using Amazon Bedrock with caching support.
    
    Args:
        model: Bedrock model ID
        prompt: User prompt
        system_prompt: Optional system prompt
        history_messages: Conversation history
        hashing_kv: Optional cache storage
        **kwargs: Additional parameters
        
    Returns:
        Generated text response
    """
    import os
    import json
    
    client_manager = get_client_manager()
    bedrock_session = client_manager.get_bedrock_session()
    
    request_id = generate_request_id()
    logger.debug(f"[{request_id}] Bedrock request: model={model}")
    
    messages = []
    messages.extend(history_messages)
    messages.append({"role": "user", "content": [{"text": prompt}]})
    
    # Check cache
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        cached_response = await hashing_kv.get_by_id(args_hash)
        if cached_response is not None:
            logger.debug(f"[{request_id}] Cache hit")
            return cached_response["return"]
        logger.debug(f"[{request_id}] Cache miss")
    
    inference_config = {
        "temperature": 0,
        "maxTokens": 4096 if "max_tokens" not in kwargs else kwargs["max_tokens"],
    }
    
    async with bedrock_session.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    ) as bedrock_runtime:
        if system_prompt:
            response = await bedrock_runtime.converse(
                modelId=model, messages=messages, inferenceConfig=inference_config,
                system=[{"text": system_prompt}]
            )
        else:
            response = await bedrock_runtime.converse(
                modelId=model, messages=messages, inferenceConfig=inference_config,
            )
    
    response_text = response["output"]["message"]["content"][0]["text"]
    
    # Store in cache
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response_text, "model": model}}
        )
        await hashing_kv.index_done_callback()
    
    logger.debug(f"[{request_id}] Bedrock response: {len(response_text)} chars")
    return response_text


async def ollama_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, Any]] = [],
    hashing_kv: Optional[BaseKVStorage] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> str:
    """Complete a prompt using Ollama with caching support.
    
    Args:
        model: Ollama model name
        prompt: User prompt
        system_prompt: Optional system prompt
        history_messages: Conversation history
        hashing_kv: Optional cache storage
        base_url: Ollama base URL
        **kwargs: Additional parameters
        
    Returns:
        Generated text response
    """
    request_id = generate_request_id()
    logger.debug(f"[{request_id}] Ollama request: model={model}")
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    # Check cache
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        cached_response = await hashing_kv.get_by_id(args_hash)
        if cached_response is not None:
            logger.debug(f"[{request_id}] Cache hit")
            return cached_response["return"]
        logger.debug(f"[{request_id}] Cache miss")
    
    # Make API call
    response_text = await ollama_complete(
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=base_url,
        **kwargs
    )
    
    # Store in cache
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response_text, "model": model}}
        )
        await hashing_kv.index_done_callback()
    
    logger.debug(f"[{request_id}] Ollama response: {len(response_text)} chars")
    return response_text


async def gemini_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, Any]] = [],
    hashing_kv: Optional[BaseKVStorage] = None,
    **kwargs
) -> str:
    """Complete a prompt using Google Gemini with caching support.
    
    Args:
        model: Gemini model name
        prompt: User prompt
        system_prompt: Optional system prompt
        history_messages: Conversation history
        hashing_kv: Optional cache storage
        **kwargs: Additional parameters
        
    Returns:
        Generated text response
    """
    request_id = generate_request_id()
    logger.debug(f"[{request_id}] Gemini request: model={model}")
    
    # Build messages for cache key (Gemini doesn't use standard message format)
    messages = [prompt]
    if system_prompt:
        messages.insert(0, system_prompt)
    
    # Check cache
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        cached_response = await hashing_kv.get_by_id(args_hash)
        if cached_response is not None:
            logger.debug(f"[{request_id}] Cache hit")
            return cached_response["return"]
        logger.debug(f"[{request_id}] Cache miss")
    
    # Make API call
    response_text = await gemini_complete(
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )
    
    # Store in cache
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response_text, "model": model}}
        )
        await hashing_kv.index_done_callback()
    
    logger.debug(f"[{request_id}] Gemini response: {len(response_text)} chars")
    return response_text


# Convenience functions for specific models
async def gpt_4o_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, Any]] = [],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> str:
    """Complete using GPT-4o.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        history_messages: Conversation history
        api_key: OpenAI API key (optional)
        base_url: Custom base URL (optional)
        **kwargs: Additional parameters
    """
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, Any]] = [],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> str:
    """Complete using GPT-4o-mini.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        history_messages: Conversation history
        api_key: OpenAI API key (optional)
        base_url: Custom base URL (optional)
        **kwargs: Additional parameters
    """
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )


async def azure_gpt_4o_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, Any]] = [],
    **kwargs
) -> str:
    """Complete using Azure GPT-4o."""
    return await azure_openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_gpt_4o_mini_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, Any]] = [],
    **kwargs
) -> str:
    """Complete using Azure GPT-4o-mini."""
    return await azure_openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


def create_amazon_bedrock_complete_function(model_id: str) -> Callable:
    """Factory function to create Bedrock completion functions.
    
    Args:
        model_id: Bedrock model identifier
        
    Returns:
        Completion function
    """
    async def bedrock_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List[Dict[str, Any]] = [],
        **kwargs
    ) -> str:
        return await amazon_bedrock_complete_if_cache(
            model_id,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs
        )
    
    bedrock_complete.__name__ = f"{model_id}_complete"
    return bedrock_complete


def create_provider_complete_function(
    provider: str,
    model: str,
    **provider_kwargs
) -> Callable:
    """Factory function to create provider-specific completion functions.
    
    Args:
        provider: Provider name (openai, azure, bedrock, gemini, ollama)
        model: Model identifier
        **provider_kwargs: Provider-specific kwargs (e.g., api_key, base_url for OpenAI)
        
    Returns:
        Completion function
    """
    async def provider_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List[Dict[str, Any]] = [],
        hashing_kv: Optional[BaseKVStorage] = None,
        **kwargs
    ) -> str:
        if provider == "openai":
            # Extract api_key and base_url from provider_kwargs or kwargs
            api_key = kwargs.pop("api_key", provider_kwargs.get("api_key"))
            base_url = kwargs.pop("base_url", provider_kwargs.get("base_url"))
            return await openai_complete_if_cache(
                model, prompt, system_prompt, history_messages, hashing_kv,
                api_key=api_key, base_url=base_url, **kwargs
            )
        elif provider == "azure":
            return await azure_openai_complete_if_cache(
                model, prompt, system_prompt, history_messages, hashing_kv, **kwargs
            )
        elif provider == "bedrock":
            return await amazon_bedrock_complete_if_cache(
                model, prompt, system_prompt, history_messages, hashing_kv, **kwargs
            )
        elif provider == "gemini":
            return await gemini_complete_if_cache(
                model, prompt, system_prompt, history_messages, hashing_kv, **kwargs
            )
        elif provider == "ollama":
            return await ollama_complete_if_cache(
                model, prompt, system_prompt, history_messages, hashing_kv,
                base_url=provider_kwargs.get("base_url"),
                **kwargs
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    provider_complete.__name__ = f"{provider}_{model}_complete"
    return provider_complete

