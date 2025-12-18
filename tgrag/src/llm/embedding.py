"""LLM embedding functions for various providers."""

import os
import json
import logging
from typing import List, Optional

import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import RateLimitError

from .client import get_client_manager
from ..utils import EmbeddingFunc, wrap_embedding_func_with_attrs

logger = logging.getLogger("temporal-graphrag.llm")


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError,)),
)
async def openai_embedding(
    texts: List[str],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> np.ndarray:
    """Generate embeddings using OpenAI.
    
    Args:
        texts: List of texts to embed
        api_key: OpenAI API key (optional, defaults to env var)
        base_url: Custom base URL (optional, defaults to env var or OpenAI API)
        
    Returns:
        NumPy array of embeddings
    """
    client_manager = get_client_manager()
    openai_client = client_manager.get_openai_client(api_key=api_key, base_url=base_url)
    
    response = await openai_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError,)),
)
async def azure_openai_embedding(texts: List[str]) -> np.ndarray:
    """Generate embeddings using Azure OpenAI.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        NumPy array of embeddings
    """
    client_manager = get_client_manager()
    azure_client = client_manager.get_azure_client()
    
    response = await azure_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError,)),
)
async def amazon_bedrock_embedding(texts: List[str]) -> np.ndarray:
    """Generate embeddings using Amazon Bedrock.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        NumPy array of embeddings
    """
    import aioboto3
    
    client_manager = get_client_manager()
    bedrock_session = client_manager.get_bedrock_session()
    
    async with bedrock_session.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    ) as bedrock_runtime:
        embeddings = []
        for text in texts:
            body = json.dumps(
                {
                    "inputText": text,
                    "dimensions": 1024,
                }
            )
            response = await bedrock_runtime.invoke_model(
                modelId="amazon.titan-embed-text-v2:0", body=body,
            )
            response_body = await response.get("body").read()
            embeddings.append(json.loads(response_body))
    return np.array([dp["embedding"] for dp in embeddings])


@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=8192)
async def ollama_embedding(
    texts: List[str],
    model: str = "nomic-embed-text",
    base_url: Optional[str] = None
) -> np.ndarray:
    """Generate embeddings using Ollama.
    
    Args:
        texts: List of texts to embed
        model: Ollama embedding model name (default: nomic-embed-text)
        base_url: Ollama base URL
        
    Returns:
        NumPy array of embeddings
    """
    import aiohttp
    
    client_manager = get_client_manager()
    ollama_url = client_manager.get_ollama_base_url(base_url)
    
    embeddings = []
    async with aiohttp.ClientSession() as session:
        for text in texts:
            payload = {
                "model": model,
                "prompt": text,
            }
            
            async with session.post(
                f"{ollama_url}/api/embeddings",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama embedding API error {response.status}: {error_text}")
                
                result = await response.json()
                embedding = result.get("embedding", [])
                embeddings.append(embedding)
    
    return np.array(embeddings)


@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=8192)
async def gemini_embedding(
    texts: List[str],
    model: str = "models/embedding-001"
) -> np.ndarray:
    """Generate embeddings using Google Gemini.
    
    Args:
        texts: List of texts to embed
        model: Gemini embedding model name
        
    Returns:
        NumPy array of embeddings
    """
    client_manager = get_client_manager()
    google_async_client = client_manager.get_gemini_client()
    
    # Gemini embed_content API expects 'contents' (plural) as a list
    # Process all texts at once (batched)
    try:
        result = await google_async_client.aio.models.embed_content(
            model=model,
            contents=texts  # 'contents' is plural and expects a list of strings
        )
        
        # Handle response - Gemini returns embeddings in result.embeddings list
        if hasattr(result, "embeddings") and result.embeddings:
            # Result has an embeddings attribute with a list
            embeddings_list = [emb.values for emb in result.embeddings]
            return np.array(embeddings_list)
        elif hasattr(result, "embedding"):
            # Single embedding (shouldn't happen with multiple texts, but handle it)
            return np.array([result.embedding])
        elif isinstance(result, dict):
            # Dict format
            if "embeddings" in result:
                return np.array(result["embeddings"])
            elif "embedding" in result:
                return np.array([result["embedding"]])
            else:
                raise ValueError(f"Unexpected result format: {type(result)}")
        else:
            raise ValueError(f"Unexpected result type: {type(result)}")
    except Exception as e:
        logger.error(f"Gemini embedding error: {e}")
        raise

