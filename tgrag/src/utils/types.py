"""
Type definitions for Temporal GraphRAG utilities.
"""

from dataclasses import dataclass
from typing import Protocol
from functools import wraps
from hashlib import md5
import numpy as np


@dataclass
class EmbeddingFunc:
    """Embedding function wrapper with metadata.
    
    Attributes:
        embedding_dim: Dimension of the embedding vectors
        max_token_size: Maximum token size for the embedding model
        func: The actual embedding function to call
    """
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        """Call the embedding function asynchronously.
        
        Args:
            *args: Positional arguments for the embedding function
            **kwargs: Keyword arguments for the embedding function
            
        Returns:
            numpy array of embeddings
        """
        return await self.func(*args, **kwargs)


def compute_args_hash(*args) -> str:
    """Compute MD5 hash of arguments for caching.
    
    Args:
        *args: Arguments to hash
        
    Returns:
        MD5 hash hex string
    """
    return md5(str(args).encode()).hexdigest()


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with EmbeddingFunc attributes.
    
    Args:
        **kwargs: Attributes to set on EmbeddingFunc (embedding_dim, max_token_size, etc.)
        
    Returns:
        Decorator function
    """
    def final_decorator(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decorator

