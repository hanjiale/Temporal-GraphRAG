"""Hashing utilities for generating content IDs."""

from hashlib import md5
from typing import Union


def compute_mdhash_id(content: Union[str, bytes], prefix: str = "") -> str:
    """
    Compute MD5 hash ID for content.
    
    Args:
        content: Content string or bytes to hash
        prefix: Optional prefix to prepend to the hash
        
    Returns:
        String with format: {prefix}{md5_hash}
        
    Example:
        >>> compute_mdhash_id("test content", prefix="chunk-")
        'chunk-a1b2c3d4e5f6...'
    """
    if isinstance(content, bytes):
        content_bytes = content
    else:
        content_bytes = content.encode()
    return prefix + md5(content_bytes).hexdigest()

__all__ = ["compute_mdhash_id"]

