"""Text chunking utilities for splitting documents into manageable pieces."""

from __future__ import annotations

import tiktoken
from typing import List, Optional, Dict, Any, Callable, Literal, Union
import logging

logger = logging.getLogger("temporal-graphrag.chunking")

# Import from new hashing module (migrated from OLD)
from ..utils.hashing import compute_mdhash_id

# Import prompts - will use PromptManager once fully integrated
try:
    from ..config.prompts import get_prompt_manager
    _prompt_manager = None
    def _get_prompts():
        global _prompt_manager
        if _prompt_manager is None:
            _prompt_manager = get_prompt_manager()
        return _prompt_manager.prompts
except ImportError:
    # PromptManager should be available - if not, provide minimal fallback
    def _get_prompts():
        logger.warning("PromptManager not available, using minimal fallback prompts")
        return {"default_text_separator": ["\n\n", "\r\n\r\n", "\n", "\r\n", ".", "!", "?", " ", "\t"]}


class SeparatorSplitter:
    """Splits token sequences using specified separators."""
    
    def __init__(
        self,
        separators: Optional[List[List[int]]] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = "end",
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[List[int]], int] = len,
    ):
        """Initialize SeparatorSplitter."""
        self._separators = separators or []
        self._keep_separator = keep_separator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function

    def split_tokens(self, tokens: List[int]) -> List[List[int]]:
        """Split tokens into chunks using separators."""
        splits = self._split_tokens_with_separators(tokens)
        return self._merge_splits(splits)

    def _split_tokens_with_separators(self, tokens: List[int]) -> List[List[int]]:
        """Split tokens at separator boundaries."""
        splits = []
        current_split = []
        i = 0
        while i < len(tokens):
            separator_found = False
            for separator in self._separators:
                if tokens[i:i+len(separator)] == separator:
                    if self._keep_separator in [True, "end"]:
                        current_split.extend(separator)
                    if current_split:
                        splits.append(current_split)
                        current_split = []
                    if self._keep_separator == "start":
                        current_split.extend(separator)
                    i += len(separator)
                    separator_found = True
                    break
            if not separator_found:
                current_split.append(tokens[i])
                i += 1
        if current_split:
            splits.append(current_split)
        return [s for s in splits if s]

    def _merge_splits(self, splits: List[List[int]]) -> List[List[int]]:
        """Merge splits into chunks respecting size constraints."""
        if not splits:
            return []

        merged_splits = []
        current_chunk = []

        for split in splits:
            if not current_chunk:
                current_chunk = split
            elif self._length_function(current_chunk) + self._length_function(split) <= self._chunk_size:
                current_chunk.extend(split)
            else:
                merged_splits.append(current_chunk)
                current_chunk = split

        if current_chunk:
            merged_splits.append(current_chunk)

        if len(merged_splits) == 1 and self._length_function(merged_splits[0]) > self._chunk_size:
            return self._split_chunk(merged_splits[0])

        if self._chunk_overlap > 0:
            return self._enforce_overlap(merged_splits)
        
        return merged_splits

    def _split_chunk(self, chunk: List[int]) -> List[List[int]]:
        """Split a single oversized chunk."""
        result = []
        for i in range(0, len(chunk), self._chunk_size - self._chunk_overlap):
            new_chunk = chunk[i:i + self._chunk_size]
            if len(new_chunk) > self._chunk_overlap:
                result.append(new_chunk)
        return result

    def _enforce_overlap(self, chunks: List[List[int]]) -> List[List[int]]:
        """Add overlap between chunks."""
        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
            else:
                overlap = chunks[i-1][-self._chunk_overlap:]
                new_chunk = overlap + chunk
                if self._length_function(new_chunk) > self._chunk_size:
                    new_chunk = new_chunk[:self._chunk_size]
                result.append(new_chunk)
        return result


def chunking_by_token_size(
    tokens_list: List[List[int]],
    doc_keys: List[str],
    tiktoken_model: tiktoken.Encoding,
    title_tokens_list: Optional[List[List[int]]] = None,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
) -> List[Dict[str, Any]]:
    """
    Chunk documents by fixed token size with optional overlap.
    
    This function splits documents into chunks of approximately max_token_size tokens,
    with optional overlap between chunks. Supports optional title tokens that are
    prepended to each chunk.
    
    Args:
        tokens_list: List of token sequences, one per document
        doc_keys: List of document identifiers
        tiktoken_model: Tiktoken encoding model for decoding
        title_tokens_list: Optional list of title token sequences
        overlap_token_size: Number of tokens to overlap between chunks
        max_token_size: Maximum tokens per chunk
        
    Returns:
        List of chunk dictionaries with keys: tokens, content, chunk_order_index, full_doc_id
        
    Example:
        >>> import tiktoken
        >>> encoder = tiktoken.encoding_for_model("gpt-4o")
        >>> tokens = [encoder.encode("This is a long document...")]
        >>> chunks = chunking_by_token_size(tokens, ["doc1"], encoder, max_token_size=100)
    """
    if not tokens_list:
        return []
    
    if len(tokens_list) != len(doc_keys):
        raise ValueError(f"tokens_list length ({len(tokens_list)}) must match doc_keys length ({len(doc_keys)})")
    
    results = []
    if not title_tokens_list:
        title_tokens_list = [[] for _ in tokens_list]
    
    if len(title_tokens_list) != len(tokens_list):
        raise ValueError(f"title_tokens_list length ({len(title_tokens_list)}) must match tokens_list length ({len(tokens_list)})")
    
    for index, (tokens, title_tokens) in enumerate(zip(tokens_list, title_tokens_list)):
        chunk_token = []
        lengths = []
        max_token_size_minus_title = max_token_size - len(title_tokens)
        
        if max_token_size_minus_title <= 0:
            raise ValueError(f"max_token_size ({max_token_size}) must be greater than title length ({len(title_tokens)})")
        
        step_size = max(1, max_token_size_minus_title - overlap_token_size)
        for start in range(0, len(tokens), step_size):
            chunk = title_tokens + tokens[start: start + max_token_size_minus_title]
            chunk_token.append(chunk)
            lengths.append(min(max_token_size, len(tokens) - start + len(title_tokens)))

        # Decode batch for efficiency
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):
            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def chunking_by_separators(
    tokens_list: List[List[int]],
    doc_keys: List[str],
    tiktoken_model: tiktoken.Encoding,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
) -> List[Dict[str, Any]]:
    """
    Chunk documents by natural separators (paragraphs, sentences, etc.).
    
    This function splits documents at natural boundaries like paragraph breaks,
    sentence endings, etc., while respecting chunk size constraints.
    
    Args:
        tokens_list: List of token sequences, one per document
        doc_keys: List of document identifiers
        tiktoken_model: Tiktoken encoding model for encoding separators and decoding chunks
        overlap_token_size: Number of tokens to overlap between chunks
        max_token_size: Maximum tokens per chunk
        
    Returns:
        List of chunk dictionaries with keys: tokens, content, chunk_order_index, full_doc_id
        
    Example:
        >>> import tiktoken
        >>> encoder = tiktoken.encoding_for_model("gpt-4o")
        >>> tokens = [encoder.encode("Paragraph 1.\\n\\nParagraph 2.")]
        >>> chunks = chunking_by_separators(tokens, ["doc1"], encoder, max_token_size=100)
    """
    if not tokens_list:
        return []
    
    if len(tokens_list) != len(doc_keys):
        raise ValueError(f"tokens_list length ({len(tokens_list)}) must match doc_keys length ({len(doc_keys)})")
    
    # Get text separators from prompts
    prompts = _get_prompts()
    text_separators = prompts.get("default_text_separator", ["\n\n", "\r\n\r\n", "\n", "\r\n", ".", "!", "?", " ", "\t"])
    
    splitter = SeparatorSplitter(
        separators=[
            tiktoken_model.encode(s) for s in text_separators
        ],
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = splitter.split_tokens(tokens)
        lengths = [len(c) for c in chunk_token]

        # Decode batch for efficiency
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):
            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def get_chunks(
    new_doc_dicts: Dict[str, Dict[str, str]],
    chunk_func: Callable = chunking_by_token_size,
    **chunk_func_params: Any
) -> Dict[str, Dict[str, Any]]:
    """
    Process documents and convert them into chunks.
    
    This function takes a dictionary of documents, encodes them using tiktoken,
    applies the specified chunking function, and returns a dictionary of chunks
    keyed by their MD5 hash IDs.
    
    Args:
        new_doc_dicts: Dictionary mapping doc_id to dict with 'doc' and optional 'title' keys
        chunk_func: Chunking function to use (default: chunking_by_token_size)
        **chunk_func_params: Additional parameters to pass to chunk_func
        
    Returns:
        Dictionary mapping chunk_id (MD5 hash) to chunk data
        
    Example:
        >>> docs = {
        ...     "doc1": {"doc": "This is a document.", "title": "Title"}
        ... }
        >>> chunks = get_chunks(docs, max_token_size=100)
    """
    if not new_doc_dicts:
        return {}
    
    inserting_chunks = {}
    new_docs_list = list(new_doc_dicts.items())
    doc_contents = [new_doc[1]['doc'] for new_doc in new_docs_list]
    
    # Check if titles are present
    has_titles = new_docs_list[0][1].get('title', '')
    if has_titles:
        doc_titles = [new_doc[1].get('title', '') for new_doc in new_docs_list]
    else:
        doc_titles = None

    doc_keys = [new_doc[0] for new_doc in new_docs_list]

    # Use gpt-4o encoding by default (can be made configurable)
    ENCODER = tiktoken.encoding_for_model("gpt-4o")
    content_tokens = ENCODER.encode_batch(doc_contents, num_threads=16)
    
    if doc_titles:
        titles_with_separator = [title + "\n" if title else "" for title in doc_titles]
        title_tokens = ENCODER.encode_batch(titles_with_separator, num_threads=16)
        chunks = chunk_func(
            content_tokens,
            title_tokens_list=title_tokens,
            doc_keys=doc_keys,
            tiktoken_model=ENCODER,
            **chunk_func_params
        )
    else:
        chunks = chunk_func(
            content_tokens,
            doc_keys=doc_keys,
            tiktoken_model=ENCODER,
            **chunk_func_params
        )

    # Create dictionary keyed by MD5 hash of content
    for chunk in chunks:
        chunk_id = compute_mdhash_id(chunk["content"], prefix="chunk-")
        inserting_chunks[chunk_id] = chunk

    return inserting_chunks

