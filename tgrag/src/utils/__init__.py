"""Utility functions and helpers."""

from .cost_tracker import CostTracker
from .types import (
    EmbeddingFunc,
    compute_args_hash,
    wrap_embedding_func_with_attrs,
)

# Import from new utility modules (migrated from OLD)
from .hashing import compute_mdhash_id
from .async_utils import always_get_an_event_loop, limit_async_func_call
from .json_utils import convert_response_to_json

# Import remaining utilities from helpers (still re-exporting from OLD for now)
from .helpers import (
    logger,
    clean_str,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    get_parent_timestamp_name,
    infer_timestamp_level,
    complete_timestamp_range_by_level,
    sort_timestamp_by_datetime,
    convert_timestamp_to_datetime,
)

__all__ = [
    "CostTracker",
    "EmbeddingFunc",
    "compute_args_hash",
    "wrap_embedding_func_with_attrs",
    # Migrated utilities (from new modules)
    "compute_mdhash_id",
    "always_get_an_event_loop",
    "limit_async_func_call",
    "convert_response_to_json",
    # Helper utilities (temporary re-exports from OLD)
    "logger",
    "clean_str",
    "decode_tokens_by_tiktoken",
    "encode_string_by_tiktoken",
    "is_float_regex",
    "list_of_list_to_csv",
    "pack_user_ass_to_openai_messages",
    "split_string_by_multi_markers",
    "truncate_list_by_token_size",
    "get_parent_timestamp_name",
    "infer_timestamp_level",
    "complete_timestamp_range_by_level",
    "sort_timestamp_by_datetime",
    "convert_timestamp_to_datetime",
]


