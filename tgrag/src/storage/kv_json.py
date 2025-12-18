"""
JSON-based key-value storage implementation.

This module provides a simple file-based key-value storage backend using JSON.
Suitable for small to medium-sized datasets that can fit in memory.
"""

import os
from dataclasses import dataclass

# Import from utils (helpers.py re-exports from OLD for now, will migrate in Phase 6)
from ..utils.helpers import load_json, write_json
from ..utils.logging import get_logger

logger = get_logger(__name__)
from .base import BaseKVStorage


@dataclass
class JsonKVStorage(BaseKVStorage):
    """JSON-based key-value storage implementation.
    
    Stores data in a JSON file on disk. All data is loaded into memory
    during initialization and persisted on index_done_callback.
    
    Attributes:
        namespace: Unique namespace identifier for this storage instance
        global_config: Global configuration dictionary
    """
    
    def __post_init__(self):
        """Initialize the JSON storage backend."""
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        """Get all keys in the storage."""
        return list(self._data.keys())

    async def index_done_callback(self):
        """Persist data to disk when indexing is complete."""
        write_json(self._data, self._file_name)

    async def get_by_id(self, id: str):
        """Get a value by its ID."""
        return self._data.get(id, None)

    async def get_by_ids(self, ids: list[str], fields: set[str] | None = None):
        """Get multiple values by their IDs.
        
        Args:
            ids: List of keys to retrieve
            fields: Optional set of field names to include in results
            
        Returns:
            List of values (or None for missing keys) in the same order as ids
        """
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        """Filter out keys that already exist in storage."""
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        """Upsert key-value pairs into storage."""
        self._data.update(data)

    async def drop(self):
        """Drop all data from storage."""
        self._data = {}

