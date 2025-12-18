"""Configuration management module.

This module provides high-level configuration management utilities:
- ConfigLoader: YAML configuration file loader and parser
- PromptManager: For managing prompt templates (to be implemented in Phase 7)
- Re-exports: Configuration classes from `core.config` for convenience

Structure:
- `core.config`: Configuration dataclasses (data structures)
- `config.config_loader`: YAML config loader (this module)
- `config/`: Unified configuration management interface
"""

# Re-export configuration classes for convenience
from ..core.config import (
    CoreConfig,
    ChunkingConfig,
    LLMConfig,
    StorageConfig,
    EmbeddingConfig,
    EntityConfig,
    GraphConfig,
    ExtensionConfig,
)

# Import config loader (now in this module)
from .config_loader import ConfigLoader

# Import PromptManager (Phase 4)
from .prompts import PromptManager, get_prompt_manager, PROMPTS, GRAPH_FIELD_SEP

__all__ = [
    # Config classes
    "CoreConfig",
    "ChunkingConfig",
    "LLMConfig",
    "StorageConfig",
    "EmbeddingConfig",
    "EntityConfig",
    "GraphConfig",
    "ExtensionConfig",
    # Config loader
    "ConfigLoader",
    # Prompt management (Phase 4)
    "PromptManager",
    "get_prompt_manager",
    "PROMPTS",
    "GRAPH_FIELD_SEP",
]


