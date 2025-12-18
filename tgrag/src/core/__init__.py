"""Core components and main TemporalGraphRAG class."""

from .types import (
    QueryParam,
    TextChunkSchema,
    SingleCommunitySchema,
    SingleTemporalSchema,
    CommunitySchema,
    TemporalSchema,
)

# Import new modules (Phase 4) - lazy import to avoid dependency issues during import
__all__ = [
    "QueryParam",
    "TextChunkSchema",
    "SingleCommunitySchema",
    "SingleTemporalSchema",
    "CommunitySchema",
    "TemporalSchema",
    # Phase 4 modules (imported lazily)
    "chunking",
    "building",
    "querying",
]

# Lazy import modules to avoid triggering dependencies during package import
# Modules are imported on first access
def __getattr__(name):
    if name == "chunking":
        from . import chunking
        return chunking
    elif name == "building":
        from . import building
        return building
    elif name == "querying":
        from . import querying
        return querying
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


