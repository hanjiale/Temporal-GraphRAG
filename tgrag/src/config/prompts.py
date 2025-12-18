"""Prompt management for loading and accessing prompt templates from YAML configuration."""

from __future__ import annotations

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

logger = logging.getLogger("temporal-graphrag.config")

# Default graph field separator (used as fallback if not in YAML)
_DEFAULT_GRAPH_FIELD_SEP = "<SEP>"


class PromptManager:
    """
    Manages prompt templates loaded from YAML configuration.
    
    Provides runtime loading, caching, and version management for prompts.
    
    Attributes:
        prompts: Dictionary of prompt templates
        version: Version string from YAML config
        _yaml_path: Path to prompts YAML file
    """
    
    def __init__(self, yaml_path: Optional[str] = None):
        """Initialize PromptManager."""
        self.prompts: Dict[str, Any] = {}
        self.version: Optional[str] = None
        self._yaml_path: Optional[str] = yaml_path
        
        # Load prompts
        self._load_prompts()
    
    def _get_default_yaml_path(self) -> str:
        """Get default path to prompts.yaml."""
        # Try to find prompts.yaml relative to this file
        current_dir = Path(__file__).parent.parent.parent
        yaml_path = current_dir / "configs" / "prompts.yaml"
        if yaml_path.exists():
            return str(yaml_path)
        
        # Fallback: try in tgrag/configs/
        alt_path = Path(__file__).parent.parent / "configs" / "prompts.yaml"
        if alt_path.exists():
            return str(alt_path)
        
        # Last resort: current directory
        return "prompts.yaml"
    
    def _load_prompts(self) -> None:
        """Load prompts from YAML file."""
        yaml_path = self._yaml_path or self._get_default_yaml_path()
        
        if not os.path.exists(yaml_path):
            logger.warning(f"Prompts YAML not found at {yaml_path}")
            raise FileNotFoundError(f"Prompts YAML file not found at {yaml_path}")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not data:
                logger.warning(f"Prompts YAML is empty at {yaml_path}")
                raise ValueError(f"Prompts YAML file is empty at {yaml_path}")
            
            # Extract version if present
            self.version = data.get('version', '1.0')
            
            # Extract graph field separator from YAML (single source of truth)
            graph_field_sep = data.get('graph_field_sep', _DEFAULT_GRAPH_FIELD_SEP)
            self.prompts['GRAPH_FIELD_SEP'] = graph_field_sep
            
            # Load defaults
            defaults = data.get('defaults', {})
            self._load_defaults(defaults)
            
            # Load entity extraction prompts
            entity_extraction = data.get('entity_extraction', {})
            self._load_entity_extraction_prompts(entity_extraction)
            
            # Load query prompts
            query = data.get('query', {})
            self._load_query_prompts(query)
            
            # Load building prompts
            building = data.get('building', {})
            self._load_building_prompts(building)
            
            # Load claim extraction prompts
            claim_extraction = data.get('claim_extraction', {})
            self._load_claim_extraction_prompts(claim_extraction)
            
            logger.info(f"Loaded {len(self.prompts)} prompts from {yaml_path} (version: {self.version})")
            
        except Exception as e:
            logger.error(f"Error loading prompts from {yaml_path}: {e}")
            raise
    
    def _load_defaults(self, defaults: Dict[str, Any]) -> None:
        """Load default configuration values."""
        # Tuple delimiter
        self.prompts['DEFAULT_TUPLE_DELIMITER'] = defaults.get('tuple_delimiter', '<|>')
        
        # Record delimiter
        self.prompts['DEFAULT_RECORD_DELIMITER'] = defaults.get('record_delimiter', '##')
        
        # Completion delimiter
        self.prompts['DEFAULT_COMPLETION_DELIMITER'] = defaults.get('completion_delimiter', '<|COMPLETE|>')
        
        # Entity types
        entity_types = defaults.get('entity_types', ['financial concept', 'business segment', 'event', 'company'])
        self.prompts['DEFAULT_ENTITY_TYPES'] = entity_types
        
        # Temporal hierarchy
        temporal_hierarchy = defaults.get('temporal_hierarchy', ['year', 'quarter', 'month', 'date'])
        self.prompts['DEFAULT_TEMPORAL_HIERARCHY'] = temporal_hierarchy
        
        # Temporal hierarchy level
        temporal_hierarchy_level = defaults.get('temporal_hierarchy_level', {
            'year': 0, 'quarter': 1, 'month': 2, 'week': 2, 'season': 2, 'date': 3, 'UNKNOWN': 3
        })
        self.prompts['DEFAULT_TEMPORAL_HIERARCHY_LEVEL'] = temporal_hierarchy_level
        
        # Timestamp format
        timestamp_format = defaults.get('timestamp_format', {
            'year': 'YYYY', 'quarter': 'YYYY-QN', 'month': 'YYYY-MM', 'date': 'YYYY-MM-DD'
        })
        self.prompts['DEFAULT_TIMESTAMP_FORMAT'] = timestamp_format
        
        # Text separators
        text_separator = defaults.get('text_separator', ['\n\n', '\r\n\r\n', '\n', '\r\n'])
        self.prompts['default_text_separator'] = text_separator
        
        # Process tickers
        process_tickers = defaults.get('process_tickers', ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.prompts['process_tickers'] = process_tickers
    
    def _load_entity_extraction_prompts(self, entity_extraction: Dict[str, Any]) -> None:
        """Load entity extraction prompts."""
        self.prompts['entity_extraction'] = entity_extraction.get('entity_extraction', '')
        self.prompts['temporal_entity_extraction_new'] = entity_extraction.get('temporal_entity_extraction_new', '')
        self.prompts['temporal_entity_extraction_old'] = entity_extraction.get('temporal_entity_extraction_old', '')
        self.prompts['entiti_continue_extraction'] = entity_extraction.get('entiti_continue_extraction', '')
        self.prompts['entiti_if_loop_extraction'] = entity_extraction.get('entiti_if_loop_extraction', '')
    
    def _load_query_prompts(self, query: Dict[str, Any]) -> None:
        """Load query prompts."""
        self.prompts['extract_timestamp_in_query'] = query.get('extract_timestamp_in_query', '')
        self.prompts['extract_temporal_hierarchy'] = query.get('extract_temporal_hierarchy', '')
        self.prompts['local_rag_response'] = query.get('local_rag_response', '')
        self.prompts['global_map_rag_points'] = query.get('global_map_rag_points', '')
        self.prompts['global_reduce_rag_response'] = query.get('global_reduce_rag_response', '')
        self.prompts['naive_rag_response'] = query.get('naive_rag_response', '')
        self.prompts['fail_response'] = query.get('fail_response', "Sorry, I'm not able to provide an answer to that question.")
    
    def _load_building_prompts(self, building: Dict[str, Any]) -> None:
        """Load building prompts."""
        self.prompts['community_report'] = building.get('community_report', '')
        self.prompts['temporal_community_report'] = building.get('temporal_community_report', '')
        self.prompts['summarize_entity_descriptions'] = building.get('summarize_entity_descriptions', '')
    
    def _load_claim_extraction_prompts(self, claim_extraction: Dict[str, Any]) -> None:
        """Load claim extraction prompts."""
        self.prompts['claim_extraction'] = claim_extraction.get('claim_extraction', '')
    
    
    def get(self, key: str, default: Optional[str] = None) -> Any:
        """
        Get a prompt template by key.
        
        Args:
            key: Prompt template key
            default: Default value if key not found
            
        Returns:
            Prompt template string or default value
        """
        return self.prompts.get(key, default)
    
    def format_prompt(self, key: str, **kwargs) -> str:
        """
        Get and format a prompt template with provided variables.
        
        Args:
            key: Prompt template key
            **kwargs: Variables to format into the template
            
        Returns:
            Formatted prompt string
        """
        template = self.get(key)
        if template is None:
            raise KeyError(f"Prompt template '{key}' not found")
        
        try:
            # Format with defaults if available
            format_vars = {
                'tuple_delimiter': self.prompts.get('DEFAULT_TUPLE_DELIMITER', '<|>'),
                'record_delimiter': self.prompts.get('DEFAULT_RECORD_DELIMITER', '##'),
                'completion_delimiter': self.prompts.get('DEFAULT_COMPLETION_DELIMITER', '<|COMPLETE|>'),
                **kwargs
            }
            return template.format(**format_vars)
        except KeyError as e:
            logger.warning(f"Missing format variable {e} in prompt '{key}', using template as-is")
            return template
    
    def validate_template(self, key: str) -> bool:
        """
        Validate that a prompt template exists and is non-empty.
        
        Args:
            key: Prompt template key
            
        Returns:
            True if template exists and is non-empty, False otherwise
        """
        template = self.get(key)
        return template is not None and len(str(template).strip()) > 0


# Global singleton instance
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager(yaml_path: Optional[str] = None) -> PromptManager:
    """Get or create the global PromptManager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager(yaml_path=yaml_path)
    return _prompt_manager


# Backward compatibility exports
def get_prompts() -> Dict[str, Any]:
    """
    Get prompts dictionary for backward compatibility.
    
    Returns:
        Dictionary of prompt templates
    """
    return get_prompt_manager().prompts


# Export PROMPTS dict and GRAPH_FIELD_SEP for backward compatibility
# PROMPTS will be initialized lazily on first access
def _get_prompts_dict() -> Dict[str, Any]:
    """Lazy initialization of PROMPTS dict for backward compatibility."""
    return get_prompt_manager().prompts

# Create a proxy object that behaves like a dict
class _PromptsDict:
    """Proxy dict that lazily loads prompts."""
    def __getitem__(self, key: str) -> Any:
        return _get_prompts_dict()[key]
    
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return _get_prompts_dict().get(key, default)
    
    def __contains__(self, key: str) -> bool:
        return key in _get_prompts_dict()
    
    def keys(self):
        return _get_prompts_dict().keys()
    
    def values(self):
        return _get_prompts_dict().values()
    
    def items(self):
        return _get_prompts_dict().items()
    
    def copy(self) -> Dict[str, Any]:
        return _get_prompts_dict().copy()

PROMPTS = _PromptsDict()


# Export GRAPH_FIELD_SEP - reads from YAML via PromptManager (single source of truth)
# This is initialized lazily on first access to ensure PromptManager is ready
def _init_graph_field_sep() -> str:
    """Initialize GRAPH_FIELD_SEP from YAML configuration."""
    return get_prompt_manager().prompts.get('GRAPH_FIELD_SEP', _DEFAULT_GRAPH_FIELD_SEP)

# Use a simple string initialized from YAML
# The PromptManager initializes on first access, so this will read from YAML
GRAPH_FIELD_SEP: str = _init_graph_field_sep()

