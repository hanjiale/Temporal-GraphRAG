"""Centralized temporal normalization module.

This module provides a centralized interface for temporal normalization using
the EnhancedTemporalNormalizer, with consistent fallback behavior and error handling.

The TemporalNormalizer class maintains a single instance of EnhancedTemporalNormalizer
to avoid redundant instantiation and ensure consistent behavior across the codebase.
"""

from __future__ import annotations

import re
import logging
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger("temporal-graphrag.temporal.normalization")

# Import enhanced normalizer from new location
try:
    from .normalizer import (
        EnhancedTemporalNormalizer,
        TemporalGranularity
    )
    ENHANCED_NORMALIZER_AVAILABLE = True
except ImportError:
    # Enhanced normalizer should be available from new location
    # If not, something is wrong with the migration
    ENHANCED_NORMALIZER_AVAILABLE = False
    EnhancedTemporalNormalizer = None
    TemporalGranularity = None
    logger.error("EnhancedTemporalNormalizer not found in new location. Migration may be incomplete.")

# Basic functions are no longer needed as fallback since we have EnhancedTemporalNormalizer
# The enhanced normalizer handles all cases, and we have basic pattern matching as final fallback
BASIC_FUNCTIONS_AVAILABLE = False


class TemporalNormalizer:
    """
    Centralized temporal normalizer using EnhancedTemporalNormalizer.
    
    This class provides a singleton-like interface for temporal normalization,
    ensuring consistent behavior and error handling across the codebase.
    The EnhancedTemporalNormalizer instance is created once and reused.
    
    Example:
        >>> normalizer = TemporalNormalizer()
        >>> level = normalizer.infer_timestamp_level("2024-Q1")
        >>> normalized = normalizer.normalize_timestamp("Q1 2024")
    """
    
    _instance: Optional['TemporalNormalizer'] = None
    _enhanced_normalizer: Optional[EnhancedTemporalNormalizer] = None
    
    def __new__(cls):
        """Create singleton instance of TemporalNormalizer."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the temporal normalizer (only once)."""
        if self._initialized:
            return
        
        if ENHANCED_NORMALIZER_AVAILABLE and EnhancedTemporalNormalizer is not None:
            try:
                self._enhanced_normalizer = EnhancedTemporalNormalizer()
                logger.debug("Initialized EnhancedTemporalNormalizer")
            except Exception as e:
                logger.warning(f"Failed to initialize EnhancedTemporalNormalizer: {e}")
                self._enhanced_normalizer = None
        else:
            self._enhanced_normalizer = None
        
        self._initialized = True
    
    def infer_timestamp_level(self, timestamp: str) -> str:
        """
        Infer the granularity level of a timestamp.
        
        Uses EnhancedTemporalNormalizer if available, with fallback to basic methods.
        
        Args:
            timestamp: Timestamp string to analyze
            
        Returns:
            Timestamp level string (e.g., "year", "quarter", "month", "date", "UNKNOWN")
            
        Example:
            >>> normalizer = TemporalNormalizer()
            >>> normalizer.infer_timestamp_level("2024-Q1")
            'quarter'
            >>> normalizer.infer_timestamp_level("2024-01-15")
            'date'
        """
        if self._enhanced_normalizer is not None:
            try:
                result = self._enhanced_normalizer.normalize_temporal_expression(timestamp)
                if result.normalized_forms and result.granularity:
                    return result.granularity.value
            except Exception as e:
                logger.warning(f"Enhanced normalizer failed for {timestamp}: {e}, using basic fallback")
        
        # Final fallback: basic pattern matching (no longer using OLD._utils functions)
        logger.warning(f"No temporal normalizer available, using basic pattern matching for {timestamp}")
        if re.match(r'^\d{4}$', timestamp):
            return "year"
        elif re.match(r'^\d{4}-Q[1-4]$', timestamp, re.IGNORECASE):
            return "quarter"
        elif re.match(r'^\d{4}-\d{2}$', timestamp):
            return "month"
        elif re.match(r'^\d{4}-\d{2}-\d{2}$', timestamp):
            return "date"
        return "UNKNOWN"
    
    def normalize_timestamp(self, timestamp: str) -> str:
        """
        Normalize a timestamp to standard format.
        
        Uses EnhancedTemporalNormalizer if available, with fallback to basic methods.
        Ensures date-level timestamps are in proper YYYY-MM-DD format.
        
        Args:
            timestamp: Timestamp string to normalize
            
        Returns:
            Normalized timestamp string
            
        Example:
            >>> normalizer = TemporalNormalizer()
            >>> normalizer.normalize_timestamp("Q1 2024")
            '2024-Q1'
            >>> normalizer.normalize_timestamp("Jan 15, 2024")
            '2024-01-15'
        """
        if self._enhanced_normalizer is not None:
            try:
                result = self._enhanced_normalizer.normalize_temporal_expression(timestamp)
                if result.normalized_forms:
                    normalized = result.normalized_forms[0]
                    # CRITICAL FIX: Additional validation for date-level timestamps
                    if result.granularity and result.granularity.value == "date":
                        # Ensure date is in proper YYYY-MM-DD format
                        if re.match(r'^\d{4}-\d{2}-\d{2}$', normalized):
                            return normalized
                        else:
                            logger.warning(f"Enhanced normalizer produced malformed date '{normalized}' for {timestamp}")
                            # Try to fix the format if it's close to correct
                            date_fix_match = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', normalized)
                            if date_fix_match:
                                year, month, day = date_fix_match.groups()
                                try:
                                    dt = datetime(int(year), int(month), int(day))
                                    corrected = f"{int(year)}-{int(month):02d}-{int(day):02d}"
                                    logger.info(f"Corrected date format from '{normalized}' to '{corrected}'")
                                    return corrected
                                except ValueError:
                                    pass
                    return normalized
            except Exception as e:
                logger.warning(f"Enhanced normalizer error for {timestamp}: {e}, using basic fallback")
        
        # Final fallback: return as-is (no longer using OLD._utils functions)
        logger.warning(f"No temporal normalizer available, returning timestamp as-is: {timestamp}")
        return timestamp
    
    def normalize_temporal_expression(self, timestamp: str):
        """
        Normalize temporal expression using EnhancedTemporalNormalizer.
        
        This method directly returns the result object from EnhancedTemporalNormalizer,
        which includes normalized_forms, granularity, confidence, etc.
        
        Args:
            timestamp: Timestamp string to normalize
            
        Returns:
            Normalization result object with normalized_forms, granularity, confidence, etc.
            Returns None if normalizer is not available.
            
        Example:
            >>> normalizer = TemporalNormalizer()
            >>> result = normalizer.normalize_temporal_expression("Q1 2024")
            >>> result.normalized_forms
            ['2024-Q1']
            >>> result.granularity.value
            'quarter'
        """
        if self._enhanced_normalizer is not None:
            try:
                return self._enhanced_normalizer.normalize_temporal_expression(timestamp)
            except Exception as e:
                logger.warning(f"Enhanced normalizer error for {timestamp}: {e}")
                return None
        
        logger.warning(f"EnhancedTemporalNormalizer not available for {timestamp}")
        return None


# Create a global instance for convenience
_global_normalizer: Optional[TemporalNormalizer] = None

def get_temporal_normalizer() -> TemporalNormalizer:
    """
    Get the global TemporalNormalizer instance.
    
    Returns:
        Singleton instance of TemporalNormalizer
        
    Example:
        >>> normalizer = get_temporal_normalizer()
        >>> level = normalizer.infer_timestamp_level("2024")
    """
    global _global_normalizer
    if _global_normalizer is None:
        _global_normalizer = TemporalNormalizer()
    return _global_normalizer

