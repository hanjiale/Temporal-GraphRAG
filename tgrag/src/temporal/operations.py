"""Temporal operations and utilities for timestamp handling.

This module provides helper functions for temporal operations used in building
and querying. Uses the centralized TemporalNormalizer for consistent behavior.
"""

from __future__ import annotations

import re
import logging
from typing import Optional, Tuple

from .normalization import get_temporal_normalizer

logger = logging.getLogger("temporal-graphrag.temporal")


def enhanced_infer_timestamp_level(timestamp: str) -> str:
    """
    Use enhanced temporal normalizer to infer timestamp level.
    
    This function uses the centralized TemporalNormalizer for consistent
    timestamp level inference, with fallback to basic methods.
    
    Args:
        timestamp: Timestamp string to analyze
        
    Returns:
        Timestamp level string (e.g., "year", "quarter", "month", "date")
        
    Example:
        >>> enhanced_infer_timestamp_level("2024-Q1")
        'quarter'
        >>> enhanced_infer_timestamp_level("2024-01-15")
        'date'
    """
    normalizer = get_temporal_normalizer()
    return normalizer.infer_timestamp_level(timestamp)


def enhanced_normalize_timestamp(timestamp: str) -> str:
    """
    Use enhanced temporal normalizer to normalize timestamp.
    
    This function uses the centralized TemporalNormalizer for consistent
    timestamp normalization, with fallback to basic methods. Ensures date-level
    timestamps are in proper YYYY-MM-DD format.
    
    Args:
        timestamp: Timestamp string to normalize
        
    Returns:
        Normalized timestamp string
        
    Example:
        >>> enhanced_normalize_timestamp("Q1 2024")
        '2024-Q1'
        >>> enhanced_normalize_timestamp("Jan 15, 2024")
        '2024-01-15'
    """
    normalizer = get_temporal_normalizer()
    return normalizer.normalize_timestamp(timestamp)


def temporal_overlap(ts1: str, ts2: str) -> bool:
    """
    Check if two timestamps overlap temporally.
    
    Args:
        ts1: First timestamp string
        ts2: Second timestamp string
    
    Returns:
        True if timestamps overlap, False otherwise
        
    Example:
        >>> temporal_overlap("2024", "2024-Q1")
        True
        >>> temporal_overlap("2024-Q1", "2025-Q1")
        False
    """
    try:
        # Handle different timestamp formats
        if ts1 == ts2:
            return True
        
        # Extract years
        year1 = extract_year(ts1)
        year2 = extract_year(ts2)
        
        if year1 and year2:
            # Check if years are the same or adjacent
            return abs(year1 - year2) <= 1
        
        # Handle quarter format (e.g., 2024Q1)
        quarter1 = extract_quarter(ts1)
        quarter2 = extract_quarter(ts2)
        
        if quarter1 and quarter2:
            year1, q1 = quarter1
            year2, q2 = quarter2
            # Same year or adjacent years
            if abs(year1 - year2) <= 1:
                return True
        
        return False
    except Exception:
        return False


def extract_year(timestamp: str) -> Optional[int]:
    """
    Extract year from timestamp string.
    
    Args:
        timestamp: Timestamp string
        
    Returns:
        Year as integer, or None if not found
        
    Example:
        >>> extract_year("2024-Q1")
        2024
        >>> extract_year("2024-01-15")
        2024
    """
    # Match 4-digit year
    year_match = re.search(r'\b(\d{4})\b', timestamp)
    if year_match:
        return int(year_match.group(1))
    return None


def extract_quarter(timestamp: str) -> Optional[Tuple[int, int]]:
    """
    Extract year and quarter from timestamp string.
    
    Args:
        timestamp: Timestamp string
        
    Returns:
        Tuple of (year, quarter), or None if not found
        
    Example:
        >>> extract_quarter("2024-Q1")
        (2024, 1)
        >>> extract_quarter("2024Q2")
        (2024, 2)
    """
    # Match quarter format like 2024Q1 or 2024-Q1
    quarter_match = re.search(r'\b(\d{4})[- ]?Q([1-4])\b', timestamp, re.IGNORECASE)
    if quarter_match:
        year = int(quarter_match.group(1))
        quarter = int(quarter_match.group(2))
        return (year, quarter)
    return None


def calculate_temporal_distance(ts1: str, ts2: str) -> float:
    """
    Calculate temporal distance between two timestamps.
    
    Args:
        ts1: First timestamp string
        ts2: Second timestamp string
    
    Returns:
        Temporal distance in years (lower = closer)
        
    Example:
        >>> calculate_temporal_distance("2024", "2025")
        1.0
        >>> calculate_temporal_distance("2024-Q1", "2024-Q3")
        0.5
    """
    try:
        year1 = extract_year(ts1)
        year2 = extract_year(ts2)
        
        if year1 and year2:
            return abs(year1 - year2)
        
        quarter1 = extract_quarter(ts1)
        quarter2 = extract_quarter(ts2)
        
        if quarter1 and quarter2:
            year1, q1 = quarter1
            year2, q2 = quarter2
            # Calculate distance in quarters
            quarter_distance = abs((year1 * 4 + q1) - (year2 * 4 + q2))
            return quarter_distance / 4.0  # Convert back to years
        
        return 10.0  # Default large distance for unparseable timestamps
    except Exception:
        return 10.0

