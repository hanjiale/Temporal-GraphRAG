"""JSON parsing and repair utilities."""

import json
import re
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("temporal-graphrag.utils.json_utils")

try:
    from json_repair import repair_json
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False
    logger.warning("json_repair not available, using fallback JSON parsing")


def parse_value(value: str) -> Any:
    """
    Convert a string value to its appropriate type (int, float, bool, None, or keep as string).
    
    Acts as a safer alternative to eval() for parsing string values.
    
    Args:
        value: String value to parse
        
    Returns:
        Parsed value (int, float, bool, None, or str)
        
    Example:
        >>> parse_value("123")
        123
        >>> parse_value("3.14")
        3.14
        >>> parse_value("true")
        True
        >>> parse_value("null")
        None
    """
    if not value:
        return value
        
    value = value.strip()

    if value.lower() == "null" or value.lower() == "none":
        return None
    elif value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        # Try to convert to int or float
        try:
            if '.' in value:  # If there's a dot, it might be a float
                return float(value)
            else:
                return int(value)
        except ValueError:
            # If conversion fails, return the value as-is (likely a string)
            return value.strip('"').strip("'")  # Remove surrounding quotes if they exist


def extract_values_from_json(json_string: str, keys: Optional[List[str]] = None, allow_no_quotes: bool = False) -> Dict[str, Any]:
    """
    Extract key values from a non-standard or malformed JSON string, handling nested objects.
    
    Args:
        json_string: JSON string to parse (may be malformed)
        keys: Optional list of keys to extract (if None, extracts all found keys)
        allow_no_quotes: Whether to allow unquoted keys
        
    Returns:
        Dictionary of extracted key-value pairs
        
    Example:
        >>> extract_values_from_json('{"title": "Test", "rating": 5.0}')
        {'title': 'Test', 'rating': 5.0}
    """
    if keys is None:
        keys = ["reasoning", "answer", "data"]
        
    if not json_string or not isinstance(json_string, str):
        logger.warning("Input is not a valid string for JSON extraction")
        return {}
        
    extracted_values = {}

    # Enhanced pattern to match both quoted and unquoted values, as well as nested objects
    regex_pattern = r'(?P<key>"?\w+"?)\s*:\s*(?P<value>{[^}]*}|".*?"|[^,}]+)'

    for match in re.finditer(regex_pattern, json_string, re.DOTALL):
        key = match.group('key').strip('"')  # Strip quotes from key
        value = match.group('value').strip()

        # If the value is another nested JSON (starts with '{' and ends with '}'), recursively parse it
        if value.startswith('{') and value.endswith('}'):
            extracted_values[key] = extract_values_from_json(value)
        else:
            # Parse the value into the appropriate type (int, float, bool, etc.)
            extracted_values[key] = parse_value(value)

    if not extracted_values:
        logger.warning("No values could be extracted from the string.")
        # Try to extract at least some basic structure
        extracted_values = _extract_basic_structure(json_string)

    return extracted_values


def _extract_basic_structure(text: str) -> Dict[str, Any]:
    """Extract basic structure when regex extraction fails."""
    result = {}
    
    # Look for key-value patterns with various separators
    patterns = [
        r'(\w+)\s*[:=]\s*([^,\n]+)',  # key: value or key=value
        r'"(\w+)"\s*:\s*"([^"]*)"',   # "key": "value"
        r"'(\w+)'\s*:\s*'([^']*)'",   # 'key': 'value'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for key, value in matches:
            if key.lower() not in ['title', 'summary', 'rating', 'rating_explanation', 'findings']:
                continue
            result[key] = value.strip()
    
    return result


def convert_response_to_json(response: str) -> Dict[str, Any]:
    """
    Convert response string to JSON using json-repair for robust parsing.
    
    Handles malformed JSON responses from LLMs by:
    1. Using json-repair to fix common JSON issues
    2. Falling back to regex-based extraction if repair fails
    3. Returning a minimal error structure if all parsing fails
    
    Args:
        response: String response that should contain JSON
        
    Returns:
        Parsed JSON dictionary
        
    Example:
        >>> convert_response_to_json('{"title": "Test", "rating": 5}')
        {'title': 'Test', 'rating': 5}
    """
    if not response or not isinstance(response, str):
        logger.warning("Response is not a valid string")
        return {}
    
    # Try json-repair first if available
    if JSON_REPAIR_AVAILABLE:
        try:
            repaired_json = repair_json(response)
            prediction_json = json.loads(repaired_json)
            logger.info("JSON data successfully extracted using json-repair.")
            return prediction_json
        except Exception as e:
            logger.warning(f"JSON repair failed: {e}. Attempting fallback extraction...")
    else:
        logger.debug("json-repair not available, using fallback extraction")
    
    # Fallback to regex-based extraction
    prediction_json = extract_values_from_json(response, allow_no_quotes=True)
    
    if not prediction_json:
        logger.error("Unable to extract meaningful data from the response.")
        # Return a minimal valid structure to prevent crashes
        prediction_json = {
            "title": "Error in report generation",
            "summary": "Failed to parse LLM response",
            "rating": 0.0,
            "rating_explanation": "Unable to generate report due to parsing error",
            "findings": []
        }
    else:
        logger.info("JSON data extracted using fallback method.")

    return prediction_json


__all__ = [
    "convert_response_to_json",
    "extract_values_from_json",
    "parse_value",
]

