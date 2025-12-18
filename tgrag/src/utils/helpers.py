"""Helper utilities migrated from OLD for compatibility."""

from __future__ import annotations

import html
import json
import logging
import os
import re
import numbers
from datetime import datetime, timedelta

import tiktoken
from dateutil.relativedelta import relativedelta

logger = logging.getLogger("temporal-graphrag.utils")

# Import from new utility modules (migrated functions)
from .hashing import compute_mdhash_id
from .async_utils import always_get_an_event_loop, limit_async_func_call
from .json_utils import convert_response_to_json

# Global encoder cache for tiktoken
_ENCODER = None

# Temporal patterns (used by temporal functions)
_SEASONAL_PATTERNS = {
    "SPRING": {"start_month": 3, "end_month": 5},
    "SUMMER": {"start_month": 6, "end_month": 8},
    "FALL": {"start_month": 9, "end_month": 11},
    "AUTUMN": {"start_month": 9, "end_month": 11},
    "WINTER": {"start_month": 12, "end_month": 2},
}

_TIMESTAMP_PATTERNS = {
    "year": re.compile(r"\d{4}"),
    "quarter": re.compile(r"\d{4}-[Qq][1-4]|[Qq][1-4]\s*\d{4}|[Qq][1-4]-\d{4}"),
    "month": re.compile(r"\d{4}-(0[1-9]|1[0-2])|(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s*\d{4}"),
    "season": re.compile(r"(?:SPRING|SUMMER|FALL|AUTUMN|WINTER)\s*\d{4}", re.IGNORECASE),
    "week": re.compile(r"\d{4}-W(0[1-9]|[1-4][0-9]|5[0-3])"),
    "date": re.compile(r"\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])"),
}


def load_json(file_name):
    """Load JSON from file."""
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    """Write JSON to file."""
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def clean_str(input: any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    if not isinstance(input, str):
        return input
    result = html.unescape(input.strip())
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    """Encode string to tokens using tiktoken."""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.encoding_for_model(model_name)
    
    if not isinstance(content, str):
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        else:
            content = str(content)
    
    try:
        tokens = _ENCODER.encode(content)
    except Exception:
        raise ValueError(f'error content: {content}')
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    """Decode tokens to string using tiktoken."""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.encoding_for_model(model_name)
    content = _ENCODER.decode(tokens)
    return content


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size."""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def pack_user_ass_to_openai_messages(prompt: str, generated_content: str, using_amazon_bedrock: bool):
    """Pack user and assistant messages for OpenAI or Bedrock format."""
    if using_amazon_bedrock:
        return [
            {"role": "user", "content": [{"text": prompt}]},
            {"role": "assistant", "content": [{"text": generated_content}]},
        ]
    else:
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": generated_content},
        ]


def is_float_regex(value):
    """Check if value matches float regex pattern."""
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers."""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def enclose_string_with_quotes(content: any) -> str:
    """Enclose a string with quotes."""
    if isinstance(content, numbers.Number):
        return str(content)
    content = str(content)
    content = content.strip().strip("'").strip('"')
    return f'"{content}"'


def list_of_list_to_csv(data: list[list]):
    """Convert list of lists to CSV format."""
    return "\n".join(
        [
            ",\t".join([f"{enclose_string_with_quotes(data_dd)}" for data_dd in data_d])
            for data_d in data
        ]
    )


def _month_name_to_number(month_name: str) -> int:
    """Convert month name to month number (1-12)."""
    month_map = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
        'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4, 'MAY': 5, 'JUNE': 6,
        'JULY': 7, 'AUGUST': 8, 'SEPTEMBER': 9, 'OCTOBER': 10, 'NOVEMBER': 11, 'DECEMBER': 12
    }
    month_upper = month_name.upper()
    if month_upper in month_map:
        return month_map[month_upper]
    raise ValueError(f"Invalid month name: {month_name}")


def _normalize_timestamp(s: str) -> str:
    """Normalize timestamp to standard format."""
    s = s.strip().strip('"').strip("'")
    
    date_match = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', s)
    if date_match:
        year, month, day = date_match.groups()
        try:
            dt = datetime(int(year), int(month), int(day))
            return f"{int(year)}-{int(month):02d}-{int(day):02d}"
        except ValueError:
            pass
    
    quarter_match = re.match(r"[Qq]([1-4])\s*[-]?\s*(\d{4})", s)
    if quarter_match:
        quarter = quarter_match.group(1)
        year = quarter_match.group(2)
        return f"{year}-Q{quarter}"
    
    quarter_match2 = re.match(r"(\d{4})\s+[Qq]([1-4])", s)
    if quarter_match2:
        year = quarter_match2.group(1)
        quarter = quarter_match2.group(2)
        return f"{year}-Q{quarter}"
    
    if re.match(r"\d{4}-q[1-4]", s):
        return s.replace("-q", "-Q")
    
    seasonal_match = re.match(r"(SPRING|SUMMER|FALL|AUTUMN|WINTER)\s*(\d{4})", s, re.IGNORECASE)
    if seasonal_match:
        season = seasonal_match.group(1).upper()
        year = seasonal_match.group(2)
        if season in _SEASONAL_PATTERNS:
            start_month = _SEASONAL_PATTERNS[season]["start_month"]
            end_month = _SEASONAL_PATTERNS[season]["end_month"]
            middle_month = start_month + (end_month - start_month) // 2
            if season == "WINTER":
                middle_month = 12
            return f"{year}-{middle_month:02d}"
    
    return s


def infer_timestamp_level(s: str) -> str:
    """Infer timestamp level from string."""
    try:
        from ..temporal.normalization import get_temporal_normalizer
        normalizer = get_temporal_normalizer()
        result = normalizer.normalize_temporal_expression(s)
        if result.granularity:
            return result.granularity.value
    except Exception:
        pass
    
    s = _normalize_timestamp(s)
    for level, pattern in _TIMESTAMP_PATTERNS.items():
        if pattern.fullmatch(s):
            return level
    raise ValueError(f"Cannot infer timestamp level from input: '{s}'")


def _convert_timestamp_to_datetime(s):
    """Convert timestamp string to datetime object."""
    s = _normalize_timestamp(s)
    
    seasonal_match = re.match(r"(SPRING|SUMMER|FALL|AUTUMN|WINTER)\s*(\d{4})", s, re.IGNORECASE)
    if seasonal_match:
        season = seasonal_match.group(1).upper()
        year = seasonal_match.group(2)
        if season in _SEASONAL_PATTERNS:
            start_month = _SEASONAL_PATTERNS[season]["start_month"]
            end_month = _SEASONAL_PATTERNS[season]["end_month"]
            middle_month = start_month + (end_month - start_month) // 2
            if season == "WINTER":
                middle_month = 12
            return datetime(int(year), middle_month, 1)
    
    if re.match(r"\d{4}$", s):
        return datetime(int(s), 1, 1)
    elif re.match(r"\d{4}-[Qq][1-4]$", s):
        year, q = s.split("-")
        q = q.upper()
        month = (int(q[1]) - 1) * 3 + 1
        return datetime(int(year), month, 1)
    elif re.match(r"\d{4}-\d{2}$", s):
        return datetime.strptime(s, "%Y-%m")
    elif re.match(r"\d{4}-\d{2}-\d{2}$", s):
        return datetime.strptime(s, "%Y-%m-%d")
    elif re.match(r"\d{4}-W\d{2}$", s):
        year, week = s.split("-W")
        return datetime.strptime(f"{year}-{week}-1", "%Y-%W-%w")
    else:
        raise ValueError(f"Unsupported timestamp format: {s}")


def convert_timestamp_to_datetime(s):
    """Convert timestamp string to datetime object."""
    try:
        from ..temporal.normalization import get_temporal_normalizer
        normalizer = get_temporal_normalizer()
        result = normalizer.normalize_temporal_expression(s)
        if result.normalized_forms:
            s = result.normalized_forms[0]
    except Exception:
        pass
    return _convert_timestamp_to_datetime(s)


def get_parent_timestamp_name(input: str, timestamp_type: str) -> str:
    """Convert input timestamp to a higher-level timestamp string based on target type."""
    try:
        from ..temporal.operations import enhanced_infer_timestamp_level, enhanced_normalize_timestamp
        level = enhanced_infer_timestamp_level(input)
        input = enhanced_normalize_timestamp(input)
    except Exception:
        level = infer_timestamp_level(input)
        input = _normalize_timestamp(input)
    
    input = html.unescape(input)
    input = input.strip('"').strip("'")
    
    try:
        if level == "year":
            year = int(input)
            dt = datetime(year, 1, 1)
        elif level == "quarter":
            if "-Q" in input:
                year, q = input.split("-Q")
            elif "-q" in input:
                year, q = input.split("-q")
            else:
                match = re.match(r"[Qq]([1-4])\s*(\d{4})", input)
                if match:
                    q, year = match.group(1), match.group(2)
                else:
                    raise ValueError(f"Invalid quarter format: {input}")
            month = (int(q) - 1) * 3 + 1
            dt = datetime(int(year), month, 1)
        elif level == "month":
            if re.match(r"\d{4}-\d{2}", input):
                dt = datetime.strptime(input, "%Y-%m")
            else:
                match = re.match(r"([A-Za-z]+)\s*(\d{4})", input)
                if match:
                    month_name, year = match.group(1), match.group(2)
                    month_num = _month_name_to_number(month_name)
                    dt = datetime(int(year), month_num, 1)
                else:
                    raise ValueError(f"Invalid month format: {input}")
        elif level == "date":
            dt = datetime.strptime(input, "%Y-%m-%d")
        elif level == "week":
            year, week = input.split("-W")
            dt = datetime.strptime(f"{year}-{week}-1", "%Y-%W-%w")
        else:
            raise ValueError(f"Unsupported input level: '{level}'")
    except Exception as e:
        raise ValueError(f"Timestamp parse error: {e}")
    
    if timestamp_type == "year":
        return dt.strftime("%Y")
    elif timestamp_type == "quarter":
        quarter = (dt.month - 1) // 3 + 1
        return f"{dt.year}-Q{quarter}"
    elif timestamp_type == "month":
        return dt.strftime("%Y-%m")
    elif timestamp_type == "week":
        year, week, _ = dt.isocalendar()
        return f"{year}-W{week:02}"
    elif timestamp_type == "date":
        return f"{dt.year}-{dt.month:02d}-{dt.day:02d}"
    else:
        raise ValueError(f"Unsupported target timestamp_type: {timestamp_type}")


def _convert_datetime_to_timestamp(dt, level):
    """Convert datetime object to timestamp string based on level."""
    if level == "year":
        return str(dt.year)
    elif level == "quarter":
        quarter = (dt.month - 1) // 3 + 1
        return f"{dt.year}-Q{quarter}"
    elif level == "month":
        return f"{dt.year}-{dt.month:02d}"
    elif level == "week":
        return f"{dt.year}-W{dt.isocalendar()[1]:02d}"
    elif level == "date":
        return f"{dt.year}-{dt.month:02d}-{dt.day:02d}"
    else:
        raise ValueError(f"Unsupported level: {level}")


def complete_timestamp_range_by_level(start_ts, end_ts, level):
    """Complete timestamp range by level."""
    try:
        from ..temporal.operations import enhanced_normalize_timestamp
        start_ts = enhanced_normalize_timestamp(start_ts)
        end_ts = enhanced_normalize_timestamp(end_ts)
    except Exception:
        start_ts = _normalize_timestamp(start_ts)
        end_ts = _normalize_timestamp(end_ts)
    
    start = convert_timestamp_to_datetime(start_ts)
    end = convert_timestamp_to_datetime(end_ts)
    
    result = []
    current = start
    
    while current <= end:
        result.append(_convert_datetime_to_timestamp(current, level))
        if level == "year":
            current += relativedelta(years=1)
        elif level == "quarter":
            current += relativedelta(months=3)
        elif level == "month":
            current += relativedelta(months=1)
        elif level == "week":
            current += timedelta(weeks=1)
        elif level == "day":
            current += timedelta(days=1)
    
    return result


def sort_timestamp_by_datetime(date_list, reverse=False):
    """Sort timestamps by datetime."""
    try:
        from ..temporal.operations import enhanced_normalize_timestamp
        normalized_list = [enhanced_normalize_timestamp(ts) for ts in date_list]
    except Exception:
        normalized_list = [_normalize_timestamp(ts) for ts in date_list]
    
    return sorted(normalized_list, key=convert_timestamp_to_datetime, reverse=reverse)


__all__ = [
    "logger",
    "clean_str",
    "compute_mdhash_id",
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
    "convert_response_to_json",
    "always_get_an_event_loop",
    "limit_async_func_call",
    "load_json",
    "write_json",
]
