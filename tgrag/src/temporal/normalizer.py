
import re
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Optional, Tuple, Union, Set
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("enhanced-temporal-normalizer")

class TemporalGranularity(Enum):
    """Temporal granularity levels"""
    YEAR = "year"
    QUARTER = "quarter"
    MONTH = "month"
    WEEK = "week"
    DATE = "date"
    UNKNOWN = "unknown"

@dataclass
class TemporalRange:
    """Represents a temporal range with start and end dates"""
    start_date: datetime
    end_date: datetime
    granularity: TemporalGranularity
    confidence: float
    original_expression: str
    
    def contains_timestamp(self, timestamp: str) -> bool:
        """Check if a timestamp falls within this range"""
        try:
            ts_date = self._parse_timestamp_to_date(timestamp)
            return self.start_date <= ts_date <= self.end_date
        except:
            return False
    
    def _parse_timestamp_to_date(self, timestamp: str) -> datetime:
        """Parse various timestamp formats to datetime"""
        timestamp = timestamp.strip().upper()
        
        # Year format
        if re.match(r'^\d{4}$', timestamp):
            return datetime(int(timestamp), 1, 1)
        
        # Quarter format (2024-Q1, Q1-2024, etc.)
        quarter_match = re.match(r'(?:(\d{4})-?Q([1-4])|Q([1-4])-?(\d{4}))', timestamp)
        if quarter_match:
            year = int(quarter_match.group(1) or quarter_match.group(4))
            quarter = int(quarter_match.group(2) or quarter_match.group(3))
            month = (quarter - 1) * 3 + 1
            return datetime(year, month, 1)
        
        # Month format (2024-01, JAN-2024, etc.)
        month_match = re.match(r'(\d{4})-(\d{1,2})', timestamp)
        if month_match:
            return datetime(int(month_match.group(1)), int(month_match.group(2)), 1)
        
        # Date format
        date_match = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})', timestamp)
        if date_match:
            return datetime(int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3)))
        
        raise ValueError(f"Cannot parse timestamp: {timestamp}")

@dataclass
class TemporalNormalizationResult:
    """Enhanced result object for temporal normalization"""
    normalized_forms: List[str]  # Multiple possible normalized forms
    temporal_ranges: List[TemporalRange]  # Temporal ranges covered
    granularity: TemporalGranularity
    confidence: float
    original_expression: str
    normalization_type: str  # 'exact', 'range', 'relative', etc.
    additional_context: Dict[str, any]  # Additional metadata

class EnhancedTemporalNormalizer:
    """
    Enhanced temporal normalizer with improved range handling and confidence scoring.
    """
    
    # Enhanced seasonal patterns with full ranges
    SEASONAL_PATTERNS = {
        "SPRING": {"months": [3, 4, 5], "start_month": 3, "end_month": 5, "primary_month": 4},
        "SUMMER": {"months": [6, 7, 8], "start_month": 6, "end_month": 8, "primary_month": 7},
        "FALL": {"months": [9, 10, 11], "start_month": 9, "end_month": 11, "primary_month": 10},
        "AUTUMN": {"months": [9, 10, 11], "start_month": 9, "end_month": 11, "primary_month": 10},
        "WINTER": {"months": [12, 1, 2], "start_month": 12, "end_month": 2, "primary_month": 12},
    }
    
    # Enhanced quarter patterns
    QUARTER_PATTERNS = {
        "Q1": {"months": [1, 2, 3], "start_month": 1, "end_month": 3},
        "Q2": {"months": [4, 5, 6], "start_month": 4, "end_month": 6},
        "Q3": {"months": [7, 8, 9], "start_month": 7, "end_month": 9},
        "Q4": {"months": [10, 11, 12], "start_month": 10, "end_month": 12},
    }
    
    # Month name mappings
    MONTH_NAMES = {
        "JANUARY": 1, "JAN": 1,
        "FEBRUARY": 2, "FEB": 2,
        "MARCH": 3, "MAR": 3,
        "APRIL": 4, "APR": 4,
        "MAY": 5,
        "JUNE": 6, "JUN": 6,
        "JULY": 7, "JUL": 7,
        "AUGUST": 8, "AUG": 8,
        "SEPTEMBER": 9, "SEP": 9, "SEPT": 9,
        "OCTOBER": 10, "OCT": 10,
        "NOVEMBER": 11, "NOV": 11,
        "DECEMBER": 12, "DEC": 12,
    }
    
    def __init__(self, reference_date: Optional[datetime] = None, enable_seasonal_matching: Optional[bool] = None):
        """Initialize the enhanced temporal normalizer.
        
        enable_seasonal_matching controls whether expressions like "SPRING 2024"
        and seasonal ranges expand to multiple month nodes. When disabled, seasonal
        patterns are ignored by this enhanced normalizer. Default is disabled, but
        can be enabled via environment variable TEMPORAL_RAG_ENABLE_SEASONAL_MATCHING=1.
        """
        self.reference_date = reference_date or datetime.now()
        if enable_seasonal_matching is None:
            env_val = os.getenv("TEMPORAL_RAG_ENABLE_SEASONAL_MATCHING", "")
            enable_seasonal_matching = env_val.strip().lower() in ("1", "true", "yes", "on")
        self.enable_seasonal_matching = bool(enable_seasonal_matching)
    
    def normalize_temporal_expression(self, expression: str) -> TemporalNormalizationResult:
        """
        Enhanced normalization with better range and confidence handling.
        """
        expression = expression.strip().upper()
        original_expression = expression
        
        try:
            # Try range-based normalization first
            result = self._try_range_normalization(expression)
            if result:
                return result
            
            # Try exact normalization
            result = self._try_exact_normalization(expression)
            if result:
                return result
            
            # Try relative normalization
            result = self._try_relative_normalization(expression)
            if result:
                return result
            
            # Try fuzzy normalization
            result = self._try_fuzzy_normalization(expression)
            if result:
                return result
            
            # Default failure case
            return TemporalNormalizationResult(
                normalized_forms=[],
                temporal_ranges=[],
                granularity=TemporalGranularity.UNKNOWN,
                confidence=0.0,
                original_expression=original_expression,
                normalization_type="failed",
                additional_context={"error": f"Unable to normalize: {expression}"}
            )
            
        except Exception as e:
            logger.error(f"Error normalizing '{expression}': {e}")
            return TemporalNormalizationResult(
                normalized_forms=[],
                temporal_ranges=[],
                granularity=TemporalGranularity.UNKNOWN,
                confidence=0.0,
                original_expression=original_expression,
                normalization_type="error",
                additional_context={"error": str(e)}
            )
    
    def _try_range_normalization(self, expression: str) -> Optional[TemporalNormalizationResult]:
        """Handle range expressions like 'Q1-Q3 2024', 'SPRING-SUMMER 2024'"""
        
        # Quarter ranges
        quarter_range_match = re.match(r'Q([1-4])-Q([1-4])\s+(\d{4})', expression)
        if quarter_range_match:
            start_q, end_q, year = int(quarter_range_match.group(1)), int(quarter_range_match.group(2)), int(quarter_range_match.group(3))
            
            start_month = (start_q - 1) * 3 + 1
            end_month = end_q * 3
            
            start_date = datetime(year, start_month, 1)
            end_date = datetime(year, end_month, 28)  # Conservative end
            if end_month == 12:
                end_date = datetime(year, 12, 31)
            else:
                end_date = datetime(year, end_month + 1, 1) - timedelta(days=1)
            
            normalized_forms = [f"{year}-Q{q}" for q in range(start_q, end_q + 1)]
            temporal_ranges = [TemporalRange(start_date, end_date, TemporalGranularity.QUARTER, 0.9, expression)]
            
            return TemporalNormalizationResult(
                normalized_forms=normalized_forms,
                temporal_ranges=temporal_ranges,
                granularity=TemporalGranularity.QUARTER,
                confidence=0.9,
                original_expression=expression,
                normalization_type="quarter_range",
                additional_context={"quarters": list(range(start_q, end_q + 1)), "year": year}
            )
        
        # Seasonal ranges (optional)
        if self.enable_seasonal_matching:
            season_range_match = re.match(r'(SPRING|SUMMER|FALL|AUTUMN|WINTER)-(SPRING|SUMMER|FALL|AUTUMN|WINTER)\s+(\d{4})', expression)
        else:
            season_range_match = None
        if season_range_match:
            start_season, end_season, year = season_range_match.groups()
            year = int(year)
            
            start_months = self.SEASONAL_PATTERNS[start_season]["months"]
            end_months = self.SEASONAL_PATTERNS[end_season]["months"]
            
            # Handle winter crossing year boundary
            all_months = set()
            season_order = ["SPRING", "SUMMER", "FALL", "WINTER"]
            start_idx = season_order.index(start_season if start_season != "AUTUMN" else "FALL")
            end_idx = season_order.index(end_season if end_season != "AUTUMN" else "FALL")
            
            if start_idx <= end_idx:
                for i in range(start_idx, end_idx + 1):
                    all_months.update(self.SEASONAL_PATTERNS[season_order[i]]["months"])
            else:  # Wrapping around year
                for i in range(start_idx, 4):
                    all_months.update(self.SEASONAL_PATTERNS[season_order[i]]["months"])
                for i in range(0, end_idx + 1):
                    all_months.update(self.SEASONAL_PATTERNS[season_order[i]]["months"])
            
            normalized_forms = [f"{year}-{month:02d}" for month in sorted(all_months) if month <= 12]
            if 12 in all_months and (1 in all_months or 2 in all_months):
                # Add next year months for winter
                next_year_months = [m for m in all_months if m <= 2]
                normalized_forms.extend([f"{year + 1}-{month:02d}" for month in next_year_months])
            
            start_month = min(all_months)
            end_month = max(all_months)
            
            start_date = datetime(year, start_month, 1)
            if end_month < start_month:  # Winter case
                end_date = datetime(year + 1, end_month, 28)
            else:
                end_date = datetime(year, end_month, 28)
            
            temporal_ranges = [TemporalRange(start_date, end_date, TemporalGranularity.MONTH, 0.8, expression)]
            
            return TemporalNormalizationResult(
                normalized_forms=normalized_forms,
                temporal_ranges=temporal_ranges,
                granularity=TemporalGranularity.MONTH,
                confidence=0.8,
                original_expression=expression,
                normalization_type="seasonal_range",
                additional_context={"seasons": [start_season, end_season], "year": year}
            )
        
        return None
    
    def _try_exact_normalization(self, expression: str) -> Optional[TemporalNormalizationResult]:
        """Handle exact temporal expressions with improved confidence"""
        
        # Year
        year_match = re.match(r'^(\d{4})$', expression)
        if year_match:
            year = int(year_match.group(1))
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
            
            return TemporalNormalizationResult(
                normalized_forms=[str(year)],
                temporal_ranges=[TemporalRange(start_date, end_date, TemporalGranularity.YEAR, 1.0, expression)],
                granularity=TemporalGranularity.YEAR,
                confidence=1.0,
                original_expression=expression,
                normalization_type="exact_year",
                additional_context={"year": year}
            )
        
        # Fiscal year patterns (common in business contexts): "FY 2024", "FISCAL YEAR 2023", etc.
        fiscal_match = re.match(r'^(FY|FISCAL|FISCAL\s+YEAR)\s*(\d{4})$', expression, re.IGNORECASE)
        if fiscal_match:
            fiscal_type = fiscal_match.group(1).upper()
            year = int(fiscal_match.group(2))
            
            # Fiscal year typically starts in July (US standard, can be customized per organization)
            # We normalize to the start month of the fiscal year
            fiscal_start_month = 7  # July
            start_date = datetime(year, fiscal_start_month, 1)
            
            # Fiscal year spans from July of current year to June of next year
            # For normalization, we use the start month (July) as the primary form
            normalized_form = f"{year}-{fiscal_start_month:02d}"
            end_date = datetime(year, 12, 31)  # End of calendar year (partial fiscal year range)
            
            return TemporalNormalizationResult(
                normalized_forms=[normalized_form],
                temporal_ranges=[TemporalRange(start_date, end_date, TemporalGranularity.MONTH, 0.8, expression)],
                granularity=TemporalGranularity.MONTH,
                confidence=0.8,
                original_expression=expression,
                normalization_type="fiscal_year",
                additional_context={"fiscal_year": year, "fiscal_start_month": fiscal_start_month, "fiscal_type": fiscal_type}
            )
        
        # Date patterns: handle "MONTH DAY YEAR" formats with ordinals (e.g., "AUGUST 31ST 2024", "December 31, 2021")
        # First, clean up the expression to handle ordinals and punctuation
        cleaned_expr = expression.replace(',', '').replace('.', '')
        cleaned_expr = re.sub(r'(\d+)(?:ST|ND|RD|TH)', r'\1', cleaned_expr)  # Remove ordinals
        
        # Pattern: MONTH DAY YEAR (e.g., "AUGUST 31 2024", "DECEMBER 31 2021")
        month_day_year_match = re.match(r'(\w+)\s+(\d{1,2})\s+(\d{4})', cleaned_expr)
        if month_day_year_match:
            month_name, day, year = month_day_year_match.groups()
            
            if month_name in self.MONTH_NAMES:
                month_num = self.MONTH_NAMES[month_name]
                day_num = int(day)
                year_num = int(year)
                
                # Validate the date
                try:
                    date_obj = datetime(year_num, month_num, day_num)
                    normalized_form = f"{year_num}-{month_num:02d}-{day_num:02d}"
                    
                    return TemporalNormalizationResult(
                        normalized_forms=[normalized_form],
                        temporal_ranges=[TemporalRange(date_obj, date_obj, TemporalGranularity.DATE, 0.95, expression)],
                        granularity=TemporalGranularity.DATE,
                        confidence=0.95,
                        original_expression=expression,
                        normalization_type="exact_date",
                        additional_context={"year": year_num, "month": month_num, "day": day_num}
                    )
                except ValueError:
                    # Invalid date, continue to other patterns
                    pass
        
        # Pattern: DAY MONTH YEAR (e.g., "31 AUGUST 2024", "31 DEC 2024")
        day_month_year_match = re.match(r'(\d{1,2})\s+(\w+)\s+(\d{4})', cleaned_expr)
        if day_month_year_match:
            day, month_name, year = day_month_year_match.groups()
            
            if month_name in self.MONTH_NAMES:
                month_num = self.MONTH_NAMES[month_name]
                day_num = int(day)
                year_num = int(year)
                
                # Validate the date
                try:
                    date_obj = datetime(year_num, month_num, day_num)
                    normalized_form = f"{year_num}-{month_num:02d}-{day_num:02d}"
                    
                    return TemporalNormalizationResult(
                        normalized_forms=[normalized_form],
                        temporal_ranges=[TemporalRange(date_obj, date_obj, TemporalGranularity.DATE, 0.95, expression)],
                        granularity=TemporalGranularity.DATE,
                        confidence=0.95,
                        original_expression=expression,
                        normalization_type="exact_date",
                        additional_context={"year": year_num, "month": month_num, "day": day_num}
                    )
                except ValueError:
                    # Invalid date, continue to other patterns
                    pass
        
        # Standard ISO date format: YYYY-MM-DD
        iso_date_match = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', expression)
        if iso_date_match:
            year, month, day = iso_date_match.groups()
            year_num, month_num, day_num = int(year), int(month), int(day)
            
            try:
                date_obj = datetime(year_num, month_num, day_num)
                normalized_form = f"{year_num}-{month_num:02d}-{day_num:02d}"
                
                return TemporalNormalizationResult(
                    normalized_forms=[normalized_form],
                    temporal_ranges=[TemporalRange(date_obj, date_obj, TemporalGranularity.DATE, 1.0, expression)],
                    granularity=TemporalGranularity.DATE,
                    confidence=1.0,
                    original_expression=expression,
                    normalization_type="exact_date_iso",
                    additional_context={"year": year_num, "month": month_num, "day": day_num}
                )
            except ValueError:
                # Invalid date format, continue
                pass
        
        # Quarter
        quarter_match = re.match(r'(?:(\d{4})-?Q([1-4])|Q([1-4])-?(\d{4}))', expression)
        if quarter_match:
            year = int(quarter_match.group(1) or quarter_match.group(4))
            quarter = int(quarter_match.group(2) or quarter_match.group(3))
            
            quarter_info = self.QUARTER_PATTERNS[f"Q{quarter}"]
            start_month = quarter_info["start_month"]
            end_month = quarter_info["end_month"]
            
            start_date = datetime(year, start_month, 1)
            end_date = datetime(year, end_month, 28)
            if end_month == 12:
                end_date = datetime(year, 12, 31)
            else:
                end_date = datetime(year, end_month + 1, 1) - timedelta(days=1)
            
            normalized_form = f"{year}-Q{quarter}"
            
            return TemporalNormalizationResult(
                normalized_forms=[normalized_form],
                temporal_ranges=[TemporalRange(start_date, end_date, TemporalGranularity.QUARTER, 1.0, expression)],
                granularity=TemporalGranularity.QUARTER,
                confidence=1.0,
                original_expression=expression,
                normalization_type="exact_quarter",
                additional_context={"year": year, "quarter": quarter}
            )
        
        # Season - support multiple formats: "SPRING 2024", etc. (optional)
        if self.enable_seasonal_matching:
            season_patterns = [
                r'(SPRING|SUMMER|FALL|AUTUMN|WINTER)\s+(\d{4})',  # "SPRING 2024"
                r'(SPRING|SUMMER|FALL|AUTUMN|WINTER)-(\d{4})',    # "SPRING-2024"
                r'(\d{4})-(SPRING|SUMMER|FALL|AUTUMN|WINTER)',    # "2024-SPRING"
            ]
        else:
            season_patterns = []
        
        for pattern in season_patterns:
            season_match = re.match(pattern, expression)
            if season_match:
                groups = season_match.groups()
                
                # Handle different group orders based on pattern
                if groups[0] in self.SEASONAL_PATTERNS:
                    # Pattern 1 or 2: season first
                    season, year = groups[0], int(groups[1])
                else:
                    # Pattern 3: year first
                    year, season = int(groups[0]), groups[1]
                
                season_info = self.SEASONAL_PATTERNS[season]
                months = season_info["months"]
                
                # Generate all month forms
                normalized_forms = []
                temporal_ranges = []
                
                # Special handling for winter season which spans across years
                if season == "WINTER":
                    # Winter: December of current year + January/February of next year
                    normalized_forms = [
                        f"{year}-12",  # December of current year
                        f"{year + 1}-01",  # January of next year
                        f"{year + 1}-02"   # February of next year
                    ]
                    
                    # Create temporal ranges for each month
                    temporal_ranges = [
                        TemporalRange(datetime(year, 12, 1), datetime(year, 12, 31), TemporalGranularity.MONTH, 0.9, expression),
                        TemporalRange(datetime(year + 1, 1, 1), datetime(year + 1, 1, 31), TemporalGranularity.MONTH, 0.9, expression),
                        TemporalRange(datetime(year + 1, 2, 1), datetime(year + 1, 2, 28), TemporalGranularity.MONTH, 0.9, expression)
                    ]
                else:
                    # Handle other seasons normally
                    for month in months:
                        normalized_forms.append(f"{year}-{month:02d}")
                        start_date = datetime(year, month, 1)
                        if month == 12:
                            end_date = datetime(year, 12, 31)
                        else:
                            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
                        temporal_ranges.append(TemporalRange(start_date, end_date, TemporalGranularity.MONTH, 0.9, expression))
                
                # Also add primary month as the main normalized form
                primary_month = season_info["primary_month"]
                primary_normalized = f"{year}-{primary_month:02d}"
                if primary_normalized not in normalized_forms:
                    normalized_forms.insert(0, primary_normalized)
                
                return TemporalNormalizationResult(
                    normalized_forms=normalized_forms,
                    temporal_ranges=temporal_ranges,
                    granularity=TemporalGranularity.MONTH,
                    confidence=0.9,
                    original_expression=expression,
                    normalization_type="exact_season",
                    additional_context={"season": season, "year": year, "primary_month": primary_month}
                )
        
        return None
    
    def _try_relative_normalization(self, expression: str) -> Optional[TemporalNormalizationResult]:
        """Handle relative temporal expressions"""
        
        # This year, last year, next year
        if expression in ["THIS YEAR", "CURRENT YEAR"]:
            year = self.reference_date.year
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
            
            return TemporalNormalizationResult(
                normalized_forms=[str(year)],
                temporal_ranges=[TemporalRange(start_date, end_date, TemporalGranularity.YEAR, 0.8, expression)],
                granularity=TemporalGranularity.YEAR,
                confidence=0.8,
                original_expression=expression,
                normalization_type="relative_year",
                additional_context={"relative_to": self.reference_date.isoformat()}
            )
        
        if expression == "LAST YEAR":
            year = self.reference_date.year - 1
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
            
            return TemporalNormalizationResult(
                normalized_forms=[str(year)],
                temporal_ranges=[TemporalRange(start_date, end_date, TemporalGranularity.YEAR, 0.8, expression)],
                granularity=TemporalGranularity.YEAR,
                confidence=0.8,
                original_expression=expression,
                normalization_type="relative_year",
                additional_context={"relative_to": self.reference_date.isoformat()}
            )
        
        # This quarter, last quarter
        if expression in ["THIS QUARTER", "CURRENT QUARTER"]:
            current_quarter = (self.reference_date.month - 1) // 3 + 1
            year = self.reference_date.year
            normalized_form = f"{year}-Q{current_quarter}"
            
            quarter_info = self.QUARTER_PATTERNS[f"Q{current_quarter}"]
            start_date = datetime(year, quarter_info["start_month"], 1)
            end_month = quarter_info["end_month"]
            if end_month == 12:
                end_date = datetime(year, 12, 31)
            else:
                end_date = datetime(year, end_month + 1, 1) - timedelta(days=1)
            
            return TemporalNormalizationResult(
                normalized_forms=[normalized_form],
                temporal_ranges=[TemporalRange(start_date, end_date, TemporalGranularity.QUARTER, 0.8, expression)],
                granularity=TemporalGranularity.QUARTER,
                confidence=0.8,
                original_expression=expression,
                normalization_type="relative_quarter",
                additional_context={"relative_to": self.reference_date.isoformat(), "quarter": current_quarter}
            )
        
        return None
    
    def _try_fuzzy_normalization(self, expression: str) -> Optional[TemporalNormalizationResult]:
        """Handle fuzzy/partial matches with lower confidence"""
        
        # Extract year from expression
        year_match = re.search(r'(\d{4})', expression)
        if year_match:
            year = int(year_match.group(1))
            
            # Check for quarter indicators
            if re.search(r'[Qq][1-4]|QUARTER|QTR', expression):
                quarter_match = re.search(r'[Qq]([1-4])', expression)
                if quarter_match:
                    quarter = int(quarter_match.group(1))
                    if 1 <= quarter <= 4:
                        normalized_form = f"{year}-Q{quarter}"
                        
                        quarter_info = self.QUARTER_PATTERNS[f"Q{quarter}"]
                        start_date = datetime(year, quarter_info["start_month"], 1)
                        end_month = quarter_info["end_month"]
                        if end_month == 12:
                            end_date = datetime(year, 12, 31)
                        else:
                            end_date = datetime(year, end_month + 1, 1) - timedelta(days=1)
                        
                        return TemporalNormalizationResult(
                            normalized_forms=[normalized_form],
                            temporal_ranges=[TemporalRange(start_date, end_date, TemporalGranularity.QUARTER, 0.6, expression)],
                            granularity=TemporalGranularity.QUARTER,
                            confidence=0.6,
                            original_expression=expression,
                            normalization_type="fuzzy_quarter",
                            additional_context={"year": year, "quarter": quarter, "fuzzy_match": True}
                        )
            
            # Check for month indicators
            for month_name, month_num in self.MONTH_NAMES.items():
                if month_name in expression:
                    
                    # Enhanced: Check if there's also a day for DATE granularity
                    # Clean ordinals from the expression first
                    cleaned_expr = re.sub(r'(\d+)(?:ST|ND|RD|TH)', r'\1', expression)
                    day_match = re.search(r'(\d{1,2})', cleaned_expr)
                    
                    if day_match:
                        day = int(day_match.group(1))
                        # Validate it's a reasonable day (1-31) and not the year
                        if 1 <= day <= 31 and day != year % 100 and day != year:
                            try:
                                date_obj = datetime(year, month_num, day)
                                normalized_form = f"{year}-{month_num:02d}-{day:02d}"
                                
                                return TemporalNormalizationResult(
                                    normalized_forms=[normalized_form],
                                    temporal_ranges=[TemporalRange(date_obj, date_obj, TemporalGranularity.DATE, 0.7, expression)],
                                    granularity=TemporalGranularity.DATE,
                                    confidence=0.7,
                                    original_expression=expression,
                                    normalization_type="fuzzy_date",
                                    additional_context={"year": year, "month": month_num, "day": day, "fuzzy_match": True}
                                )
                            except ValueError:
                                # Invalid date, fall back to month-only
                                pass
                    
                    # Fall back to month-only normalization
                    normalized_form = f"{year}-{month_num:02d}"
                    start_date = datetime(year, month_num, 1)
                    if month_num == 12:
                        end_date = datetime(year, 12, 31)
                    else:
                        end_date = datetime(year, month_num + 1, 1) - timedelta(days=1)
                    
                    return TemporalNormalizationResult(
                        normalized_forms=[normalized_form],
                        temporal_ranges=[TemporalRange(start_date, end_date, TemporalGranularity.MONTH, 0.7, expression)],
                        granularity=TemporalGranularity.MONTH,
                        confidence=0.7,
                        original_expression=expression,
                        normalization_type="fuzzy_month",
                        additional_context={"year": year, "month": month_num, "fuzzy_match": True}
                    )
            
            # Default to year if no other indicators
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
            
            return TemporalNormalizationResult(
                normalized_forms=[str(year)],
                temporal_ranges=[TemporalRange(start_date, end_date, TemporalGranularity.YEAR, 0.5, expression)],
                granularity=TemporalGranularity.YEAR,
                confidence=0.5,
                original_expression=expression,
                normalization_type="fuzzy_year",
                additional_context={"year": year, "fuzzy_match": True}
            )
        
        return None
    
    def get_temporal_expansions(self, expression: str, max_expansions: int = 10) -> List[str]:
        """
        Get temporal expansions for enumeration queries.
        For example, "2024" -> ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"]
        """
        result = self.normalize_temporal_expression(expression)
        
        if not result.normalized_forms:
            return []
        
        expansions = set(result.normalized_forms)
        
        # Add temporal expansions based on granularity
        for norm_form in result.normalized_forms[:max_expansions]:
            if result.granularity == TemporalGranularity.YEAR:
                # Expand year to quarters and months
                year_match = re.match(r'^(\d{4})$', norm_form)
                if year_match:
                    year = int(year_match.group(1))
                    # Add quarters
                    for q in range(1, 5):
                        expansions.add(f"{year}-Q{q}")
                    # Add months
                    for m in range(1, 13):
                        expansions.add(f"{year}-{m:02d}")
            
            elif result.granularity == TemporalGranularity.QUARTER:
                # Expand quarter to months
                quarter_match = re.match(r'^(\d{4})-Q([1-4])$', norm_form)
                if quarter_match:
                    year, quarter = int(quarter_match.group(1)), int(quarter_match.group(2))
                    quarter_info = self.QUARTER_PATTERNS[f"Q{quarter}"]
                    for month in quarter_info["months"]:
                        expansions.add(f"{year}-{month:02d}")
        
        return list(expansions)[:max_expansions]

# Convenience functions for backwards compatibility
def normalize_temporal_expression(expression: str, reference_date: Optional[datetime] = None) -> TemporalNormalizationResult:
    """Convenience function for single expression normalization"""
    normalizer = EnhancedTemporalNormalizer(reference_date)
    return normalizer.normalize_temporal_expression(expression)

def batch_normalize_expressions(expressions: List[str], reference_date: Optional[datetime] = None) -> List[TemporalNormalizationResult]:
    """Convenience function for batch normalization"""
    normalizer = EnhancedTemporalNormalizer(reference_date)
    return [normalizer.normalize_temporal_expression(expr) for expr in expressions]
