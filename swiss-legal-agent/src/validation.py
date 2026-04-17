"""
Parameter validation and default substitution utilities.

This module provides centralized validation functions and default value
management for all system components. When invalid parameters are detected,
the system substitutes valid defaults and logs corrections.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar, Union, Optional, Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')


def validate_and_default_positive_int(
    value: Any, 
    default: int, 
    param_name: str, 
    component: str = "System"
) -> int:
    """
    Validate that a parameter is a positive integer, substitute default if invalid.
    
    Args:
        value: Parameter value to validate
        default: Default value to use if invalid
        param_name: Name of the parameter for logging
        component: Component name for logging context
        
    Returns:
        Valid positive integer (original value or default)
    """
    try:
        if value is None:
            logger.info(
                f"{component}.{param_name}: Parameter is None. "
                f"Using default value: {default}. "
                f"Expected: positive integer > 0"
            )
            return default
            
        int_value = int(value)
        if int_value <= 0:
            logger.warning(
                f"{component}.{param_name}: Invalid value '{int_value}' is not positive. "
                f"Expected: integer > 0, got: {int_value} (type: {type(value).__name__}). "
                f"Using default: {default}. "
                f"Suggestion: Provide a positive integer value like 1, 10, or 100."
            )
            return default
            
        return int_value
        
    except (ValueError, TypeError) as e:
        logger.error(
            f"{component}.{param_name}: Failed to convert '{value}' to integer. "
            f"Input type: {type(value).__name__}, input value: '{value}'. "
            f"Expected: positive integer > 0. "
            f"Conversion error: {type(e).__name__}: {str(e)}. "
            f"Using default: {default}. "
            f"Suggestion: Provide a numeric value that can be converted to a positive integer."
        )
        return default


def validate_and_default_non_negative_int(
    value: Any, 
    default: int, 
    param_name: str, 
    component: str = "System"
) -> int:
    """
    Validate that a parameter is a non-negative integer, substitute default if invalid.
    
    Args:
        value: Parameter value to validate
        default: Default value to use if invalid
        param_name: Name of the parameter for logging
        component: Component name for logging context
        
    Returns:
        Valid non-negative integer (original value or default)
    """
    try:
        if value is None:
            logger.info(
                f"{component}.{param_name}: Parameter is None. "
                f"Using default value: {default}. "
                f"Expected: non-negative integer >= 0"
            )
            return default
            
        int_value = int(value)
        if int_value < 0:
            logger.warning(
                f"{component}.{param_name}: Invalid value '{int_value}' is negative. "
                f"Expected: integer >= 0, got: {int_value} (type: {type(value).__name__}). "
                f"Using default: {default}. "
                f"Suggestion: Provide a non-negative integer value like 0, 5, or 200."
            )
            return default
            
        return int_value
        
    except (ValueError, TypeError) as e:
        logger.error(
            f"{component}.{param_name}: Failed to convert '{value}' to integer. "
            f"Input type: {type(value).__name__}, input value: '{value}'. "
            f"Expected: non-negative integer >= 0. "
            f"Conversion error: {type(e).__name__}: {str(e)}. "
            f"Using default: {default}. "
            f"Suggestion: Provide a numeric value that can be converted to a non-negative integer."
        )
        return default


def validate_and_default_float_range(
    value: Any, 
    default: float, 
    min_val: float, 
    max_val: float, 
    param_name: str, 
    component: str = "System"
) -> float:
    """
    Validate that a parameter is a float within specified range, substitute default if invalid.
    
    Args:
        value: Parameter value to validate
        default: Default value to use if invalid
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        param_name: Name of the parameter for logging
        component: Component name for logging context
        
    Returns:
        Valid float within range (original value or default)
    """
    try:
        if value is None:
            logger.info(
                f"{component}.{param_name}: Parameter is None. "
                f"Using default value: {default}. "
                f"Expected: float in range [{min_val}, {max_val}]"
            )
            return default
            
        float_value = float(value)
        
        # Check for NaN or infinity
        if not (float_value == float_value):  # NaN check
            logger.error(
                f"{component}.{param_name}: Invalid value is NaN (Not a Number). "
                f"Input: '{value}' (type: {type(value).__name__}). "
                f"Expected: finite float in range [{min_val}, {max_val}]. "
                f"Using default: {default}. "
                f"Suggestion: Provide a valid numeric value within the allowed range."
            )
            return default
            
        if float_value == float('inf') or float_value == float('-inf'):
            logger.error(
                f"{component}.{param_name}: Invalid value is infinite. "
                f"Input: '{value}' -> {float_value} (type: {type(value).__name__}). "
                f"Expected: finite float in range [{min_val}, {max_val}]. "
                f"Using default: {default}. "
                f"Suggestion: Provide a finite numeric value within the allowed range."
            )
            return default
            
        if not (min_val <= float_value <= max_val):
            logger.warning(
                f"{component}.{param_name}: Value '{float_value}' is outside allowed range. "
                f"Input: {float_value} (type: {type(value).__name__}). "
                f"Expected: float in range [{min_val}, {max_val}]. "
                f"Using default: {default}. "
                f"Suggestion: Provide a value between {min_val} and {max_val} inclusive."
            )
            return default
            
        return float_value
        
    except (ValueError, TypeError) as e:
        logger.error(
            f"{component}.{param_name}: Failed to convert '{value}' to float. "
            f"Input type: {type(value).__name__}, input value: '{value}'. "
            f"Expected: numeric value convertible to float in range [{min_val}, {max_val}]. "
            f"Conversion error: {type(e).__name__}: {str(e)}. "
            f"Using default: {default}. "
            f"Suggestion: Provide a numeric value like {min_val}, {default}, or {max_val}."
        )
        return default


def validate_and_default_string(
    value: Any, 
    default: str, 
    param_name: str, 
    component: str = "System",
    allow_empty: bool = False
) -> str:
    """
    Validate that a parameter is a non-empty string, substitute default if invalid.
    
    Args:
        value: Parameter value to validate
        default: Default value to use if invalid
        param_name: Name of the parameter for logging
        component: Component name for logging context
        allow_empty: Whether to allow empty strings
        
    Returns:
        Valid string (original value or default)
    """
    try:
        if value is None:
            logger.info(
                f"{component}.{param_name}: Parameter is None. "
                f"Using default value: '{default}'. "
                f"Expected: {'string (can be empty)' if allow_empty else 'non-empty string'}"
            )
            return default
            
        str_value = str(value)
        
        if not allow_empty and not str_value.strip():
            logger.warning(
                f"{component}.{param_name}: String is empty or contains only whitespace. "
                f"Input: '{str_value}' (length: {len(str_value)}, type: {type(value).__name__}). "
                f"Expected: non-empty string with meaningful content. "
                f"Using default: '{default}'. "
                f"Suggestion: Provide a string with actual content, not just spaces or empty text."
            )
            return default
            
        return str_value
        
    except Exception as e:
        logger.error(
            f"{component}.{param_name}: Failed to convert '{value}' to string. "
            f"Input type: {type(value).__name__}, input value: '{value}'. "
            f"Expected: value convertible to {'string (can be empty)' if allow_empty else 'non-empty string'}. "
            f"Conversion error: {type(e).__name__}: {str(e)}. "
            f"Using default: '{default}'. "
            f"Suggestion: Provide a value that can be converted to string (text, number, etc.)."
        )
        return default


def validate_and_default_path(
    value: Any, 
    default: Union[str, Path], 
    param_name: str, 
    component: str = "System",
    must_exist: bool = False
) -> Path:
    """
    Validate that a parameter is a valid path, substitute default if invalid.
    
    Args:
        value: Parameter value to validate
        default: Default path to use if invalid
        param_name: Name of the parameter for logging
        component: Component name for logging context
        must_exist: Whether the path must exist
        
    Returns:
        Valid Path object (original value or default)
    """
    try:
        if value is None:
            logger.info(
                f"{component}.{param_name}: Parameter is None. "
                f"Using default path: '{default}'. "
                f"Expected: valid file/directory path {'that exists' if must_exist else '(may not exist yet)'}"
            )
            return Path(default)
            
        path_value = Path(value)
        
        if must_exist and not path_value.exists():
            logger.error(
                f"{component}.{param_name}: Path does not exist. "
                f"Input path: '{path_value}' (absolute: '{path_value.absolute()}'). "
                f"Expected: existing file or directory path. "
                f"Current working directory: '{Path.cwd()}'. "
                f"Using default: '{default}'. "
                f"Suggestion: Check if the path exists and is accessible, or create the required file/directory."
            )
            return Path(default)
            
        return path_value
        
    except Exception as e:
        logger.error(
            f"{component}.{param_name}: Failed to create Path object from '{value}'. "
            f"Input type: {type(value).__name__}, input value: '{value}'. "
            f"Expected: valid path string or Path object. "
            f"Path creation error: {type(e).__name__}: {str(e)}. "
            f"Using default: '{default}'. "
            f"Suggestion: Provide a valid file path string like 'data/file.csv' or '/absolute/path/file.txt'."
        )
        return Path(default)


def validate_and_default_iterable(
    value: Any, 
    default: list, 
    param_name: str, 
    component: str = "System",
    allow_empty: bool = True
) -> list:
    """
    Validate that a parameter is an iterable, substitute default if invalid.
    
    Args:
        value: Parameter value to validate
        default: Default list to use if invalid
        param_name: Name of the parameter for logging
        component: Component name for logging context
        allow_empty: Whether to allow empty iterables
        
    Returns:
        Valid list (converted from original value or default)
    """
    try:
        if value is None:
            logger.info(
                f"{component}.{param_name}: Parameter is None. "
                f"Using default list: {default}. "
                f"Expected: iterable (list, tuple, etc.) {'that can be empty' if allow_empty else 'with at least one element'}"
            )
            return default
            
        # Try to convert to list
        if isinstance(value, str):
            # Don't treat strings as iterables of characters
            logger.warning(
                f"{component}.{param_name}: String provided instead of iterable. "
                f"Input: '{value}' (type: {type(value).__name__}, length: {len(value)}). "
                f"Expected: iterable like list, tuple, or set, not string. "
                f"Using default: {default}. "
                f"Suggestion: Wrap string in a list like ['{value}'] or provide a proper iterable."
            )
            return default
            
        list_value = list(value)
        
        if not allow_empty and not list_value:
            logger.warning(
                f"{component}.{param_name}: Iterable is empty. "
                f"Input: {value} (type: {type(value).__name__}, length: 0). "
                f"Expected: non-empty iterable with at least one element. "
                f"Using default: {default}. "
                f"Suggestion: Provide an iterable with content like ['item1', 'item2'] or (1, 2, 3)."
            )
            return default
            
        return list_value
        
    except (TypeError, ValueError) as e:
        logger.error(
            f"{component}.{param_name}: Failed to convert '{value}' to iterable. "
            f"Input type: {type(value).__name__}, input value: '{value}'. "
            f"Expected: iterable object (list, tuple, set, etc.) {'that can be empty' if allow_empty else 'with at least one element'}. "
            f"Conversion error: {type(e).__name__}: {str(e)}. "
            f"Using default: {default}. "
            f"Suggestion: Provide an iterable like [1, 2, 3], ('a', 'b'), or {{'x', 'y'}}."
        )
        return default


def validate_chunk_parameters(
    chunk_size: Any, 
    overlap: Any, 
    component: str = "TextChunker"
) -> tuple[int, int]:
    """
    Validate chunking parameters with interdependent constraints.
    
    Args:
        chunk_size: Chunk size parameter
        overlap: Overlap parameter
        component: Component name for logging
        
    Returns:
        Tuple of (validated_chunk_size, validated_overlap)
    """
    # Default values
    default_chunk_size = 4000
    default_overlap = 200
    
    # Validate chunk_size first
    validated_chunk_size = validate_and_default_positive_int(
        chunk_size, default_chunk_size, "chunk_size", component
    )
    
    # Validate overlap with constraint relative to chunk_size
    validated_overlap = validate_and_default_non_negative_int(
        overlap, default_overlap, "overlap", component
    )
    
    # Ensure overlap < chunk_size
    if validated_overlap >= validated_chunk_size:
        logger.error(
            f"{component}: Invalid chunking configuration detected. "
            f"Overlap ({validated_overlap}) must be less than chunk_size ({validated_chunk_size}). "
            f"Current values: chunk_size={validated_chunk_size}, overlap={validated_overlap}. "
            f"This would cause infinite loops or invalid chunks. "
            f"Using safe default overlap: {default_overlap}. "
            f"Suggestion: Ensure overlap < chunk_size, e.g., chunk_size=4000 with overlap=200."
        )
        validated_overlap = min(default_overlap, validated_chunk_size - 1)
    
    return validated_chunk_size, validated_overlap


def log_parameter_correction(
    original_value: Any, 
    corrected_value: Any, 
    param_name: str, 
    component: str, 
    reason: str = "invalid parameter"
) -> None:
    """
    Log parameter correction for debugging and monitoring.
    
    Args:
        original_value: Original parameter value
        corrected_value: Corrected parameter value
        param_name: Name of the parameter
        component: Component name
        reason: Reason for correction
    """
    logger.info(
        f"{component}: Parameter auto-correction applied. "
        f"Parameter: {param_name}, "
        f"Original: '{original_value}' (type: {type(original_value).__name__}), "
        f"Corrected: '{corrected_value}' (type: {type(corrected_value).__name__}), "
        f"Reason: {reason}. "
        f"Operation will continue with corrected value."
    )