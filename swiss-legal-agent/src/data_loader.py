from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Any, Dict

import pandas as pd

from .validation import (
    validate_and_default_path,
    validate_and_default_iterable,
    log_parameter_correction
)

logger = logging.getLogger(__name__)


def read_csv_file(path: str | Path, **kwargs) -> pd.DataFrame:
    """
    Read CSV file with comprehensive error handling for missing files.
    
    Args:
        path: Path to CSV file
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        DataFrame with loaded data, or empty DataFrame if file doesn't exist
    """
    csv_path = Path(path)
    
    if not csv_path.exists():
        logger.warning(
            f"DataLoader.read_csv_file: CSV file not found. "
            f"Requested path: '{csv_path}' (absolute: '{csv_path.absolute()}'). "
            f"Current working directory: '{Path.cwd()}'. "
            f"File existence check failed. "
            f"Returning empty DataFrame to allow graceful continuation. "
            f"Suggestion: Verify the file path exists or create the required CSV file."
        )
        return pd.DataFrame()
    
    if not csv_path.is_file():
        logger.error(
            f"DataLoader.read_csv_file: Path exists but is not a file. "
            f"Path: '{csv_path}' (absolute: '{csv_path.absolute()}'). "
            f"Path type: {'directory' if csv_path.is_dir() else 'other'}. "
            f"Expected: regular file with .csv extension. "
            f"Returning empty DataFrame. "
            f"Suggestion: Ensure the path points to a CSV file, not a directory."
        )
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path, **kwargs)
        logger.info(
            f"DataLoader.read_csv_file: Successfully loaded CSV file. "
            f"Path: '{csv_path}', "
            f"Shape: {df.shape} (rows: {len(df)}, columns: {len(df.columns)}), "
            f"Columns: {list(df.columns)}, "
            f"Memory usage: ~{df.memory_usage(deep=True).sum() / 1024:.1f} KB"
        )
        return df
        
    except pd.errors.EmptyDataError as e:
        logger.error(
            f"DataLoader.read_csv_file: CSV file is empty or has no data. "
            f"Path: '{csv_path}', "
            f"File size: {csv_path.stat().st_size} bytes. "
            f"Error: {str(e)}. "
            f"Returning empty DataFrame. "
            f"Suggestion: Check if the CSV file contains data and proper headers."
        )
        return pd.DataFrame()
        
    except pd.errors.ParserError as e:
        logger.error(
            f"DataLoader.read_csv_file: CSV parsing failed due to malformed data. "
            f"Path: '{csv_path}', "
            f"File size: {csv_path.stat().st_size} bytes. "
            f"Parser error: {str(e)}. "
            f"Returning empty DataFrame. "
            f"Suggestion: Check CSV format, ensure proper delimiters and escaping."
        )
        return pd.DataFrame()
        
    except PermissionError as e:
        logger.error(
            f"DataLoader.read_csv_file: Permission denied accessing file. "
            f"Path: '{csv_path}' (absolute: '{csv_path.absolute()}'). "
            f"Permission error: {str(e)}. "
            f"Current user may lack read permissions. "
            f"Returning empty DataFrame. "
            f"Suggestion: Check file permissions or run with appropriate privileges."
        )
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(
            f"DataLoader.read_csv_file: Unexpected error reading CSV file. "
            f"Path: '{csv_path}', "
            f"File size: {csv_path.stat().st_size if csv_path.exists() else 'unknown'} bytes. "
            f"Error type: {type(e).__name__}, "
            f"Error message: {str(e)}. "
            f"Returning empty DataFrame to prevent system failure. "
            f"Suggestion: Check file format, encoding, or contact system administrator."
        )
        return pd.DataFrame()


def clean_text(value: object) -> str:
    """
    Clean text by removing extra whitespace and normalizing format.
    
    Args:
        value: Input value to clean (can be None, string, or other type)
        
    Returns:
        Cleaned string with normalized whitespace
    """
    if value is None:
        return ""
    text = str(value)
    # Remove extra whitespace and normalize
    text = " ".join(text.split())
    return text.strip()


def normalize_text_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Normalize text columns by removing extra whitespace with comprehensive error handling.
    
    Args:
        frame: Input DataFrame
        columns: Iterable of column names to normalize
        
    Returns:
        DataFrame with normalized text columns (structure preserved)
    """
    if frame.empty:
        logger.info(
            f"DataLoader.normalize_text_columns: Input DataFrame is empty. "
            f"Columns to normalize: {list(columns)}. "
            f"Returning empty DataFrame unchanged. "
            f"No normalization needed for empty data."
        )
        return frame.copy()

    # Validate columns parameter
    original_columns = columns
    validated_columns = validate_and_default_iterable(
        columns, [], "columns", "DataLoader.normalize_text_columns", allow_empty=True
    )
    if validated_columns != list(original_columns):
        log_parameter_correction(original_columns, validated_columns, "columns", "DataLoader.normalize_text_columns", "invalid iterable")
    
    # Capture original structure for validation
    original_structure = capture_dataframe_structure(frame)
    
    cleaned = frame.copy()
    normalized_count = 0
    skipped_columns = []
    
    for column in validated_columns:
        if column in cleaned.columns:
            try:
                original_values = cleaned[column].copy()
                cleaned[column] = cleaned[column].map(clean_text)
                normalized_count += 1
                
                # Log normalization statistics
                null_count = original_values.isnull().sum()
                changed_count = (original_values != cleaned[column]).sum()
                logger.debug(
                    f"DataLoader.normalize_text_columns: Normalized column '{column}'. "
                    f"Total values: {len(original_values)}, "
                    f"Null values: {null_count}, "
                    f"Values changed: {changed_count}, "
                    f"Sample before: '{original_values.iloc[0] if len(original_values) > 0 else 'N/A'}', "
                    f"Sample after: '{cleaned[column].iloc[0] if len(cleaned) > 0 else 'N/A'}'"
                )
                
            except Exception as e:
                logger.error(
                    f"DataLoader.normalize_text_columns: Failed to normalize column '{column}'. "
                    f"Column type: {frame[column].dtype}, "
                    f"Column shape: {frame[column].shape}, "
                    f"Error: {type(e).__name__}: {str(e)}. "
                    f"Skipping this column and continuing with others. "
                    f"Suggestion: Check if column contains data that can be converted to text."
                )
                skipped_columns.append(column)
                continue
        else:
            logger.warning(
                f"DataLoader.normalize_text_columns: Column '{column}' not found in DataFrame. "
                f"Available columns: {list(frame.columns)} (total: {len(frame.columns)}). "
                f"Skipping normalization for this column. "
                f"Suggestion: Check column name spelling or verify DataFrame structure."
            )
            skipped_columns.append(column)
    
    # Log summary of normalization operation
    logger.info(
        f"DataLoader.normalize_text_columns: Text normalization completed. "
        f"Columns processed: {normalized_count}/{len(validated_columns)}, "
        f"Columns skipped: {len(skipped_columns)} {skipped_columns if skipped_columns else ''}, "
        f"DataFrame shape: {cleaned.shape} (unchanged)"
    )
    
    # Validate structure preservation
    try:
        validate_structure_preservation(original_structure, cleaned, f"text normalization of columns {list(validated_columns)}")
    except ValueError as e:
        logger.error(
            f"DataLoader.normalize_text_columns: Structure preservation validation failed. "
            f"Original shape: {original_structure['shape']}, "
            f"Current shape: {cleaned.shape}. "
            f"Validation error: {str(e)}. "
            f"This indicates a serious data integrity issue."
        )
        raise
    
    return cleaned


def load_dataset(path: str | Path, text_columns: Iterable[str] | None = None, **kwargs) -> pd.DataFrame:
    """
    Load dataset from CSV file with optional text column normalization.
    
    Args:
        path: Path to CSV file
        text_columns: Optional iterable of column names to normalize
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        DataFrame with loaded and optionally normalized data
    """
    # Validate and correct path parameter
    original_path = path
    validated_path = validate_and_default_path(
        path, "data/raw/train.csv", "path", "DataLoader", must_exist=False
    )
    if str(validated_path) != str(original_path):
        log_parameter_correction(original_path, validated_path, "path", "DataLoader", "invalid path")
    
    # Validate text_columns parameter
    if text_columns is not None:
        original_text_columns = text_columns
        validated_text_columns = validate_and_default_iterable(
            text_columns, [], "text_columns", "DataLoader", allow_empty=True
        )
        if validated_text_columns != list(original_text_columns):
            log_parameter_correction(original_text_columns, validated_text_columns, "text_columns", "DataLoader", "invalid iterable")
        text_columns = validated_text_columns
    
    frame = read_csv_file(validated_path, **kwargs)
    if text_columns and not frame.empty:
        original_structure = capture_dataframe_structure(frame)
        frame = normalize_text_columns(frame, text_columns)
        validate_structure_preservation(original_structure, frame, "text normalization")
    return frame


def capture_dataframe_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Capture the structure of a DataFrame for validation purposes.
    
    Args:
        df: DataFrame to capture structure from
        
    Returns:
        Dictionary containing structure information
    """
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': dict(df.dtypes),
        'index': df.index.tolist() if len(df) < 1000 else len(df.index),  # Store full index for small DataFrames
        'row_count': len(df),
        'column_count': len(df.columns)
    }


def validate_structure_preservation(original_structure: Dict[str, Any], processed_df: pd.DataFrame, operation_name: str = "processing") -> None:
    """
    Validate that DataFrame structure is preserved after processing with detailed error messages.
    
    Args:
        original_structure: Structure captured before processing
        processed_df: DataFrame after processing
        operation_name: Name of the operation for error messages
        
    Raises:
        ValueError: If structure preservation is violated
    """
    current_structure = capture_dataframe_structure(processed_df)
    
    # Validate row count preservation
    if original_structure['row_count'] != current_structure['row_count']:
        raise ValueError(
            f"DataLoader.validate_structure_preservation: Row count changed during {operation_name}. "
            f"Expected: {original_structure['row_count']} rows, "
            f"Got: {current_structure['row_count']} rows. "
            f"Difference: {current_structure['row_count'] - original_structure['row_count']} rows. "
            f"Original shape: {original_structure['shape']}, "
            f"Current shape: {current_structure['shape']}. "
            f"This indicates data loss or unexpected data addition during processing. "
            f"Operation: {operation_name}"
        )
    
    # Validate column structure preservation
    if original_structure['columns'] != current_structure['columns']:
        missing_cols = set(original_structure['columns']) - set(current_structure['columns'])
        added_cols = set(current_structure['columns']) - set(original_structure['columns'])
        
        raise ValueError(
            f"DataLoader.validate_structure_preservation: Column structure changed during {operation_name}. "
            f"Expected columns: {original_structure['columns']}, "
            f"Got columns: {current_structure['columns']}. "
            f"Missing columns: {list(missing_cols) if missing_cols else 'none'}, "
            f"Added columns: {list(added_cols) if added_cols else 'none'}. "
            f"Column count change: {len(current_structure['columns']) - len(original_structure['columns'])}. "
            f"This indicates unexpected column modification during processing. "
            f"Operation: {operation_name}"
        )
    
    # Validate column count preservation
    if original_structure['column_count'] != current_structure['column_count']:
        raise ValueError(
            f"DataLoader.validate_structure_preservation: Column count changed during {operation_name}. "
            f"Expected: {original_structure['column_count']} columns, "
            f"Got: {current_structure['column_count']} columns. "
            f"Difference: {current_structure['column_count'] - original_structure['column_count']} columns. "
            f"Original columns: {original_structure['columns']}, "
            f"Current columns: {current_structure['columns']}. "
            f"This indicates structural modification during processing. "
            f"Operation: {operation_name}"
        )
    
    # Validate index preservation (for small DataFrames)
    if isinstance(original_structure['index'], list) and isinstance(current_structure['index'], list):
        if original_structure['index'] != current_structure['index']:
            raise ValueError(
                f"DataLoader.validate_structure_preservation: Row index changed during {operation_name}. "
                f"Expected index: {original_structure['index'][:10]}{'...' if len(original_structure['index']) > 10 else ''}, "
                f"Got index: {current_structure['index'][:10]}{'...' if len(current_structure['index']) > 10 else ''}. "
                f"Index length: original={len(original_structure['index'])}, current={len(current_structure['index'])}. "
                f"Row relationships and identifiers are not preserved. "
                f"This may affect data integrity and downstream processing. "
                f"Operation: {operation_name}"
            )
    
    logger.debug(
        f"DataLoader.validate_structure_preservation: Structure validation passed for {operation_name}. "
        f"Preserved: {current_structure['row_count']} rows, {current_structure['column_count']} columns. "
        f"Shape: {current_structure['shape']}, "
        f"Columns: {current_structure['columns']}"
    )


def validate_dataframe_integrity(df: pd.DataFrame, expected_columns: Iterable[str] | None = None, min_rows: int = 0) -> bool:
    """
    Validate DataFrame integrity and structure with comprehensive error messages.
    
    Args:
        df: DataFrame to validate
        expected_columns: Optional list of expected column names
        min_rows: Minimum expected number of rows
        
    Returns:
        True if DataFrame passes validation
        
    Raises:
        ValueError: If validation fails
    """
    if df is None:
        raise ValueError(
            f"DataLoader.validate_dataframe_integrity: DataFrame cannot be None. "
            f"Expected: pandas DataFrame object. "
            f"Got: None (NoneType). "
            f"Suggestion: Ensure DataFrame is properly initialized before validation."
        )
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            f"DataLoader.validate_dataframe_integrity: Invalid object type. "
            f"Expected: pandas DataFrame. "
            f"Got: {type(df).__name__} with value '{df}'. "
            f"Suggestion: Convert data to pandas DataFrame using pd.DataFrame() constructor."
        )
    
    if len(df) < min_rows:
        raise ValueError(
            f"DataLoader.validate_dataframe_integrity: Insufficient rows in DataFrame. "
            f"Expected: at least {min_rows} rows. "
            f"Got: {len(df)} rows. "
            f"DataFrame shape: {df.shape}. "
            f"Shortage: {min_rows - len(df)} rows. "
            f"Suggestion: Ensure data source contains sufficient records or adjust minimum row requirement."
        )
    
    if expected_columns:
        expected_cols = list(expected_columns)
        actual_cols = list(df.columns)
        missing_cols = set(expected_cols) - set(actual_cols)
        
        if missing_cols:
            extra_cols = set(actual_cols) - set(expected_cols)
            raise ValueError(
                f"DataLoader.validate_dataframe_integrity: Missing expected columns. "
                f"Expected columns: {expected_cols} (count: {len(expected_cols)}). "
                f"Actual columns: {actual_cols} (count: {len(actual_cols)}). "
                f"Missing columns: {sorted(missing_cols)} (count: {len(missing_cols)}). "
                f"Extra columns: {sorted(extra_cols) if extra_cols else 'none'}. "
                f"Suggestion: Check data source schema or update expected column list."
            )
    
    # Check for duplicate columns
    if len(df.columns) != len(set(df.columns)):
        duplicates = [col for col in df.columns if list(df.columns).count(col) > 1]
        unique_duplicates = list(set(duplicates))
        duplicate_counts = {col: list(df.columns).count(col) for col in unique_duplicates}
        
        raise ValueError(
            f"DataLoader.validate_dataframe_integrity: Duplicate columns detected. "
            f"Total columns: {len(df.columns)}, "
            f"Unique columns: {len(set(df.columns))}. "
            f"Duplicate columns: {unique_duplicates}. "
            f"Duplicate counts: {duplicate_counts}. "
            f"All columns: {list(df.columns)}. "
            f"Suggestion: Remove or rename duplicate columns to ensure unique column names."
        )
    
    logger.debug(
        f"DataLoader.validate_dataframe_integrity: DataFrame validation passed. "
        f"Shape: {df.shape} (rows: {len(df)}, columns: {len(df.columns)}). "
        f"Columns: {list(df.columns)}. "
        f"Memory usage: ~{df.memory_usage(deep=True).sum() / 1024:.1f} KB. "
        f"Data types: {dict(df.dtypes)}"
    )
    return True
