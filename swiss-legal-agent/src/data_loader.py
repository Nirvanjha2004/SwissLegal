from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def read_csv_file(path: str | Path, **kwargs) -> pd.DataFrame:
    """
    Read CSV file with error handling for missing files.
    
    Args:
        path: Path to CSV file
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        DataFrame with loaded data, or empty DataFrame if file doesn't exist
    """
    csv_path = Path(path)
    if not csv_path.exists():
        logger.warning(f"CSV file not found: {csv_path}. Returning empty DataFrame.")
        return pd.DataFrame()
    
    try:
        return pd.read_csv(csv_path, **kwargs)
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {e}. Returning empty DataFrame.")
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
    Normalize text columns by removing extra whitespace.
    
    Args:
        frame: Input DataFrame
        columns: Iterable of column names to normalize
        
    Returns:
        DataFrame with normalized text columns
    """
    if frame.empty:
        return frame.copy()

    cleaned = frame.copy()
    for column in columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].map(clean_text)
        else:
            logger.warning(f"Column '{column}' not found in DataFrame. Skipping normalization.")
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
    frame = read_csv_file(path, **kwargs)
    if text_columns and not frame.empty:
        frame = normalize_text_columns(frame, text_columns)
    return frame
