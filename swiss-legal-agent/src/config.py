"""
Configuration module with parameter validation and default substitution.

This module provides centralized configuration management with validation
and default value substitution for all system parameters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .validation import (
    validate_and_default_path,
    validate_and_default_positive_int,
    validate_and_default_string,
    log_parameter_correction
)

logger = logging.getLogger(__name__)

# Base paths with validation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Default file paths
TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"
VAL_FILE = RAW_DATA_DIR / "val.csv"
LAWS_FILE = RAW_DATA_DIR / "laws_de.csv"
SUBMISSION_FILE = PROJECT_ROOT / "submission.csv"

# Default configuration values
DEFAULT_BM25_TOP_K = 10
DEFAULT_VECTOR_TOP_K = 10
DEFAULT_CHUNK_SIZE = 4000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MODEL_NAME = "local-hf-model"


class SystemConfig:
    """
    System configuration with parameter validation and default substitution.
    """
    
    def __init__(
        self,
        bm25_top_k: Any = None,
        vector_top_k: Any = None,
        chunk_size: Any = None,
        chunk_overlap: Any = None,
        model_name: Any = None,
        train_file: Any = None,
        test_file: Any = None,
        laws_file: Any = None,
        submission_file: Any = None
    ):
        """
        Initialize system configuration with parameter validation.
        
        Args:
            bm25_top_k: Number of top results for BM25 retrieval
            vector_top_k: Number of top results for vector retrieval
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            model_name: Name of the language model
            train_file: Path to training data file
            test_file: Path to test data file
            laws_file: Path to laws data file
            submission_file: Path to submission output file
        """
        # Validate and set retrieval parameters
        self.bm25_top_k = validate_and_default_positive_int(
            bm25_top_k, DEFAULT_BM25_TOP_K, "bm25_top_k", "SystemConfig"
        )
        
        self.vector_top_k = validate_and_default_positive_int(
            vector_top_k, DEFAULT_VECTOR_TOP_K, "vector_top_k", "SystemConfig"
        )
        
        # Validate chunking parameters with interdependent constraints
        validated_chunk_size = validate_and_default_positive_int(
            chunk_size, DEFAULT_CHUNK_SIZE, "chunk_size", "SystemConfig"
        )
        
        validated_chunk_overlap = validate_and_default_positive_int(
            chunk_overlap, DEFAULT_CHUNK_OVERLAP, "chunk_overlap", "SystemConfig"
        )
        
        # Ensure overlap < chunk_size
        if validated_chunk_overlap >= validated_chunk_size:
            logger.warning(
                f"SystemConfig: chunk_overlap={validated_chunk_overlap} >= chunk_size={validated_chunk_size}, "
                f"using default overlap: {DEFAULT_CHUNK_OVERLAP}"
            )
            validated_chunk_overlap = min(DEFAULT_CHUNK_OVERLAP, validated_chunk_size - 1)
        
        self.chunk_size = validated_chunk_size
        self.chunk_overlap = validated_chunk_overlap
        
        # Validate model name
        self.model_name = validate_and_default_string(
            model_name, DEFAULT_MODEL_NAME, "model_name", "SystemConfig", allow_empty=False
        )
        
        # Validate file paths
        self.train_file = validate_and_default_path(
            train_file, TRAIN_FILE, "train_file", "SystemConfig", must_exist=False
        )
        
        self.test_file = validate_and_default_path(
            test_file, TEST_FILE, "test_file", "SystemConfig", must_exist=False
        )
        
        self.laws_file = validate_and_default_path(
            laws_file, LAWS_FILE, "laws_file", "SystemConfig", must_exist=False
        )
        
        self.submission_file = validate_and_default_path(
            submission_file, SUBMISSION_FILE, "submission_file", "SystemConfig", must_exist=False
        )
        
        # Log configuration summary
        logger.info(f"SystemConfig initialized: chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, "
                   f"bm25_top_k={self.bm25_top_k}, vector_top_k={self.vector_top_k}, model_name='{self.model_name}'")
    
    def validate_and_update_parameter(self, param_name: str, value: Any) -> Any:
        """
        Validate and update a single configuration parameter.
        
        Args:
            param_name: Name of the parameter to update
            value: New value for the parameter
            
        Returns:
            Validated parameter value
        """
        original_value = getattr(self, param_name, None)
        
        if param_name == "bm25_top_k":
            validated_value = validate_and_default_positive_int(
                value, DEFAULT_BM25_TOP_K, param_name, "SystemConfig"
            )
        elif param_name == "vector_top_k":
            validated_value = validate_and_default_positive_int(
                value, DEFAULT_VECTOR_TOP_K, param_name, "SystemConfig"
            )
        elif param_name == "chunk_size":
            validated_value = validate_and_default_positive_int(
                value, DEFAULT_CHUNK_SIZE, param_name, "SystemConfig"
            )
            # Re-validate overlap if chunk_size changes
            if hasattr(self, 'chunk_overlap') and self.chunk_overlap >= validated_value:
                self.chunk_overlap = min(DEFAULT_CHUNK_OVERLAP, validated_value - 1)
                logger.warning(f"SystemConfig: Adjusted chunk_overlap to {self.chunk_overlap} due to chunk_size change")
        elif param_name == "chunk_overlap":
            validated_value = validate_and_default_positive_int(
                value, DEFAULT_CHUNK_OVERLAP, param_name, "SystemConfig"
            )
            # Ensure overlap < chunk_size
            if hasattr(self, 'chunk_size') and validated_value >= self.chunk_size:
                validated_value = min(DEFAULT_CHUNK_OVERLAP, self.chunk_size - 1)
                logger.warning(f"SystemConfig: Adjusted chunk_overlap to {validated_value} to be less than chunk_size")
        elif param_name == "model_name":
            validated_value = validate_and_default_string(
                value, DEFAULT_MODEL_NAME, param_name, "SystemConfig", allow_empty=False
            )
        elif param_name in ["train_file", "test_file", "laws_file", "submission_file"]:
            default_path = getattr(self, param_name, Path("default.csv"))
            validated_value = validate_and_default_path(
                value, default_path, param_name, "SystemConfig", must_exist=False
            )
        else:
            logger.warning(f"SystemConfig: Unknown parameter '{param_name}', no validation applied")
            validated_value = value
        
        if validated_value != original_value:
            log_parameter_correction(original_value, validated_value, param_name, "SystemConfig", "parameter update")
            setattr(self, param_name, validated_value)
        
        return validated_value


# Global default configuration instance
default_config = SystemConfig()
