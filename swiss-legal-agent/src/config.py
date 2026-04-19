"""
Configuration module with parameter validation and default substitution.

This module provides centralized configuration management with validation
and default value substitution for all system parameters.
"""

from __future__ import annotations

import logging
import os
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
DEFAULT_ENABLE_TRAIN_EVAL = False
DEFAULT_EVAL_PROGRESS_INTERVAL = 100


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
        enable_train_eval: Any = None,
        eval_progress_interval: Any = None,
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
            enable_train_eval: Whether optional train evaluation runs
            eval_progress_interval: Progress logging interval for train evaluation
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

        enable_train_eval_normalized = str(enable_train_eval).strip().lower() if enable_train_eval is not None else None
        if enable_train_eval_normalized is None:
            self.enable_train_eval = DEFAULT_ENABLE_TRAIN_EVAL
        elif enable_train_eval_normalized in {"1", "true", "yes", "on"}:
            self.enable_train_eval = True
        elif enable_train_eval_normalized in {"0", "false", "no", "off"}:
            self.enable_train_eval = False
        else:
            logger.warning(
                "SystemConfig: enable_train_eval=%s is invalid, using default=%s",
                enable_train_eval,
                DEFAULT_ENABLE_TRAIN_EVAL,
            )
            self.enable_train_eval = DEFAULT_ENABLE_TRAIN_EVAL

        self.eval_progress_interval = validate_and_default_positive_int(
            eval_progress_interval,
            DEFAULT_EVAL_PROGRESS_INTERVAL,
            "eval_progress_interval",
            "SystemConfig",
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
        logger.info(
            "SystemConfig initialized: chunk_size=%s, chunk_overlap=%s, bm25_top_k=%s, "
            "vector_top_k=%s, model_name='%s', enable_train_eval=%s, eval_progress_interval=%s",
            self.chunk_size,
            self.chunk_overlap,
            self.bm25_top_k,
            self.vector_top_k,
            self.model_name,
            self.enable_train_eval,
            self.eval_progress_interval,
        )
    
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


def _get_env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning(
            "Config environment variable '%s' is not a valid integer: '%s'. Using default.",
            name,
            value,
        )
        return None


def _get_env_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _get_env_path(name: str) -> Path | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return Path(stripped)


def load_config_from_env() -> SystemConfig:
    """
    Build system configuration from environment variables with safe fallback.

    Supported environment variables:
    - BM25_TOP_K
    - VECTOR_TOP_K
    - CHUNK_SIZE
    - CHUNK_OVERLAP
    - MODEL_NAME
    - ENABLE_TRAIN_EVAL
    - EVAL_PROGRESS_INTERVAL
    - TRAIN_FILE
    - TEST_FILE
    - LAWS_FILE
    - SUBMISSION_FILE
    """
    config = SystemConfig(
        bm25_top_k=_get_env_int("BM25_TOP_K"),
        vector_top_k=_get_env_int("VECTOR_TOP_K"),
        chunk_size=_get_env_int("CHUNK_SIZE"),
        chunk_overlap=_get_env_int("CHUNK_OVERLAP"),
        model_name=_get_env_str("MODEL_NAME"),
        enable_train_eval=_get_env_str("ENABLE_TRAIN_EVAL"),
        eval_progress_interval=_get_env_int("EVAL_PROGRESS_INTERVAL"),
        train_file=_get_env_path("TRAIN_FILE"),
        test_file=_get_env_path("TEST_FILE"),
        laws_file=_get_env_path("LAWS_FILE"),
        submission_file=_get_env_path("SUBMISSION_FILE"),
    )
    logger.info(
        "SystemConfig loaded from environment with validation and default substitution. "
        "bm25_top_k=%s vector_top_k=%s chunk_size=%s chunk_overlap=%s model_name=%s "
        "enable_train_eval=%s eval_progress_interval=%s",
        config.bm25_top_k,
        config.vector_top_k,
        config.chunk_size,
        config.chunk_overlap,
        config.model_name,
        config.enable_train_eval,
        config.eval_progress_interval,
    )
    return config


def validate_config_on_startup(config: SystemConfig) -> dict[str, Any]:
    """
    Validate and summarize effective configuration at startup.

    Returns:
        Dictionary summary that can be logged or emitted for monitoring.
    """
    summary = {
        "bm25_top_k": config.bm25_top_k,
        "vector_top_k": config.vector_top_k,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "model_name": config.model_name,
        "enable_train_eval": config.enable_train_eval,
        "eval_progress_interval": config.eval_progress_interval,
        "train_file": str(config.train_file),
        "test_file": str(config.test_file),
        "laws_file": str(config.laws_file),
        "submission_file": str(config.submission_file),
    }
    logger.info("Startup configuration validated: %s", summary)
    return summary
