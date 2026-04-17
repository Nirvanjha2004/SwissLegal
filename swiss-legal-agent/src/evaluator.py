from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd
from sklearn.metrics import f1_score

from .validation import (
    validate_and_default_string,
    validate_and_default_iterable,
    log_parameter_correction
)

logger = logging.getLogger(__name__)


def macro_f1(y_true: Iterable, y_pred: Iterable) -> float:
    """
    Calculate macro F1 score for multi-class evaluation with comprehensive parameter validation.
    
    The macro F1 score calculates F1 scores for each class independently
    and then takes the unweighted average, which is useful for multi-class
    evaluation scenarios where all classes should be treated equally.
    
    Args:
        y_true: True labels (ground truth)
        y_pred: Predicted labels
        
    Returns:
        float: Macro F1 score bounded between 0.0 and 1.0
        
    Raises:
        ValueError: If inputs are empty or have mismatched lengths
    """
    # Validate y_true parameter
    original_y_true = y_true
    validated_y_true = validate_and_default_iterable(
        y_true, [], "y_true", "Evaluator.macro_f1", allow_empty=False
    )
    if validated_y_true != list(original_y_true):
        log_parameter_correction(original_y_true, validated_y_true, "y_true", "Evaluator.macro_f1", "invalid iterable")
    
    # Validate y_pred parameter
    original_y_pred = y_pred
    validated_y_pred = validate_and_default_iterable(
        y_pred, [], "y_pred", "Evaluator.macro_f1", allow_empty=False
    )
    if validated_y_pred != list(original_y_pred):
        log_parameter_correction(original_y_pred, validated_y_pred, "y_pred", "Evaluator.macro_f1", "invalid iterable")
    
    # Convert to lists to handle various iterable types
    y_true_list = list(validated_y_true)
    y_pred_list = list(validated_y_pred)
    
    # Handle edge cases
    if not y_true_list or not y_pred_list:
        logger.warning(
            f"Evaluator.macro_f1: Empty input data detected. "
            f"y_true length: {len(y_true_list)}, "
            f"y_pred length: {len(y_pred_list)}. "
            f"Cannot compute F1 score with empty data. "
            f"Returning 0.0. "
            f"Suggestion: Ensure both ground truth and predictions contain data."
        )
        return 0.0
    
    if len(y_true_list) != len(y_pred_list):
        logger.error(
            f"Evaluator.macro_f1: Length mismatch between ground truth and predictions. "
            f"y_true length: {len(y_true_list)}, "
            f"y_pred length: {len(y_pred_list)}, "
            f"Difference: {abs(len(y_true_list) - len(y_pred_list))} elements. "
            f"y_true sample: {y_true_list[:5] if y_true_list else 'empty'}, "
            f"y_pred sample: {y_pred_list[:5] if y_pred_list else 'empty'}. "
            f"Evaluation requires equal-length arrays. "
            f"Suggestion: Ensure predictions are generated for all ground truth samples."
        )
        raise ValueError(f"Length mismatch: y_true has {len(y_true_list)} elements, y_pred has {len(y_pred_list)} elements")
    
    # Analyze class distribution
    true_classes = set(y_true_list)
    pred_classes = set(y_pred_list)
    all_classes = true_classes.union(pred_classes)
    
    logger.debug(
        f"Evaluator.macro_f1: Class analysis. "
        f"True classes: {sorted(true_classes)} (count: {len(true_classes)}), "
        f"Predicted classes: {sorted(pred_classes)} (count: {len(pred_classes)}), "
        f"All classes: {sorted(all_classes)} (count: {len(all_classes)}), "
        f"Missing in predictions: {sorted(true_classes - pred_classes)}, "
        f"Extra in predictions: {sorted(pred_classes - true_classes)}"
    )
    
    # Handle single-class scenarios
    if len(set(y_true_list)) == 1 and len(set(y_pred_list)) == 1:
        # If both have the same single class, perfect score
        if y_true_list[0] == y_pred_list[0]:
            logger.info(
                f"Evaluator.macro_f1: Single-class perfect match scenario. "
                f"Class: '{y_true_list[0]}', "
                f"Samples: {len(y_true_list)}. "
                f"Returning F1 score: 1.0"
            )
            return 1.0
        # If different single classes, zero score
        else:
            logger.info(
                f"Evaluator.macro_f1: Single-class mismatch scenario. "
                f"True class: '{y_true_list[0]}', "
                f"Predicted class: '{y_pred_list[0]}', "
                f"Samples: {len(y_true_list)}. "
                f"Returning F1 score: 0.0"
            )
            return 0.0
    
    try:
        # Calculate macro F1 score using sklearn
        score = float(f1_score(y_true_list, y_pred_list, average="macro", zero_division=0))
        
        # Ensure bounds are respected (should always be true with sklearn, but explicit check)
        score = max(0.0, min(1.0, score))
        
        logger.info(
            f"Evaluator.macro_f1: F1 score calculation completed. "
            f"Samples: {len(y_true_list)}, "
            f"Classes: {len(all_classes)}, "
            f"Macro F1 score: {score:.4f}, "
            f"Score bounded: [0.0, 1.0]"
        )
        
        return score
        
    except Exception as e:
        logger.error(
            f"Evaluator.macro_f1: F1 score calculation failed. "
            f"Samples: {len(y_true_list)}, "
            f"True classes: {sorted(true_classes)}, "
            f"Pred classes: {sorted(pred_classes)}, "
            f"Error type: {type(e).__name__}, "
            f"Error message: {str(e)}. "
            f"Returning 0.0 as fallback. "
            f"Suggestion: Check data format and sklearn compatibility."
        )
        return 0.0


def evaluate_predictions(frame: pd.DataFrame, target_column: str, prediction_column: str) -> float:
    """
    Evaluate predictions using macro F1 score with comprehensive parameter validation.
    
    Args:
        frame: DataFrame containing target and prediction columns
        target_column: Name of the column containing ground truth labels
        prediction_column: Name of the column containing predicted labels
        
    Returns:
        float: Macro F1 score between 0.0 and 1.0
        
    Raises:
        KeyError: If required columns are missing from the DataFrame
        ValueError: If data lengths mismatch or other validation errors
    """
    # Validate DataFrame parameter
    if frame is None:
        logger.error(
            f"Evaluator.evaluate_predictions: DataFrame is None. "
            f"Expected: pandas DataFrame with target and prediction columns. "
            f"Got: None (NoneType). "
            f"Cannot perform evaluation without data. "
            f"Suggestion: Provide a valid DataFrame with evaluation data."
        )
        raise ValueError("DataFrame cannot be None")
    
    if not isinstance(frame, pd.DataFrame):
        logger.error(
            f"Evaluator.evaluate_predictions: Invalid data type. "
            f"Expected: pandas DataFrame. "
            f"Got: {type(frame).__name__} with value '{frame}'. "
            f"Evaluation requires DataFrame format. "
            f"Suggestion: Convert data to pandas DataFrame before evaluation."
        )
        raise ValueError(f"Expected pandas DataFrame, got {type(frame)}")
    
    # Handle empty DataFrame
    if frame.empty:
        logger.warning(
            f"Evaluator.evaluate_predictions: Empty DataFrame provided. "
            f"Shape: {frame.shape}, "
            f"Columns: {list(frame.columns)}, "
            f"Target column: '{target_column}', "
            f"Prediction column: '{prediction_column}'. "
            f"Cannot evaluate with no data. "
            f"Returning F1 score: 0.0. "
            f"Suggestion: Ensure DataFrame contains evaluation data."
        )
        return 0.0
    
    # Validate column parameters
    original_target_column = target_column
    validated_target_column = validate_and_default_string(
        target_column, "target", "target_column", "Evaluator.evaluate_predictions", allow_empty=False
    )
    if validated_target_column != original_target_column:
        log_parameter_correction(original_target_column, validated_target_column, "target_column", "Evaluator.evaluate_predictions", "invalid column name")
    
    original_prediction_column = prediction_column
    validated_prediction_column = validate_and_default_string(
        prediction_column, "prediction", "prediction_column", "Evaluator.evaluate_predictions", allow_empty=False
    )
    if validated_prediction_column != original_prediction_column:
        log_parameter_correction(original_prediction_column, validated_prediction_column, "prediction_column", "Evaluator.evaluate_predictions", "invalid column name")
    
    # Validate required columns exist
    available_columns = list(frame.columns)
    
    if validated_target_column not in frame.columns:
        logger.error(
            f"Evaluator.evaluate_predictions: Target column not found. "
            f"Requested target column: '{validated_target_column}', "
            f"Available columns: {available_columns} (count: {len(available_columns)}), "
            f"DataFrame shape: {frame.shape}. "
            f"Column name case-sensitive match required. "
            f"Suggestion: Check column name spelling or use one of the available columns."
        )
        raise KeyError(f"Target column '{validated_target_column}' not found in DataFrame. Available columns: {list(frame.columns)}")
    
    if validated_prediction_column not in frame.columns:
        logger.error(
            f"Evaluator.evaluate_predictions: Prediction column not found. "
            f"Requested prediction column: '{validated_prediction_column}', "
            f"Available columns: {available_columns} (count: {len(available_columns)}), "
            f"DataFrame shape: {frame.shape}. "
            f"Column name case-sensitive match required. "
            f"Suggestion: Check column name spelling or use one of the available columns."
        )
        raise KeyError(f"Prediction column '{validated_prediction_column}' not found in DataFrame. Available columns: {list(frame.columns)}")
    
    # Extract the data and analyze NaN values
    original_target = frame[validated_target_column]
    original_pred = frame[validated_prediction_column]
    
    target_nan_count = original_target.isnull().sum()
    pred_nan_count = original_pred.isnull().sum()
    
    logger.debug(
        f"Evaluator.evaluate_predictions: Data extraction completed. "
        f"Total rows: {len(frame)}, "
        f"Target NaN values: {target_nan_count}, "
        f"Prediction NaN values: {pred_nan_count}, "
        f"Target column type: {original_target.dtype}, "
        f"Prediction column type: {original_pred.dtype}"
    )
    
    # Drop NaN values and find common indices
    y_true = original_target.dropna()
    y_pred = original_pred.dropna()
    
    # Ensure we have the same indices after dropping NaN
    common_indices = y_true.index.intersection(y_pred.index)
    
    if len(common_indices) == 0:
        logger.warning(
            f"Evaluator.evaluate_predictions: No valid data pairs after removing NaN values. "
            f"Original rows: {len(frame)}, "
            f"Target valid: {len(y_true)}, "
            f"Prediction valid: {len(y_pred)}, "
            f"Common valid indices: 0. "
            f"Target NaN: {target_nan_count}, "
            f"Prediction NaN: {pred_nan_count}. "
            f"Returning F1 score: 0.0. "
            f"Suggestion: Check data quality and ensure both columns have valid values."
        )
        return 0.0
    
    # Extract aligned data
    y_true = y_true.loc[common_indices]
    y_pred = y_pred.loc[common_indices]
    
    # Log evaluation summary
    excluded_rows = len(frame) - len(common_indices)
    logger.info(
        f"Evaluator.evaluate_predictions: Evaluation data prepared. "
        f"Total rows: {len(frame)}, "
        f"Valid pairs: {len(y_true)}, "
        f"Excluded rows: {excluded_rows} ({excluded_rows/len(frame)*100:.1f}%), "
        f"Target column: '{validated_target_column}', "
        f"Prediction column: '{validated_prediction_column}'"
    )
    
    try:
        return macro_f1(y_true, y_pred)
    except Exception as e:
        logger.error(
            f"Evaluator.evaluate_predictions: F1 calculation failed. "
            f"Valid pairs: {len(y_true)}, "
            f"Target sample: {list(y_true.head())}, "
            f"Prediction sample: {list(y_pred.head())}, "
            f"Error type: {type(e).__name__}, "
            f"Error message: {str(e)}. "
            f"Returning 0.0 as fallback."
        )
        return 0.0
