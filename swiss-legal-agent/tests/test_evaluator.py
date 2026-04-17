"""
Test cases for the evaluator module.

These tests verify the macro F1 score calculation functionality
and edge case handling as specified in requirements 5.1 and 5.4.
"""

import pytest
import pandas as pd
from src.evaluator import macro_f1, evaluate_predictions


class TestMacroF1:
    """Test cases for macro_f1 function."""
    
    def test_perfect_predictions(self):
        """Test that perfect predictions return F1 score of 1.0."""
        y_true = ['A', 'B', 'C', 'A', 'B', 'C']
        y_pred = ['A', 'B', 'C', 'A', 'B', 'C']
        score = macro_f1(y_true, y_pred)
        assert score == 1.0
    
    def test_completely_wrong_predictions(self):
        """Test that completely wrong predictions return F1 score of 0.0."""
        y_true = ['A', 'A', 'A']
        y_pred = ['B', 'B', 'B']
        score = macro_f1(y_true, y_pred)
        assert score == 0.0
    
    def test_mixed_predictions(self):
        """Test macro F1 with mixed correct and incorrect predictions."""
        y_true = ['A', 'B', 'A', 'C', 'B']
        y_pred = ['A', 'B', 'C', 'C', 'A']
        score = macro_f1(y_true, y_pred)
        # Score should be between 0 and 1
        assert 0.0 <= score <= 1.0
    
    def test_single_class_same(self):
        """Test single class scenario where predictions match."""
        y_true = ['A', 'A', 'A']
        y_pred = ['A', 'A', 'A']
        score = macro_f1(y_true, y_pred)
        assert score == 1.0
    
    def test_single_class_different(self):
        """Test single class scenario where predictions don't match."""
        y_true = ['A', 'A', 'A']
        y_pred = ['B', 'B', 'B']
        score = macro_f1(y_true, y_pred)
        assert score == 0.0
    
    def test_empty_inputs(self):
        """Test that empty inputs return 0.0."""
        score = macro_f1([], [])
        assert score == 0.0
    
    def test_length_mismatch(self):
        """Test that mismatched lengths raise ValueError."""
        y_true = ['A', 'B', 'C']
        y_pred = ['A', 'B']
        with pytest.raises(ValueError, match="Length mismatch"):
            macro_f1(y_true, y_pred)
    
    def test_score_bounds(self):
        """Test that F1 scores are always bounded between 0.0 and 1.0."""
        # Test various scenarios to ensure bounds
        test_cases = [
            (['A', 'B', 'C'], ['A', 'B', 'C']),  # Perfect
            (['A', 'B', 'C'], ['C', 'A', 'B']),  # Partial
            (['A', 'A', 'A'], ['B', 'C', 'D']),  # Poor
        ]
        
        for y_true, y_pred in test_cases:
            score = macro_f1(y_true, y_pred)
            assert 0.0 <= score <= 1.0, f"Score {score} not in bounds for {y_true}, {y_pred}"


class TestEvaluatePredictions:
    """Test cases for evaluate_predictions function."""
    
    def test_valid_dataframe(self):
        """Test evaluation with valid DataFrame."""
        df = pd.DataFrame({
            'target': ['A', 'B', 'A', 'C', 'B'],
            'prediction': ['A', 'B', 'C', 'C', 'A']
        })
        score = evaluate_predictions(df, 'target', 'prediction')
        assert 0.0 <= score <= 1.0
    
    def test_empty_dataframe(self):
        """Test that empty DataFrame returns 0.0."""
        df = pd.DataFrame()
        score = evaluate_predictions(df, 'target', 'prediction')
        assert score == 0.0
    
    def test_missing_target_column(self):
        """Test that missing target column raises KeyError."""
        df = pd.DataFrame({
            'prediction': ['A', 'B', 'C']
        })
        with pytest.raises(KeyError, match="Target column 'target' not found"):
            evaluate_predictions(df, 'target', 'prediction')
    
    def test_missing_prediction_column(self):
        """Test that missing prediction column raises KeyError."""
        df = pd.DataFrame({
            'target': ['A', 'B', 'C']
        })
        with pytest.raises(KeyError, match="Prediction column 'prediction' not found"):
            evaluate_predictions(df, 'target', 'prediction')
    
    def test_nan_handling(self):
        """Test that NaN values are handled properly."""
        df = pd.DataFrame({
            'target': ['A', 'B', None, 'C'],
            'prediction': ['A', None, 'C', 'C']
        })
        score = evaluate_predictions(df, 'target', 'prediction')
        # Should only evaluate non-NaN pairs
        assert 0.0 <= score <= 1.0
    
    def test_all_nan_data(self):
        """Test that all NaN data returns 0.0."""
        df = pd.DataFrame({
            'target': [None, None, None],
            'prediction': [None, None, None]
        })
        score = evaluate_predictions(df, 'target', 'prediction')
        assert score == 0.0


if __name__ == "__main__":
    # Simple test runner for manual verification
    print("Running basic macro_f1 tests...")
    
    # Test perfect predictions
    score = macro_f1(['A', 'B', 'C'], ['A', 'B', 'C'])
    print(f"Perfect predictions F1: {score} (expected: 1.0)")
    
    # Test wrong predictions
    score = macro_f1(['A', 'A', 'A'], ['B', 'B', 'B'])
    print(f"Wrong predictions F1: {score} (expected: 0.0)")
    
    # Test mixed predictions
    score = macro_f1(['A', 'B', 'A', 'C', 'B'], ['A', 'B', 'C', 'C', 'A'])
    print(f"Mixed predictions F1: {score} (expected: 0.0 <= score <= 1.0)")
    
    print("Basic tests completed!")