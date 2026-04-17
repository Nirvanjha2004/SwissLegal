"""
Unit tests for DataLoader functionality.
Tests Requirements 1.1, 1.2, 1.3, 1.4, 1.5
"""

import tempfile
import os
import pytest
import pandas as pd
from pathlib import Path

from src.data_loader import (
    load_dataset, read_csv_file, normalize_text_columns, clean_text,
    capture_dataframe_structure, validate_structure_preservation, validate_dataframe_integrity
)


class TestCleanText:
    """Test clean_text function"""
    
    def test_clean_text_whitespace_normalization(self):
        """Test that clean_text normalizes whitespace correctly"""
        assert clean_text("  hello   world  ") == "hello world"
        assert clean_text("\t\nhello\n\tworld\t\n") == "hello world"
        
    def test_clean_text_none_handling(self):
        """Test that clean_text handles None values"""
        assert clean_text(None) == ""
        
    def test_clean_text_type_conversion(self):
        """Test that clean_text converts non-string types"""
        assert clean_text(123) == "123"
        assert clean_text(45.67) == "45.67"
        
    def test_clean_text_empty_string(self):
        """Test that clean_text handles empty strings"""
        assert clean_text("") == ""
        assert clean_text("   ") == ""


class TestReadCsvFile:
    """Test read_csv_file function"""
    
    def test_read_existing_file(self):
        """Test reading an existing CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\nvalue1,value2\n")
            temp_path = f.name
        
        try:
            df = read_csv_file(temp_path)
            assert not df.empty
            assert len(df) == 1
            assert list(df.columns) == ['col1', 'col2']
            assert df.iloc[0]['col1'] == 'value1'
        finally:
            os.unlink(temp_path)
    
    def test_read_missing_file(self):
        """Test reading a non-existent file returns empty DataFrame"""
        df = read_csv_file('nonexistent_file.csv')
        assert df.empty
        assert isinstance(df, pd.DataFrame)
    
    def test_read_invalid_csv(self):
        """Test reading an invalid CSV file returns empty DataFrame"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("invalid,csv,content\nwith,mismatched\n")
            temp_path = f.name
        
        try:
            df = read_csv_file(temp_path)
            # Should handle gracefully and return empty DataFrame on error
            assert isinstance(df, pd.DataFrame)
        finally:
            os.unlink(temp_path)


class TestNormalizeTextColumns:
    """Test normalize_text_columns function"""
    
    def test_normalize_existing_columns(self):
        """Test normalizing existing text columns"""
        df = pd.DataFrame({
            'text': ['  hello   world  ', '  test  '],
            'other': [1, 2]
        })
        
        normalized = normalize_text_columns(df, ['text'])
        assert normalized['text'].iloc[0] == 'hello world'
        assert normalized['text'].iloc[1] == 'test'
        assert normalized['other'].iloc[0] == 1  # Other columns unchanged
    
    def test_normalize_missing_columns(self):
        """Test normalizing non-existent columns"""
        df = pd.DataFrame({'col1': ['value1'], 'col2': ['value2']})
        
        # Should handle missing columns gracefully
        normalized = normalize_text_columns(df, ['nonexistent'])
        pd.testing.assert_frame_equal(normalized, df)
    
    def test_normalize_empty_dataframe(self):
        """Test normalizing empty DataFrame"""
        df = pd.DataFrame()
        normalized = normalize_text_columns(df, ['text'])
        assert normalized.empty
        assert isinstance(normalized, pd.DataFrame)
    
    def test_normalize_multiple_columns(self):
        """Test normalizing multiple text columns"""
        df = pd.DataFrame({
            'text1': ['  hello  ', '  world  '],
            'text2': ['  foo  ', '  bar  '],
            'number': [1, 2]
        })
        
        normalized = normalize_text_columns(df, ['text1', 'text2'])
        assert normalized['text1'].iloc[0] == 'hello'
        assert normalized['text2'].iloc[0] == 'foo'
        assert normalized['number'].iloc[0] == 1


class TestLoadDataset:
    """Test load_dataset function"""
    
    def test_load_dataset_without_normalization(self):
        """Test loading dataset without text normalization"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("text,number\n  hello world  ,1\n")
            temp_path = f.name
        
        try:
            df = load_dataset(temp_path)
            assert not df.empty
            assert df['text'].iloc[0] == '  hello world  '  # Not normalized
        finally:
            os.unlink(temp_path)
    
    def test_load_dataset_with_normalization(self):
        """Test loading dataset with text normalization"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("text,number\n  hello world  ,1\n")
            temp_path = f.name
        
        try:
            df = load_dataset(temp_path, text_columns=['text'])
            assert not df.empty
            assert df['text'].iloc[0] == 'hello world'  # Normalized
        finally:
            os.unlink(temp_path)
    
    def test_load_missing_file(self):
        """Test loading non-existent file"""
        df = load_dataset('nonexistent.csv', text_columns=['text'])
        assert df.empty
    
    def test_load_empty_file_with_normalization(self):
        """Test loading empty file with text normalization"""
        df = load_dataset('nonexistent.csv', text_columns=['text'])
        assert df.empty
        # Should not attempt normalization on empty DataFrame


class TestDataStructurePreservation:
    """Test that data structure is preserved through processing"""
    
    def test_row_count_preservation(self):
        """Test that row count is preserved"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("text,id\n  hello  ,1\n  world  ,2\n  test  ,3\n")
            temp_path = f.name
        
        try:
            df = load_dataset(temp_path, text_columns=['text'])
            assert len(df) == 3  # All rows preserved
            assert list(df['id']) == [1, 2, 3]  # IDs preserved
        finally:
            os.unlink(temp_path)
    
    def test_column_structure_preservation(self):
        """Test that column structure is preserved"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("text,number,category\n  hello  ,1,A\n  world  ,2,B\n")
            temp_path = f.name
        
        try:
            df = load_dataset(temp_path, text_columns=['text'])
            expected_columns = ['text', 'number', 'category']
            assert list(df.columns) == expected_columns
            assert df['number'].dtype == 'int64'  # Type preserved
        finally:
            os.unlink(temp_path)


class TestCaptureDataFrameStructure:
    """Test capture_dataframe_structure function"""
    
    def test_capture_basic_structure(self):
        """Test capturing basic DataFrame structure"""
        df = pd.DataFrame({
            'text': ['hello', 'world'],
            'number': [1, 2],
            'category': ['A', 'B']
        })
        
        structure = capture_dataframe_structure(df)
        
        assert structure['shape'] == (2, 3)
        assert structure['columns'] == ['text', 'number', 'category']
        assert structure['row_count'] == 2
        assert structure['column_count'] == 3
        assert structure['index'] == [0, 1]
    
    def test_capture_empty_dataframe(self):
        """Test capturing structure of empty DataFrame"""
        df = pd.DataFrame()
        structure = capture_dataframe_structure(df)
        
        assert structure['shape'] == (0, 0)
        assert structure['columns'] == []
        assert structure['row_count'] == 0
        assert structure['column_count'] == 0
    
    def test_capture_large_dataframe_index(self):
        """Test that large DataFrames store index count instead of full index"""
        # Create DataFrame with more than 1000 rows
        df = pd.DataFrame({'col': range(1500)})
        structure = capture_dataframe_structure(df)
        
        assert structure['row_count'] == 1500
        assert structure['index'] == 1500  # Should store count, not full list


class TestValidateStructurePreservation:
    """Test validate_structure_preservation function"""
    
    def test_valid_structure_preservation(self):
        """Test that valid structure preservation passes validation"""
        df = pd.DataFrame({'text': ['hello', 'world'], 'id': [1, 2]})
        original_structure = capture_dataframe_structure(df)
        
        # Process DataFrame (normalize text)
        processed_df = df.copy()
        processed_df['text'] = processed_df['text'].str.upper()
        
        # Should not raise any exception
        validate_structure_preservation(original_structure, processed_df, "text processing")
    
    def test_row_count_change_detection(self):
        """Test that row count changes are detected"""
        df = pd.DataFrame({'text': ['hello', 'world'], 'id': [1, 2]})
        original_structure = capture_dataframe_structure(df)
        
        # Create processed DataFrame with different row count
        processed_df = df.iloc[:1].copy()  # Remove one row
        
        with pytest.raises(ValueError, match="Row count changed during"):
            validate_structure_preservation(original_structure, processed_df, "row removal")
    
    def test_column_structure_change_detection(self):
        """Test that column structure changes are detected"""
        df = pd.DataFrame({'text': ['hello', 'world'], 'id': [1, 2]})
        original_structure = capture_dataframe_structure(df)
        
        # Create processed DataFrame with different columns
        processed_df = df.copy()
        processed_df['new_col'] = ['a', 'b']
        
        with pytest.raises(ValueError, match="Column structure changed during"):
            validate_structure_preservation(original_structure, processed_df, "column addition")
    
    def test_column_count_change_detection(self):
        """Test that column count changes are detected"""
        df = pd.DataFrame({'text': ['hello', 'world'], 'id': [1, 2]})
        original_structure = capture_dataframe_structure(df)
        
        # Create processed DataFrame with fewer columns
        processed_df = df[['text']].copy()
        
        with pytest.raises(ValueError, match="Column count changed during"):
            validate_structure_preservation(original_structure, processed_df, "column removal")
    
    def test_index_preservation_validation(self):
        """Test that index changes are detected for small DataFrames"""
        df = pd.DataFrame({'text': ['hello', 'world'], 'id': [1, 2]})
        original_structure = capture_dataframe_structure(df)
        
        # Create processed DataFrame with different index
        processed_df = df.copy()
        processed_df.index = [10, 20]
        
        with pytest.raises(ValueError, match="Row index changed during"):
            validate_structure_preservation(original_structure, processed_df, "index modification")


class TestValidateDataFrameIntegrity:
    """Test validate_dataframe_integrity function"""
    
    def test_valid_dataframe_passes(self):
        """Test that valid DataFrame passes integrity check"""
        df = pd.DataFrame({'text': ['hello', 'world'], 'id': [1, 2]})
        
        assert validate_dataframe_integrity(df) is True
        assert validate_dataframe_integrity(df, expected_columns=['text', 'id']) is True
        assert validate_dataframe_integrity(df, min_rows=2) is True
    
    def test_none_dataframe_fails(self):
        """Test that None DataFrame fails validation"""
        with pytest.raises(ValueError, match="DataFrame cannot be None"):
            validate_dataframe_integrity(None)
    
    def test_wrong_type_fails(self):
        """Test that non-DataFrame objects fail validation"""
        with pytest.raises(ValueError, match="Expected pandas DataFrame"):
            validate_dataframe_integrity([1, 2, 3])
    
    def test_insufficient_rows_fails(self):
        """Test that DataFrames with insufficient rows fail validation"""
        df = pd.DataFrame({'text': ['hello']})
        
        with pytest.raises(ValueError, match="DataFrame has 1 rows, expected at least 3"):
            validate_dataframe_integrity(df, min_rows=3)
    
    def test_missing_columns_fails(self):
        """Test that DataFrames with missing expected columns fail validation"""
        df = pd.DataFrame({'text': ['hello', 'world']})
        
        with pytest.raises(ValueError, match="Missing expected columns"):
            validate_dataframe_integrity(df, expected_columns=['text', 'id', 'category'])
    
    def test_duplicate_columns_fails(self):
        """Test that DataFrames with duplicate columns fail validation"""
        # Create DataFrame with duplicate columns (this is possible in pandas)
        df = pd.DataFrame([[1, 2, 3]], columns=['A', 'B', 'A'])
        
        with pytest.raises(ValueError, match="Duplicate columns found"):
            validate_dataframe_integrity(df)
    
    def test_empty_dataframe_passes_basic_validation(self):
        """Test that empty DataFrame passes basic validation"""
        df = pd.DataFrame()
        
        assert validate_dataframe_integrity(df) is True
        assert validate_dataframe_integrity(df, min_rows=0) is True


class TestIntegratedDataStructurePreservation:
    """Test data structure preservation in integrated workflows"""
    
    def test_load_dataset_preserves_structure_with_normalization(self):
        """Test that load_dataset preserves structure during text normalization"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("text,id,category\n  hello world  ,1,A\n  test data  ,2,B\n  sample text  ,3,C\n")
            temp_path = f.name
        
        try:
            df = load_dataset(temp_path, text_columns=['text'])
            
            # Verify structure preservation
            assert len(df) == 3  # Row count preserved
            assert list(df.columns) == ['text', 'id', 'category']  # Column structure preserved
            assert list(df['id']) == [1, 2, 3]  # Row relationships preserved
            assert list(df['category']) == ['A', 'B', 'C']  # Non-text columns preserved
            
            # Verify text normalization occurred
            assert df['text'].iloc[0] == 'hello world'  # Whitespace normalized
            assert df['text'].iloc[1] == 'test data'
            assert df['text'].iloc[2] == 'sample text'
        finally:
            os.unlink(temp_path)
    
    def test_normalize_text_columns_preserves_relationships(self):
        """Test that normalize_text_columns preserves row relationships"""
        df = pd.DataFrame({
            'text1': ['  hello  ', '  world  ', '  test  '],
            'text2': ['  foo  ', '  bar  ', '  baz  '],
            'id': [1, 2, 3],
            'value': [10.5, 20.3, 30.1]
        })
        
        normalized = normalize_text_columns(df, ['text1', 'text2'])
        
        # Verify structure preservation
        assert len(normalized) == 3
        assert list(normalized.columns) == ['text1', 'text2', 'id', 'value']
        
        # Verify row relationships preserved
        for i in range(3):
            assert normalized.iloc[i]['id'] == df.iloc[i]['id']
            assert normalized.iloc[i]['value'] == df.iloc[i]['value']
        
        # Verify normalization occurred
        assert normalized['text1'].iloc[0] == 'hello'
        assert normalized['text2'].iloc[0] == 'foo'


if __name__ == "__main__":
    pytest.main([__file__])