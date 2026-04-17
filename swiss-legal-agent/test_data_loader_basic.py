#!/usr/bin/env python3
"""
Basic test for DataLoader functionality without external dependencies.
This test validates the core logic and error handling.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_without_pandas():
    """Test DataLoader logic without pandas dependency"""
    print("Testing DataLoader implementation...")
    
    # Test 1: Check if file exists
    try:
        from src.data_loader import read_csv_file
        print("✓ read_csv_file function imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test 2: Check clean_text function
    try:
        from src.data_loader import clean_text
        
        # Test clean_text with various inputs
        assert clean_text("  hello   world  ") == "hello world"
        assert clean_text(None) == ""
        assert clean_text(123) == "123"
        assert clean_text("") == ""
        print("✓ clean_text function works correctly")
    except Exception as e:
        print(f"✗ clean_text test failed: {e}")
        return False
    
    print("✓ All basic tests passed!")
    return True

if __name__ == "__main__":
    success = test_without_pandas()
    sys.exit(0 if success else 1)