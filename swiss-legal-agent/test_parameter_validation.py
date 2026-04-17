#!/usr/bin/env python3
"""
Simple test script to verify parameter validation and default substitution.
This script demonstrates the parameter validation functionality across all components.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import load_dataset, normalize_text_columns
from src.chunker import chunk_text, chunk_records, TextChunk
from src.retriever import BM25Retriever, TfidfRetriever
from src.agent import AgentConfig, build_prompt, run_agent
from src.evaluator import macro_f1, evaluate_predictions
from src.config import SystemConfig
import pandas as pd

# Configure logging to see validation messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

def test_data_loader_validation():
    """Test DataLoader parameter validation."""
    print("\n=== Testing DataLoader Parameter Validation ===")
    
    # Test invalid path (should use default and log correction)
    df = load_dataset(None, text_columns=["text"])
    print(f"DataLoader with None path returned DataFrame with shape: {df.shape}")
    
    # Test invalid text_columns (should use default and log correction)
    df = load_dataset("nonexistent.csv", text_columns="invalid_string")
    print(f"DataLoader with invalid text_columns returned DataFrame with shape: {df.shape}")

def test_chunker_validation():
    """Test TextChunker parameter validation."""
    print("\n=== Testing TextChunker Parameter Validation ===")
    
    # Test invalid chunk_size (should use default and log correction)
    chunks = chunk_text("This is a test text.", chunk_size=-100, overlap=50)
    print(f"Chunker with invalid chunk_size returned {len(chunks)} chunks")
    
    # Test invalid overlap >= chunk_size (should correct and log)
    chunks = chunk_text("This is a test text.", chunk_size=100, overlap=150)
    print(f"Chunker with invalid overlap returned {len(chunks)} chunks")
    
    # Test invalid records (should handle gracefully)
    chunks = chunk_records("invalid_records", chunk_size=100, overlap=50)
    print(f"Chunker with invalid records returned {len(chunks)} chunks")

def test_retriever_validation():
    """Test Retriever parameter validation."""
    print("\n=== Testing Retriever Parameter Validation ===")
    
    # Test BM25 with invalid documents (should use default and log correction)
    retriever = BM25Retriever(None)
    results = retriever.search("test query", top_k=-5)
    print(f"BM25Retriever with invalid parameters returned {len(results)} results")
    
    # Test TF-IDF with invalid documents
    retriever = TfidfRetriever("invalid_documents")
    results = retriever.search(None, top_k="invalid_top_k")
    print(f"TfidfRetriever with invalid parameters returned {len(results)} results")

def test_agent_validation():
    """Test Agent parameter validation."""
    print("\n=== Testing Agent Parameter Validation ===")
    
    # Test AgentConfig with invalid parameters (should correct and log)
    config = AgentConfig(temperature=-1.0, max_context_chunks=-5)
    print(f"AgentConfig with invalid parameters: temperature={config.temperature}, max_context_chunks={config.max_context_chunks}")
    
    # Test build_prompt with invalid parameters
    prompt = build_prompt(None, "invalid_contexts")
    print(f"build_prompt with invalid parameters returned prompt length: {len(prompt)}")
    
    # Test run_agent with invalid parameters (mock LLM)
    class MockLLM:
        def invoke(self, prompt, **kwargs):
            return "Mock response"
    
    answer = run_agent(None, None, MockLLM(), config=None)
    print(f"run_agent with invalid parameters returned: '{answer[:50]}...'")

def test_evaluator_validation():
    """Test Evaluator parameter validation."""
    print("\n=== Testing Evaluator Parameter Validation ===")
    
    # Test macro_f1 with invalid parameters
    score = macro_f1(None, "invalid_pred")
    print(f"macro_f1 with invalid parameters returned score: {score}")
    
    # Test evaluate_predictions with invalid DataFrame
    df = pd.DataFrame({"target": [1, 2, 3], "prediction": [1, 2, 2]})
    score = evaluate_predictions(df, None, "invalid_column")
    print(f"evaluate_predictions with invalid parameters returned score: {score}")

def test_config_validation():
    """Test SystemConfig parameter validation."""
    print("\n=== Testing SystemConfig Parameter Validation ===")
    
    # Test SystemConfig with invalid parameters (should correct and log)
    config = SystemConfig(
        bm25_top_k=-10,
        chunk_size="invalid",
        chunk_overlap=5000,  # > chunk_size
        model_name=None
    )
    print(f"SystemConfig with invalid parameters: bm25_top_k={config.bm25_top_k}, "
          f"chunk_size={config.chunk_size}, chunk_overlap={config.chunk_overlap}, "
          f"model_name='{config.model_name}'")

def main():
    """Run all parameter validation tests."""
    print("Starting Parameter Validation Tests")
    print("=" * 50)
    
    try:
        test_data_loader_validation()
        test_chunker_validation()
        test_retriever_validation()
        test_agent_validation()
        test_evaluator_validation()
        test_config_validation()
        
        print("\n" + "=" * 50)
        print("All parameter validation tests completed successfully!")
        print("Check the log messages above to see parameter corrections in action.")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()