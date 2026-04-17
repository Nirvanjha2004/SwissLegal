"""
Test empty corpus handling for retrieval systems.
This test verifies that both BM25 and TF-IDF retrievers handle empty corpus scenarios gracefully.
"""

import logging
from unittest.mock import patch
import pytest

from src.retriever import BM25Retriever, TfidfRetriever


class TestEmptyCorpusHandling:
    """Test cases for empty corpus scenarios in retrieval systems."""
    
    def test_bm25_empty_corpus_initialization(self):
        """Test BM25Retriever initialization with empty corpus."""
        with patch('src.retriever.logger') as mock_logger:
            retriever = BM25Retriever([])
            
            # Verify empty corpus is detected
            assert retriever.documents == []
            assert retriever.tokenized_documents == []
            assert retriever.model is None
            
            # Verify warning is logged
            mock_logger.warning.assert_called_once_with(
                "BM25Retriever initialized with empty document corpus. Search operations will return empty results."
            )
    
    def test_bm25_empty_corpus_search(self):
        """Test BM25Retriever search with empty corpus."""
        with patch('src.retriever.logger') as mock_logger:
            retriever = BM25Retriever([])
            results = retriever.search("test query", top_k=5)
            
            # Verify empty results
            assert results == []
            
            # Verify search warning is logged
            mock_logger.warning.assert_called_with(
                "BM25 search attempted on empty corpus for query: 'test query'. Returning empty results."
            )
    
    def test_tfidf_empty_corpus_initialization(self):
        """Test TfidfRetriever initialization with empty corpus."""
        with patch('src.retriever.logger') as mock_logger:
            retriever = TfidfRetriever([])
            
            # Verify empty corpus is detected
            assert retriever.documents == []
            assert retriever.matrix is None
            
            # Verify warning is logged
            mock_logger.warning.assert_called_once_with(
                "TfidfRetriever initialized with empty document corpus. Search operations will return empty results."
            )
    
    def test_tfidf_empty_corpus_search(self):
        """Test TfidfRetriever search with empty corpus."""
        with patch('src.retriever.logger') as mock_logger:
            retriever = TfidfRetriever([])
            results = retriever.search("test query", top_k=5)
            
            # Verify empty results
            assert results == []
            
            # Verify search warning is logged
            mock_logger.warning.assert_called_with(
                "TF-IDF search attempted on empty corpus for query: 'test query'. Returning empty results."
            )
    
    def test_bm25_non_empty_corpus_no_warnings(self):
        """Test BM25Retriever with non-empty corpus doesn't log empty corpus warnings."""
        with patch('src.retriever.logger') as mock_logger:
            documents = ["This is a test document", "Another document for testing"]
            retriever = BM25Retriever(documents)
            
            # Verify no empty corpus warnings
            mock_logger.warning.assert_not_called()
            
            # Verify proper initialization
            assert len(retriever.documents) == 2
            assert retriever.model is not None
    
    def test_tfidf_non_empty_corpus_no_warnings(self):
        """Test TfidfRetriever with non-empty corpus doesn't log empty corpus warnings."""
        with patch('src.retriever.logger') as mock_logger:
            documents = ["This is a test document", "Another document for testing"]
            retriever = TfidfRetriever(documents)
            
            # Verify no empty corpus warnings
            mock_logger.warning.assert_not_called()
            
            # Verify proper initialization
            assert len(retriever.documents) == 2
            assert retriever.matrix is not None


if __name__ == "__main__":
    # Simple test runner for verification
    test_class = TestEmptyCorpusHandling()
    
    print("Testing BM25 empty corpus initialization...")
    test_class.test_bm25_empty_corpus_initialization()
    print("✓ Passed")
    
    print("Testing BM25 empty corpus search...")
    test_class.test_bm25_empty_corpus_search()
    print("✓ Passed")
    
    print("Testing TF-IDF empty corpus initialization...")
    test_class.test_tfidf_empty_corpus_initialization()
    print("✓ Passed")
    
    print("Testing TF-IDF empty corpus search...")
    test_class.test_tfidf_empty_corpus_search()
    print("✓ Passed")
    
    print("Testing BM25 non-empty corpus...")
    test_class.test_bm25_non_empty_corpus_no_warnings()
    print("✓ Passed")
    
    print("Testing TF-IDF non-empty corpus...")
    test_class.test_tfidf_non_empty_corpus_no_warnings()
    print("✓ Passed")
    
    print("\nAll empty corpus handling tests passed! ✓")