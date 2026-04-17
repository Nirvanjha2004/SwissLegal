from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .validation import (
    validate_and_default_positive_int,
    validate_and_default_string,
    validate_and_default_iterable,
    log_parameter_correction
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    index: int
    score: float
    text: str
    
    def __post_init__(self):
        if self.index < 0:
            raise ValueError(
                f"RetrievalResult validation failed: index must be non-negative. "
                f"Got: {self.index} (type: {type(self.index).__name__}). "
                f"Expected: integer >= 0 representing document position in corpus. "
                f"Result details: score={self.score}, text_length={len(self.text)}. "
                f"Suggestion: Ensure index corresponds to a valid document position in the corpus."
            )
        if not math.isfinite(self.score):
            score_type = "NaN" if math.isnan(self.score) else "infinite" if math.isinf(self.score) else "invalid"
            raise ValueError(
                f"RetrievalResult validation failed: score must be finite. "
                f"Got: {self.score} (type: {type(self.score).__name__}, classification: {score_type}). "
                f"Expected: finite float representing relevance score. "
                f"Result details: index={self.index}, text_length={len(self.text)}. "
                f"Suggestion: Check retrieval algorithm for numerical stability issues."
            )
        if not isinstance(self.text, str):
            raise ValueError(
                f"RetrievalResult validation failed: text must be a string. "
                f"Got: {type(self.text).__name__} with value '{self.text}'. "
                f"Expected: string containing document content. "
                f"Result details: index={self.index}, score={self.score}. "
                f"Suggestion: Ensure document content is properly converted to string format."
            )


class BM25Retriever:
    def __init__(self, documents: Sequence[str]):
        """
        Initialize BM25 retriever with comprehensive parameter validation and error handling.
        
        Args:
            documents: Sequence of document strings
        """
        # Validate documents parameter
        original_documents = documents
        validated_documents = validate_and_default_iterable(
            documents, [], "documents", "BM25Retriever", allow_empty=True
        )
        if validated_documents != list(original_documents):
            log_parameter_correction(original_documents, validated_documents, "documents", "BM25Retriever", "invalid document sequence")
        
        # Convert to list and validate each document
        self.documents = []
        invalid_docs = 0
        total_chars = 0
        
        for i, doc in enumerate(validated_documents):
            validated_doc = validate_and_default_string(
                doc, f"empty_document_{i}", f"document[{i}]", "BM25Retriever", allow_empty=True
            )
            if validated_doc != doc:
                log_parameter_correction(doc, validated_doc, f"document[{i}]", "BM25Retriever", "invalid document string")
                invalid_docs += 1
            
            self.documents.append(validated_doc)
            total_chars += len(validated_doc)
        
        # Handle empty corpus scenario
        if not self.documents:
            logger.warning(
                f"BM25Retriever.__init__: Initialized with empty document corpus. "
                f"Input documents: {len(validated_documents)}, "
                f"Valid documents: 0. "
                f"Search operations will return empty results. "
                f"Suggestion: Provide a non-empty sequence of document strings for indexing."
            )
            self.tokenized_documents = []
            self.model = None
        else:
            try:
                # Tokenize documents and build BM25 index
                self.tokenized_documents = []
                empty_docs = 0
                
                for i, document in enumerate(self.documents):
                    tokens = self._tokenize(document)
                    if not tokens:
                        empty_docs += 1
                        logger.debug(f"BM25Retriever.__init__: Document {i} produced no tokens after tokenization")
                    self.tokenized_documents.append(tokens)
                
                self.model = BM25Okapi(self.tokenized_documents)
                
                logger.info(
                    f"BM25Retriever.__init__: Successfully initialized BM25 index. "
                    f"Total documents: {len(self.documents)}, "
                    f"Invalid documents corrected: {invalid_docs}, "
                    f"Empty documents after tokenization: {empty_docs}, "
                    f"Total characters: {total_chars}, "
                    f"Average document length: {total_chars / len(self.documents):.1f} chars, "
                    f"Index ready for search operations."
                )
                
            except Exception as e:
                logger.error(
                    f"BM25Retriever.__init__: Failed to build BM25 index. "
                    f"Documents count: {len(self.documents)}, "
                    f"Error type: {type(e).__name__}, "
                    f"Error message: {str(e)}. "
                    f"Falling back to empty index. "
                    f"Suggestion: Check document content and BM25 library compatibility."
                )
                self.tokenized_documents = []
                self.model = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """
        Search documents with comprehensive parameter validation and error handling.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            
        Returns:
            List of RetrievalResult objects
        """
        # Validate query parameter
        original_query = query
        validated_query = validate_and_default_string(
            query, "", "query", "BM25Retriever.search", allow_empty=True
        )
        if validated_query != original_query:
            log_parameter_correction(original_query, validated_query, "query", "BM25Retriever.search", "invalid query string")
        
        # Validate top_k parameter
        original_top_k = top_k
        validated_top_k = validate_and_default_positive_int(
            top_k, 10, "top_k", "BM25Retriever.search"
        )
        if validated_top_k != original_top_k:
            log_parameter_correction(original_top_k, validated_top_k, "top_k", "BM25Retriever.search", "invalid top_k")
        
        # Handle empty corpus scenario
        if not self.model or not self.documents:
            logger.warning(
                f"BM25Retriever.search: Search attempted on empty corpus. "
                f"Query: '{validated_query}' (length: {len(validated_query)}), "
                f"Requested results: {validated_top_k}, "
                f"Available documents: {len(self.documents) if self.documents else 0}. "
                f"Returning empty results. "
                f"Suggestion: Initialize retriever with non-empty document corpus before searching."
            )
            return []
        
        # Handle empty query
        if not validated_query.strip():
            logger.warning(
                f"BM25Retriever.search: Empty query provided. "
                f"Query: '{validated_query}' (length: {len(validated_query)}), "
                f"Available documents: {len(self.documents)}, "
                f"Requested results: {validated_top_k}. "
                f"Returning empty results. "
                f"Suggestion: Provide a meaningful search query with actual content."
            )
            return []
        
        try:
            # Tokenize query and get scores
            query_tokens = self._tokenize(validated_query)
            if not query_tokens:
                logger.warning(
                    f"BM25Retriever.search: Query produced no tokens after tokenization. "
                    f"Original query: '{validated_query}', "
                    f"Tokens: {query_tokens}. "
                    f"Returning empty results. "
                    f"Suggestion: Use query with meaningful words that can be tokenized."
                )
                return []
            
            scores = self.model.get_scores(query_tokens)
            
            if len(scores) != len(self.documents):
                logger.error(
                    f"BM25Retriever.search: Score count mismatch. "
                    f"Documents: {len(self.documents)}, "
                    f"Scores: {len(scores)}. "
                    f"Query: '{validated_query}', "
                    f"Tokens: {query_tokens}. "
                    f"This indicates an indexing inconsistency. "
                    f"Returning empty results."
                )
                return []
            
            # Rank documents by score
            ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:validated_top_k]
            
            results = []
            for rank, index in enumerate(ranked_indices):
                try:
                    result = RetrievalResult(
                        index=index, 
                        score=float(scores[index]), 
                        text=self.documents[index]
                    )
                    results.append(result)
                except (ValueError, IndexError) as e:
                    logger.error(
                        f"BM25Retriever.search: Failed to create RetrievalResult for rank {rank}. "
                        f"Index: {index}, "
                        f"Score: {scores[index] if index < len(scores) else 'out_of_bounds'}, "
                        f"Document available: {index < len(self.documents)}, "
                        f"Error: {type(e).__name__}: {str(e)}. "
                        f"Skipping this result."
                    )
                    continue
            
            logger.debug(
                f"BM25Retriever.search: Search completed successfully. "
                f"Query: '{validated_query}' ({len(query_tokens)} tokens), "
                f"Results returned: {len(results)}/{validated_top_k} requested, "
                f"Top score: {results[0].score if results else 'N/A'}, "
                f"Score range: {min(r.score for r in results) if results else 'N/A'} - {max(r.score for r in results) if results else 'N/A'}"
            )
            
            return results
            
        except Exception as e:
            logger.error(
                f"BM25Retriever.search: Unexpected error during search. "
                f"Query: '{validated_query}', "
                f"Top-k: {validated_top_k}, "
                f"Documents: {len(self.documents)}, "
                f"Error type: {type(e).__name__}, "
                f"Error message: {str(e)}. "
                f"Returning empty results. "
                f"Suggestion: Check BM25 model state and query format."
            )
            return []


class TfidfRetriever:
    def __init__(self, documents: Sequence[str]):
        """
        Initialize TF-IDF retriever with parameter validation and default substitution.
        
        Args:
            documents: Sequence of document strings
        """
        # Validate documents parameter
        original_documents = documents
        validated_documents = validate_and_default_iterable(
            documents, [], "documents", "TfidfRetriever", allow_empty=True
        )
        if validated_documents != list(original_documents):
            log_parameter_correction(original_documents, validated_documents, "documents", "TfidfRetriever", "invalid document sequence")
        
        # Convert to list and validate each document
        self.documents = []
        for i, doc in enumerate(validated_documents):
            validated_doc = validate_and_default_string(
                doc, f"empty_document_{i}", f"document[{i}]", "TfidfRetriever", allow_empty=True
            )
            if validated_doc != doc:
                log_parameter_correction(doc, validated_doc, f"document[{i}]", "TfidfRetriever", "invalid document string")
            self.documents.append(validated_doc)
        
        # Handle empty corpus scenario
        if not self.documents:
            logger.warning("TfidfRetriever initialized with empty document corpus. Search operations will return empty results.")
            self.vectorizer = TfidfVectorizer(tokenizer=self._tokenize, lowercase=False)
            self.matrix = None
        else:
            # Use consistent tokenization with BM25Retriever
            self.vectorizer = TfidfVectorizer(tokenizer=self._tokenize, lowercase=False)
            self.matrix = self.vectorizer.fit_transform(self.documents)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Consistent tokenization with BM25Retriever"""
        return text.lower().split()

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """
        Search documents with parameter validation and default substitution.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            
        Returns:
            List of RetrievalResult objects
        """
        # Validate query parameter
        original_query = query
        validated_query = validate_and_default_string(
            query, "", "query", "TfidfRetriever.search", allow_empty=True
        )
        if validated_query != original_query:
            log_parameter_correction(original_query, validated_query, "query", "TfidfRetriever.search", "invalid query string")
        
        # Validate top_k parameter
        original_top_k = top_k
        validated_top_k = validate_and_default_positive_int(
            top_k, 10, "top_k", "TfidfRetriever.search"
        )
        if validated_top_k != original_top_k:
            log_parameter_correction(original_top_k, validated_top_k, "top_k", "TfidfRetriever.search", "invalid top_k")
        
        # Handle empty corpus scenario
        if self.matrix is None or not self.documents:
            logger.warning(f"TF-IDF search attempted on empty corpus for query: '{validated_query}'. Returning empty results.")
            return []
        
        # Handle empty query
        if not validated_query.strip():
            logger.warning("TF-IDF search attempted with empty query. Returning empty results.")
            return []
            
        query_vector = self.vectorizer.transform([validated_query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:validated_top_k]
        return [RetrievalResult(index=index, score=float(scores[index]), text=self.documents[index]) for index in ranked_indices]
