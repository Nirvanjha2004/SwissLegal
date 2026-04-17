from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RetrievalResult:
    index: int
    score: float
    text: str


class BM25Retriever:
    def __init__(self, documents: Sequence[str]):
        self.documents = list(documents)
        self.tokenized_documents = [self._tokenize(document) for document in self.documents]
        self.model = BM25Okapi(self.tokenized_documents) if self.documents else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if not self.model:
            return []
        scores = self.model.get_scores(self._tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]
        return [RetrievalResult(index=index, score=float(scores[index]), text=self.documents[index]) for index in ranked_indices]


class TfidfRetriever:
    def __init__(self, documents: Sequence[str]):
        self.documents = list(documents)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.documents) if self.documents else None

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if self.matrix is None:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]
        return [RetrievalResult(index=index, score=float(scores[index]), text=self.documents[index]) for index in ranked_indices]
