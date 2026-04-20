from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class LocalSemanticReranker:
    """Offline semantic reranker backed by a local SentenceTransformer snapshot."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.model: SentenceTransformer | None = None

        if not self.model_path.exists():
            logger.warning(
                "LocalSemanticReranker: model path does not exist, skipping semantic reranking. Path: %s",
                self.model_path,
            )
            return

        try:
            self.model = SentenceTransformer(str(self.model_path))
            logger.info("LocalSemanticReranker: loaded offline model from %s", self.model_path)
        except Exception as error:
            logger.warning(
                "LocalSemanticReranker: failed to load model from %s (%s: %s). Falling back to BM25 only.",
                self.model_path,
                type(error).__name__,
                error,
            )
            self.model = None

    def rerank(self, query: str, contexts: Iterable[str], top_k: int | None = None) -> list[str]:
        context_list = [context for context in contexts if str(context).strip()]
        if not context_list:
            return []

        ranked_positions = self.rerank_indices(query, context_list, top_k=top_k)
        return [context_list[position] for position in ranked_positions]

    def rerank_indices(self, query: str, contexts: Iterable[str], top_k: int | None = None) -> list[int]:
        context_list = [context for context in contexts if str(context).strip()]
        if not context_list:
            return []

        effective_top_k = len(context_list) if top_k is None else min(top_k, len(context_list))
        if effective_top_k <= 0:
            return []

        if self.model is None:
            return list(range(effective_top_k))

        try:
            query_embedding = self.model.encode(
                [f"query: {query}"],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            context_embeddings = self.model.encode(
                [f"passage: {context}" for context in context_list],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            scores = np.dot(context_embeddings, query_embedding)
            ranked_indices = np.argsort(scores)[::-1][:effective_top_k]
            return ranked_indices.tolist()
        except Exception as error:
            logger.warning(
                "LocalSemanticReranker: rerank failed (%s: %s). Returning BM25 order.",
                type(error).__name__,
                error,
            )
            return list(range(effective_top_k))


class LocalSemanticRetriever:
    """Dense retriever backed by the same local SentenceTransformer model."""

    def __init__(self, reranker: LocalSemanticReranker, corpus_texts: Iterable[str], batch_size: int = 256):
        self.model = reranker.model
        self.corpus_embeddings: np.ndarray | None = None
        self.corpus_size = 0

        texts = [str(text).strip() for text in corpus_texts if str(text).strip()]
        self.corpus_size = len(texts)
        if self.model is None:
            logger.info("LocalSemanticRetriever: model unavailable, dense retrieval disabled.")
            return
        if not texts:
            logger.info("LocalSemanticRetriever: empty corpus, dense retrieval disabled.")
            return

        try:
            logger.info("LocalSemanticRetriever: encoding corpus for dense retrieval (size=%s)", len(texts))
            embeddings = self.model.encode(
                [f"passage: {text}" for text in texts],
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size,
            )
            self.corpus_embeddings = np.asarray(embeddings, dtype=np.float32)
            logger.info(
                "LocalSemanticRetriever: corpus embedding index ready (rows=%s dim=%s)",
                self.corpus_embeddings.shape[0],
                self.corpus_embeddings.shape[1] if self.corpus_embeddings.ndim == 2 else 0,
            )
        except Exception as error:
            logger.warning(
                "LocalSemanticRetriever: failed to build dense index (%s: %s).",
                type(error).__name__,
                error,
            )
            self.corpus_embeddings = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.corpus_embeddings is not None and self.corpus_embeddings.size > 0

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if not self.ready or top_k <= 0:
            return []

        try:
            query_embedding = self.model.encode(
                [f"query: {query}"],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            scores = np.dot(self.corpus_embeddings, np.asarray(query_embedding, dtype=np.float32))
            effective_top_k = min(top_k, len(scores))
            if effective_top_k <= 0:
                return []

            if effective_top_k >= len(scores):
                ranked_indices = np.argsort(scores)[::-1]
            else:
                candidate_indices = np.argpartition(scores, -effective_top_k)[-effective_top_k:]
                ranked_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

            return [(int(index), float(scores[index])) for index in ranked_indices.tolist()]
        except Exception as error:
            logger.warning(
                "LocalSemanticRetriever: search failed (%s: %s).",
                type(error).__name__,
                error,
            )
            return []