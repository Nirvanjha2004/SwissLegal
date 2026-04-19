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

        if self.model is None:
            return context_list[:top_k] if top_k is not None else context_list

        effective_top_k = len(context_list) if top_k is None else min(top_k, len(context_list))
        if effective_top_k <= 0:
            return []

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
            return [context_list[index] for index in ranked_indices.tolist()]
        except Exception as error:
            logger.warning(
                "LocalSemanticReranker: rerank failed (%s: %s). Returning BM25 order.",
                type(error).__name__,
                error,
            )
            return context_list[:effective_top_k]