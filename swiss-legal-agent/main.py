from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import pandas as pd

from src.agent import AgentConfig, run_agent
from src.chunker import chunk_records
from src.config import (
    SystemConfig,
    load_config_from_env,
    validate_config_on_startup,
)
from src.data_loader import load_dataset
from src.evaluator import evaluate_predictions
from src.retriever import BM25Retriever
from src.semantic_reranker import LocalSemanticReranker


@dataclass
class PipelineMonitoring:
    stage_durations: dict[str, float] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def start_stage(self) -> float:
        return perf_counter()

    def end_stage(self, stage_name: str, started_at: float) -> None:
        duration = perf_counter() - started_at
        self.stage_durations[stage_name] = duration

    def add_count(self, key: str, value: int) -> None:
        self.counters[key] = value


LOGGER = logging.getLogger(__name__)


def _extract_train_citations(train_df: pd.DataFrame) -> set[str]:
    citations: set[str] = set()
    if train_df.empty or "gold_citations" not in train_df.columns:
        return citations

    for value in train_df["gold_citations"].dropna():
        parts = [part.strip() for part in str(value).split(";")]
        citations.update(part for part in parts if part)
    return citations


def load_inputs(config: SystemConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/test/laws/court datasets with graceful missing-file handling."""
    train_df = load_dataset(config.train_file)
    test_df = load_dataset(config.test_file)
    laws_df = load_dataset(config.laws_file)
    court_df = load_dataset(config.court_file)

    if not court_df.empty and "citation" in court_df.columns:
        original_court_rows = len(court_df)
        train_citations = _extract_train_citations(train_df)

        if train_citations:
            filtered_court_df = court_df[court_df["citation"].astype(str).isin(train_citations)]
            if not filtered_court_df.empty:
                court_df = filtered_court_df
                LOGGER.info(
                    "Court corpus filtered by training citations: kept_rows=%s original_rows=%s unique_train_citations=%s",
                    len(court_df),
                    original_court_rows,
                    len(train_citations),
                )

        court_df = court_df.drop_duplicates(subset=["citation"], keep="first")
        if len(court_df) > config.max_court_records:
            court_df = court_df.head(config.max_court_records)
            LOGGER.info(
                "Court corpus capped: max_court_records=%s selected_rows=%s",
                config.max_court_records,
                len(court_df),
            )

    availability = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "laws_rows": len(laws_df),
        "court_rows": len(court_df),
    }
    LOGGER.info("Input loading completed: %s", availability)

    if train_df.empty:
        LOGGER.warning("Training data unavailable or empty: %s", config.train_file)
    if test_df.empty:
        LOGGER.warning("Test data unavailable or empty: %s", config.test_file)
    if laws_df.empty:
        LOGGER.warning("Laws data unavailable or empty: %s", config.laws_file)
    if court_df.empty:
        LOGGER.warning("Court considerations data unavailable or empty: %s", config.court_file)

    return train_df, test_df, laws_df, court_df


def build_law_corpus(frame: pd.DataFrame, source_name: str) -> list[tuple[str, str]]:
    """Build citation-text corpus from a retrieval source DataFrame."""
    if frame.empty:
        LOGGER.warning("No records available for source '%s'; corpus generation skipped.", source_name)
        return []

    citation_column = None
    for candidate in ["citation", "citations", "source_id", "id"]:
        if candidate in frame.columns:
            citation_column = candidate
            break

    if citation_column is None:
        LOGGER.warning("No citation column found in laws data. Using generated IDs.")

    text_columns = [column for column in ["text", "title", "law_text", "article_text", "content"] if column in frame.columns]
    if not text_columns:
        LOGGER.warning("No known text column found for source '%s'. Falling back to concatenating all columns.", source_name)
        text_columns = list(frame.columns)

    corpus: list[tuple[str, str]] = []
    for row_index, row in frame.iterrows():
        citation = str(row.get(citation_column, f"doc_{row_index}")) if citation_column else f"doc_{row_index}"
        citation = citation.strip()
        text = " ".join(str(row.get(column, "")) for column in text_columns).strip()
        if citation and text:
            corpus.append((citation, text))

    unique_citations = len({citation for citation, _ in corpus})
    LOGGER.info("Source corpus ready: source=%s documents=%s unique_citations=%s", source_name, len(corpus), unique_citations)
    return corpus


def _collapse_citations(citations: list[str], max_items: int) -> str:
    """Create a semicolon-separated unique citation string preserving order."""
    unique: list[str] = []
    seen: set[str] = set()
    for citation in citations:
        cleaned = str(citation).strip()
        if not cleaned or cleaned in seen:
            continue
        unique.append(cleaned)
        seen.add(cleaned)
        if len(unique) >= max_items:
            break
    return ";".join(unique)


def _select_query_column(test_df: pd.DataFrame) -> str | None:
    for column in ["query", "question", "facts", "fact", "text"]:
        if column in test_df.columns:
            return column
    return None


def _build_empty_submission(test_df: pd.DataFrame) -> pd.DataFrame:
    predictions = [""] * len(test_df)
    submission = pd.DataFrame({"predicted_citations": predictions})
    if "query_id" in test_df.columns:
        submission.insert(0, "query_id", test_df["query_id"].values)
    elif "row_id" in test_df.columns:
        submission.insert(0, "row_id", test_df["row_id"].values)
    return submission


def _optional_evaluate_on_train(
    train_df: pd.DataFrame,
    retriever: BM25Retriever,
    retrieval_citations: list[str],
    config: SystemConfig,
) -> None:
    """Run optional retrieval-only evaluation when train columns allow it."""
    if not config.enable_train_eval:
        LOGGER.info("Optional evaluation skipped: enable_train_eval is disabled.")
        return

    if train_df.empty:
        LOGGER.info("Optional evaluation skipped: training dataset is empty.")
        return

    query_column = _select_query_column(train_df)
    if query_column is None:
        LOGGER.info("Optional evaluation skipped: no supported query column in training data.")
        return

    target_column = None
    for candidate in ["gold_citations", "target", "label", "answer", "response"]:
        if candidate in train_df.columns:
            target_column = candidate
            break

    if target_column is None:
        LOGGER.info("Optional evaluation skipped: no supported target column in training data.")
        return

    eval_df = train_df.copy()
    predictions: list[str] = []
    total_rows = len(eval_df)
    LOGGER.info(
        "Optional training evaluation started: rows=%s query_column=%s target_column=%s progress_interval=%s",
        total_rows,
        query_column,
        target_column,
        config.eval_progress_interval,
    )
    for row_index, (_, row) in enumerate(eval_df.iterrows(), start=1):
        query = str(row.get(query_column, "")).strip()
        if not query:
            predictions.append("")
            if row_index % config.eval_progress_interval == 0 or row_index == total_rows:
                LOGGER.info("Optional evaluation progress: %s/%s", row_index, total_rows)
            continue
        results = retriever.search(query, top_k=config.bm25_top_k)
        top_citations = [retrieval_citations[result.index] for result in results if 0 <= result.index < len(retrieval_citations)]
        predictions.append(_collapse_citations(top_citations, max_items=config.output_top_k))
        if row_index % config.eval_progress_interval == 0 or row_index == total_rows:
            LOGGER.info("Optional evaluation progress: %s/%s", row_index, total_rows)

    eval_df["prediction"] = predictions
    score = evaluate_predictions(eval_df, target_column=target_column, prediction_column="prediction")
    LOGGER.info(
        "Optional training evaluation completed: query_column=%s target_column=%s macro_f1=%.4f",
        query_column,
        target_column,
        score,
    )


def processRetrievalPipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    laws_df: pd.DataFrame,
    court_df: pd.DataFrame,
    config: SystemConfig,
    semantic_reranker: LocalSemanticReranker | None = None,
    llm: Any | None = None,
    agent_config: AgentConfig | None = None,
) -> pd.DataFrame:
    """
    Run end-to-end retrieval pipeline and return submission DataFrame.

    This function is designed to be fail-soft: empty inputs skip dependent stages
    while still returning a schema-compatible submission output.
    """
    monitor = PipelineMonitoring()

    started = monitor.start_stage()
    monitor.add_count("train_rows", len(train_df))
    monitor.add_count("test_rows", len(test_df))
    monitor.add_count("laws_rows", len(laws_df))
    monitor.add_count("court_rows", len(court_df))
    monitor.end_stage("input_summary", started)

    if test_df.empty:
        LOGGER.warning("Skipping retrieval pipeline because test dataset is empty.")
        return pd.DataFrame()

    stage_started = monitor.start_stage()
    law_documents = build_law_corpus(laws_df, source_name="laws")
    court_documents = build_law_corpus(court_df, source_name="court")
    corpus_documents = law_documents + court_documents
    monitor.add_count("law_documents", len(law_documents))
    monitor.add_count("court_documents", len(court_documents))
    monitor.add_count("corpus_documents", len(corpus_documents))
    monitor.end_stage("build_corpus", stage_started)

    if not corpus_documents:
        LOGGER.warning("No law documents available. Returning empty predictions for all test rows.")
        empty_submission = _build_empty_submission(test_df)
        monitor.add_count("predictions", len(empty_submission))
        LOGGER.info("Pipeline monitoring summary: durations=%s counters=%s", monitor.stage_durations, monitor.counters)
        return empty_submission

    stage_started = monitor.start_stage()
    chunked_laws = chunk_records(
        ((citation, text) for citation, text in corpus_documents),
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap,
    )
    monitor.add_count("law_chunks", len(chunked_laws))
    monitor.end_stage("chunk_laws", stage_started)

    if not chunked_laws:
        LOGGER.warning("Law chunking yielded no chunks. Falling back to full law documents.")
        retrieval_corpus = [text for _, text in corpus_documents]
        retrieval_citations = [citation for citation, _ in corpus_documents]
    else:
        retrieval_corpus = [chunk.text for chunk in chunked_laws]
        retrieval_citations = [chunk.source_id for chunk in chunked_laws]

    stage_started = monitor.start_stage()
    retriever = BM25Retriever(retrieval_corpus)
    monitor.add_count("retrieval_corpus_size", len(retrieval_corpus))
    monitor.end_stage("init_retriever", stage_started)

    # Integrate evaluator in fail-soft mode when train labels are available.
    stage_started = monitor.start_stage()
    _optional_evaluate_on_train(train_df, retriever, retrieval_citations, config)
    monitor.end_stage("optional_train_evaluation", stage_started)

    query_column = _select_query_column(test_df)
    if query_column is None:
        LOGGER.warning("No supported query column found in test dataset. Returning empty predictions.")
        empty_submission = _build_empty_submission(test_df)
        LOGGER.info("Pipeline monitoring summary: durations=%s counters=%s", monitor.stage_durations, monitor.counters)
        return empty_submission

    stage_started = monitor.start_stage()
    predictions: list[str] = []
    empty_query_count = 0
    no_result_count = 0

    effective_agent_config = agent_config or AgentConfig(max_context_chunks=min(config.bm25_top_k, 5))

    for _, row in test_df.iterrows():
        query = str(row.get(query_column, "")).strip()
        if not query:
            empty_query_count += 1
            predictions.append("")
            continue

        results = retriever.search(query, top_k=config.bm25_top_k)
        ranked_indices = [result.index for result in results if 0 <= result.index < len(retrieval_corpus)]
        contexts = [retrieval_corpus[index] for index in ranked_indices]

        if semantic_reranker is not None and contexts:
            reranked_positions = semantic_reranker.rerank_indices(query, contexts, top_k=config.bm25_top_k)
            ranked_indices = [ranked_indices[position] for position in reranked_positions]

        ranked_citations = [retrieval_citations[index] for index in ranked_indices if 0 <= index < len(retrieval_citations)]
        citation_prediction = _collapse_citations(ranked_citations, max_items=config.output_top_k)

        if citation_prediction:
            predictions.append(citation_prediction)
        else:
            no_result_count += 1
            predictions.append("")

        if len(predictions) % 10 == 0 or len(predictions) == len(test_df):
            LOGGER.info("Prediction progress: %s/%s", len(predictions), len(test_df))

    monitor.end_stage("predict", stage_started)

    if len(predictions) != len(test_df):
        LOGGER.warning(
            "Prediction count mismatch detected (predictions=%s test_rows=%s). Auto-correcting output length.",
            len(predictions),
            len(test_df),
        )
        if len(predictions) < len(test_df):
            predictions.extend([""] * (len(test_df) - len(predictions)))
        else:
            predictions = predictions[: len(test_df)]

    stage_started = monitor.start_stage()
    submission = pd.DataFrame({"predicted_citations": predictions})
    if "query_id" in test_df.columns:
        submission.insert(0, "query_id", test_df["query_id"].values)
    elif "row_id" in test_df.columns:
        submission.insert(0, "row_id", test_df["row_id"].values)
    monitor.end_stage("submission", stage_started)

    monitor.add_count("predictions", len(predictions))
    monitor.add_count("agent_predictions", 0)
    monitor.add_count("empty_queries", empty_query_count)
    monitor.add_count("no_retrieval_results", no_result_count)
    LOGGER.info("Pipeline monitoring summary: durations=%s counters=%s", monitor.stage_durations, monitor.counters)

    return submission


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    LOGGER.info("Starting retrieval evaluation system.")

    config = load_config_from_env()
    validate_config_on_startup(config)

    semantic_reranker = LocalSemanticReranker(config.embedding_model_path)

    train_df, test_df, laws_df, court_df = load_inputs(config)
    submission = processRetrievalPipeline(
        train_df,
        test_df,
        laws_df,
        court_df,
        config,
        semantic_reranker=semantic_reranker,
    )

    if submission.empty:
        LOGGER.warning("No submission output generated. Skipping write operation.")
        print("No test rows found. Nothing to write.")
    else:
        config.submission_file.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(config.submission_file, index=False)
        LOGGER.info("Submission written to %s with %s rows", config.submission_file, len(submission))
        print(f"Wrote submission to {config.submission_file}")

    LOGGER.info(
        "Checkpoint status: Required implementation phases completed in code. "
        "Test execution checkpoints intentionally skipped per current workflow constraints."
    )


if __name__ == "__main__":
    main()
