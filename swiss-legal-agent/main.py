from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.chunker import chunk_records
from src.config import DEFAULT_BM25_TOP_K, LAWS_FILE, MODELS_DIR, RAW_DATA_DIR, SUBMISSION_FILE, TEST_FILE, TRAIN_FILE
from src.data_loader import load_dataset
from src.retriever import BM25Retriever


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = load_dataset(TRAIN_FILE)
    test_df = load_dataset(TEST_FILE)
    laws_df = load_dataset(LAWS_FILE)
    return train_df, test_df, laws_df


def build_law_corpus(laws_df: pd.DataFrame) -> list[str]:
    if laws_df.empty:
        return []
    text_columns = [column for column in ["text", "law_text", "article_text", "content"] if column in laws_df.columns]
    if not text_columns:
        return laws_df.astype(str).agg(" ".join, axis=1).tolist()
    return laws_df[text_columns].astype(str).agg(" ".join, axis=1).tolist()


def run_baseline(test_df: pd.DataFrame, law_documents: list[str]) -> pd.DataFrame:
    if test_df.empty:
        return pd.DataFrame()

    retriever = BM25Retriever(law_documents)
    query_columns = [column for column in ["question", "facts", "fact", "text"] if column in test_df.columns]
    query_column = query_columns[0] if query_columns else None

    predictions: list[str] = []
    for _, row in test_df.iterrows():
        query = str(row.get(query_column, "")) if query_column else ""
        results = retriever.search(query, top_k=DEFAULT_BM25_TOP_K)
        predictions.append(results[0].text if results else "")

    submission = pd.DataFrame({"prediction": predictions})
    if "row_id" in test_df.columns:
        submission.insert(0, "row_id", test_df["row_id"].values)
    return submission


def main() -> None:
    train_df, test_df, laws_df = load_inputs()
    law_documents = build_law_corpus(laws_df)

    if law_documents:
        chunked_laws = chunk_records((str(index), document) for index, document in enumerate(law_documents))
        print(f"Loaded {len(law_documents)} law rows and created {len(chunked_laws)} chunks.")
    else:
        print("No law documents found. Check data/raw/laws_de.csv.")

    submission = run_baseline(test_df, law_documents)
    if not submission.empty:
        submission.to_csv(SUBMISSION_FILE, index=False)
        print(f"Wrote submission to {SUBMISSION_FILE}")
    else:
        print("No test rows found. Nothing to write.")

    if not train_df.empty:
        print(f"Loaded train set with {len(train_df)} rows.")


if __name__ == "__main__":
    main()
