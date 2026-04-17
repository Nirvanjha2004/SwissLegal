from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def read_csv_file(path: str | Path, **kwargs) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path, **kwargs)


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = " ".join(text.split())
    return text.strip()


def normalize_text_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    cleaned = frame.copy()
    for column in columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].map(clean_text)
    return cleaned


def load_dataset(path: str | Path, text_columns: Iterable[str] | None = None, **kwargs) -> pd.DataFrame:
    frame = read_csv_file(path, **kwargs)
    if text_columns:
        frame = normalize_text_columns(frame, text_columns)
    return frame
