from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.metrics import f1_score


def macro_f1(y_true: Iterable, y_pred: Iterable) -> float:
    return float(f1_score(list(y_true), list(y_pred), average="macro"))


def evaluate_predictions(frame: pd.DataFrame, target_column: str, prediction_column: str) -> float:
    if frame.empty:
        return 0.0
    if target_column not in frame.columns or prediction_column not in frame.columns:
        raise KeyError(f"Expected columns {target_column!r} and {prediction_column!r}")
    return macro_f1(frame[target_column], frame[prediction_column])
