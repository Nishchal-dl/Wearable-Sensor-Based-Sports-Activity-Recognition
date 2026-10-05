"""MVP benchmark: shallow decision tree on two features, plus a majority-class floor."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

BENCHMARK_FEATURES = ["chest_acc16_mag_std", "hr_mean"]


def _matrix(df: pd.DataFrame, medians: dict) -> np.ndarray:
    missing = [c for c in BENCHMARK_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    return df[BENCHMARK_FEATURES].astype(float).fillna(pd.Series(medians)).to_numpy()


def fit_benchmark(train_df: pd.DataFrame, label_col: str = "label") -> dict:
    """Fit on train; NaNs filled with train medians. Returns model bundle."""
    missing = [c for c in BENCHMARK_FEATURES if c not in train_df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    if len(train_df) == 0:
        raise ValueError("empty training frame")
    medians = {c: float(train_df[c].astype(float).median()) for c in BENCHMARK_FEATURES}
    medians = {c: (0.0 if np.isnan(v) else v) for c, v in medians.items()}
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(_matrix(train_df, medians), train_df[label_col].astype(int))
    return {"tree": tree, "medians": medians,
            "majority": int(train_df[label_col].astype(int).mode().iloc[0])}


def predict_benchmark(model: dict, df: pd.DataFrame) -> np.ndarray:
    return model["tree"].predict(_matrix(df, model["medians"]))


def predict_majority(model: dict, df: pd.DataFrame) -> np.ndarray:
    return np.full(len(df), model["majority"], dtype=int)
