"""Metric, comparison and payload helpers shared by benchmark and XGBoost (pure, no AWS)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_label_map(raw) -> dict[int, int]:
    """label_map.json (list of {activity_id,label}) -> {class_index: activity_id}."""
    return {int(r["label"]): int(r["activity_id"]) for r in raw}


def classification_metrics(y_true, y_pred, labels) -> dict:
    """Macro-F1, accuracy, per-class P/R/F1/support and KxK confusion matrix over `labels`.

    Classes with zero support are reported (support 0) but excluded from the macro average;
    classes never predicted get precision 0 (zero_division=0).
    """
    labels = [int(x) for x in labels]
    idx = {l: i for i, l in enumerate(labels)}
    yt, yp = (np.asarray(y, dtype=float) for y in (y_true, y_pred))
    if np.isnan(yt).any() or np.isnan(yp).any():
        raise ValueError("NaN in y_true/y_pred")
    yt, yp = yt.astype(int), yp.astype(int)
    unknown = sorted((set(yt.tolist()) | set(yp.tolist())) - set(labels))
    if unknown:
        raise ValueError(f"labels not in `labels`: {unknown}")
    k = len(labels)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(yt, yp):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    per_class, f1s = {}, []
    for l, i in idx.items():
        tp, support, predicted = cm[i, i], cm[i].sum(), cm[:, i].sum()
        prec = float(tp / predicted) if predicted else 0.0
        rec = float(tp / support) if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[l] = {"precision": prec, "recall": rec, "f1": f1, "support": int(support)}
        if support > 0:
            f1s.append(f1)
    n = len(yt)
    return {
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "accuracy": float((yt == yp).sum() / n) if n else 0.0,
        "n": int(n),
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def probs_to_labels(probs, k: int) -> np.ndarray:
    """Argmax class index from softprob rows; raises if column count != k."""
    arr = np.asarray(probs, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != k:
        raise ValueError(f"expected {k} probability columns, got shape {arr.shape}")
    return arr.argmax(axis=1)


def compare(benchmark: dict, model: dict) -> dict:
    """Side-by-side table of headline metrics with delta; `better` is True when model macro-F1 wins."""
    rows = []
    for m in ("macro_f1", "accuracy"):
        rows.append({"metric": m, "benchmark": benchmark[m], "model": model[m],
                     "delta": model[m] - benchmark[m]})
    return {"table": pd.DataFrame(rows), "better": bool(model["macro_f1"] > benchmark["macro_f1"])}


def check_feature_columns(df: pd.DataFrame, cols) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")


def parse_probs(raw, k: int) -> np.ndarray:
    """Parse batch-transform CSV output (bytes or str) into an (n, k) probability array.

    Takes the last k columns, so joined output (input features then k probs) and plain output both work.
    Newer XGBoost containers write each softprob row as a bracketed list ("[p1, p2, ...]"); the
    brackets and quotes are stripped first.
    """
    import io
    text = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
    text = text.translate(str.maketrans("", "", '[]"'))
    df = pd.read_csv(io.StringIO(text), header=None, skipinitialspace=True)
    if df.shape[1] < k:
        raise ValueError(f"expected at least {k} columns, got {df.shape[1]}")
    return df.iloc[:, -k:].to_numpy(float)
