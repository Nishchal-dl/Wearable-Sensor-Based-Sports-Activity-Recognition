"""Pipeline evaluation step: score the trained XGBoost model on the test CSV and write evaluation.json.

Self-contained (stdlib, numpy; xgboost/pandas only in __main__); runs in the XGBoost container.
Metric semantics match src.pamap2.evaluation.classification_metrics.
"""
import csv
import json
import os
import tarfile

import numpy as np


def classification_metrics(y_true, y_pred, k):
    """Macro-F1 (classes with support > 0), accuracy, per-class P/R/F1/support, KxK confusion matrix."""
    yt, yp = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(yt, yp):
        cm[t, p] += 1
    per_class, f1s = {}, []
    for i in range(k):
        tp, support, predicted = cm[i, i], cm[i].sum(), cm[:, i].sum()
        prec = float(tp / predicted) if predicted else 0.0
        rec = float(tp / support) if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[i] = {"precision": prec, "recall": rec, "f1": f1, "support": int(support)}
        if support > 0:
            f1s.append(f1)
    n = len(yt)
    return {"macro_f1": float(np.mean(f1s)) if f1s else 0.0,
            "accuracy": float((yt == yp).sum() / n) if n else 0.0,
            "n": int(n), "labels": list(range(k)), "per_class": per_class,
            "confusion_matrix": cm.tolist()}


def build_report(y_true, probs):
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"expected 2-D probabilities, got shape {probs.shape}")
    m = classification_metrics(y_true, probs.argmax(axis=1), probs.shape[1])
    return {"multiclass_classification_metrics": {
                "macro_f1": {"value": m["macro_f1"], "standard_deviation": "NaN"},
                "accuracy": {"value": m["accuracy"], "standard_deviation": "NaN"}},
            "per_class": m["per_class"], "confusion_matrix": m["confusion_matrix"],
            "labels": m["labels"], "n": m["n"]}


def prediction_rows(y_true, probs):
    """Rows (label, predicted_class, confidence) per test window; confidence is the row max probability."""
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"expected 2-D probabilities, got shape {probs.shape}")
    labels = np.asarray(y_true).astype(int)
    if len(labels) != len(probs):
        raise ValueError(f"{len(labels)} labels but {len(probs)} probability rows")
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    return [(int(t), int(p), float(c)) for t, p, c in zip(labels, pred, conf)]


def write_outputs(y_true, probs, out_dir):
    """Write evaluation.json and test_predictions.csv (header label,predicted_class,confidence); returns the report."""
    report = build_report(y_true, probs)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "evaluation.json"), "w") as f:
        json.dump(report, f)
    with open(os.path.join(out_dir, "test_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "predicted_class", "confidence"])
        w.writerows(prediction_rows(y_true, probs))
    return report


if __name__ == "__main__":
    import pandas as pd
    import xgboost as xgb

    model_dir = "/opt/ml/processing/model"
    with tarfile.open(os.path.join(model_dir, "model.tar.gz")) as tar:
        tar.extractall(model_dir)
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, "xgboost-model"))
    df = pd.read_csv("/opt/ml/processing/test/test.csv", header=None)
    y_true, X = df.iloc[:, 0].to_numpy(), df.iloc[:, 1:].to_numpy()
    probs = booster.predict(xgb.DMatrix(X))
    out = "/opt/ml/processing/evaluation"
    print(write_outputs(y_true, probs, out)["multiclass_classification_metrics"])
