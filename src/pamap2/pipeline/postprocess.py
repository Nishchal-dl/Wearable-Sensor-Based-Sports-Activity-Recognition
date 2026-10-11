"""Pipeline post-process step: batch-transform probabilities -> predictions Parquet.

Self-contained (stdlib, numpy, pandas; no `src.` imports); runs in the scikit-learn processing container.
Rows are joined to the production windows by position (the batch CSV is features only), with a hard row-count check.
"""
import glob
import io
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd


def parse_probs(text, k):
    """Parse batch-transform CSV output (features-only input) into an (n, k) array.

    Brackets and quotes are stripped; any column count other than k raises ValueError.
    """
    if isinstance(text, (bytes, bytearray)):
        text = text.decode()
    text = text.translate(str.maketrans("", "", '[]"'))
    df = pd.read_csv(io.StringIO(text), header=None, skipinitialspace=True)
    if df.shape[1] != k:
        raise ValueError(f"expected exactly {k} probability columns, got {df.shape[1]}")
    return df.to_numpy(float)


def build_predictions(prod, probs, label_map, activity_meta):
    """Predictions contract: one row per production window, in input order.

    label_map: list of {"activity_id", "label"}; activity_meta: {"<activity_id>": {"name", "intensity"}}.
    """
    probs = np.asarray(probs, dtype=float)
    k = len(label_map)
    if probs.ndim != 2 or probs.shape[1] != k:
        raise ValueError(f"expected {k} probability columns, got shape {probs.shape}")
    if len(probs) != len(prod):
        raise ValueError(f"probability rows {len(probs)} != production rows {len(prod)}")
    missing = [c for c in ("window_id", "subject_id", "activity_id") if c not in prod.columns]
    if missing:
        raise ValueError(f"production split is missing columns: {missing}")
    to_activity = {int(r["label"]): int(r["activity_id"]) for r in label_map}
    cls = probs.argmax(axis=1)
    act = [to_activity[int(c)] for c in cls]
    unknown = sorted({a for a in act if str(a) not in activity_meta})
    if unknown:
        raise ValueError(f"predicted activity ids missing from activity metadata: {unknown}")
    prod = prod.reset_index(drop=True)
    return pd.DataFrame({
        "window_id": prod["window_id"].astype(str),
        "subject_id": prod["subject_id"],
        "predicted_class": cls,
        "predicted_activity_id": act,
        "activity_name": [activity_meta[str(a)]["name"] for a in act],
        "intensity": [activity_meta[str(a)]["intensity"] for a in act],
        "confidence": probs.max(axis=1),
        "true_activity_id": prod["activity_id"],
    })


def run(transform_dir, input_dir, meta_dir, out_dir):
    """Read the single *.out file, production.parquet, label_map.json and activity_meta.json; write predictions.parquet."""
    outs = glob.glob(os.path.join(transform_dir, "*.out"))
    if len(outs) != 1:
        raise ValueError(f"expected exactly one *.out file in {transform_dir}, found {len(outs)}")
    with open(os.path.join(input_dir, "label_map.json")) as f:
        label_map = json.load(f)
    with open(os.path.join(meta_dir, "activity_meta.json")) as f:
        meta = json.load(f)
    with open(outs[0]) as f:
        probs = parse_probs(f.read(), len(label_map))
    prod = pd.read_parquet(os.path.join(input_dir, "production.parquet"))
    preds = build_predictions(prod, probs, label_map, meta)
    os.makedirs(out_dir, exist_ok=True)
    preds.to_parquet(os.path.join(out_dir, "predictions.parquet"), index=False)
    return len(preds)


if __name__ == "__main__":
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyarrow"])
    n = run("/opt/ml/processing/transform", "/opt/ml/processing/input", "/opt/ml/processing/meta",
            "/opt/ml/processing/predictions")
    print(n, "predictions written")
