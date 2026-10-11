"""Pipeline preprocessing step: validate split Parquet files and emit XGBoost CSVs.

Self-contained (stdlib, pandas, numpy); runs inside the SKLearn processing container.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

SPLITS = ("train", "validation", "test", "production")


def _read(input_dir, name):
    path = os.path.join(input_dir, f"{name}.parquet")
    if not os.path.exists(path):
        raise ValueError(f"missing split file: {path}")
    return pd.read_parquet(path)


def _check_labels(df, name, k, required):
    if "label" not in df.columns:
        if required:
            raise ValueError(f"{name}: missing 'label' column")
        return
    lab = df["label"]
    if lab.isna().any():
        raise ValueError(f"{name}: {int(lab.isna().sum())} null labels")
    vals = lab.to_numpy(dtype=float)
    bad = sorted(set(vals[(vals < 0) | (vals >= k) | (vals != np.floor(vals))].tolist()))
    if bad:
        raise ValueError(f"{name}: labels outside 0..{k - 1}: {bad[:10]}")


def build_outputs(input_dir, output_dir):
    """Read splits from `input_dir`; write CSVs under `output_dir`. Returns row counts per output."""
    with open(os.path.join(input_dir, "feature_columns.json")) as f:
        features = json.load(f)
    with open(os.path.join(input_dir, "label_map.json")) as f:
        k = len(json.load(f))
    counts = {}
    for name in SPLITS:
        df = _read(input_dir, name)
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: missing feature columns: {missing}")
        _check_labels(df, name, k, required=name != "production")
        if name == "production":
            sub, path = df[features], os.path.join(output_dir, "batch", "features.csv")
        else:
            sub = pd.concat([df["label"].astype(int), df[features]], axis=1)
            path = os.path.join(output_dir, name, f"{name}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sub.to_csv(path, header=False, index=False)
        counts[name] = len(sub)
    return counts


if __name__ == "__main__":
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyarrow"])
    print(build_outputs("/opt/ml/processing/input", "/opt/ml/processing"))
