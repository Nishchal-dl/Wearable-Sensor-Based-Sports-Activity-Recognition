"""PAMAP2 column layout (54 columns), invalid-column handling, parser, label map."""
from __future__ import annotations

import pandas as pd

IMUS = ("hand", "chest", "ankle")

ACTIVITY_NAMES = {
    0: "other (transient)", 1: "lying", 2: "sitting", 3: "standing", 4: "walking",
    5: "running", 6: "cycling", 7: "Nordic walking", 9: "watching TV",
    10: "computer work", 11: "car driving", 12: "ascending stairs",
    13: "descending stairs", 16: "vacuum cleaning", 17: "ironing",
    18: "folding laundry", 19: "house cleaning", 20: "playing soccer", 24: "rope jumping",
}


def _imu_columns(imu: str) -> list[str]:
    cols = [f"{imu}_temp"]
    for name in ("acc16", "acc6", "gyro", "mag"):
        cols += [f"{imu}_{name}_{a}" for a in "xyz"]
    cols += [f"{imu}_orient_{i}" for i in range(1, 5)]
    return cols


COLUMNS = ["timestamp", "activity_id", "heart_rate"] + [c for i in IMUS for c in _imu_columns(i)]
INVALID_COLUMNS = [f"{i}_orient_{n}" for i in IMUS for n in range(1, 5)]
CURATED_COLUMNS = [c for c in COLUMNS if c not in INVALID_COLUMNS]
assert len(COLUMNS) == 54 and len(CURATED_COLUMNS) == 42

DTYPES = {c: "double" for c in COLUMNS}
DTYPES["activity_id"] = "int"


def parse_dat(path_or_buf, subject_id: int | None = None, session: str | None = None,
              drop_invalid: bool = True) -> pd.DataFrame:
    """Parse a space-separated PAMAP2 .dat file; keeps NaN (no imputation)."""
    df = pd.read_csv(path_or_buf, sep=r"\s+", header=None, na_values=["NaN"])
    if df.shape[1] != 54:
        raise ValueError(f"expected 54 columns, got {df.shape[1]}")
    df.columns = COLUMNS
    bad = df["activity_id"].isna() | df["timestamp"].isna()
    if bad.any():
        print(f"parse_dat: dropping {int(bad.sum())} rows with NaN activity_id/timestamp")
        df = df[~bad].reset_index(drop=True)
    df["activity_id"] = df["activity_id"].astype("int32")
    if drop_invalid:
        df = df.drop(columns=INVALID_COLUMNS)
    if subject_id is not None:
        df["subject_id"] = int(subject_id)
    if session is not None:
        df["session"] = session
    return df
