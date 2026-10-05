"""Deterministic chronological split: 40/10/10/40 train/validation/test/production."""
from __future__ import annotations

import pandas as pd

SPLITS = ("train", "validation", "test", "production")
FRACTIONS = (0.40, 0.10, 0.10)  # production takes the remainder
GROUP = ["subject_id", "activity_id"]
SESSION_ORDER = {"protocol": 0, "optional": 1}  # PAMAP2 recorded the optional session after protocol


def _block_labels(n: int) -> list:
    a = round(FRACTIONS[0] * n)
    b = a + round(FRACTIONS[1] * n)
    c = b + round(FRACTIONS[2] * n)
    out = []
    for lo, hi, name in ((0, a, "train"), (a, b, "validation"), (b, c, "test"), (c, n, "production")):
        if hi > lo:
            # first window of every block after train is the dropped 1-window gap
            out += ([None] if lo > 0 else []) + [name] * (hi - lo - (1 if lo > 0 else 0))
    return out


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``split`` column; windows in the 1-window gaps between blocks are dropped.

    Within each subject+activity, windows are ordered by (session order protocol < optional, start_idx, window_id) and cut
    into consecutive blocks: train 40%, validation 10%, test 10%, production 40%. One window is
    dropped at each block boundary so 50%-overlapping windows never straddle a boundary.
    Fully deterministic (no randomness), so repeated runs give identical output.
    """
    ordered = (df.assign(_so=df["session"].map(SESSION_ORDER).fillna(len(SESSION_ORDER)))
               .sort_values(GROUP + ["_so", "start_idx", "window_id"], kind="mergesort")
               .drop(columns="_so"))
    parts = []
    for _, g in ordered.groupby(GROUP, sort=True):
        labels = _block_labels(len(g))
        g = g.assign(split=labels)
        parts.append(g[g["split"].notna()])
    if not parts:
        return df.iloc[0:0].assign(split=pd.Series(dtype="object"))
    return pd.concat(parts).reset_index(drop=True)


def split_shares(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Window-count shares (%) per split, overall or per ``by`` column."""
    if by is None:
        return (df["split"].value_counts(normalize=True).reindex(SPLITS).fillna(0) * 100).round(1)
    return (pd.crosstab(df[by], df["split"], normalize="index").reindex(columns=SPLITS)
            .fillna(0) * 100).round(1)


def restrict_to_train_classes(df: pd.DataFrame):
    """Drop non-train rows whose activity_id is absent from train.

    Returns (filtered frame, Series of dropped counts per split, train class list).
    """
    classes = sorted(df.loc[df["split"] == "train", "activity_id"].unique().tolist())
    bad = (df["split"] != "train") & ~df["activity_id"].isin(classes)
    return df[~bad].reset_index(drop=True), df[bad]["split"].value_counts(), classes


def class_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Windows per activity_id x split."""
    return pd.crosstab(df["activity_id"], df["split"]).reindex(columns=SPLITS, fill_value=0)
