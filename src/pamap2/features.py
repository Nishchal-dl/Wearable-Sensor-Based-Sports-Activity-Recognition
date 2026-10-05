"""Windowing and feature extraction for PAMAP2 (pure functions, no AWS)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .schema import ACTIVITY_NAMES, IMUS

FS = 100.0
MAX_GAP_S = 0.1
WINDOW_SIZE = 512
WINDOW_STEP = 256

# Ainsworth Compendium of Physical Activities (approximate METs) for each PAMAP2 activity.
ACTIVITY_MET = {
    1: 1.0, 2: 1.3, 3: 1.8, 4: 3.5, 5: 8.0, 6: 6.0, 7: 6.8, 9: 1.3, 10: 1.5,
    11: 2.0, 12: 8.8, 13: 3.5, 16: 3.5, 17: 1.8, 18: 2.0, 19: 3.5, 20: 7.0, 24: 12.3,
}
assert set(ACTIVITY_MET) == set(ACTIVITY_NAMES) - {0}  # all 18 labelled activities


def intensity_of(activity_id: int) -> str:
    """light (< 3 METs), moderate (3-6), vigorous (> 6)."""
    met = ACTIVITY_MET[int(activity_id)]
    return "light" if met < 3 else ("moderate" if met <= 6 else "vigorous")


def _channels(imu: str) -> list[str]:
    return ([f"{imu}_acc16_{a}" for a in "xyz"] + [f"{imu}_acc16_mag"]
            + [f"{imu}_gyro_{a}" for a in "xyz"])


FEATURE_STATS = ("mean", "std", "min", "max", "energy", "domfreq")
FEATURE_COLUMNS = ([f"{ch}_{s}" for imu in IMUS for ch in _channels(imu) for s in FEATURE_STATS]
                   + ["hr_mean", "hr_max"])


def segment(df: pd.DataFrame, max_gap: float = MAX_GAP_S) -> list[pd.DataFrame]:
    """Split a time-ordered frame at timestamp gaps > max_gap. Index labels are kept."""
    if df.empty:
        return []
    gaps = np.diff(df["timestamp"].to_numpy(dtype=float)) > max_gap + 1e-9
    bounds = [0, *(np.flatnonzero(gaps) + 1).tolist(), len(df)]
    return [df.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def make_windows(df: pd.DataFrame, size: int = WINDOW_SIZE, step: int = WINDOW_STEP):
    """Yield (start_idx, window) for sliding windows inside each gap-free segment.

    Rows are stable-sorted by timestamp first; start_idx is the positional index in that sorted
    frame. Short tails are dropped.
    """
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    for seg in segment(df):
        for off in range(0, len(seg) - size + 1, step):
            win = seg.iloc[off:off + size]
            yield int(win.index[0]), win


def _stats(x: np.ndarray) -> list[float]:
    if np.isnan(x).all():
        return [np.nan] * len(FEATURE_STATS)
    nan = np.isnan(x)
    if nan.any():
        x = np.where(nan, np.nanmean(x), x)  # keep FFT/energy defined for partial dropout
    spec = np.abs(np.fft.rfft(x - x.mean()))
    spec[0] = 0.0
    dom = float(np.fft.rfftfreq(len(x), 1.0 / FS)[int(np.argmax(spec))])
    return [float(x.mean()), float(x.std()), float(x.min()), float(x.max()),
            float(np.mean(x ** 2)), dom]


def window_features(win: pd.DataFrame) -> dict:
    """Flat feature dict for one window (see FEATURE_COLUMNS)."""
    out: dict[str, float] = {}
    for imu in IMUS:
        acc = np.stack([win[f"{imu}_acc16_{a}"].to_numpy(float) for a in "xyz"])
        data = {f"{imu}_acc16_{a}": acc[i] for i, a in enumerate("xyz")}
        data[f"{imu}_acc16_mag"] = np.sqrt((acc ** 2).sum(axis=0))
        for a in "xyz":
            data[f"{imu}_gyro_{a}"] = win[f"{imu}_gyro_{a}"].to_numpy(float)
        for ch in _channels(imu):
            for name, v in zip(FEATURE_STATS, _stats(data[ch])):
                out[f"{ch}_{name}"] = v
    hr = win["heart_rate"].astype(float).ffill().bfill().to_numpy()
    allnan = np.isnan(hr).all()
    out["hr_mean"] = np.nan if allnan else float(hr.mean())
    out["hr_max"] = np.nan if allnan else float(hr.max())
    return out


def build_feature_frame(df: pd.DataFrame, subject_id: int, session: str,
                        ingest_time: float | None = None) -> pd.DataFrame:
    """One row per window with metadata + features. Majority-label-0 windows are dropped.

    ``event_time`` is the epoch time of the ingest run (``ingest_time``, default now; pass one
    value per run); the window start (session-relative seconds) is kept as ``t_start``.
    """
    if ingest_time is None:
        ingest_time = time.time()
    rows = []
    for start, win in make_windows(df):
        counts = win["activity_id"].value_counts()
        act = int(counts[counts == counts.max()].index.min())  # deterministic tie-break
        if act == 0:
            continue
        rec = {
            "window_id": f"{subject_id}_{session}_{start}",
            "subject_id": int(subject_id),
            "session": str(session),
            "start_idx": int(start),
            "activity_id": act,
            "intensity": intensity_of(act),
            "event_time": float(ingest_time),
            "t_start": float(win["timestamp"].iloc[0]),
        }
        rec.update(window_features(win))
        rows.append(rec)
    cols = ["window_id", "subject_id", "session", "start_idx", "activity_id", "intensity",
            "event_time", "t_start", *FEATURE_COLUMNS]
    out = pd.DataFrame(rows, columns=cols)
    for c in ("window_id", "session", "intensity"):
        out[c] = out[c].astype("string")
    for c in ("subject_id", "start_idx", "activity_id"):
        out[c] = out[c].astype("int64")
    for c in ["event_time", "t_start", *FEATURE_COLUMNS]:
        out[c] = out[c].astype("float64")
    return out
