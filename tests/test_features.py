import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pamap2 import features as F
from src.pamap2.schema import ACTIVITY_NAMES, CURATED_COLUMNS


def synth(n=1024, act=4, t0=0.0, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(CURATED_COLUMNS))), columns=CURATED_COLUMNS)
    df["timestamp"] = t0 + np.arange(n) * 0.01
    df["activity_id"] = act
    df["heart_rate"] = np.nan
    df.loc[::11, "heart_rate"] = 100.0
    return df


def test_intensity_covers_all_activities():
    ids = [a for a in ACTIVITY_NAMES if a != 0]
    assert len(ids) == 18
    assert {F.intensity_of(a) for a in ids} == {"light", "moderate", "vigorous"}
    assert F.intensity_of(1) == "light" and F.intensity_of(4) == "moderate"
    assert F.intensity_of(24) == "vigorous"


def test_normal_window():
    out = F.build_feature_frame(synth(512), 101, "protocol")
    assert len(out) == 1
    r = out.iloc[0]
    assert r.window_id == "101_protocol_0" and r.activity_id == 4 and r.intensity == "moderate"
    assert list(out.columns[8:]) == F.FEATURE_COLUMNS
    assert r.hr_mean == 100.0 and r.hr_max == 100.0


def test_overlap_and_short_tail():
    assert [s for s, _ in F.make_windows(synth(1000))] == [0, 256]  # 1000-512=488 -> 0,256
    assert list(F.make_windows(synth(500))) == []


def test_mixed_labels_majority():
    df = synth(512)
    df.loc[:100, "activity_id"] = 1
    assert F.build_feature_frame(df, 1, "p").activity_id.iloc[0] == 4


def test_mostly_transient_dropped():
    df = synth(512, act=0)
    df.loc[:100, "activity_id"] = 4
    assert F.build_feature_frame(df, 1, "p").empty


def test_gap_splits_segment():
    a, b = synth(600), synth(600, t0=100.0)
    df = pd.concat([a, b], ignore_index=True)
    starts = [s for s, _ in F.make_windows(df)]
    assert starts == [0, 600]  # no window crosses the gap
    for _, w in F.make_windows(df):
        assert w["timestamp"].diff().max() <= 0.1


def test_all_nan_hr():
    df = synth(512)
    df["heart_rate"] = np.nan
    r = F.build_feature_frame(df, 1, "p").iloc[0]
    assert np.isnan(r.hr_mean) and np.isnan(r.hr_max)


def test_all_nan_axis_keeps_row():
    df = synth(512)
    df["hand_gyro_x"] = np.nan
    out = F.build_feature_frame(df, 1, "p")
    assert len(out) == 1
    assert np.isnan(out.iloc[0]["hand_gyro_x_mean"]) and not np.isnan(out.iloc[0]["hand_gyro_y_mean"])


def test_dominant_frequency():
    df = synth(512)
    df["chest_gyro_z"] = np.sin(2 * np.pi * 5.0 * df["timestamp"])
    r = F.window_features(df)
    assert abs(r["chest_gyro_z_domfreq"] - 5.0) < 0.3


def test_shuffled_rows_same_result():
    df = synth(1024)
    a = F.build_feature_frame(df, 1, "p", ingest_time=5.0)
    b = F.build_feature_frame(df.sample(frac=1, random_state=1), 1, "p", ingest_time=5.0)
    pd.testing.assert_frame_equal(a, b)


def test_event_time_is_ingest_time():
    r = F.build_feature_frame(synth(512, t0=7.0), 1, "p", ingest_time=1.7e9).iloc[0]
    assert r.event_time == 1.7e9 and r.t_start == 7.0


def _check_dtypes(out):
    assert str(out.window_id.dtype) == "string" and str(out.session.dtype) == "string"
    assert str(out.intensity.dtype) == "string"
    for c in ("subject_id", "start_idx", "activity_id"):
        assert out[c].dtype == "int64"
    for c in ["event_time", "t_start", *F.FEATURE_COLUMNS]:
        assert out[c].dtype == "float64"


def test_dtype_contract():
    _check_dtypes(F.build_feature_frame(synth(512), 1, "p"))
    _check_dtypes(F.build_feature_frame(synth(100), 1, "p"))
