import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pamap2.split import (SESSION_ORDER, SPLITS, _block_labels, assign_splits,
                              restrict_to_train_classes, split_shares)


def frame():
    rows = [(s, a, "protocol", i, f"{s}_protocol_{a}_{i}")
            for s in (101, 102) for a in (1, 4) for i in range(200)]
    return pd.DataFrame(rows, columns=["subject_id", "activity_id", "session", "start_idx", "window_id"])


def test_deterministic_and_unique():
    df = frame()
    a, b = assign_splits(df), assign_splits(df.sample(frac=1, random_state=3))
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    assert a.window_id.is_unique


def test_proportions_and_order():
    out = assign_splits(frame())
    sh = split_shares(out)
    for k, v in zip(SPLITS, (40, 10, 10, 40)):
        assert abs(sh[k] - v) <= 5
    g = out[(out.subject_id == 101) & (out.activity_id == 1)]
    first = {s: g[g.split == s].start_idx for s in SPLITS}
    assert first["train"].max() < first["validation"].min()
    assert first["validation"].max() < first["test"].min() < first["production"].min()
    # dropped window at every boundary
    for a, b in (("train", "validation"), ("validation", "test"), ("test", "production")):
        assert first[b].min() - first[a].max() >= 2


def test_tiny_group():
    df = frame().query("start_idx < 3")
    out = assign_splits(df)
    assert len(out) <= len(df) and out.window_id.is_unique


def test_block_labels_small_n():
    for n in (0, 1, 2, 5):
        assert len(_block_labels(n)) == n
    assert _block_labels(0) == []


def test_empty_input():
    out = assign_splits(frame().iloc[0:0])
    assert out.empty and "split" in out.columns


def test_session_order_protocol_before_optional():
    assert SESSION_ORDER["protocol"] < SESSION_ORDER["optional"]
    rows = [(1, 4, s, i, f"{s}{i}") for s in ("optional", "protocol") for i in range(20)]
    df = pd.DataFrame(rows, columns=["subject_id", "activity_id", "session", "start_idx", "window_id"])
    out = assign_splits(df)
    train = out[out.split == "train"]
    assert (train.session == "protocol").all()
    assert (out[out.split == "production"].session == "optional").all()


def test_restrict_to_train_classes():
    df = pd.DataFrame({"activity_id": [1, 1, 2, 2], "split": ["train", "test", "validation", "production"]})
    out, dropped, classes = restrict_to_train_classes(df)
    assert classes == [1] and len(out) == 2
    assert dropped["validation"] == 1 and dropped["production"] == 1
