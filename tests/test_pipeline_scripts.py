import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pamap2 import evaluation as E
from src.pamap2.pipeline import evaluate as ev
from src.pamap2.pipeline import postprocess as pp_post
from src.pamap2.pipeline import preprocess as pp

FEATS = ["f_b", "f_a", "f_c"]
K = 3


def write_splits(d, mutate=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "feature_columns.json").write_text(json.dumps(FEATS))
    (d / "label_map.json").write_text(json.dumps([{"activity_id": 10 + i, "label": i} for i in range(K)]))
    rng = np.random.default_rng(0)
    for name in pp.SPLITS:
        df = pd.DataFrame({"subject": 1, "label": [0, 1, 2, 1], **{c: rng.random(4) for c in ["f_a", "f_b", "f_c"]}})
        if mutate:
            df = mutate(name, df)
        df.to_parquet(d / f"{name}.parquet")


def test_outputs_shape_and_order(tmp_path):
    write_splits(tmp_path / "in")
    counts = pp.build_outputs(str(tmp_path / "in"), str(tmp_path / "out"))
    assert counts == {n: 4 for n in pp.SPLITS}
    tr = pd.read_csv(tmp_path / "out/train/train.csv", header=None)
    src = pd.read_parquet(tmp_path / "in/train.parquet")
    assert tr.shape == (4, 1 + len(FEATS))
    assert tr[0].tolist() == src["label"].tolist()
    assert np.allclose(tr.iloc[:, 1:].to_numpy(), src[FEATS].to_numpy())
    assert (tmp_path / "out/validation/validation.csv").exists() and (tmp_path / "out/test/test.csv").exists()
    batch = pd.read_csv(tmp_path / "out/batch/features.csv", header=None)
    assert batch.shape == (4, len(FEATS))
    prod = pd.read_parquet(tmp_path / "in/production.parquet")
    assert np.allclose(batch.to_numpy(), prod[FEATS].to_numpy())


def test_missing_feature_column(tmp_path):
    write_splits(tmp_path, lambda n, df: df.drop(columns=["f_c"]) if n == "test" else df)
    with pytest.raises(ValueError, match="f_c"):
        pp.build_outputs(str(tmp_path), str(tmp_path / "o"))


@pytest.mark.parametrize("bad", [None, 3, -1])
def test_bad_labels(tmp_path, bad):
    def mut(n, df):
        if n == "train":
            df = df.copy()
            df["label"] = df["label"].astype(float)
            df.loc[0, "label"] = np.nan if bad is None else bad
        return df
    write_splits(tmp_path, mut)
    with pytest.raises(ValueError):
        pp.build_outputs(str(tmp_path), str(tmp_path / "o"))


def test_report_matches_project_metrics():
    k = 4  # class 3 has zero support
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 60)
    probs = rng.random((60, k))
    rep = ev.build_report(y, probs)
    ref = E.classification_metrics(y, probs.argmax(axis=1), list(range(k)))
    m = rep["multiclass_classification_metrics"]
    assert set(rep) == {"multiclass_classification_metrics", "per_class", "confusion_matrix", "labels", "n"}
    assert m["macro_f1"]["standard_deviation"] == "NaN" and m["accuracy"]["standard_deviation"] == "NaN"
    assert m["macro_f1"]["value"] == pytest.approx(ref["macro_f1"])
    assert m["accuracy"]["value"] == pytest.approx(ref["accuracy"])
    assert rep["confusion_matrix"] == ref["confusion_matrix"]
    assert rep["per_class"][3]["support"] == 0
    json.dumps(rep)


def test_report_rejects_1d_probs():
    with pytest.raises(ValueError):
        ev.build_report([0, 1, 2], np.array([0.2, 0.3, 0.5]))


def test_prediction_rows():
    probs = np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]])
    rows = ev.prediction_rows([1, 2], probs)
    assert rows == [(1, 1, pytest.approx(0.7)), (2, 0, pytest.approx(0.5))]
    with pytest.raises(ValueError):
        ev.prediction_rows([1], probs)
    with pytest.raises(ValueError):
        ev.prediction_rows([1, 2, 3], np.array([0.1, 0.2, 0.7]))


# ---- post-process step
LMAP = [{"activity_id": 10 + i, "label": i} for i in range(K)]
META = {str(10 + i): {"name": f"act{i}", "intensity": "light"} for i in range(K)}
PROBS = np.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])


def prod_frame(n=3):
    return pd.DataFrame({"window_id": [f"w{i}" for i in range(n)], "subject_id": [1, 2, 3][:n],
                         "activity_id": [12, 10, 11][:n]})


def test_postprocess_contract():
    out = pp_post.build_predictions(prod_frame(), PROBS, LMAP, META)
    assert list(out.columns) == ["window_id", "subject_id", "predicted_class", "predicted_activity_id",
                                 "activity_name", "intensity", "confidence", "true_activity_id"]
    assert out["window_id"].tolist() == ["w0", "w1", "w2"] and out["window_id"].map(type).eq(str).all()
    assert out["predicted_class"].tolist() == [2, 0, 1]
    assert out["predicted_activity_id"].tolist() == [12, 10, 11]
    assert out["activity_name"].tolist() == ["act2", "act0", "act1"]
    assert out["confidence"].tolist() == pytest.approx([0.7, 0.6, 0.5])
    assert out["true_activity_id"].tolist() == [12, 10, 11]
    assert pd.api.types.is_integer_dtype(out["predicted_class"]) and pd.api.types.is_float_dtype(out["confidence"])
    plain = "0.1,0.2,0.7\n0.6,0.3,0.1\n0.2,0.5,0.3\n"
    bracket = '"[0.1, 0.2, 0.7]"\n"[0.6, 0.3, 0.1]"\n"[0.2, 0.5, 0.3]"\n'
    for text in (plain, bracket):
        got = pp_post.parse_probs(text, K)
        assert np.allclose(got, PROBS) and np.allclose(got, E.parse_probs(text, K))


def test_postprocess_row_mismatch():
    with pytest.raises(ValueError, match="rows"):
        pp_post.build_predictions(prod_frame(2), PROBS, LMAP, META)


def test_postprocess_wrong_class_count():
    with pytest.raises(ValueError, match="columns"):
        pp_post.build_predictions(prod_frame(), PROBS[:, :2], LMAP, META)


def test_postprocess_unknown_activity():
    meta = {k: v for k, v in META.items() if k != "12"}
    with pytest.raises(ValueError, match="12"):
        pp_post.build_predictions(prod_frame(), PROBS, LMAP, meta)


@pytest.mark.parametrize("col", ["window_id", "subject_id", "activity_id"])
def test_postprocess_missing_id_column(col):
    with pytest.raises(ValueError, match=col):
        pp_post.build_predictions(prod_frame().drop(columns=[col]), PROBS, LMAP, META)


def test_postprocess_wider_than_k():
    with pytest.raises(ValueError, match="exactly"):
        pp_post.parse_probs("9.9,8.8,0.1,0.2,0.7\n", K)
    with pytest.raises(ValueError, match="exactly"):
        pp_post.parse_probs("0.5,0.5\n", K)


def test_evaluate_write_outputs(tmp_path):
    probs = np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]])
    rep = ev.write_outputs([1, 2], probs, str(tmp_path / "ev"))
    assert sorted(p.name for p in (tmp_path / "ev").iterdir()) == ["evaluation.json", "test_predictions.csv"]
    assert json.loads((tmp_path / "ev/evaluation.json").read_text()) == json.loads(json.dumps(rep))
    lines = (tmp_path / "ev/test_predictions.csv").read_text().splitlines()
    assert lines[0] == "label,predicted_class,confidence" and len(lines) == 3


def test_postprocess_run(tmp_path):
    pytest.importorskip("pyarrow")
    t, i, m, o = (tmp_path / n for n in ("t", "i", "m", "o"))
    for d in (t, i, m):
        d.mkdir()
    (t / "features.csv.out").write_text("0.1,0.2,0.7\n0.6,0.3,0.1\n0.2,0.5,0.3\n")
    (i / "label_map.json").write_text(json.dumps(LMAP))
    (m / "activity_meta.json").write_text(json.dumps(META))
    prod_frame().to_parquet(i / "production.parquet")
    assert pp_post.run(str(t), str(i), str(m), str(o)) == 3
    out = pd.read_parquet(o / "predictions.parquet")
    assert list(out.columns) == ["window_id", "subject_id", "predicted_class", "predicted_activity_id",
                                 "activity_name", "intensity", "confidence", "true_activity_id"]
    (t / "second.out").write_text("x")
    with pytest.raises(ValueError, match="exactly one"):
        pp_post.run(str(t), str(i), str(m), str(o))


def test_postprocess_run_needs_single_out(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        pp_post.run(str(tmp_path), str(tmp_path), str(tmp_path), str(tmp_path / "o"))
