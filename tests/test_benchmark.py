import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pamap2.benchmark import BENCHMARK_FEATURES, fit_benchmark, predict_benchmark, predict_majority


def frame(n=60):
    rng = np.random.default_rng(0)
    lab = np.repeat([0, 1, 2], n // 3)
    return pd.DataFrame({"chest_acc16_mag_std": lab * 2.0 + rng.normal(0, .1, n),
                         "hr_mean": 70 + lab * 20.0 + rng.normal(0, 1, n), "label": lab})


def test_fit_predict():
    df = frame()
    m = fit_benchmark(df)
    assert m["tree"].get_depth() <= 3
    assert (predict_benchmark(m, df) == df.label).mean() > 0.9


def test_nan_hr_filled_with_train_median():
    df = frame()
    m = fit_benchmark(df)
    assert m["medians"]["hr_mean"] == pytest.approx(df.hr_mean.median())
    test = df.copy()
    test["hr_mean"] = np.nan
    assert len(predict_benchmark(m, test)) == len(test)
    tr = df.copy()
    tr.loc[:5, "hr_mean"] = np.nan
    fit_benchmark(tr)


def test_majority_and_columns():
    df = frame()
    df.loc[df.index < 10, "label"] = 2
    m = fit_benchmark(df)
    assert (predict_majority(m, df) == m["majority"]).all()
    with pytest.raises(ValueError, match="hr_mean"):
        predict_benchmark(m, df.drop(columns=["hr_mean"]))
    assert BENCHMARK_FEATURES == ["chest_acc16_mag_std", "hr_mean"]


def test_imputation_equals_median_and_edge_cases():
    df = frame()
    m = fit_benchmark(df)
    nan_rows = df.copy()
    nan_rows["hr_mean"] = np.nan
    med_rows = df.copy()
    med_rows["hr_mean"] = m["medians"]["hr_mean"]
    assert (predict_benchmark(m, nan_rows) == predict_benchmark(m, med_rows)).all()
    allnan = df.copy()
    allnan["hr_mean"] = np.nan
    assert fit_benchmark(allnan)["medians"]["hr_mean"] == 0.0
    with pytest.raises(ValueError):
        fit_benchmark(df.iloc[0:0])
