import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pamap2.config import build_config, with_defaults
from src.pamap2.evaluation import (check_feature_columns, classification_metrics, compare,
                                   load_label_map, parse_probs, probs_to_labels)


def test_happy_path():
    m = classification_metrics([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0], [0, 1, 2])
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["per_class"][1]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"][1]["recall"] == 1.0
    assert np.array(m["confusion_matrix"]).shape == (3, 3)
    assert np.array(m["confusion_matrix"]).sum() == 6
    assert 0 < m["macro_f1"] < 1


def test_class_absent_in_predictions():
    m = classification_metrics([0, 1, 1], [0, 0, 0], [0, 1])
    assert m["per_class"][1]["precision"] == 0.0 and m["per_class"][1]["recall"] == 0.0
    assert len(m["confusion_matrix"]) == 2


def test_class_absent_in_test():
    m = classification_metrics([0, 0, 1, 1], [0, 0, 1, 1], [0, 1, 2])
    assert m["per_class"][2]["support"] == 0
    assert m["macro_f1"] == 1.0  # class 2 excluded from average
    assert len(m["confusion_matrix"]) == 3


def test_probs_to_labels():
    out = probs_to_labels([[0.1, 0.9], [0.8, 0.2]], 2)
    assert out.tolist() == [1, 0]
    with pytest.raises(ValueError):
        probs_to_labels([[0.1, 0.2, 0.7]], 2)


def test_label_map():
    assert load_label_map([{"activity_id": 4, "label": 0}, {"activity_id": 12, "label": 1}]) == {0: 4, 1: 12}


def test_compare():
    c = compare({"macro_f1": 0.4, "accuracy": 0.5}, {"macro_f1": 0.8, "accuracy": 0.9})
    assert c["better"] is True
    assert c["table"].loc[0, "delta"] == pytest.approx(0.4)
    assert compare({"macro_f1": 0.9, "accuracy": 0.9}, {"macro_f1": 0.8, "accuracy": 0.9})["better"] is False


def test_feature_mismatch():
    with pytest.raises(ValueError, match="b"):
        check_feature_columns(pd.DataFrame({"a": [1]}), ["a", "b"])
    check_feature_columns(pd.DataFrame({"a": [1]}), ["a"])


def test_config_defaults():
    cfg = with_defaults({"s3_prefixes": {}})
    assert cfg["model_package_group"] == "pamap2-har-xgboost" and cfg["xgboost_version"] == "1.7-1"
    assert build_config("1")["model_package_group"] == "pamap2-har-xgboost"


def test_unknown_and_nan_labels():
    with pytest.raises(ValueError):
        classification_metrics([0, 5], [0, 0], [0, 1])
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0, 7], [0, 1])
    with pytest.raises(ValueError):
        classification_metrics([0, np.nan], [0, 1], [0, 1])


def test_parse_probs():
    joined = "1.5,2.5,0.1,0.9" + chr(10) + "3.5,,0.8,0.2" + chr(10)
    out = parse_probs(joined.encode(), 2)
    assert out.shape == (2, 2) and out[1].tolist() == [0.8, 0.2]
    plain = parse_probs("0.1,0.2,0.7" + chr(10) + "0.3,0.3,0.4" + chr(10), 3)
    assert plain.shape == (2, 3)
    with pytest.raises(ValueError):
        parse_probs("0.5,0.5" + chr(10), 3)
    bracketed = parse_probs(("[0.1, 0.2, 0.7]" + chr(10) + "[0.3, 0.3, 0.4]" + chr(10)).encode(), 3)
    assert bracketed.shape == (2, 3) and bracketed[0, 2] == 0.7
    joined_br = parse_probs("1.5,2.5," + chr(34) + "[0.4, 0.6]" + chr(34) + chr(10), 2)
    assert joined_br.tolist() == [[0.4, 0.6]]
