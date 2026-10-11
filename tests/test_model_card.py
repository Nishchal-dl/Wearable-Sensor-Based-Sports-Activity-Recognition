import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pamap2.model_card import MAX_TEXT, card_text, eval_metric_groups

EVAL = {
    "macro_f1": 0.9, "accuracy": 0.95, "n": 100, "labels": [0, 1],
    "per_class": {"0": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "support": 60},
                  "1": {"precision": 1.0, "recall": 0.9, "f1": 0.95, "support": 40}},
    "confusion_matrix": [[42, 18], [4, 36]], "better_than_benchmark": True,
}
BENCH = {"macro_f1": 0.5, "accuracy": 0.6}
NAMES = {0: "lying", 1: "walking"}


def groups():
    return eval_metric_groups(EVAL, BENCH, NAMES)


def test_group_shapes_and_types():
    g = groups()
    assert set(g) == {"headline", "per_class", "confusion_matrix"}
    head = {m["name"]: m for m in g["headline"]}
    assert head["Test macro-F1"]["value"] == 0.9
    assert abs(head["Macro-F1 delta vs benchmark"]["value"] - 0.4) < 1e-12
    assert head["Better than benchmark"]["type"] == "boolean" and head["Better than benchmark"]["value"] is True
    assert type(head["Test windows"]["value"]) is int
    assert all(m["type"] in ("number", "boolean") for m in g["headline"] + g["per_class"])


def test_per_class_string_keys_and_names():
    pc = {m["name"]: m["value"] for m in groups()["per_class"]}
    assert len(pc) == 8
    assert pc["lying recall"] == 0.7 and pc["walking support"] == 40


def test_confusion_matrix_axes():
    (cm,) = groups()["confusion_matrix"]
    assert cm["type"] == "matrix" and cm["value"] == [[42, 18], [4, 36]]
    assert cm["x_axis_name"] == cm["y_axis_name"] == ["lying", "walking"]


def test_string_keyed_names_and_native_types():
    g = eval_metric_groups(EVAL, BENCH, {"0": "lying", "1": "walking"})
    assert all(type(m["value"]) in (float, int, bool) for m in g["headline"] + g["per_class"])


def test_card_text_limits():
    t = card_text()
    assert t and all(0 < len(v) <= MAX_TEXT for v in t.values())
    assert "Medium" in t["explanations_for_risk_rating"]
    assert "medical" in t["caveats_and_recommendations"]
