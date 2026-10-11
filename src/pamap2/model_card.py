"""Pure helpers for the SageMaker Model Card (no sagemaker/boto3 import, testable without AWS).

``card_text()`` holds the static narrative sections; ``eval_metric_groups()`` turns the evaluation
and benchmark reports into plain metric dicts that the notebook wraps in ``sagemaker.model_card``
objects (``{"name", "type", "value", ...}`` with ``type`` in number / boolean / matrix).
"""
from __future__ import annotations

MAX_TEXT = 1000   # self-imposed cap per text field (service limit is 2048)


def card_text() -> dict:
    """Static card strings, sourced from the README and the project brief (constraints and risks)."""
    return {
        "model_description": (
            "Multiclass activity classifier for wearable-sensor windows (512 samples / 5.12 s, 50% overlap) "
            "from the PAMAP2 dataset: statistical and frequency-domain features from the hand/chest/ankle IMU accelerometer (16g) "
            "and gyroscope plus heart rate. Built-in SageMaker XGBoost (multi:softprob), fixed hyperparameters, early "
            "stopping on a validation split. Each predicted activity maps to a light / moderate / vigorous "
            "intensity category downstream."),
        "purpose_of_model": (
            "Recognise which physical activity a person is performing from short windows of wearable "
            "sensor data, as the core model of an end-to-end MLOps demonstration (AAI-540 capstone)."),
        "intended_uses": (
            "Offline batch scoring of recorded wearable sessions for activity summaries and intensity "
            "estimates, and as a documented example of a registered, approved, monitored model. Inputs must "
            "be feature rows in the training column order. Not a real-time service: no endpoint is deployed."),
        "factors_affecting_model_efficiency": (
            "Trained on 9 subjects only, so it may not generalise to people with different age, fitness, "
            "body type or sensor placement. Heart rate is sparse (about 9 Hz against 100 Hz IMU) and is "
            "interpolated. Windows labelled as transient / other (activity 0) are dropped, so that class is not predicted. Uneven class sizes and similar activities "
            "(for example walking vs Nordic walking, ascending vs descending stairs) lower per-class scores."),
        "explanations_for_risk_rating": (
            "Medium: the inputs include heart rate, which is health-adjacent data, and the training "
            "population is a narrow 9-subject demographic, so errors could be systematically worse for "
            "people unlike the subjects. Outputs are activity labels and intensity categories, not "
            "diagnoses, and there is no automated decision-making on top of them."),
        "business_problem": (
            "Raw wearable output (acceleration and rotation at about 100 Hz, heart rate at about 9 Hz) is only "
            "useful once interpreted into what a person is doing and how hard they are working. This model "
            "turns sensor windows into an activity label that supports an exertion-intensity estimate."),
        "training_observations": (
            "Chronological 40/10/10/40 train / validation / test / production split per subject and "
            "activity, with a one-window gap between blocks. Unweighted multi:softprob objective with early "
            "stopping on validation (hyperparameters are listed in the training details). SageMaker Debugger rules (LossNotDecreasing, "
            "Overfit, ClassImbalance) were attached; see the Debugger artifacts path in the "
            "additional information section."),
        "evaluation_observation": (
            "Test scores are optimistic: test windows come from subjects that also appear in training "
            "(chronological split, not leave-one-subject-out), so they overstate performance on a new "
            "person. Macro-F1 is the headline metric because of class imbalance; a depth-3 decision-tree "
            "benchmark is reported alongside for comparison."),
        "ethical_considerations": (
            "Small 9-subject demographic: performance on other populations is unknown and may be worse. "
            "Heart-rate and movement data can reveal health or behaviour information; PAMAP2 is public "
            "(CC BY 4.0) and de-identified, but any new data should be handled with consent and privacy "
            "controls. Do not use the model for medical, clinical or safety decisions."),
        "caveats_and_recommendations": (
            "Scores come from seen subjects and are optimistic; validate with leave-one-subject-out or on "
            "new people before relying on it. Not for medical or safety decisions. Re-evaluate and "
            "re-approve after any retraining, and watch the data-quality and model-quality monitors for "
            "drift. Class set is limited to the activities present in the training split."),
    }


def _num(x) -> float:
    return float(x)


def eval_metric_groups(eval_json: dict, bench: dict, names: dict) -> dict:
    """Build the three evaluation metric groups.

    ``eval_json``: ``xgboost_eval.json`` (JSON keys of ``per_class`` are strings).
    ``bench``: ``benchmark.json["test_tree"]``.
    ``names``: class index -> activity name (int or str keys accepted).
    Returns ``{"headline": [...], "per_class": [...], "confusion_matrix": [...]}``; every metric is a
    plain dict with native ``float`` / ``int`` / ``bool`` / ``list`` values.
    """
    nm = {int(k): str(v) for k, v in names.items()}
    macro, bench_macro = _num(eval_json["macro_f1"]), _num(bench["macro_f1"])
    headline = [
        {"name": "Test macro-F1", "type": "number", "value": macro},
        {"name": "Test accuracy", "type": "number", "value": _num(eval_json["accuracy"])},
        {"name": "Test windows", "type": "number", "value": int(eval_json["n"])},
        {"name": "Benchmark macro-F1", "type": "number", "value": bench_macro,
         "notes": "Depth-3 decision tree on two features"},
        {"name": "Benchmark accuracy", "type": "number", "value": _num(bench["accuracy"])},
        {"name": "Macro-F1 delta vs benchmark", "type": "number", "value": macro - bench_macro},
        {"name": "Better than benchmark", "type": "boolean",
         "value": bool(eval_json.get("better_than_benchmark", macro > bench_macro))},
    ]
    per_class = []
    for k in sorted(eval_json["per_class"], key=int):
        row, label = eval_json["per_class"][k], nm[int(k)]
        for stat in ("precision", "recall", "f1"):
            per_class.append({"name": f"{label} {stat}", "type": "number", "value": _num(row[stat])})
        per_class.append({"name": f"{label} support", "type": "number", "value": int(row["support"])})
    labels = [nm[int(i)] for i in eval_json.get("labels", sorted(nm))]
    matrix = [[int(v) for v in r] for r in eval_json["confusion_matrix"]]
    confusion = [{"name": "Test confusion matrix", "type": "matrix", "value": matrix,
                  "x_axis_name": labels, "y_axis_name": labels,
                  "notes": "Rows = true activity, columns = predicted activity"}]
    return {"headline": headline, "per_class": per_class, "confusion_matrix": confusion}
