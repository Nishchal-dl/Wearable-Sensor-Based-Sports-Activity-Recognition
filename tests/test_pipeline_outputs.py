import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pamap2.model_card import eval_metric_groups
from src.pamap2.pipeline_outputs import eval_report, step_resources, training_report

ARN = "arn:aws:sagemaker:us-east-1:1:{}/{}"


def step(name, status, meta=None, reason=None):
    s = {"StepName": name, "StepStatus": status}
    if meta:
        s["Metadata"] = meta
    if reason:
        s["FailureReason"] = reason
    return s


SUCCEEDED = [
    step("PreprocessSplits", "Succeeded", {"ProcessingJob": {"Arn": ARN.format("processing-job", "pipelines-x-Prep-1")}}),
    step("TrainModel", "Succeeded", {"TrainingJob": {"Arn": ARN.format("training-job", "pipelines-x-Train-2")}}),
    step("EvaluateModel", "Succeeded", {"ProcessingJob": {"Arn": ARN.format("processing-job", "pipelines-x-Eval-3")}}),
    step("RegisterModel-RegisterModel", "Succeeded", {"RegisterModel": {"Arn": ARN.format("model-package/g", "7")}}),
    step("CreateModel-CreateModel", "Succeeded", {"Model": {"Arn": ARN.format("model", "m")}}),
]


def test_step_resources_succeeded():
    r = step_resources(SUCCEEDED)
    assert r["training_job"] == "pipelines-x-Train-2"
    assert r["processing_jobs"] == {"PreprocessSplits": "pipelines-x-Prep-1", "EvaluateModel": "pipelines-x-Eval-3"}
    assert r["model_package_arn"].endswith("model-package/g/7")
    assert r["failed"] == [] and set(r["statuses"].values()) == {"Succeeded"}


def test_step_resources_gate_failed():
    steps = SUCCEEDED[:3] + [step("CheckMacroF1", "Succeeded"),
                              step("MacroF1BelowThreshold", "Failed", reason="below threshold")]
    r = step_resources(steps)
    assert r["model_package_arn"] is None
    assert r["failed"] == [{"step": "MacroF1BelowThreshold", "reason": "below threshold"}]
    assert "EvaluateModel" in r["processing_jobs"]


JOB = {"TrainingJobName": "job1", "ModelArtifacts": {"S3ModelArtifacts": "s3://b/m/model.tar.gz"},
       "FinalMetricDataList": [{"MetricName": "validation:mlogloss", "Value": 0.1},
                               {"MetricName": "train:merror", "Value": 0.02}],
       "DebugHookConfig": {"S3OutputPath": "s3://b/models/pipeline/xgboost/debugger/"}}


def test_training_report_keys():
    r = training_report(JOB, "img", "arn:pkg", "arn:exec")
    assert {"training_job", "model_data", "image_uri", "model_package_arn", "final_metrics",
            "debugger_artifacts"} <= set(r)
    assert r["final_metrics"] == {"validation:mlogloss": 0.1, "train:merror": 0.02}
    assert r["debugger_artifacts"] == "s3://b/models/pipeline/xgboost/debugger/job1/debug-output"
    no_hook = training_report({k: v for k, v in JOB.items() if k != "DebugHookConfig"}, "img", "p")
    assert no_hook["debugger_artifacts"] is None
    json.dumps(r)


EVALUATION = {"multiclass_classification_metrics": {"macro_f1": {"value": 0.9, "standard_deviation": "NaN"},
                                                     "accuracy": {"value": 0.95, "standard_deviation": "NaN"}},
              "per_class": {0: {"precision": 0.8, "recall": 0.7, "f1": 0.75, "support": 60},
                            1: {"precision": 1.0, "recall": 0.9, "f1": 0.95, "support": 40}},
              "confusion_matrix": [[42, 18], [4, 36]], "labels": [0, 1], "n": 100}


def test_eval_report_feeds_model_card():
    bench = {"macro_f1": 0.5, "accuracy": 0.6}
    r = eval_report(EVALUATION, {0: 10, 1: 11}, bench, "arn:pkg", pipeline_execution_arn="arn:exec",
                    predictions_uri="s3://b/p")
    assert r["better_than_benchmark"] is True and r["benchmark_macro_f1"] == 0.5
    assert r["label_map"] == {"0": 10, "1": 11} and r["predictions_uri"] == "s3://b/p"
    r = json.loads(json.dumps(r))
    groups = eval_metric_groups(r, bench, {0: "lying", 1: "walking"})
    head = {m["name"]: m["value"] for m in groups["headline"]}
    assert head["Test macro-F1"] == 0.9 and head["Test windows"] == 100


def test_eval_report_rejects_missing_metrics():
    with pytest.raises(ValueError):
        eval_report({"n": 1}, {0: 10}, {"macro_f1": 0.5}, "arn")
    bad = {**EVALUATION, "multiclass_classification_metrics": {"macro_f1": {"value": 0.9}}}
    with pytest.raises(ValueError):
        eval_report(bad, {0: 10, 1: 11}, {"macro_f1": 0.5}, "arn")
