import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pamap2.config import MONITORING_DEFAULTS, build_config, with_defaults
from src.pamap2.monitoring import (PIPELINE_NAMESPACE, alarm_spec, dashboard_body, parse_violations,
                                   pipeline_alarm_spec, pipeline_status, quality_drift, report_index,
                                   to_model_quality_frame, to_monitor_csv)


def test_monitor_csv_order_and_header():
    df = pd.DataFrame({"b": [1], "a": [2], "label": [3]})
    assert to_monitor_csv(df, ["a", "b"]).splitlines() == ["a,b", "2,1"]


def test_monitor_csv_missing():
    with pytest.raises(ValueError, match="zz"):
        to_monitor_csv(pd.DataFrame({"a": [1]}), ["a", "zz"])


def test_model_quality_frame_and_nan_drop():
    p = pd.DataFrame({"predicted_activity_id": [1, 2, 3], "confidence": [.9, .8, .7],
                      "true_activity_id": [1, None, 3.0]})
    f, dropped = to_model_quality_frame(p)
    assert list(f.columns) == ["prediction", "probability", "label"]
    assert dropped == 1 and len(f) == 2
    assert f["label"].tolist() == [1, 3] and str(f["label"].dtype).startswith("int")


def test_model_quality_missing_label():
    with pytest.raises(ValueError, match="true_activity_id"):
        to_model_quality_frame(pd.DataFrame({"predicted_activity_id": [1], "confidence": [.5]}))


def test_parse_violations():
    raw = json.dumps({"violations": [{"feature_name": "f1", "constraint_check_type": "data_type_check",
                                       "description": "bad"}]})
    r = parse_violations(raw)
    assert r["count"] == 1 and r["violations"][0] == {"feature": "f1", "check_type": "data_type_check",
                                                      "description": "bad"}
    assert r["status"] == "violations"


def test_missing_and_empty_violations():
    assert parse_violations(None)["count"] == 0
    r = parse_violations(b'{"violations": []}')
    assert r["count"] == 0 and r["status"] == "ok"


def test_quality_drift():
    assert quality_drift(0.9, 0.80)["degraded"] is True
    assert quality_drift(0.9, 0.87)["degraded"] is False
    assert quality_drift(0.9, 0.80, threshold=0.2)["degraded"] is False


def test_alarm_spec():
    a = alarm_spec("training", "CPUUtilization", 90)
    assert a["AlarmName"] == "pamap2-har-training-cpuutilization"
    assert a["Metrics"][0]["Expression"] == 'SELECT MAX(CPUUtilization) FROM "/aws/sagemaker/TrainingJobs"'
    assert a["Threshold"] == 90.0 and "AlarmActions" not in a and "Namespace" not in a
    c = alarm_spec("custom", "JobFailed", 0, namespace="ns")
    assert c["AlarmName"] == "pamap2-har-JobFailed" and c["TreatMissingData"] == "ignore"
    with pytest.raises(ValueError):
        alarm_spec("endpoint", "CPUUtilization", 90)
    with pytest.raises(ValueError):
        alarm_spec("training", "Bogus", 90)


def test_dashboard_body():
    jm = [{"title": "t", "namespace": "/aws/sagemaker/TrainingJobs", "metric": "CPUUtilization"}] * 3
    body = json.loads(dashboard_body("us-east-1", jm, ["ViolationCount"], ["arn:a"], "ns", {"r": "s3://x"}))
    ws = body["widgets"]
    assert [w["type"] for w in ws].count("metric") == 4 and "alarm" in [w["type"] for w in ws]
    assert "SEARCH(" in ws[1]["properties"]["metrics"][0][0]["expression"]
    assert all(w["x"] + w["width"] <= 24 for w in ws)
    for i, a in enumerate(ws):   # no rectangle overlap
        for b in ws[i + 1:]:
            assert (a["x"] + a["width"] <= b["x"] or b["x"] + b["width"] <= a["x"]
                    or a["y"] + a["height"] <= b["y"] or b["y"] + b["height"] <= a["y"])
    ys = [w["y"] for w in ws]
    assert ys == sorted(ys)
    huge = [{"title": "x" * 1000, "namespace": "n", "metric": "m"}] * 2000
    with pytest.raises(ValueError):
        dashboard_body("us-east-1", huge, [], [], "ns")


def test_report_index():
    r = report_index({"a": "s3://x/a", "b": None})
    assert r["reports"] == {"a": "s3://x/a"} and r["missing"] == ["b"]


def test_config_defaults():
    c = with_defaults({"bucket": "b", "s3_prefixes": {}})
    b = build_config("1")
    for k, v in MONITORING_DEFAULTS.items():
        assert c[k] == v and b[k] == v
    assert len(MONITORING_DEFAULTS) == 4


def test_with_defaults_preserves_user_values():
    c = with_defaults({"bucket": "b", "s3_prefixes": {}, "monitoring_dashboard": "mine"})
    assert c["monitoring_dashboard"] == "mine"


def test_parse_violations_dict():
    r = parse_violations({"violations": [{"feature_name": "a"}]})
    assert r["count"] == 1 and r["violations"][0]["feature"] == "a"


def test_quality_drift_exact_threshold():
    assert quality_drift(0.5, 0.25, threshold=0.25)["degraded"] is False


def test_model_quality_nan_prediction_probability():
    p = pd.DataFrame({"predicted_activity_id": [1, None, 3], "confidence": [.9, .8, None],
                      "true_activity_id": [1, 2, 3]})
    f, dropped = to_model_quality_frame(p)
    assert dropped == 2 and f["prediction"].tolist() == [1]


T0 = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)


def _exec(arn, status, minutes_ago, minutes=10):
    start = T0 - timedelta(minutes=minutes_ago)
    return {"PipelineExecutionArn": arn, "PipelineExecutionStatus": status, "StartTime": start,
            "LastModifiedTime": start + timedelta(minutes=minutes)}


def test_pipeline_status_succeeded():
    st = pipeline_status([_exec("old", "Failed", 120), _exec("new", "Succeeded", 5)],
                         [{"StepName": "TrainModel", "StepStatus": "Succeeded"}])
    assert st["executions"] == 2 and st["latest_arn"] == "new" and st["latest_status"] == "Succeeded"
    assert st["failed"] == 0 and st["failed_steps"] == [] and st["duration_s"] == 600.0
    assert st["status_counts"] == {"Failed": 1, "Succeeded": 1}
    assert st["metrics"] == {"PipelineExecutionFailed": 0, "PipelineFailedSteps": 0}


def test_pipeline_status_failed_steps():
    steps = [{"StepName": "EvaluateModel", "StepStatus": "Succeeded"},
             {"StepName": "MacroF1BelowThreshold", "StepStatus": "Failed", "FailureReason": "below"}]
    st = pipeline_status([_exec("e", "Failed", 1)], steps)
    assert st["failed"] == 1 and st["failed_steps"] == [{"step": "MacroF1BelowThreshold", "reason": "below"}]
    assert st["metrics"] == {"PipelineExecutionFailed": 1, "PipelineFailedSteps": 1}


def test_pipeline_status_no_executions():
    st = pipeline_status([], [])
    assert st["executions"] == 0 and st["metrics"] == {} and st["failed"] == 0 and st["failed_steps"] == []


def test_pipeline_alarm_spec():
    a = pipeline_alarm_spec("pipe")
    assert a["AlarmName"] == "pamap2-har-pipeline-executionfailed"
    assert a["Namespace"] == PIPELINE_NAMESPACE == "AWS/Sagemaker/ModelBuildingPipeline"
    assert a["Dimensions"] == [{"Name": "PipelineName", "Value": "pipe"}]
    assert a["Statistic"] == "Sum" and a["TreatMissingData"] == "notBreaching" and a["Threshold"] == 0.0
    assert pipeline_alarm_spec("pipe", prefix="x", metric="ExecutionStopped")["AlarmName"] == "x-pipeline-executionstopped"


def test_dashboard_body_pipeline_widget():
    jm = [{"title": "t", "namespace": "/aws/sagemaker/TrainingJobs", "metric": "CPUUtilization"}] * 3
    for custom, alarms in ((["V"], ["arn:a"]), ([], [])):
        base = json.loads(dashboard_body("us-east-1", jm, custom, alarms, "ns"))["widgets"]
        ws = json.loads(dashboard_body("us-east-1", jm, custom, alarms, "ns", pipeline_name="pipe"))["widgets"]
        assert len(ws) == len(base) + 1 and ws[:-1] == base
        w = ws[-1]
        assert w["properties"]["title"] == "Pipeline executions" and w["properties"]["stat"] == "Sum"
        assert [m[1] for m in w["properties"]["metrics"]] == ["ExecutionStarted", "ExecutionSucceeded", "ExecutionFailed"]
        assert all(m[0] == PIPELINE_NAMESPACE and m[3] == "pipe" for m in w["properties"]["metrics"])
        assert all(x["x"] + x["width"] <= 24 for x in ws)
        for i, a in enumerate(ws):
            for b in ws[i + 1:]:
                assert (a["x"] + a["width"] <= b["x"] or b["x"] + b["width"] <= a["x"]
                        or a["y"] + a["height"] <= b["y"] or b["y"] + b["height"] <= a["y"])
        assert [x["y"] for x in ws] == sorted(x["y"] for x in ws)
