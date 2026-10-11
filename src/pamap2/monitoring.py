"""Pure logic for data/model/infrastructure monitoring (no AWS calls)."""
from __future__ import annotations

import json

import pandas as pd

from .config import MONITORING_DEFAULTS

JOB_NAMESPACES = {
    "training": "/aws/sagemaker/TrainingJobs",
    "transform": "/aws/sagemaker/TransformJobs",
    "processing": "/aws/sagemaker/ProcessingJobs",
}
PIPELINE_NAMESPACE = "AWS/Sagemaker/ModelBuildingPipeline"
JOB_METRICS = ("CPUUtilization", "MemoryUtilization", "DiskUtilization")
DASHBOARD_MAX_BYTES = 800_000  # CloudWatch limit is 1 MB; keep headroom
DEFAULT_DRIFT_THRESHOLD = MONITORING_DEFAULTS["monitoring_drift_threshold"]


def to_monitor_csv(df: pd.DataFrame, feature_columns) -> str:
    """Header CSV in exact feature_columns order, no label (data-quality monitor input)."""
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    return df[list(feature_columns)].to_csv(index=False)


def to_model_quality_frame(preds: pd.DataFrame, prediction_col: str = "predicted_activity_id",
                           probability_col: str = "confidence",
                           label_col: str = "true_activity_id") -> tuple[pd.DataFrame, int]:
    """Frame with prediction/probability/label (activity_id ints). Returns (frame, n_dropped_nan_rows)."""
    missing = [c for c in (prediction_col, probability_col, label_col) if c not in preds.columns]
    if missing:
        raise ValueError(f"missing columns for model-quality dataset: {missing}")
    out = pd.DataFrame({"prediction": preds[prediction_col].to_numpy(),
                        "probability": preds[probability_col].to_numpy(),
                        "label": preds[label_col].to_numpy()})
    keep = out.notna().all(axis=1)
    dropped = int((~keep).sum())
    out = out[keep].copy()
    out["prediction"] = out["prediction"].astype(int)
    out["label"] = out["label"].astype(int)
    return out.reset_index(drop=True), dropped


def parse_violations(raw) -> dict:
    """constraint_violations.json (bytes/str/dict/None) -> {violations, count, status}.

    Each violation is {feature, check_type, description}. None (file absent) means no violations.
    """
    if raw is None:
        data = {}
    elif isinstance(raw, dict):
        data = raw
    else:
        data = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
    items = [{"feature": v.get("feature_name"), "check_type": v.get("constraint_check_type"),
              "description": v.get("description")} for v in data.get("violations", [])]
    return {"violations": items, "count": len(items), "status": "ok" if not items else "violations"}


def quality_drift(baseline_f1: float, production_f1: float,
                  threshold: float = DEFAULT_DRIFT_THRESHOLD) -> dict:
    """`degraded` is True when production macro-F1 falls more than `threshold` below baseline."""
    drop = float(baseline_f1) - float(production_f1)
    return {"baseline_macro_f1": float(baseline_f1), "production_macro_f1": float(production_f1),
            "drop": drop, "threshold": threshold, "degraded": bool(drop > threshold)}


def alarm_spec(job_type: str, metric: str, threshold: float, prefix: str = "pamap2-har",
               namespace: str | None = None, comparison: str = "GreaterThanThreshold",
               period: int = 300, evaluation_periods: int = 1, actions=None) -> dict:
    """put_metric_alarm kwargs.

    Job types (training|transform|processing) use a Metrics Insights query over the whole job namespace, so
    every current and future job is watched. 'custom' alarms a metric in `namespace` and ignores missing data
    so the last evaluated state persists between manual runs.
    """
    if job_type == "custom":
        if not namespace:
            raise ValueError("custom alarms need a namespace")
        spec = {"AlarmName": f"{prefix}-{metric}", "Namespace": namespace, "MetricName": metric,
                "Statistic": "Maximum", "Period": period, "TreatMissingData": "ignore"}
    elif job_type in JOB_NAMESPACES:
        if metric not in JOB_METRICS:
            raise ValueError(f"unknown metric {metric!r} for job type {job_type!r}; use {JOB_METRICS}")
        query = f'SELECT MAX({metric}) FROM "{JOB_NAMESPACES[job_type]}"'
        spec = {"AlarmName": f"{prefix}-{job_type}-{metric}".lower(),
                "Metrics": [{"Id": "q1", "Expression": query, "Period": period, "ReturnData": True}],
                "TreatMissingData": "notBreaching"}
    else:
        raise ValueError(f"unknown job type {job_type!r}; use {sorted(JOB_NAMESPACES)} or 'custom'")
    spec.update({"EvaluationPeriods": evaluation_periods, "Threshold": float(threshold),
                 "ComparisonOperator": comparison})
    if actions:
        spec["AlarmActions"] = list(actions)
    return spec


def dashboard_body(region: str, job_metrics: list[dict], custom_metrics: list[str], alarm_arns: list[str],
                   namespace: str, links: dict[str, str] | None = None,
                   pipeline_name: str | None = None) -> str:
    """CloudWatch dashboard JSON. job_metrics: [{title, namespace, metric}] drawn as SEARCH expressions."""
    md = "## pamap2-har ML monitoring\n" + "\n".join(f"- {k}: `{v}`" for k, v in (links or {}).items())
    y = 4
    widgets = [{"type": "text", "x": 0, "y": 0, "width": 24, "height": 4, "properties": {"markdown": md}}]
    for i, jm in enumerate(job_metrics):
        expr = "SEARCH('{" + jm["namespace"] + ",Host} MetricName=\"" + jm["metric"] + "\"', 'Maximum', 300)"
        widgets.append({"type": "metric", "x": (i % 2) * 12, "y": y + (i // 2) * 6, "width": 12, "height": 6,
                        "properties": {"title": jm["title"], "region": region, "period": 300, "view": "timeSeries",
                                       "metrics": [[{"expression": expr, "id": f"e{i}", "label": jm["metric"]}]]}})
    y += ((len(job_metrics) + 1) // 2) * 6
    if custom_metrics:
        widgets.append({"type": "metric", "x": 0, "y": y, "width": 12, "height": 6,
                        "properties": {"title": "Monitoring metrics", "region": region, "stat": "Maximum",
                                       "period": 300, "view": "timeSeries",
                                       "metrics": [[namespace, m] for m in custom_metrics]}})
    if alarm_arns:
        widgets.append({"type": "alarm", "x": 12, "y": y, "width": 12, "height": 6,
                        "properties": {"title": "Alarm status", "alarms": list(alarm_arns)}})
    if pipeline_name:
        y += 6 if (custom_metrics or alarm_arns) else 0
        widgets.append({"type": "metric", "x": 0, "y": y, "width": 12, "height": 6,
                        "properties": {"title": "Pipeline executions", "region": region, "stat": "Sum",
                                       "period": 300, "view": "timeSeries",
                                       "metrics": [[PIPELINE_NAMESPACE, m, "PipelineName", pipeline_name]
                                                   for m in ("ExecutionStarted", "ExecutionSucceeded",
                                                             "ExecutionFailed")]}})
    body = json.dumps({"widgets": widgets})
    if len(body.encode()) > DASHBOARD_MAX_BYTES:
        raise ValueError(f"dashboard body {len(body.encode())} bytes exceeds {DASHBOARD_MAX_BYTES}")
    return body


def pipeline_status(executions, steps) -> dict:
    """Summarise pipeline executions (list_pipeline_executions items) and the latest execution's steps.

    `failed` is 1 when the newest execution failed. Empty `executions` gives executions=0 and no metrics.
    """
    execs = sorted(executions, key=lambda e: e.get("StartTime") or 0, reverse=True)
    if not execs:
        return {"executions": 0, "latest_arn": None, "latest_status": None, "failed": 0, "failed_steps": [],
                "duration_s": None, "status_counts": {}, "metrics": {}}
    latest = execs[0]
    counts: dict = {}
    for e in execs:
        counts[e["PipelineExecutionStatus"]] = counts.get(e["PipelineExecutionStatus"], 0) + 1
    failed_steps = [{"step": st["StepName"], "reason": st.get("FailureReason", "")}
                    for st in steps if st.get("StepStatus") == "Failed"]
    start, end = latest.get("StartTime"), latest.get("LastModifiedTime")
    duration = float((end - start).total_seconds()) if start and end else None
    failed = int(latest["PipelineExecutionStatus"] == "Failed")
    return {"executions": len(execs), "latest_arn": latest.get("PipelineExecutionArn"),
            "latest_status": latest["PipelineExecutionStatus"], "failed": failed, "failed_steps": failed_steps,
            "duration_s": duration, "status_counts": counts,
            "metrics": {"PipelineExecutionFailed": failed, "PipelineFailedSteps": len(failed_steps)}}


def pipeline_alarm_spec(pipeline_name: str, metric: str = "ExecutionFailed", threshold: float = 0,
                        prefix: str = "pamap2-har", period: int = 300) -> dict:
    """put_metric_alarm kwargs for a native SageMaker pipeline metric (dimension PipelineName)."""
    return {"AlarmName": f"{prefix}-pipeline-{metric}".lower(), "Namespace": PIPELINE_NAMESPACE,
            "MetricName": metric, "Dimensions": [{"Name": "PipelineName", "Value": pipeline_name}],
            "Statistic": "Sum", "Period": period, "EvaluationPeriods": 1, "Threshold": float(threshold),
            "ComparisonOperator": "GreaterThanThreshold", "TreatMissingData": "notBreaching"}


def report_index(entries: dict) -> dict:
    """{name: s3_uri}; entries with no URI are listed under `missing`."""
    return {"reports": {k: v for k, v in entries.items() if v},
            "missing": sorted(k for k, v in entries.items() if not v)}
