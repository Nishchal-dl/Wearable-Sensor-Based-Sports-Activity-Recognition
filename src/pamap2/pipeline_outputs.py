"""Pure helpers that turn SageMaker Pipeline execution data into the stable report contracts (no AWS calls)."""
from __future__ import annotations


def _job_name(arn: str) -> str:
    return arn.rsplit("/", 1)[-1]


def step_resources(steps) -> dict:
    """Summarise `list_pipeline_execution_steps` items.

    Returns statuses {step: status}, the training job name, processing job names per step, the registered
    model package ARN (found by the `RegisterModel` metadata key, whatever the ModelStep sub-step is called)
    and the failed steps with their reasons.
    """
    out = {"statuses": {}, "training_job": None, "processing_jobs": {}, "model_package_arn": None, "failed": []}
    for s in steps:
        name, status = s["StepName"], s.get("StepStatus")
        out["statuses"][name] = status
        meta = s.get("Metadata") or {}
        if "TrainingJob" in meta:
            out["training_job"] = _job_name(meta["TrainingJob"]["Arn"])
        if "ProcessingJob" in meta:
            out["processing_jobs"][name] = _job_name(meta["ProcessingJob"]["Arn"])
        if "RegisterModel" in meta:
            out["model_package_arn"] = meta["RegisterModel"]["Arn"]
        if status == "Failed":
            out["failed"].append({"step": name, "reason": s.get("FailureReason", "")})
    return out


def training_report(job_desc: dict, image_uri: str, model_package_arn: str, execution_arn: str | None = None) -> dict:
    """`xgboost_training.json` from a describe_training_job response."""
    job = job_desc["TrainingJobName"]
    hook = job_desc.get("DebugHookConfig") or {}
    base = hook.get("S3OutputPath")
    return {
        "training_job": job,
        "model_data": job_desc["ModelArtifacts"]["S3ModelArtifacts"],
        "image_uri": image_uri,
        "model_package_arn": model_package_arn,
        "final_metrics": {m["MetricName"]: m["Value"] for m in job_desc.get("FinalMetricDataList", [])},
        "debugger_artifacts": f"{base.rstrip('/')}/{job}/debug-output" if base else None,
        "pipeline_execution_arn": execution_arn,
    }


def eval_report(evaluation: dict, label_map: dict, bench: dict, model_package_arn: str, **pointers) -> dict:
    """`xgboost_eval.json` from the evaluate step's evaluation.json.

    `label_map`: class index -> activity id. `bench`: benchmark.json["test_tree"]. `pointers`: extra keys
    (execution ARN, S3 URIs) stored alongside.
    """
    try:
        m = evaluation["multiclass_classification_metrics"]
        macro_f1, accuracy = float(m["macro_f1"]["value"]), float(m["accuracy"]["value"])
    except (KeyError, TypeError) as e:
        raise ValueError(f"evaluation lacks multiclass_classification_metrics macro_f1/accuracy: {e!r}") from e
    for key in ("per_class", "confusion_matrix", "labels", "n"):
        if key not in evaluation:
            raise ValueError(f"evaluation lacks {key!r}")
    return {
        "macro_f1": macro_f1, "accuracy": accuracy, "n": int(evaluation["n"]),
        "labels": [int(x) for x in evaluation["labels"]],
        "per_class": {str(k): v for k, v in evaluation["per_class"].items()},
        "confusion_matrix": evaluation["confusion_matrix"],
        "label_map": {str(k): v for k, v in label_map.items()},
        "benchmark_macro_f1": bench["macro_f1"],
        "better_than_benchmark": bool(macro_f1 > bench["macro_f1"]),
        "model_package_arn": model_package_arn,
        **pointers,
    }
