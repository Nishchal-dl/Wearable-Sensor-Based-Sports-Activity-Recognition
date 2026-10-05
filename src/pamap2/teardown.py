"""Teardown of every project AWS resource, scoped strictly by config names / project prefix.

Pure and client-injected: ``clients`` is a dict of boto3-like clients with keys
``s3, sagemaker, glue, athena, cloudwatch, logs``. Nothing here touches IAM.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from botocore.exceptions import ClientError

LOG_GROUPS = ("/aws/sagemaker/TrainingJobs", "/aws/sagemaker/ProcessingJobs",
              "/aws/sagemaker/TransformJobs", "/aws/sagemaker/Endpoints")
KEEP_ZONES = ("raw/", "curated/")
NOT_FOUND_CODES = {"ResourceNotFound", "ResourceNotFoundException", "NoSuchBucket", "NoSuchKey",
                   "EntityNotFoundException", "NotFound", "404", "NoSuchEntity"}
NOT_FOUND_MSG_CODES = {"ValidationException", "InvalidRequestException"}
NOT_FOUND_MSGS = ("does not exist", "not found", "could not find")

# Deletion order (spec: Boundaries & Constraints).
ORDER = ("stop_training_job", "stop_processing_job", "stop_transform_job", "monitoring_schedule",
         "endpoint", "endpoint_config", "model", "model_package", "model_package_group",
         "feature_group", "glue_table", "glue_database", "athena_workgroup", "alarm",
         "dashboard", "log_stream", "s3_objects", "s3_bucket")


@dataclass
class Resource:
    kind: str
    name: str
    extra: dict = field(default_factory=dict)


def is_project_name(name: str, prefix: str) -> bool:
    return name.startswith(f"{prefix}-")


def is_not_found(exc: Exception) -> bool:
    if not isinstance(exc, ClientError):
        return False
    err = exc.response.get("Error", {})
    code, msg = str(err.get("Code", "")), str(err.get("Message", "")).lower()
    if code in NOT_FOUND_CODES:
        return True
    return code in NOT_FOUND_MSG_CODES and any(m in msg for m in NOT_FOUND_MSGS)


def _pages(fn, items_key, token_in="NextToken", token_out=None, **kw):
    """Yield items from a manually paginated list call."""
    token_out = token_out or token_in
    while True:
        resp = fn(**kw)
        yield from resp.get(items_key, [])
        tok = resp.get(token_out)
        if not tok:
            return
        kw = {**kw, token_in: tok}


def _safe(fn, *a, **kw):
    """Listing helper: not-found means nothing to list."""
    try:
        return list(fn(*a, **kw))
    except ClientError as e:
        if is_not_found(e):
            return []
        raise


def _keep(key: str) -> bool:
    return key.startswith(KEEP_ZONES)


def discover(clients: dict, cfg: dict, keep_data: bool = False, errors: list | None = None) -> list[Resource]:
    """List every in-scope resource. Fixed-name resources are always listed (deletion skips if absent)."""
    prefix, bucket = cfg["prefix"], cfg["bucket"]
    sm, out = clients["sagemaker"], []

    def collect(label, fn):
        try:
            out.extend(fn())
        except Exception as e:  # discovery failure must not abort the rest
            if errors is not None:
                errors.append(Resource(label, "<discovery>", {"error": str(e)}))
            else:
                raise

    def sm_list(kind, method, key, name_key, **kw):
        def run():
            items = _safe(_pages, getattr(sm, method), key, NameContains=prefix, **kw)
            return [Resource(kind, i[name_key]) for i in items if is_project_name(i[name_key], prefix)]
        return run

    collect("stop_training_job", sm_list("stop_training_job", "list_training_jobs",
                                         "TrainingJobSummaries", "TrainingJobName", StatusEquals="InProgress"))
    collect("stop_processing_job", sm_list("stop_processing_job", "list_processing_jobs",
                                           "ProcessingJobSummaries", "ProcessingJobName", StatusEquals="InProgress"))
    collect("stop_transform_job", sm_list("stop_transform_job", "list_transform_jobs",
                                          "TransformJobSummaries", "TransformJobName", StatusEquals="InProgress"))
    collect("monitoring_schedule", sm_list("monitoring_schedule", "list_monitoring_schedules",
                                           "MonitoringScheduleSummaries", "MonitoringScheduleName"))
    collect("endpoint", sm_list("endpoint", "list_endpoints", "Endpoints", "EndpointName"))
    collect("endpoint_config", sm_list("endpoint_config", "list_endpoint_configs", "EndpointConfigs",
                                       "EndpointConfigName"))
    collect("model", sm_list("model", "list_models", "Models", "ModelName"))

    group = cfg["model_package_group"]

    def packages():
        items = _safe(_pages, sm.list_model_packages, "ModelPackageSummaryList", ModelPackageGroupName=group)
        return [Resource("model_package", i["ModelPackageArn"]) for i in items]
    collect("model_package", packages)
    out.append(Resource("model_package_group", group))

    def fg():
        status = None
        try:
            status = sm.describe_feature_group(FeatureGroupName=cfg["feature_group"]).get("FeatureGroupStatus")
        except ClientError as e:
            if not is_not_found(e):
                raise
        return [Resource("feature_group", cfg["feature_group"], {"status": status})]
    collect("feature_group", fg)

    db = cfg["glue_database"]

    def tables():
        items = _safe(_pages, clients["glue"].get_tables, "TableList", DatabaseName=db)
        return [Resource("glue_table", t["Name"], {"database": db}) for t in items]
    collect("glue_table", tables)
    out.append(Resource("glue_database", db))
    out.append(Resource("athena_workgroup", cfg["athena_workgroup"]))

    def alarms():
        pre, found, kw = cfg["monitoring_alarm_prefix"], [], {
            "AlarmNamePrefix": cfg["monitoring_alarm_prefix"], "AlarmTypes": ["MetricAlarm", "CompositeAlarm"]}
        try:
            while True:
                resp = clients["cloudwatch"].describe_alarms(**kw)
                found += resp.get("CompositeAlarms", []) + resp.get("MetricAlarms", [])
                if not resp.get("NextToken"):
                    break
                kw = {**kw, "NextToken": resp["NextToken"]}
        except ClientError as e:
            if not is_not_found(e):
                raise
        return [Resource("alarm", a["AlarmName"]) for a in found if is_project_name(a["AlarmName"], pre)]
    collect("alarm", alarms)
    out.append(Resource("dashboard", cfg["monitoring_dashboard"]))

    def streams():
        res = []
        for lg in LOG_GROUPS:
            items = _safe(_pages, clients["logs"].describe_log_streams, "logStreams",
                          token_in="nextToken", logGroupName=lg, logStreamNamePrefix=prefix)
            res += [Resource("log_stream", s["logStreamName"], {"group": lg})
                    for s in items if is_project_name(s["logStreamName"], prefix)]
        return res
    collect("log_stream", streams)

    out.append(Resource("s3_objects", bucket, {"keep_data": keep_data}))
    out.append(Resource("s3_bucket", bucket, {"keep_data": keep_data}))
    return out


def plan_order(resources: list[Resource]) -> list[Resource]:
    rank = {k: i for i, k in enumerate(ORDER)}
    return sorted(resources, key=lambda r: rank.get(r.kind, len(ORDER)))  # stable within a kind


def _empty_bucket(s3, bucket: str, keep_data: bool) -> int:
    """Delete all object versions + delete markers (outside kept zones if keep_data). Returns count."""
    batch, n = [], 0

    def flush():
        nonlocal batch, n
        if batch:
            resp = s3.delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True})
            if resp and resp.get("Errors"):
                raise RuntimeError(f"delete_objects errors: {resp['Errors'][:3]}")
            n += len(batch)
            batch = []

    kw = {"Bucket": bucket}
    while True:
        resp = s3.list_object_versions(**kw)
        for item in list(resp.get("Versions", [])) + list(resp.get("DeleteMarkers", [])):
            if keep_data and _keep(item["Key"]):
                continue
            batch.append({"Key": item["Key"], "VersionId": item["VersionId"]})
            if len(batch) >= 1000:
                flush()
        if not resp.get("IsTruncated"):
            break
        kw = {"Bucket": bucket, "KeyMarker": resp.get("NextKeyMarker"),
              "VersionIdMarker": resp.get("NextVersionIdMarker")}
        kw = {k: v for k, v in kw.items() if v is not None}
    flush()
    return n


def _wait_gone(sm, name: str, wait_s: float, sleep, poll_s: float):
    elapsed = 0.0
    while True:
        try:
            sm.describe_feature_group(FeatureGroupName=name)
        except ClientError as e:
            if is_not_found(e):
                return
            raise
        if elapsed >= wait_s:
            raise TimeoutError(f"feature group {name} still present after {wait_s}s")
        sleep(poll_s)
        elapsed += poll_s


def _delete(clients: dict, cfg: dict, r: Resource, keep_data: bool, wait_s, sleep, poll_s) -> str:
    sm, k = clients["sagemaker"], r.kind
    if k == "stop_training_job":
        sm.stop_training_job(TrainingJobName=r.name)
    elif k == "stop_processing_job":
        sm.stop_processing_job(ProcessingJobName=r.name)
    elif k == "stop_transform_job":
        sm.stop_transform_job(TransformJobName=r.name)
    elif k == "monitoring_schedule":
        sm.delete_monitoring_schedule(MonitoringScheduleName=r.name)
    elif k == "endpoint":
        sm.delete_endpoint(EndpointName=r.name)
    elif k == "endpoint_config":
        sm.delete_endpoint_config(EndpointConfigName=r.name)
    elif k == "model":
        sm.delete_model(ModelName=r.name)
    elif k == "model_package":
        sm.delete_model_package(ModelPackageName=r.name)
    elif k == "model_package_group":
        sm.delete_model_package_group(ModelPackageGroupName=r.name)
    elif k == "feature_group":
        if r.extra.get("status") != "Deleting":
            sm.delete_feature_group(FeatureGroupName=r.name)
        _wait_gone(sm, r.name, wait_s, sleep, poll_s)
    elif k == "glue_table":
        clients["glue"].delete_table(DatabaseName=r.extra["database"], Name=r.name)
    elif k == "glue_database":
        clients["glue"].delete_database(Name=r.name)
    elif k == "athena_workgroup":
        clients["athena"].delete_work_group(WorkGroup=r.name, RecursiveDeleteOption=True)
    elif k == "alarm":
        clients["cloudwatch"].delete_alarms(AlarmNames=[r.name])
    elif k == "dashboard":
        clients["cloudwatch"].delete_dashboards(DashboardNames=[r.name])
    elif k == "log_stream":
        clients["logs"].delete_log_stream(logGroupName=r.extra["group"], logStreamName=r.name)
    elif k == "s3_objects":
        _empty_bucket(clients["s3"], r.name, keep_data)
    elif k == "s3_bucket":
        if keep_data:
            return "skipped"
        clients["s3"].delete_bucket(Bucket=r.name)
    else:
        raise ValueError(f"unknown resource kind {k}")
    return "deleted"


def teardown(clients: dict, cfg: dict, dry_run: bool = True, confirm: str = "",
             keep_data: bool = False, wait_s: float = 900, sleep=time.sleep, poll_s: float = 10) -> list[dict]:
    """Plan (dry_run) or delete in dependency order. Returns summary rows
    ``{kind, name, status, detail}`` with status in planned/deleted/skipped/failed."""
    if not dry_run and (not cfg.get("bucket") or confirm != cfg["bucket"]):
        raise ValueError(f"CONFIRM must equal the bucket name {cfg['bucket']!r} to delete anything.")
    disc_errors: list[Resource] = []
    plan = plan_order(discover(clients, cfg, keep_data, disc_errors))
    rows = [{"kind": e.kind, "name": e.name, "status": "failed", "detail": e.extra["error"]}
            for e in disc_errors]
    for r in plan:
        if dry_run:
            detail = ("kept (KEEP_DATA)" if r.kind == "s3_bucket" else
                      "keeping raw/, curated/" if r.kind == "s3_objects" else "") if keep_data else ""
            rows.append({"kind": r.kind, "name": r.name, "status": "planned", "detail": detail})
            continue
        try:
            status, detail = _delete(clients, cfg, r, keep_data, wait_s, sleep, poll_s), ""
            if status == "skipped":
                detail = "kept (KEEP_DATA)"
            elif keep_data and r.kind == "s3_objects":
                detail = "keeping raw/, curated/"
        except Exception as e:
            if is_not_found(e):
                status, detail = "skipped", "not found"
            else:
                status, detail = "failed", str(e)
        rows.append({"kind": r.kind, "name": r.name, "status": status, "detail": detail})
    return rows
