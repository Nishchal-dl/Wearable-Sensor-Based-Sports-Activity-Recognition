import sys
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pamap2.config import build_config, load_config
from src.pamap2.teardown import discover, is_project_name, plan_order, teardown

CFG = build_config("123456789012")
BUCKET = CFG["bucket"]


def err(code, msg="x"):
    return ClientError({"Error": {"Code": code, "Message": msg}}, "op")


class Fake:
    """Records every call to a shared log; responses come from `resp` (value or callable)."""

    def __init__(self, svc, log, resp):
        self.svc, self.log, self.resp = svc, log, resp

    def __getattr__(self, op):
        def call(**kw):
            self.log.append((self.svc, op, kw))
            r = self.resp.get((self.svc, op), {})
            if callable(r):
                r = r(**kw)
            if isinstance(r, Exception):
                raise r
            return r
        return call


def make(resp=None):
    log, resp = [], resp or {}
    return {s: Fake(s, log, resp) for s in ("s3", "sagemaker", "glue", "athena", "cloudwatch", "logs")}, log


def full_resp():
    return {
        ("sagemaker", "list_training_jobs"): {"TrainingJobSummaries": [{"TrainingJobName": "pamap2-har-xgb-1"}]},
        ("sagemaker", "list_processing_jobs"): {"ProcessingJobSummaries": [{"ProcessingJobName": "pamap2-har-mon-2"}]},
        ("sagemaker", "list_transform_jobs"): {"TransformJobSummaries": [{"TransformJobName": "pamap2-har-bt-3"}]},
        ("sagemaker", "list_monitoring_schedules"): {"MonitoringScheduleSummaries": [{"MonitoringScheduleName": "pamap2-har-ms"}]},
        ("sagemaker", "list_endpoints"): {"Endpoints": [{"EndpointName": "pamap2-har-ep"}]},
        ("sagemaker", "list_endpoint_configs"): {"EndpointConfigs": [{"EndpointConfigName": "pamap2-har-cfg"}]},
        ("sagemaker", "list_models"): {"Models": [{"ModelName": "pamap2-har-prod-1"}, {"ModelName": "other-x"}]},
        ("sagemaker", "list_model_packages"): {"ModelPackageSummaryList": [{"ModelPackageArn": "arn:pkg/1"}]},
        ("sagemaker", "list_model_cards"): {"ModelCardSummaries": [{"ModelCardName": "pamap2-har-xgboost-card"}, {"ModelCardName": "other-card"}]},
        ("sagemaker", "describe_feature_group"): {"FeatureGroupStatus": "Created"},
        ("glue", "get_tables"): {"TableList": [{"Name": "raw_pamap2"}]},
        ("cloudwatch", "describe_alarms"): {"CompositeAlarms": [{"AlarmName": "pamap2-har-comp"}], "MetricAlarms": [{"AlarmName": "pamap2-har-a1"}, {"AlarmName": "pamap2-harx"}]},
        ("logs", "describe_log_streams"): {"logStreams": [{"logStreamName": "pamap2-har-xgb-1/algo-1"}, {"logStreamName": "other/1"}]},
        ("s3", "list_object_versions"): {"Versions": [{"Key": "raw/a", "VersionId": "v1"}, {"Key": "models/b", "VersionId": "v2"}],
                                          "DeleteMarkers": [{"Key": "models/b", "VersionId": "d1"}]},
    }


def deletes(log):
    return [(s, o) for s, o, _ in log if o.startswith(("delete_", "stop_"))]


def test_is_project_name():
    assert is_project_name("pamap2-har-ml", "pamap2-har")
    assert not is_project_name("pamap2-harx", "pamap2-har")
    assert not is_project_name("my-pamap2-har-x", "pamap2-har")


def test_dry_run_no_deletes_and_lists_everything():
    resp = full_resp()
    clients, log = make(resp)
    rows = teardown(clients, CFG, dry_run=True)
    assert deletes(log) == []
    kinds = {r["kind"] for r in rows}
    assert {"stop_training_job", "model", "model_package", "model_card", "feature_group", "glue_table", "alarm",
            "log_stream", "s3_objects", "s3_bucket", "dashboard", "athena_workgroup"} <= kinds
    assert all(r["status"] == "planned" for r in rows)


def test_wrong_confirm_raises_before_any_call():
    clients, log = make()
    with pytest.raises(ValueError):
        teardown(clients, CFG, dry_run=False, confirm="nope")
    assert log == []


def test_foreign_resources_not_listed_or_deleted():
    clients, log = make(full_resp())
    rows = teardown(clients, CFG, dry_run=False, confirm=BUCKET, sleep=lambda s: None)
    names = {r["name"] for r in rows}
    assert "other-x" not in names and "pamap2-harx" not in names and "other-card" not in names and "other/1" not in names
    assert not any(k.get("ModelName") == "other-x" for _, _, k in log)
    cards = [k["ModelCardName"] for _, o, k in log if o == "delete_model_card"]
    assert cards == ["pamap2-har-xgboost-card"]


def test_confirmed_order_and_s3_before_bucket():
    resp = full_resp()
    state = {"fg": True}
    resp[("sagemaker", "delete_feature_group")] = lambda **kw: state.update(fg=False) or {}
    resp[("sagemaker", "describe_feature_group")] = lambda **kw: (
        {"FeatureGroupStatus": "Created"} if state["fg"] else err("ResourceNotFound"))
    clients, log = make(resp)
    rows = teardown(clients, CFG, dry_run=False, confirm=BUCKET, sleep=lambda s: None)
    assert not [r for r in rows if r["status"] == "failed"]
    seq = [o for _, o in deletes(log)]
    expected = ["stop_training_job", "stop_processing_job", "stop_transform_job",
                "delete_monitoring_schedule", "delete_endpoint", "delete_endpoint_config", "delete_model",
                "delete_model_package", "delete_model_package_group", "delete_model_card",
                "delete_feature_group",
                "delete_table", "delete_database", "delete_work_group", "delete_alarms",
                "delete_dashboards", "delete_log_stream", "delete_objects", "delete_bucket"]
    firsts = [seq.index(o) for o in expected]
    assert firsts == sorted(firsts)
    obj = [k for _, o, k in log if o == "delete_objects"][0]
    assert {(i["Key"], i["VersionId"]) for i in obj["Delete"]["Objects"]} == {
        ("raw/a", "v1"), ("models/b", "v2"), ("models/b", "d1")}
    wg = [k for _, o, k in log if o == "delete_work_group"][0]
    assert wg["RecursiveDeleteOption"] is True


def test_already_torn_down_all_skipped():
    nf = err("ResourceNotFound")
    resp = {("sagemaker", "describe_feature_group"): nf, ("sagemaker", "delete_feature_group"): nf,
            ("sagemaker", "delete_model_package_group"): err("ValidationException", "Group does not exist"),
            ("sagemaker", "delete_pipeline"): err("ResourceNotFound"),
            ("glue", "delete_database"): err("EntityNotFoundException"),
            ("athena", "delete_work_group"): err("InvalidRequestException", "WorkGroup is not found"),
            ("cloudwatch", "delete_dashboards"): err("ResourceNotFound"),
            ("s3", "list_object_versions"): err("NoSuchBucket"),
            ("s3", "delete_bucket"): err("NoSuchBucket")}
    for op in ("list_training_jobs", "list_models", "list_model_packages", "list_pipeline_executions"):
        resp[("sagemaker", op)] = nf
    clients, _ = make(resp)
    rows = teardown(clients, CFG, dry_run=False, confirm=BUCKET, sleep=lambda s: None)
    assert rows and all(r["status"] == "skipped" for r in rows), rows


def test_feature_group_deleting_polls_then_timeout_fails():
    resp = {("sagemaker", "describe_feature_group"): {"FeatureGroupStatus": "Deleting"}}
    clients, log = make(resp)
    sleeps = []
    rows = teardown(clients, CFG, dry_run=False, confirm=BUCKET, wait_s=30, sleep=sleeps.append, poll_s=10)
    fg = [r for r in rows if r["kind"] == "feature_group"][0]
    assert fg["status"] == "failed" and len(sleeps) == 3
    assert ("sagemaker", "delete_feature_group") not in [(s, o) for s, o, _ in log]


def test_failure_does_not_abort_later_steps():
    resp = full_resp()
    resp[("sagemaker", "delete_model")] = err("AccessDenied", "nope")
    resp[("sagemaker", "describe_feature_group")] = err("ResourceNotFound")
    clients, log = make(resp)
    rows = teardown(clients, CFG, dry_run=False, confirm=BUCKET, sleep=lambda s: None)
    assert [r for r in rows if r["status"] == "failed"][0]["kind"] == "model"
    assert ("s3", "delete_bucket") in deletes(log)


def test_keep_data_preserves_raw_curated_and_bucket():
    resp = full_resp()
    resp[("sagemaker", "describe_feature_group")] = err("ResourceNotFound")
    clients, log = make(resp)
    rows = teardown(clients, CFG, dry_run=False, confirm=BUCKET, keep_data=True, sleep=lambda s: None)
    obj = [k for _, o, k in log if o == "delete_objects"][0]
    assert {i["Key"] for i in obj["Delete"]["Objects"]} == {"models/b"}
    assert ("s3", "delete_bucket") not in deletes(log)
    assert [r for r in rows if r["kind"] == "s3_bucket"][0]["status"] == "skipped"


def test_s3_pagination_and_batching():
    pages = [{"Versions": [{"Key": f"k{i}", "VersionId": "1"} for i in range(1000)], "IsTruncated": True,
              "NextKeyMarker": "k999", "NextVersionIdMarker": "1"},
             {"Versions": [{"Key": "z", "VersionId": "1"}], "IsTruncated": False}]
    resp = {("s3", "list_object_versions"): lambda **kw: pages.pop(0),
            ("sagemaker", "describe_feature_group"): err("ResourceNotFound")}
    clients, log = make(resp)
    teardown(clients, CFG, dry_run=False, confirm=BUCKET, sleep=lambda s: None)
    sizes = [len(k["Delete"]["Objects"]) for _, o, k in log if o == "delete_objects"]
    assert sizes == [1000, 1]
    lv = [k for _, o, k in log if o == "list_object_versions"]
    assert lv[1]["KeyMarker"] == "k999"


def test_plan_order_stable():
    from src.pamap2.teardown import Resource
    out = plan_order([Resource("s3_bucket", "b"), Resource("model", "m"), Resource("stop_training_job", "t")])
    assert [r.kind for r in out] == ["stop_training_job", "model", "s3_bucket"]


def test_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="1_setup"):
        load_config(tmp_path / "none.json")


def test_in_progress_filter_on_job_lists():
    clients, log = make()
    discover(clients, CFG)
    for op in ("list_training_jobs", "list_processing_jobs", "list_transform_jobs"):
        assert [k for _, o, k in log if o == op][0]["StatusEquals"] == "InProgress"


def test_log_stream_token_pagination():
    pages = {"a": {"logStreams": [{"logStreamName": "pamap2-har-1"}], "nextToken": "t"},
             "b": {"logStreams": [{"logStreamName": "pamap2-har-2"}]}}
    resp = {("logs", "describe_log_streams"):
            lambda **kw: pages["b" if kw.get("nextToken") else "a"] if kw["logGroupName"].endswith("TrainingJobs") else {}}
    clients, log = make(resp)
    res = [r for r in discover(clients, CFG) if r.kind == "log_stream"]
    assert [r.name for r in res] == ["pamap2-har-1", "pamap2-har-2"]
    assert "/aws/sagemaker/Endpoints" in [k["logGroupName"] for _, o, k in log if o == "describe_log_streams"]


def test_discovery_error_row_and_later_steps_run():
    clients, log = make({("sagemaker", "list_models"): err("AccessDenied", "no"),
                         ("sagemaker", "describe_feature_group"): err("ResourceNotFound")})
    rows = teardown(clients, CFG, dry_run=False, confirm=BUCKET, sleep=lambda s: None)
    bad = [r for r in rows if r["status"] == "failed"]
    assert [(r["kind"], r["name"]) for r in bad] == [("model", "<discovery>")]
    assert ("s3", "delete_bucket") in deletes(log)


def test_delete_objects_errors_mark_failed():
    resp = full_resp()
    resp[("s3", "delete_objects")] = {"Errors": [{"Key": "a", "Code": "AccessDenied"}]}
    resp[("sagemaker", "describe_feature_group")] = err("ResourceNotFound")
    clients, _ = make(resp)
    rows = teardown(clients, CFG, dry_run=False, confirm=BUCKET, sleep=lambda s: None)
    assert [r for r in rows if r["kind"] == "s3_objects"][0]["status"] == "failed"


def test_composite_alarm_discovered():
    clients, log = make(full_resp())
    names = [r.name for r in discover(clients, CFG) if r.kind == "alarm"]
    assert "pamap2-har-comp" in names and "pamap2-harx" not in names
    kw = [k for _, o, k in log if o == "describe_alarms"][0]
    assert kw["AlarmTypes"] == ["MetricAlarm", "CompositeAlarm"]


def test_empty_bucket_and_confirm_raises():
    clients, log = make()
    with pytest.raises(ValueError):
        teardown(clients, {**CFG, "bucket": ""}, dry_run=False, confirm="")
    assert log == []


def test_package_group_retried_until_packages_gone():
    left = {"n": 2}
    def group_delete(**kw):
        if left["n"]:
            left["n"] -= 1
            return err("ValidationException", "Model Package Group x cannot be deleted because it still contains Model Packages.")
        return {}
    resp = full_resp()
    resp[("sagemaker", "delete_model_package_group")] = group_delete
    resp[("sagemaker", "describe_feature_group")] = err("ResourceNotFound")
    clients, log = make(resp)
    rows = teardown(clients, CFG, dry_run=False, confirm=BUCKET, sleep=lambda s: None)
    grp = [r for r in rows if r["kind"] == "model_package_group"][0]
    assert grp["status"] == "deleted", grp
    assert sum(o == "delete_model_package_group" for _, o, _ in log) == 3


EX = "arn:aws:sagemaker:us-east-1:123456789012:pipeline/pamap2-har-pipeline/execution/abc123"
EX2 = "arn:aws:sagemaker:us-east-1:123456789012:pipeline/pamap2-har-pipeline/execution/def456"


def pipeline_resp():
    resp = full_resp()
    resp[("sagemaker", "describe_feature_group")] = err("ResourceNotFound")
    resp[("sagemaker", "list_pipeline_executions")] = {"PipelineExecutionSummaries": [
        {"PipelineExecutionArn": EX, "PipelineExecutionStatus": "Executing"},
        {"PipelineExecutionArn": EX2, "PipelineExecutionStatus": "Succeeded"}]}
    resp[("sagemaker", "list_pipeline_execution_steps")] = {"PipelineExecutionSteps": [
        {"StepName": "CreateModel", "Metadata": {"Model": {"Arn": "arn:aws:sagemaker:us-east-1:123456789012:model/pipelines-abc-CreateModel-x"}}},
        {"StepName": "Train", "Metadata": {"TrainingJob": {"Arn": "arn:training-job/t"}}}]}
    resp[("logs", "describe_log_streams")] = lambda **kw: ({"logStreams": [
        {"logStreamName": kw["logStreamNamePrefix"] + "Train-1/algo-1"}, {"logStreamName": "other/1"}]}
        if kw["logGroupName"].endswith("TrainingJobs") and kw["logStreamNamePrefix"].startswith("pipelines-") else {})
    return resp


def test_pipeline_resources_discovered_and_ordered():
    clients, log = make(pipeline_resp())
    rows = teardown(clients, CFG, dry_run=False, confirm=BUCKET, sleep=lambda s: None)
    assert not [r for r in rows if r["status"] == "failed"], rows
    seq = [o for _, o in deletes(log)]
    assert seq.index("stop_pipeline_execution") < seq.index("delete_pipeline") < seq.index("delete_model") \
        < seq.index("delete_model_package")
    stops = [k["PipelineExecutionArn"] for _, o, k in log if o == "stop_pipeline_execution"]
    assert stops == [EX]
    models = [k["ModelName"] for _, o, k in log if o == "delete_model"]
    assert "pipelines-abc-CreateModel-x" in models and "other-x" not in models
    streams = [k["logStreamName"] for _, o, k in log if o == "delete_log_stream"]
    assert "pipelines-abc123-Train-1/algo-1" in streams and "pipelines-def456-Train-1/algo-1" in streams
    assert "other/1" not in streams
    pre = {k["logStreamNamePrefix"] for _, o, k in log if o == "describe_log_streams"}
    assert {"pipelines-abc123-", "pipelines-def456-"} <= pre


def test_pipeline_not_found_skipped():
    resp = full_resp()
    resp[("sagemaker", "describe_feature_group")] = err("ResourceNotFound")
    resp[("sagemaker", "list_pipeline_executions")] = err("ResourceNotFound", "Pipeline does not exist")
    resp[("sagemaker", "delete_pipeline")] = err("ResourceNotFound", "Pipeline does not exist")
    clients, _ = make(resp)
    rows = teardown(clients, CFG, dry_run=False, confirm=BUCKET, sleep=lambda s: None)
    assert [r["status"] for r in rows if r["kind"] == "pipeline"] == ["skipped"]
    assert not [r for r in rows if r["kind"] == "stop_pipeline_execution"]


def test_pipeline_delete_retried_while_executions_stopping():
    left = {"n": 2}

    def delete_pipeline(**kw):
        if left["n"]:
            left["n"] -= 1
            return err("ValidationException", "Cannot delete pipeline while executions are running.")
        return {}
    resp = pipeline_resp()
    resp[("sagemaker", "delete_pipeline")] = delete_pipeline
    clients, log = make(resp)
    rows = teardown(clients, CFG, dry_run=False, confirm=BUCKET, sleep=lambda s: None)
    assert [r["status"] for r in rows if r["kind"] == "pipeline"] == ["deleted"]
    assert sum(o == "delete_pipeline" for _, o, _ in log) == 3


def test_pipeline_dry_run_plans_only():
    clients, log = make(pipeline_resp())
    rows = teardown(clients, CFG, dry_run=True)
    assert deletes(log) == []
    assert {"pipeline", "stop_pipeline_execution"} <= {r["kind"] for r in rows}


def test_pipeline_delete_timeout_fails_after_bounded_sleeps():
    resp = pipeline_resp()
    resp[("sagemaker", "delete_pipeline")] = err("ValidationException", "Cannot delete pipeline while executions are running.")
    clients, log = make(resp)
    sleeps = []
    rows = teardown(clients, CFG, dry_run=False, confirm=BUCKET, wait_s=30, sleep=sleeps.append, poll_s=10)
    assert [r["status"] for r in rows if r["kind"] == "pipeline"] == ["failed"]
    assert len(sleeps) == 3
    assert sum(o == "delete_pipeline" for _, o, _ in log) == 4


def test_pipeline_delete_other_error_not_retried():
    resp = pipeline_resp()
    resp[("sagemaker", "delete_pipeline")] = err("AccessDeniedException", "User is not authorized")
    clients, log = make(resp)
    sleeps = []
    rows = teardown(clients, CFG, dry_run=False, confirm=BUCKET, sleep=sleeps.append)
    assert [r["status"] for r in rows if r["kind"] == "pipeline"] == ["failed"]
    assert sum(o == "delete_pipeline" for _, o, _ in log) == 1 and not sleeps
