# Wearable Sensor-Based Sports Activity Recognition

An end-to-end machine learning system on AWS SageMaker. It classifies the **physical
activity** a person is doing from synchronized wearable sensor data (accelerometer,
gyroscope, magnetometer and heart rate). It uses the labeled
[PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
dataset. Each feature window also carries a derived **intensity** label (light /
moderate / vigorous, from approximate METs).

>  Project for **AAI-540 — Machine Learning Operations**, Team 5.

**Team:** Nishchal, Manideep

**Design document:** [docs/Final-Project_Team-5_Deliverable-1.docx](docs/Final-Project_Team-5_Deliverable-1.docx)

**Repository:** https://github.com/Nishchal-dl/Wearable-Sensor-Based-Sports-Activity-Recognition

---

## Problem

Raw wearable output (~100 Hz motion signals plus heart rate) is only useful once it is
interpreted as *what a person is doing*. The data is heavily imbalanced across
activities, so **macro-F1** is the headline metric.

## What is implemented

- **Data foundation:** S3 data lake, Glue catalog and Athena tables over raw and curated data, with EDA charts.
- **Features:** 512-sample (5.12 s) windows with 50% overlap. Each window gets statistical features from the IMU channels plus heart rate, stored in an offline SageMaker Feature Store group.
- **Splits:** chronological 40/10/10/40 train/validation/test/production split per subject and activity. Test and production use subjects the model has already seen, so scores are optimistic compared with a subject-disjoint evaluation.
- **Models:** a depth-3 decision-tree benchmark (plus a majority-class floor), and SageMaker built-in XGBoost 1.7-1 with fixed hyperparameters, early stopping and Debugger rules.
- **SageMaker Pipeline** `pamap2-har-pipeline`: preprocess → train → evaluate → a macro-F1 gate (default 0.85) → register in the model registry (`PendingManualApproval`) → batch transform → post-process predictions.
- **Governance:** a SageMaker Model Card filled from the training job, model package and reports.
- **Monitoring:** Model Monitor data-quality and model-quality jobs, pipeline execution monitoring, CloudWatch alarms and dashboard, and SHAP explainability.
- **Teardown:** removes every project resource (dry run by default).
- **Unit tests** for the shared code under `tests/`.

## Dataset

| Attribute | Detail |
|---|---|
| Source  | PAMAP2 Physical Activity Monitoring — UCI ML Repository (id=231), DOI 10.24432/C5NW2H |
| Volume  | 3,850,505 instances; 54 columns; ~656 MB; 9 subjects; 18 activities |
| Sampling| 100 Hz IMU + ~9 Hz heart rate |
| Sensors | Hand/chest/ankle IMUs (accelerometer, gyroscope, magnetometer), heart rate, activity label |
| License | CC BY 4.0 |

The data is not committed to git. `2_data_pipeline.ipynb` downloads it and stores it in S3.

## Data in S3

Everything lives in one bucket, `pamap2-har-<account-id>-us-east-1`, created by `1_setup.ipynb`:

| Prefix | Contents |
|---|---|
| `raw/` | Original PAMAP2 `.dat` files |
| `curated/` | Cleaned Parquet, partitioned by session and subject |
| `athena-results/` | Athena query output |
| `features/` | Feature Store offline store; `features/splits/` holds the train/validation/test/production Parquet, XGBoost CSVs, `feature_columns.json` and `label_map.json` |
| `models/` | Benchmark and XGBoost reports, and pipeline outputs (`models/pipeline/`): model artifacts, evaluation, predictions |
| `monitoring/` | Data-quality, model-quality and pipeline monitoring runs, and `monitoring/reports/index.json` |

Catalog: the Glue database `pamap2_har` (tables `raw_pamap2` and `curated_pamap2`), the Athena workgroup `pamap2-har-wg`, and the Feature Group `pamap2-har-windows`.

## Repository layout

```
notebooks/   1_setup → 2_data_pipeline → 3_train_evaluate_deploy → 4_monitoring_reports → 5_teardown
src/pamap2/  shared code (config, schema, features, split, evaluation, benchmark, monitoring, model card, teardown)
src/pamap2/pipeline/  SageMaker Pipeline step scripts (preprocess, evaluate, postprocess)
tests/       pytest unit tests
docs/        design document (Deliverable 1), video, presentation, architecture diagrams
```

## Future work

These items are not implemented: subject-disjoint (leave-one-subject-out) evaluation,
a deep-learning model to compare against XGBoost, hyperparameter tuning, a model that
predicts intensity directly, infrastructure as code, a CI/CD workflow, and a real-time
endpoint.

## Running the notebooks

Run in SageMaker Studio, in order:

1. `notebooks/1_setup.ipynb` - idempotently provisions S3 bucket, Glue database, Athena workgroup; writes `config/project_config.json` (git-ignored).
2. `notebooks/2_data_pipeline.ipynb` - sections:
   - ingest: downloads PAMAP2, lands raw `.dat` in S3 and curated Parquet partitioned by session/subject.
   - Athena catalog: creates `raw_pamap2` and `curated_pamap2` Athena tables.
   - EDA via Athena queries.
   - feature engineering: builds 512-sample/50%-overlap window features, creates (or reuses) the offline-only SageMaker Feature Store group `pamap2-har-windows`, ingests, and verifies the offline row count via Athena (the offline store lags several minutes).
   - data split: reads the offline store, assigns chronological 40/10/10/40 train/validation/test/production splits per subject+activity (1-window gap between blocks; test/production use seen subjects, so scores are optimistic vs. LOSO), writes `features/splits/{train,validation,test,production}.parquet`, XGBoost CSVs (label first, no header), `feature_columns.json` and `label_map.json` (train classes only).
3. `notebooks/3_train_evaluate_deploy.ipynb` - sections:
   - benchmark: local depth-3 decision tree on chest acc16 magnitude std + HR mean (plus a majority-class floor); writes `models/reports/benchmark.json`.
   - SageMaker Pipeline `pamap2-har-pipeline` (scripts in `src/pamap2/pipeline/`, outputs under `models/pipeline/`): preprocess (validates the split Parquet files, emits XGBoost CSVs), train (built-in XGBoost 1.7-1, `multi:softprob`, fixed hyperparameters, early stopping, one `ml.m5.xlarge`, Debugger rules LossNotDecreasing/Overfit/ClassImbalance), evaluate (test macro-F1/accuracy/confusion matrix to `evaluation.json` plus `test_predictions.csv`), then a gate on test macro-F1 >= `MacroF1Threshold` (default 0.85): pass registers a `PendingManualApproval` version with model-quality metrics in `pamap2-har-xgboost`, creates the model, batch-transforms the production features and post-processes the probabilities into `predictions.parquet`; fail ends in a `FailStep` and registers, transforms and publishes nothing. Then definition, upsert, start/wait and step table. Training, evaluation, registration and batch transform happen only through the pipeline. Billable (minutes of `ml.m5.large`/`ml.m5.xlarge`); no endpoint is created.
   - publish: for a succeeded execution writes `models/reports/xgboost_training.json` and `models/reports/xgboost_eval.json` (with pointers to the execution, evaluation, test predictions and predictions file); any other status prints the failed steps and raises `RuntimeError`, leaving earlier reports untouched. Then Debugger rule statuses, benchmark-vs-XGBoost table (macro-F1 headline), confusion matrix, predictions summary, lineage and an optional low-threshold re-execution (`RUN_FAIL_DEMO`, default `False`).
   - model card: creates or updates one SageMaker Model Card `pamap2-har-xgboost-card` (Governance > Model cards) with overview, intended uses, business details, training details, evaluation details and additional information, filled from the real training job, model package and S3 reports; status mirrors the registry (`Approved`/`Draft`). Not billable, safe to re-run.
4. `notebooks/4_monitoring_reports.ipynb` - sections:
   - data monitor: Model Monitor data-quality baseline on train features, then a one-off analyzer job on production features; writes `monitoring/data-quality/{baseline,runs/<ts>}/` (statistics, constraints, violations) and publishes `DataViolationCount`. Two `ml.m5.large` processing jobs.
   - model monitor: multiclass ModelQualityMonitor baseline on the pipeline's test predictions, one-off job on the pipeline's production predictions file vs `true_activity_id`, drift check vs `xgboost_eval.json`; writes `monitoring/model-quality/{baseline,runs/<ts>}/` and publishes macro-F1/accuracy. Two `ml.m5.large` jobs.
   - pipeline monitor: lists the executions and the steps of the latest one (failure reasons included), writes `monitoring/pipeline/runs/<ts>/summary.json` and publishes `PipelineExecutionFailed` and `PipelineFailedSteps`. Reads only; with no executions it prints that there is nothing to monitor and publishes nothing.
   - infrastructure monitor: CloudWatch alarms (`pamap2-har-*`) on SageMaker job CPU/memory/disk (these also cover the `pipelines-*` jobs), on the custom `pamap2-har/Monitoring` metrics (incl. job-failed 0/1 and the two pipeline metrics) and a native pipeline alarm on `ExecutionFailed`. Job alarms are Metrics Insights query alarms per job namespace (they cover future jobs); custom-metric alarms use `TreatMissingData=ignore`.
   - dashboard: `put_dashboard` for `pamap2-har-ml` (job utilisation, monitoring metrics, pipeline executions widget, alarm status, report links); prints the console URL. Job widgets use SEARCH expressions; the report-index link is only populated after the reports section.
   - reports: local exact TreeSHAP explainability with XGBoost `pred_contribs` and the consolidated `monitoring/reports/index.json` (now also the pipeline evaluation, predictions and status summary).

Run order: `1_setup`, `2_data_pipeline`, `3_train_evaluate_deploy`, `4_monitoring_reports`, then `5_teardown.ipynb` as the last step.

Teardown: `notebooks/5_teardown.ipynb` (logic in `src/pamap2/teardown.py`) removes every project resource (SageMaker Pipeline `pamap2-har-pipeline` with its executions and pipeline-created models, S3 bucket incl. versions, Feature Group, Glue/Athena, SageMaker models/packages, Model Card, CloudWatch alarms/dashboard/log streams). It is a dry run by default; set `DRY_RUN=False` and `CONFIRM` to the bucket name to delete, `KEEP_DATA=True` to keep the bucket and `raw/`+`curated/`. 
