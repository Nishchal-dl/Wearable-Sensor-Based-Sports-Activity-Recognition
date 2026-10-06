# Wearable Sensor-Based Sports Activity Recognition & Intensity Estimation

End-to-end, cloud-native machine learning system that ingests synchronized wearable
sensor data (tri-axial accelerometer, gyroscope, magnetometer, and heart rate) and
predicts both the **physical activity** being performed and its **exertion intensity**
(light / moderate / vigorous), using the fully labeled
[PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
dataset.

The point of this project is focused on
**production-grade MLOps rigor**: a reproducible pipeline provisioned as real cloud
infrastructure (AWS/SageMaker via Python CDK) covering ingestion, preprocessing, training
and hyperparameter tuning, an evaluation-gated model registry, batch deployment,
monitoring, and CI/CD.

> Built as the capstone deliverable for **AAI-540 — Machine Learning Operations**.

---

## Problem

Raw wearable output — acceleration, rotation, magnetometer, and heart rate at ~100 Hz —
is only useful once it is interpreted into *what a person is doing* and *how hard they are
working*. Two things make this hard:

1. **The ML problem is non-trivial** — severe class imbalance (transient/other activity
   dominates), and models that score well on seen users often fail to generalize to a
   **new person** whose movement signature differs.
2. Reproducibility, deployment path, monitoring."

## What the system does

- **Two prediction heads** — multi-class activity recognition, plus a derived 3-class
  intensity estimate (each activity mapped to a light / moderate / vigorous MET category).
- **Generalization measurement** — Leave-One-Subject-Out (LOSO) cross-validation as
  the primary evaluation scheme, so scores reflect performance on *unseen people*, with
  **macro-F1** as the headline metric (because of class imbalance).
- **A modeling trade-off** — an interpretable classical baseline compared against
  a higher-capacity deep model.
- **MLOps infrastructure** — S3 data lakes, SageMaker processing / training / HPO jobs,
  an evaluation-gated model registry, batch-transform deployment, CloudWatch alerts, and a CI/CD pipeline.

## Dataset

| Attribute | Detail |
|---|---|
| Source  | PAMAP2 Physical Activity Monitoring — UCI ML Repository (id=231), DOI 10.24432/C5NW2H |
| Volume  | 3,850,505 instances; 54 columns; ~656 MB; 9 subjects; 18 activities |
| Sampling| 100 Hz IMU + ~9 Hz heart rate |
| Sensors | Tri-axial accelerometer (±16g / ±6g), gyroscope, magnetometer (hand/chest/ankle IMUs), heart rate, activity label |
| Files   | Space-separated `.dat` text files, one per subject per session (Protocol + Optional) |
| License | CC BY 4.0 |

The dataset is **not committed to this repository** (see [`.gitignore`](.gitignore)).
Download it from the UCI link above and place the `.dat` files under `data/raw/`
(or point the pipeline at your S3 raw bucket).

## Pipeline overview

1. **Ingest** — parse the 54 columns; drop the 4 invalid orientation columns per IMU.
2. **Clean** — handle sparse heart rate (forward-fill / interpolate); optional downsample
   from 100 Hz; normalize/standardize per channel; remove/relabel transient `activityID 0`.
3. **Window** — fixed sliding windows (e.g. 512 samples / 5.12 s, 50% overlap) with
   majority-vote labels.
4. **Feature-engineer** — statistical + frequency-domain features (mean, std, min, max,
   energy, dominant frequency) from acceleration/gyroscope; mean/max heart rate.
5. **Train** — classical baseline and deep model, with HPO.
6. **Evaluate** — LOSO cross-validation; macro-F1, accuracy, per-class precision/recall,
   confusion matrix under a subject-disjoint split.
7. **Gate & register** — promote to the model registry only past a macro-F1 threshold.
8. **Deploy** — batch transform for session summaries.
9. **Monitor** — model quality, data drift vs. training baseline, infrastructure metrics.

## Success criteria

**ML quality**
- Primary: macro-F1 under LOSO **competitive with published benchmarks (~0.85+)**.
- Supporting: accuracy, per-class precision/recall, confusion matrix on a subject-disjoint
  final split; a documented baseline-vs-deep-model trade-off.

**MLOps / engineering (the real bar)**
- Infrastructure reproducibly provisioned from Python CDK.
- Every release passes automated gates: schema validation, preprocessing unit tests, a
  macro-F1 quality threshold on a held-out subject, and an endpoint smoke test.
- Live monitoring: model-quality tracking, data-drift detection, CloudWatch alerts.

## Scope

**In scope** — the full pipeline (ingestion → windowing → features → training + HPO →
evaluation gate → registry → batch deployment → monitoring) on PAMAP2; real AWS/SageMaker
infrastructure via Python CDK; both classical and deep models; LOSO evaluation; intensity
estimation; monitoring; CI/CD; reproducible experiment tracking.

**Out of scope** — a consumer mobile app or on-device/firmware inference; medical diagnosis
or clinical decision-making; identifying individuals from gait/biometrics; collecting new
sensor data or activities beyond PAMAP2's 18 labels; a real-time streaming endpoint
(batch is primary; real-time is a future enhancement).

## Running the notebooks

Run in SageMaker Studio (region `us-east-1`, existing `LabRole`; no IAM resources are created), in order:

1. `notebooks/1_setup.ipynb` - idempotently provisions S3 bucket, Glue database, Athena workgroup; writes `config/project_config.json` (git-ignored).
2. `notebooks/2_data_pipeline.ipynb` - sections:
   - 1.1 ingest: downloads PAMAP2, lands raw `.dat` in S3 and curated Parquet partitioned by session/subject.
   - 1.2 Athena catalog: creates `raw_pamap2` and `curated_pamap2` Athena tables.
   - 2.0 EDA via Athena queries.
   - 3.0 feature engineering: builds 512-sample/50%-overlap window features, creates (or reuses) the offline-only SageMaker Feature Store group `pamap2-har-windows`, ingests, and verifies the offline row count via Athena (the offline store lags several minutes).
   - 3.1 data split: reads the offline store, assigns chronological 40/10/10/40 train/validation/test/production splits per subject+activity (1-window gap between blocks; test/production use seen subjects, so scores are optimistic vs. LOSO), writes `features/splits/{train,validation,test,production}.parquet`, XGBoost CSVs (label first, no header), `feature_columns.json` and `label_map.json` (train classes only).
3. `notebooks/3_train_evaluate_deploy.ipynb` - sections:
   - 4.0 benchmark: local depth-3 decision tree on chest acc16 magnitude std + HR mean (plus a majority-class floor); writes `models/reports/benchmark.json`. Negligible cost.
   - 4.1 train: built-in SageMaker XGBoost 1.7-1 (`multi:softprob`, fixed hyperparameters, early stopping) on one `ml.m5.xlarge` with Debugger rules (LossNotDecreasing, Overfit, ClassImbalance, XGBoost report); registers a `PendingManualApproval` package in `pamap2-har-xgboost`. Billable training job (minutes).
   - 4.2 evaluate: batch transform (`ml.m5.large` x1) on the test split, benchmark-vs-XGBoost table (macro-F1 headline), confusion matrix, `models/reports/xgboost_eval.json`. Ends with the manual approval gate: set `APPROVE = True` only if satisfied (default `False` approves nothing).
   - 5.0 deploy: batch transform on the production split using the evaluated package, which must be Approved (otherwise the section stops with a message); input/output join enabled for Part D; writes `models/predictions/production/`. No real-time endpoint is created, so nothing keeps running or billing after the notebooks finish.
4. `notebooks/4_monitoring_reports.ipynb` - sections:
   - 6.0 data monitor: Model Monitor data-quality baseline on train features, then a one-off analyzer job on production features; writes `monitoring/data-quality/{baseline,runs/<ts>}/` (statistics, constraints, violations) and publishes `DataViolationCount`. Two `ml.m5.large` processing jobs.
   - 6.1 model monitor: multiclass ModelQualityMonitor baseline on test predictions, one-off job on production predictions vs `true_activity_id`, drift check vs `xgboost_eval.json`; writes `monitoring/model-quality/{baseline,runs/<ts>}/` and publishes macro-F1/accuracy. Two `ml.m5.large` jobs. Scores are optimistic (seen subjects).
   - 6.2 infrastructure monitor: CloudWatch alarms (`pamap2-har-*`) on SageMaker job CPU/memory/disk and on the custom `pamap2-har/Monitoring` metrics (incl. job-failed 0/1). Job alarms are Metrics Insights query alarms per job namespace (they cover future jobs); custom-metric alarms use `TreatMissingData=ignore`. No SNS/EventBridge.
   - 6.3 dashboard: `put_dashboard` for `pamap2-har-ml` (job utilisation, monitoring metrics, alarm status, report links); prints the console URL. Job widgets use SEARCH expressions; the report-index link is only populated after 7.0.
   - 7.0 reports: local exact TreeSHAP explainability with XGBoost `pred_contribs` (Clarify is closed to new accounts; not billable) and the consolidated `monitoring/reports/index.json`.

Monitoring cost/cleanup: all jobs are one-off (no MonitoringSchedule, no endpoint). The monitoring notebook ends with a single cleanup cell that lists (never deletes) monitoring schedules, deletes temporary `pamap2-har-mon*` model. Alarms and the dashboard are kept until you run teardown.

Teardown: `notebooks/5_teardown.ipynb` (logic in `src/pamap2/teardown.py`) removes every project resource (S3 bucket incl. versions, Feature Group, Glue/Athena, SageMaker models/packages, CloudWatch alarms/dashboard/log streams). It is a dry run by default; set `DRY_RUN=False` and `CONFIRM` to the bucket name to delete, `KEEP_DATA=True` to keep the bucket and `raw/`+`curated/`. WARNING: irreversible. IAM and `config/project_config.json` are never touched.

Shared code lives in `src/pamap2/` (`config.py`, `schema.py`, `features.py`, `split.py`, `evaluation.py`, `benchmark.py`, `monitoring.py`). Tests: `uv run pytest tests -q`. These notebooks create billable AWS resources.
