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