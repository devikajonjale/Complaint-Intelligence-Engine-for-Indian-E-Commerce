# Model Selection Summary

This file is auto-populated by the extension modules via `models/selected_models.json`.

## Selection Policy

- Severity: highest weighted F1 (with calibration metrics tracked)
- Router: highest macro F1
- Churn: highest PR-AUC
- Auto-response: highest average response quality score
- Drift: weekly centroid + distribution divergence alert framework

## Artifacts

- Registry: `models/selected_models.json`
- Module metrics:
  - `reports/metrics_severity.csv`
  - `reports/metrics_router.csv`
  - `reports/metrics_churn.csv`
  - `reports/metrics_response.csv`
  - `reports/metrics_drift.csv`

Run `python run_pipeline.py` to regenerate this experiment set and refresh the selected-model registry.
