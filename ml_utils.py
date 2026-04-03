"""Shared utilities for extension ML modules."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def update_selected_model(task: str, model_name: str, primary_metric: float, extra: dict | None = None) -> None:
    ensure_dirs()
    selected_path = MODELS_DIR / "selected_models.json"
    payload = {}
    if selected_path.exists():
        payload = json.loads(selected_path.read_text(encoding="utf-8"))
    payload[task] = {
        "selected_model": model_name,
        "primary_metric": float(primary_metric),
        **(extra or {}),
    }
    selected_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def measure_latency_ms(predict_fn: Callable[[np.ndarray], np.ndarray], x_sample: np.ndarray, repeats: int = 100) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        predict_fn(x_sample)
    elapsed = (time.perf_counter() - start) * 1000.0
    return elapsed / repeats


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    out = {
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    if y_proba is not None:
        y_bin = pd.get_dummies(pd.Series(y_true)).reindex(columns=range(y_proba.shape[1]), fill_value=0).values
        try:
            out["roc_auc_ovr"] = float(roc_auc_score(y_bin, y_proba, multi_class="ovr", average="weighted"))
        except Exception:
            out["roc_auc_ovr"] = np.nan
        out["brier_like"] = float(np.mean((y_bin - y_proba) ** 2))
    else:
        out["roc_auc_ovr"] = np.nan
        out["brier_like"] = np.nan
    return out


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    out = {
        "f1": float(f1_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = np.nan
    return out
