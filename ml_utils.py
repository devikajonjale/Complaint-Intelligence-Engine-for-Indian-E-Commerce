"""Shared utilities for extension ML modules."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
ML_FIGURES_DIR = REPORTS_DIR / "ml_figures"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ML_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


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


def measure_latency_ms(predict_fn: Callable[..., np.ndarray], x_sample: np.ndarray, repeats: int = 100) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        predict_fn(x_sample)
    elapsed = (time.perf_counter() - start) * 1000.0
    return elapsed / repeats


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    out = {
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    if y_proba is not None:
        y_bin = pd.get_dummies(pd.Series(y_true)).reindex(columns=range(y_proba.shape[1]), fill_value=0).values
        try:
            out["roc_auc_ovr"] = float(roc_auc_score(y_bin, y_proba, multi_class="ovr", average="weighted"))
        except Exception:
            out["roc_auc_ovr"] = np.nan
        out["brier_like"] = float(np.mean((y_bin - y_proba) ** 2))
        try:
            out["log_loss"] = float(log_loss(y_true, y_proba, labels=list(range(y_proba.shape[1]))))
        except Exception:
            out["log_loss"] = np.nan
    else:
        out["roc_auc_ovr"] = np.nan
        out["brier_like"] = np.nan
        out["log_loss"] = np.nan
    return out


def multiclass_metrics_extended(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    base = multiclass_metrics(y_true, y_pred, y_proba)
    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    base["accuracy"] = float(accuracy_score(y_true, y_pred))
    base["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    try:
        base["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        base["matthews_corrcoef"] = np.nan
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    base["macro_precision"] = float(prec)
    base["macro_recall"] = float(rec)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    base["weighted_precision"] = float(prec_w)
    base["weighted_recall"] = float(rec_w)
    return base


def per_class_classification_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    labels = list(range(len(class_names)))
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame(
        {
            "class": class_names,
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": sup.astype(int),
        }
    )


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    from sklearn.metrics import confusion_matrix

    out = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = np.nan
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    try:
        out["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        out["matthews_corrcoef"] = np.nan
    try:
        out["brier_score"] = float(brier_score_loss(y_true, y_score))
    except Exception:
        out["brier_score"] = np.nan
    prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out["precision"] = float(prec)
    out["recall"] = float(rec)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else np.nan
    return out


def group_or_stratified_split_indices(
    y: np.ndarray,
    groups: np.ndarray | None,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prefer GroupShuffleSplit (hold out entire platforms / groups) to reduce same-corpus leakage.
    Falls back to stratified index split if not enough groups or test set lacks class diversity.
    """
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    y = np.asarray(y)
    n = len(y)
    idx = np.arange(n)
    classes = np.unique(y)
    need_both = len(classes) >= 2

    if groups is not None:
        groups = np.asarray(groups)
        if len(np.unique(groups)) >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(idx, y, groups))
            if not need_both or len(np.unique(y[test_idx])) >= 2:
                return train_idx, test_idx

    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y if need_both else None,
    )
    return train_idx, test_idx


def cross_val_mean_std_multiclass(model, x: np.ndarray, y: np.ndarray, cv: int = 5, random_state: int = 42) -> dict[str, float]:
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    if len(np.unique(y)) < 2 or len(y) < cv * 2:
        return {"cv_f1_weighted_mean": np.nan, "cv_f1_weighted_std": np.nan, "cv_f1_macro_mean": np.nan, "cv_f1_macro_std": np.nan}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    w = cross_val_score(model, x, y, cv=skf, scoring="f1_weighted", n_jobs=1)
    m = cross_val_score(model, x, y, cv=skf, scoring="f1_macro", n_jobs=1)
    return {
        "cv_f1_weighted_mean": float(np.mean(w)),
        "cv_f1_weighted_std": float(np.std(w)),
        "cv_f1_macro_mean": float(np.mean(m)),
        "cv_f1_macro_std": float(np.std(m)),
    }


def cross_val_mean_std_binary(model, x: np.ndarray, y: np.ndarray, cv: int = 5, random_state: int = 42) -> dict[str, float]:
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    if len(np.unique(y)) < 2 or len(y) < cv * 2:
        return {"cv_pr_auc_mean": np.nan, "cv_pr_auc_std": np.nan, "cv_roc_auc_mean": np.nan, "cv_roc_auc_std": np.nan}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    try:
        pr = cross_val_score(model, x, y, cv=skf, scoring="average_precision", n_jobs=1)
        pr_m, pr_s = float(np.mean(pr)), float(np.std(pr))
    except Exception:
        pr_m, pr_s = np.nan, np.nan
    try:
        roc = cross_val_score(model, x, y, cv=skf, scoring="roc_auc", n_jobs=1)
        roc_m, roc_s = float(np.mean(roc)), float(np.std(roc))
    except Exception:
        roc_m, roc_s = np.nan, np.nan
    return {
        "cv_pr_auc_mean": pr_m,
        "cv_pr_auc_std": pr_s,
        "cv_roc_auc_mean": roc_m,
        "cv_roc_auc_std": roc_s,
    }
