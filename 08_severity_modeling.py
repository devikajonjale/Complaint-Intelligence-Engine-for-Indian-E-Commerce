"""Module 08: Severity predictor — reduced leakage, group holdout, regularized models."""

from __future__ import annotations

import re

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ml_evaluation_plots import save_confusion_matrix_heatmap, save_model_comparison_bars
from ml_utils import (
    DATA_DIR,
    ML_FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    cross_val_mean_std_multiclass,
    ensure_dirs,
    group_or_stratified_split_indices,
    measure_latency_ms,
    multiclass_metrics_extended,
    per_class_classification_df,
    update_selected_model,
)

import warnings

warnings.filterwarnings("ignore")

LEGAL_PAT = re.compile(r"\b(fraud|legal|police|consumer court|ncdrc|chargeback|harassment|scam)\b", re.I)
MONEY_PAT = re.compile(r"\b(refund|money|charged|payment|upi|wallet|bank|transaction|deducted)\b", re.I)


def weak_label(row: pd.Series) -> int:
    txt = str(row.get("content", ""))
    score = float(row.get("score", 3))
    thumbs = float(row.get("thumbs_up", 0))
    if LEGAL_PAT.search(txt):
        return 2
    if score <= 2 and thumbs >= 20:
        return 1
    if score <= 2:
        return 1
    return 0


def gold_label(row: pd.Series) -> int:
    txt = str(row.get("content", ""))
    score = float(row.get("score", 3))
    if LEGAL_PAT.search(txt) or ("double" in txt.lower() and "charged" in txt.lower()):
        return 2
    if score <= 2:
        return 1
    return 0


def tabular_features_no_rule_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude features that duplicate weak_label rules (legal/money regex, star rating, thumbs).
    Model must rely on SBERT semantics plus coarse text-shape cues — realistic generalization eval.
    """
    txt = df["content"].fillna("").astype(str)
    return pd.DataFrame(
        {
            "review_length": df["review_length"].fillna(0),
            "exclamation_count": txt.str.count("!"),
            "capital_ratio": txt.apply(lambda s: (sum(1 for c in s if c.isupper()) / max(len(s), 1))),
        }
    )


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(DATA_DIR / "final_reviews.csv", encoding="utf-8-sig")
    df["severity_label_weak"] = df.apply(weak_label, axis=1)
    df["severity_label_gold"] = df.apply(gold_label, axis=1)
    df["gold_split"] = "train"
    if len(df) >= 120:
        gold_parts = [g.sample(min(40, len(g)), random_state=42) for _, g in df.groupby("severity_label_gold")]
        idx = pd.concat(gold_parts).index
        df.loc[idx, "gold_split"] = "gold_eval"

    num = tabular_features_no_rule_leakage(df)
    embeddings_path = DATA_DIR / "embeddings.npy"
    emb = np.load(embeddings_path) if embeddings_path.exists() else np.zeros((len(df), 16))
    x_all = np.hstack([num.values.astype(np.float64), emb[: len(df)]])
    y = df["severity_label_weak"].values
    groups = df["platform"].fillna("unknown").astype(str).values

    train_idx, test_idx = group_or_stratified_split_indices(y, groups, test_size=0.25, random_state=42)
    x_train, x_test = x_all[train_idx], x_all[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    models = {
        "logreg": LogisticRegression(max_iter=3000, class_weight="balanced", C=0.25, solver="lbfgs"),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            max_depth=10,
            min_samples_leaf=6,
            max_features="sqrt",
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=42,
            max_depth=2,
            subsample=0.75,
            n_estimators=120,
            learning_rate=0.04,
            min_samples_leaf=12,
        ),
        "linear_svc_calibrated": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=42, C=0.35, max_iter=4000)
        ),
        "hist_gbm": HistGradientBoostingClassifier(
            random_state=42,
            max_depth=4,
            learning_rate=0.05,
            max_iter=150,
            l2_regularization=0.25,
            min_samples_leaf=20,
        ),
        "sgd_log": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        max_iter=2500,
                        random_state=42,
                        class_weight="balanced",
                        alpha=0.02,
                        penalty="l2",
                    ),
                ),
            ]
        ),
    }

    rows = []
    best_name = None
    best_score = -1.0
    best_key = None
    for name, model in models.items():
        mdl = clone(model)
        mdl.fit(x_train, y_train)
        y_pred = mdl.predict(x_test)
        y_proba = mdl.predict_proba(x_test) if hasattr(mdl, "predict_proba") else None
        m = multiclass_metrics_extended(y_test, y_pred, y_proba)
        m["latency_ms"] = measure_latency_ms(mdl.predict, x_test[:1], repeats=80)
        m["model"] = name
        m["eval_split"] = "group_holdout_platform"
        cv_stats = cross_val_mean_std_multiclass(clone(model), x_train, y_train, cv=3)
        m.update(cv_stats)
        rows.append(m)
        if m["weighted_f1"] > best_score:
            best_score = m["weighted_f1"]
            best_name = name
            best_key = name

    metrics_df = pd.DataFrame(rows).sort_values("weighted_f1", ascending=False)
    metrics_df.to_csv(REPORTS_DIR / "metrics_severity.csv", index=False, encoding="utf-8-sig")

    class_names = ["Low", "Medium", "High"]
    if best_key is None and len(metrics_df):
        best_key = str(metrics_df.iloc[0]["model"])
        best_name = best_key
        best_score = float(metrics_df.iloc[0]["weighted_f1"])
    best_model = clone(models[best_key]) if best_key else None
    if best_model is not None:
        best_model.fit(x_train, y_train)
        y_pred_best = best_model.predict(x_test)
        per_class = per_class_classification_df(y_test, y_pred_best, class_names)
        per_class.to_csv(REPORTS_DIR / "metrics_severity_per_class.csv", index=False, encoding="utf-8-sig")
        cm = confusion_matrix(y_test, y_pred_best, labels=[0, 1, 2])
        save_confusion_matrix_heatmap(
            cm,
            class_names,
            ML_FIGURES_DIR / "severity_confusion_matrix.png",
            "Severity — confusion matrix (platform group holdout)",
        )

    bar_cols = [
        "weighted_f1",
        "macro_f1",
        "accuracy",
        "balanced_accuracy",
        "roc_auc_ovr",
        "cv_f1_weighted_mean",
    ]
    save_model_comparison_bars(
        metrics_df,
        [c for c in bar_cols if c in metrics_df.columns],
        ML_FIGURES_DIR / "severity_model_comparison.png",
        "Severity models — group holdout + CV (reduced feature leakage)",
    )

    production = clone(models[best_key]) if best_key else None
    if production is not None:
        production.fit(x_all, y)
        joblib.dump(production, MODELS_DIR / "severity_model.pkl")
        pred_all = production.predict(x_all)
        if hasattr(production, "predict_proba"):
            proba_all = production.predict_proba(x_all).max(axis=1)
        else:
            proba_all = np.ones(len(df)) * 0.5
    else:
        pred_all = np.zeros(len(df), dtype=int)
        proba_all = np.ones(len(df)) * 0.33

    label_map = {0: "Low", 1: "Medium", 2: "High"}
    df["severity_label_ml"] = pd.Series(pred_all).map(label_map)
    df["severity_score_ml"] = proba_all
    df.to_csv(DATA_DIR / "final_reviews_scored.csv", index=False, encoding="utf-8-sig")

    update_selected_model("severity", best_name or "unknown", best_score, {"metrics_file": "reports/metrics_severity.csv"})
    print(f"Severity module complete. Selected model: {best_name} (weighted_f1={best_score:.4f}, group holdout)")


if __name__ == "__main__":
    main()
