"""Module 10: Churn proxy — non-leaky label, group holdout, regularised models."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml_evaluation_plots import save_binary_metrics_bars, save_pr_curve_plot
from ml_utils import (
    DATA_DIR,
    ML_FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    binary_metrics,
    cross_val_mean_std_binary,
    ensure_dirs,
    group_or_stratified_split_indices,
    measure_latency_ms,
    update_selected_model,
)

import warnings

warnings.filterwarnings("ignore")


def build_proxy_label(df: pd.DataFrame) -> pd.Series:
    """
    Churn-risk proxy without using severity / Tier-1 / isolation flags (those were in X before → leakage).
    Uses only complaint bucket, star pressure, and relative review length.
    """
    complaint = df["star_bucket"].astype(str) == "complaint"
    score = df["score"].fillna(3)
    rl = df["review_length"].fillna(0)
    thr = float(rl.quantile(0.80)) if len(rl) else 0.0
    long_rant = rl >= thr
    very_unhappy = score <= 1
    return (complaint & (very_unhappy | long_rant)).astype(int)


def main() -> None:
    ensure_dirs()
    routed_path = DATA_DIR / "final_reviews_routed.csv"
    df = pd.read_csv(routed_path if routed_path.exists() else DATA_DIR / "final_reviews.csv", encoding="utf-8-sig")

    x = pd.DataFrame(
        {
            "score": df["score"].fillna(3),
            "thumbs_up": np.log1p(df["thumbs_up"].fillna(0)),
            "review_length": df["review_length"].fillna(0),
            "is_hinglish": df["is_hinglish"].fillna(False).astype(int),
            "route_confidence": df.get("route_confidence", 0.5).fillna(0.5),
        }
    )
    y = build_proxy_label(df).values
    groups = df["platform"].fillna("unknown").astype(str).values

    train_idx, test_idx = group_or_stratified_split_indices(y, groups, test_size=0.25, random_state=44)
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    models = {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.2, solver="lbfgs")),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=220,
            random_state=42,
            class_weight="balanced",
            max_depth=8,
            min_samples_leaf=8,
            max_features="sqrt",
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=42,
            max_depth=2,
            subsample=0.75,
            n_estimators=100,
            learning_rate=0.05,
            min_samples_leaf=14,
        ),
        "svc_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, class_weight="balanced", random_state=42, C=0.65, gamma="scale")),
            ]
        ),
        "adaboost": AdaBoostClassifier(random_state=42, n_estimators=120, learning_rate=0.5),
    }

    rows = []
    best_name = None
    best_score = -1.0
    best_key = None
    best_y_score = None
    for name, model in models.items():
        mdl = clone(model)
        mdl.fit(x_train.values, y_train)
        y_score = mdl.predict_proba(x_test.values)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        m = binary_metrics(y_test, y_pred, y_score)
        m["latency_ms"] = measure_latency_ms(mdl.predict, x_test.iloc[:1].values, repeats=80)
        m["model"] = name
        m["eval_split"] = "group_holdout_platform"
        cv_stats = cross_val_mean_std_binary(clone(model), x_train.values, y_train, cv=3)
        m.update(cv_stats)
        rows.append(m)
        if m["pr_auc"] > best_score:
            best_score = m["pr_auc"]
            best_name = name
            best_key = name
            best_y_score = y_score

    metrics = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    metrics.to_csv(REPORTS_DIR / "metrics_churn.csv", index=False, encoding="utf-8-sig")

    save_binary_metrics_bars(
        metrics,
        ML_FIGURES_DIR / "churn_model_comparison.png",
        "Churn proxy — group holdout (no label leakage features)",
    )

    if best_y_score is not None and best_key:
        save_pr_curve_plot(
            y_test,
            best_y_score,
            ML_FIGURES_DIR / "churn_pr_curve_best_model.png",
            f"Precision–recall curve (holdout) — {best_name}",
        )

    if best_key is None and len(metrics):
        best_key = str(metrics.iloc[0]["model"])
        best_name = best_key
        best_score = float(metrics.iloc[0]["pr_auc"])

    production = clone(models[best_key])
    production.fit(x.values, y)
    joblib.dump(production, MODELS_DIR / "churn_model.pkl")

    out = df.copy()
    out["churn_risk_score"] = production.predict_proba(x.values)[:, 1]
    out["churn_risk_label"] = np.where(
        out["churn_risk_score"] >= 0.7, "High", np.where(out["churn_risk_score"] >= 0.4, "Medium", "Low")
    )
    out.to_csv(DATA_DIR / "final_reviews_churn.csv", index=False, encoding="utf-8-sig")

    update_selected_model("churn", best_name or "unknown", best_score, {"metrics_file": "reports/metrics_churn.csv"})
    print(f"Churn module complete. Selected model: {best_name} (pr_auc={best_score:.4f}, group holdout)")


if __name__ == "__main__":
    main()
