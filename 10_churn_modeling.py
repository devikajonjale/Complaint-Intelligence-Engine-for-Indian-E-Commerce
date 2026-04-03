"""Module 10: Churn risk modeling with candidate comparison."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml_utils import DATA_DIR, MODELS_DIR, REPORTS_DIR, binary_metrics, ensure_dirs, measure_latency_ms, update_selected_model

import warnings
warnings.filterwarnings("ignore")


def build_proxy_label(df: pd.DataFrame) -> pd.Series:
    # Proxy churn risk when explicit user identifiers are unavailable.
    return (
        (df["star_bucket"].astype(str) == "complaint")
        & (
            (df.get("severity_label_ml", "Low").astype(str).eq("High"))
            | (df.get("tier1_critical", False).astype(bool))
            | (df.get("thumbs_up", 0).fillna(0) >= 10)
            | (df.get("isolation_anomaly", False).astype(bool))
        )
    ).astype(int)


def main() -> None:
    ensure_dirs()
    routed_path = DATA_DIR / "final_reviews_routed.csv"
    df = pd.read_csv(routed_path if routed_path.exists() else DATA_DIR / "final_reviews.csv", encoding="utf-8-sig")
    if "severity_label_ml" not in df.columns:
        df["severity_label_ml"] = "Low"

    severity_map = {"Low": 0, "Medium": 1, "High": 2}
    x = pd.DataFrame(
        {
            "score": df["score"].fillna(3),
            "thumbs_up": np.log1p(df["thumbs_up"].fillna(0)),
            "review_length": df["review_length"].fillna(0),
            "is_hinglish": df["is_hinglish"].fillna(False).astype(int),
            "severity_ord": df["severity_label_ml"].map(severity_map).fillna(0).astype(int),
            "isolation_score": df.get("isolation_score", 0).fillna(0),
            "route_confidence": df.get("route_confidence", 0.5).fillna(0.5),
        }
    )
    y = build_proxy_label(df).values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "logreg": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]
        ),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "svc_rbf": Pipeline(
            [("scaler", StandardScaler()), ("clf", SVC(probability=True, class_weight="balanced", random_state=42))]
        ),
    }

    rows = []
    best_name = None
    best_score = -1.0
    best_model = None
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_score = model.predict_proba(x_test)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        m = binary_metrics(y_test, y_pred, y_score)
        m["latency_ms"] = measure_latency_ms(model.predict, x_test.iloc[:1].values, repeats=100)
        m["model"] = name
        rows.append(m)
        if m["pr_auc"] > best_score:
            best_score = m["pr_auc"]
            best_name = name
            best_model = model

    metrics = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    metrics.to_csv(REPORTS_DIR / "metrics_churn.csv", index=False, encoding="utf-8-sig")
    joblib.dump(best_model, MODELS_DIR / "churn_model.pkl")

    out = df.copy()
    out["churn_risk_score"] = best_model.predict_proba(x)[:, 1]
    out["churn_risk_label"] = np.where(out["churn_risk_score"] >= 0.7, "High", np.where(out["churn_risk_score"] >= 0.4, "Medium", "Low"))
    out.to_csv(DATA_DIR / "final_reviews_churn.csv", index=False, encoding="utf-8-sig")

    update_selected_model("churn", best_name or "unknown", best_score, {"metrics_file": "reports/metrics_churn.csv"})
    print(f"Churn module complete. Selected model: {best_name} (pr_auc={best_score:.4f})")


if __name__ == "__main__":
    main()
