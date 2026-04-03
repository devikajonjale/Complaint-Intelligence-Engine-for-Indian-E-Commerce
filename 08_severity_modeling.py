"""Module 08: Severity predictor with model comparison."""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ml_utils import DATA_DIR, MODELS_DIR, REPORTS_DIR, ensure_dirs, measure_latency_ms, multiclass_metrics, update_selected_model

import warnings
warnings.filterwarnings("ignore")

LEGAL_PAT = re.compile(r"\b(fraud|legal|police|consumer court|ncdrc|chargeback|harassment|scam)\b", re.I)
MONEY_PAT = re.compile(r"\b(refund|money|charged|payment|upi|wallet|bank|transaction|deducted)\b", re.I)


def weak_label(row: pd.Series) -> int:
    txt = str(row.get("content", ""))
    score = float(row.get("score", 3))
    thumbs = float(row.get("thumbs_up", 0))
    if LEGAL_PAT.search(txt):
        return 2  # high
    if score <= 2 and thumbs >= 20:
        return 1  # medium
    if score <= 2:
        return 1
    return 0  # low


def gold_label(row: pd.Series) -> int:
    txt = str(row.get("content", ""))
    score = float(row.get("score", 3))
    if LEGAL_PAT.search(txt) or ("double" in txt.lower() and "charged" in txt.lower()):
        return 2
    if score <= 2:
        return 1
    return 0


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    txt = df["content"].fillna("").astype(str)
    return pd.DataFrame(
        {
            "review_length": df["review_length"].fillna(0),
            "score": df["score"].fillna(3),
            "thumbs_up_log": np.log1p(df["thumbs_up"].fillna(0)),
            "has_legal_keyword": txt.str.contains(LEGAL_PAT, regex=True).astype(int),
            "has_money_keyword": txt.str.contains(MONEY_PAT, regex=True).astype(int),
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
        idx = (
            df.groupby("severity_label_gold", group_keys=False)
            .apply(lambda x: x.sample(min(40, len(x)), random_state=42))
            .index
        )
        df.loc[idx, "gold_split"] = "gold_eval"

    num = feature_frame(df)
    embeddings_path = DATA_DIR / "embeddings.npy"
    emb = np.load(embeddings_path) if embeddings_path.exists() else np.zeros((len(df), 16))
    x_all = np.hstack([num.values, emb[: len(df)]])
    y = df["severity_label_weak"].values

    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x_all, y, np.arange(len(df)), test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "linear_svc_calibrated": CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=42)),
    }

    rows = []
    best_name = None
    best_score = -1.0
    best_model = None
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None
        m = multiclass_metrics(y_test, y_pred, y_proba)
        latency = measure_latency_ms(model.predict, x_test[:1], repeats=120)
        m["latency_ms"] = latency
        m["model"] = name
        rows.append(m)
        if m["weighted_f1"] > best_score:
            best_score = m["weighted_f1"]
            best_name = name
            best_model = model

    metrics_df = pd.DataFrame(rows).sort_values("weighted_f1", ascending=False)
    metrics_df.to_csv(REPORTS_DIR / "metrics_severity.csv", index=False, encoding="utf-8-sig")
    joblib.dump(best_model, MODELS_DIR / "severity_model.pkl")

    pred_all = best_model.predict(x_all)
    if hasattr(best_model, "predict_proba"):
        proba_all = best_model.predict_proba(x_all).max(axis=1)
    else:
        proba_all = np.ones(len(df)) * 0.5
    label_map = {0: "Low", 1: "Medium", 2: "High"}
    df["severity_label_ml"] = pd.Series(pred_all).map(label_map)
    df["severity_score_ml"] = proba_all
    df.to_csv(DATA_DIR / "final_reviews_scored.csv", index=False, encoding="utf-8-sig")

    update_selected_model("severity", best_name or "unknown", best_score, {"metrics_file": "reports/metrics_severity.csv"})
    print(f"Severity module complete. Selected model: {best_name} (weighted_f1={best_score:.4f})")


if __name__ == "__main__":
    main()
