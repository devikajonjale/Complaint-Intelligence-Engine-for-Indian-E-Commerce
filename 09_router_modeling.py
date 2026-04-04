"""Module 09: Complaint router with extended metrics, CV, and benchmark charts."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ml_evaluation_plots import save_confusion_matrix_heatmap, save_model_comparison_bars
from ml_utils import (
    DATA_DIR,
    ML_FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    cross_val_mean_std_multiclass,
    ensure_dirs,
    measure_latency_ms,
    multiclass_metrics_extended,
    per_class_classification_df,
    update_selected_model,
)

import warnings

warnings.filterwarnings("ignore")


def main() -> None:
    ensure_dirs()
    base_path = DATA_DIR / "final_reviews_scored.csv"
    df = pd.read_csv(base_path if base_path.exists() else DATA_DIR / "final_reviews.csv", encoding="utf-8-sig")
    text = df["text_for_model"].fillna(df["cleaned_content"]).fillna("").astype(str)
    y_raw = df["cluster_name"].fillna("Other").astype(str)
    y, classes = pd.factorize(y_raw)
    class_names = classes.tolist()

    emb_path = DATA_DIR / "embeddings.npy"
    if emb_path.exists() and "kmeans_cluster" in df.columns:
        emb = np.load(emb_path)[: len(df)]
        cluster_id = df["kmeans_cluster"].fillna(-1).astype(int).values
        centroid = {}
        for c in np.unique(cluster_id):
            if c < 0:
                continue
            centroid[c] = emb[cluster_id == c].mean(axis=0)
        dist = np.array(
            [np.linalg.norm(emb[i] - centroid.get(cluster_id[i], emb[i])) for i in range(len(df))]
        )
        threshold = np.quantile(dist, 0.4)
        keep = dist <= threshold
    else:
        keep = np.ones(len(df), dtype=bool)

    x_train, x_test, y_train, y_test = train_test_split(
        text[keep], y[keep], test_size=0.2, random_state=42, stratify=y[keep]
    )

    models = {
        "logreg_tfidf": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "linear_svc_tfidf": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=42))),
            ]
        ),
        "rf_tfidf": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=3000)),
                ("clf", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
            ]
        ),
        "nb_tfidf": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=6000)),
                ("clf", MultinomialNB()),
            ]
        ),
        "extra_trees_tfidf": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=4000, ngram_range=(1, 2))),
                ("clf", ExtraTreesClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
            ]
        ),
    }

    rows = []
    best_name = None
    best_score = -1.0
    best_model = None
    for name, model in models.items():
        mdl = clone(model)
        mdl.fit(x_train, y_train)
        y_pred = mdl.predict(x_test)
        y_proba = mdl.predict_proba(x_test) if hasattr(mdl, "predict_proba") else None
        m = multiclass_metrics_extended(y_test, y_pred, y_proba)
        m["latency_ms"] = measure_latency_ms(mdl.predict, x_train.iloc[:1], repeats=50)
        m["model"] = name
        cv_stats = cross_val_mean_std_multiclass(clone(model), x_train, y_train, cv=3)
        m.update(cv_stats)
        rows.append(m)
        if m["macro_f1"] > best_score:
            best_score = m["macro_f1"]
            best_name = name
            best_model = mdl

    metrics = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    metrics.to_csv(REPORTS_DIR / "metrics_router.csv", index=False, encoding="utf-8-sig")

    if best_model is not None:
        y_pred_best = best_model.predict(x_test)
        per_class = per_class_classification_df(y_test, y_pred_best, class_names)
        per_class.to_csv(REPORTS_DIR / "metrics_router_per_class.csv", index=False, encoding="utf-8-sig")
        labels_idx = list(range(len(class_names)))
        cm = confusion_matrix(y_test, y_pred_best, labels=labels_idx)
        save_confusion_matrix_heatmap(
            cm,
            [str(c)[:24] for c in class_names],
            ML_FIGURES_DIR / "router_confusion_matrix.png",
            "Router — confusion matrix (holdout)",
        )

    save_model_comparison_bars(
        metrics,
        [c for c in ["macro_f1", "weighted_f1", "accuracy", "balanced_accuracy", "roc_auc_ovr", "cv_f1_macro_mean"] if c in metrics.columns],
        ML_FIGURES_DIR / "router_model_comparison.png",
        "Router models — macro/weighted F1, accuracy, and CV stability",
    )

    joblib.dump({"model": best_model, "classes": class_names}, MODELS_DIR / "router_model.pkl")

    pred = best_model.predict(text)
    out = df.copy()
    out["route_category"] = pd.Series(pred).map(lambda i: class_names[int(i)] if int(i) < len(class_names) else "Other")
    if hasattr(best_model, "predict_proba"):
        out["route_confidence"] = best_model.predict_proba(text).max(axis=1)
    else:
        out["route_confidence"] = 0.5
    out.to_csv(DATA_DIR / "final_reviews_routed.csv", index=False, encoding="utf-8-sig")
    out[["review_id", "platform", "content", "route_category", "route_confidence"]].to_csv(
        DATA_DIR / "router_predictions.csv", index=False, encoding="utf-8-sig"
    )

    update_selected_model("router", best_name or "unknown", best_score, {"metrics_file": "reports/metrics_router.csv"})
    print(f"Router module complete. Selected model: {best_name} (macro_f1={best_score:.4f})")


if __name__ == "__main__":
    main()
