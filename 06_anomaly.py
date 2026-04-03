"""
Step 6 — Anomaly Detection and Spike Tracking
Input:  data/clustered_reviews.csv
Output: data/final_reviews.csv, data/spike_report.csv
        models/isoforest.pkl, models/ocsvm.pkl
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def build_weekly_spikes(df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        df.groupby(["platform", "cluster_name", "week"], as_index=False)
        .size()
        .rename(columns={"size": "complaint_count"})
        .sort_values(["platform", "cluster_name", "week"])
    )
    weekly["rolling_mean"] = weekly.groupby(["platform", "cluster_name"])["complaint_count"].transform(
        lambda s: s.rolling(window=4, min_periods=2).mean()
    )
    weekly["rolling_std"] = weekly.groupby(["platform", "cluster_name"])["complaint_count"].transform(
        lambda s: s.rolling(window=4, min_periods=2).std()
    )
    weekly["z_score"] = (weekly["complaint_count"] - weekly["rolling_mean"]) / (
        weekly["rolling_std"].replace(0, np.nan)
    )
    weekly["is_spike"] = weekly["z_score"] > 2.5
    return weekly


def main() -> None:
    path = DATA_DIR / "clustered_reviews.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run 05_cluster.py first")

    df = pd.read_csv(path, encoding="utf-8-sig")
    features = df[[f"umap_{i+1}" for i in range(10)]].to_numpy()

    iso = IsolationForest(contamination=0.05, n_estimators=200, random_state=42)
    iso.fit(features)
    iso_pred = iso.predict(features)
    iso_score = -iso.decision_function(features)
    df["isolation_anomaly"] = iso_pred == -1
    df["isolation_score"] = iso_score

    positive_mask = df["cluster_name"].str.contains("Positive", case=False, na=False)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features)
    if positive_mask.sum() >= 20:
        ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
        ocsvm.fit(x_scaled[positive_mask.to_numpy()])
        svm_pred = ocsvm.predict(x_scaled)
        df["ocsvm_anomaly"] = svm_pred == -1
        decision = ocsvm.decision_function(x_scaled)
        df["ocsvm_score"] = -decision
    else:
        log.warning("Not enough positive cluster rows for One-Class SVM; using neutral defaults.")
        ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
        df["ocsvm_anomaly"] = False
        df["ocsvm_score"] = 0.0

    df["tier1_critical"] = df["isolation_anomaly"] & df["hdbscan_is_noise"]
    df["severity"] = np.where(df["tier1_critical"], "RED", np.where(df["isolation_anomaly"], "AMBER", "GREEN"))

    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    df["week"] = df["at"].dt.to_period("W").astype(str)

    spike_df = build_weekly_spikes(df)

    df.to_csv(DATA_DIR / "final_reviews.csv", index=False, encoding="utf-8-sig")
    spike_df.to_csv(DATA_DIR / "spike_report.csv", index=False, encoding="utf-8-sig")
    joblib.dump(iso, MODEL_DIR / "isoforest.pkl")
    joblib.dump(ocsvm, MODEL_DIR / "ocsvm.pkl")
    log.info("Saved final outputs and anomaly models.")


if __name__ == "__main__":
    main()
