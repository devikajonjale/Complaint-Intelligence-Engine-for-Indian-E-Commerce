"""Module 12: Topic drift detection from weekly embeddings and distributions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_utils import DATA_DIR, REPORTS_DIR, ensure_dirs, update_selected_model

import warnings
warnings.filterwarnings("ignore")


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(p.sum(), 1e-9)
    q = q / max(q.sum(), 1e-9)
    m = 0.5 * (p + q)
    eps = 1e-9
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return float(0.5 * (kl_pm + kl_qm))


def main() -> None:
    ensure_dirs()
    path = DATA_DIR / "final_reviews_response.csv"
    df = pd.read_csv(path if path.exists() else DATA_DIR / "final_reviews.csv", encoding="utf-8-sig")
    emb = np.load(DATA_DIR / "embeddings.npy")[: len(df)]

    df["at"] = pd.to_datetime(df["at"], errors="coerce", utc=True)
    df["week_start"] = df["at"].dt.to_period("W").astype(str)
    df["topic"] = df.get("route_category", df.get("cluster_name", "Unknown")).fillna("Unknown").astype(str)
    weeks = sorted(df["week_start"].dropna().unique().tolist())
    rows = []
    for i in range(1, len(weeks)):
        prev_w, cur_w = weeks[i - 1], weeks[i]
        prev_idx = df.index[df["week_start"] == prev_w].to_numpy()
        cur_idx = df.index[df["week_start"] == cur_w].to_numpy()
        if len(prev_idx) < 5 or len(cur_idx) < 5:
            continue
        prev_cent = emb[prev_idx].mean(axis=0)
        cur_cent = emb[cur_idx].mean(axis=0)
        centroid_shift = float(np.linalg.norm(cur_cent - prev_cent))

        prev_dist = df.loc[prev_idx, "topic"].value_counts(normalize=True)
        cur_dist = df.loc[cur_idx, "topic"].value_counts(normalize=True)
        all_topics = sorted(set(prev_dist.index) | set(cur_dist.index))
        p = np.array([prev_dist.get(t, 0.0) for t in all_topics], dtype=float)
        q = np.array([cur_dist.get(t, 0.0) for t in all_topics], dtype=float)
        js = jensen_shannon(p, q)
        rows.append({"week": cur_w, "prev_week": prev_w, "centroid_shift": centroid_shift, "js_divergence": js})

    drift = pd.DataFrame(rows)
    if len(drift):
        drift["shift_z"] = (drift["centroid_shift"] - drift["centroid_shift"].mean()) / (drift["centroid_shift"].std() + 1e-9)
        drift["js_z"] = (drift["js_divergence"] - drift["js_divergence"].mean()) / (drift["js_divergence"].std() + 1e-9)
        drift["is_drift_alert"] = (drift["shift_z"] > 1.5) | (drift["js_z"] > 1.5)
    else:
        drift["is_drift_alert"] = []

    drift.to_csv(DATA_DIR / "topic_drift_report.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(
        [
            {
                "weeks_evaluated": int(len(drift)),
                "drift_alert_weeks": int(drift["is_drift_alert"].sum()) if len(drift) else 0,
                "alert_rate": float(drift["is_drift_alert"].mean()) if len(drift) else 0.0,
            }
        ]
    )
    summary.to_csv(REPORTS_DIR / "metrics_drift.csv", index=False, encoding="utf-8-sig")
    update_selected_model("drift", "weekly_centroid_plus_js", float(summary.iloc[0]["alert_rate"]), {"metrics_file": "reports/metrics_drift.csv"})
    print("Drift module complete.")


if __name__ == "__main__":
    main()
