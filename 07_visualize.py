"""
Step 7 — Ten unique pipeline visualizations (darkgrid, titled, legends).
Input:  richest CSV in data/ (expects final_reviews_response.csv when run after step 12)
Output: figures/viz_01_*.png … figures/viz_10_*.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (10, 6)


def _save(name: str) -> None:
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    log.info("Saved %s", path)
    plt.close()


def load_richest_df() -> pd.DataFrame:
    candidates = [
        DATA_DIR / "final_reviews_response.csv",
        DATA_DIR / "final_reviews_churn.csv",
        DATA_DIR / "final_reviews_routed.csv",
        DATA_DIR / "final_reviews_scored.csv",
        DATA_DIR / "final_reviews.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("No dataset found in data/")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "content" in df.columns and "review_length" not in df.columns:
        df["review_length"] = df["content"].astype(str).str.len()
    for col in ("score", "thumbs_up", "isolation_score", "route_confidence", "churn_risk_score", "severity_score_ml"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("umap_1", "umap_2"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "tier1_critical" in df.columns:
        df["tier1_critical"] = df["tier1_critical"].fillna(False).astype(bool)
    if "is_hinglish" in df.columns:
        df["is_hinglish"] = df["is_hinglish"].fillna(False).astype(bool)
    return df


def main() -> None:
    df = load_richest_df()

    # --- 1) Platform × star rating (categorical × categorical, counts) ---
    if "platform" in df.columns and "score" in df.columns:
        plt.figure(figsize=(10, 6))
        sub = df.dropna(subset=["platform", "score"]).copy()
        sub["star_rating"] = sub["score"].clip(1, 5).astype(int).astype(str)
        ct = pd.crosstab(sub["platform"], sub["star_rating"])
        sns.heatmap(ct, annot=True, fmt="d", cmap="viridis", cbar_kws={"label": "Review count"})
        plt.title("Review volume heatmap: platform × star rating")
        plt.xlabel("Star rating (categorical)")
        plt.ylabel("Platform (categorical)")
        _save("viz_01_platform_star_rating_heatmap.png")

    # --- 2) Mean review length by cluster with 95% CI ---
    if "cluster_name" in df.columns and "review_length" in df.columns:
        plt.figure(figsize=(11, 6))
        order = df.groupby("cluster_name")["review_length"].mean().sort_values(ascending=False).index.tolist()
        sns.barplot(
            data=df,
            x="cluster_name",
            y="review_length",
            order=order,
            estimator=np.mean,
            errorbar=("ci", 95),
            hue="cluster_name",
            palette="husl",
            dodge=False,
            legend=False,
        )
        plt.title("Mean review length by complaint cluster (95% CI)")
        plt.xlabel("Cluster (categorical)")
        plt.ylabel("Review length (characters)")
        handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=sns.color_palette("husl", len(order))[i], markersize=8, label=o) for i, o in enumerate(order)]
        plt.legend(handles=handles, title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=30, ha="right")
        _save("viz_02_mean_length_by_cluster.png")

    # --- 3) Isolation score KDE by Tier-1 flag ---
    if "isolation_score" in df.columns and "tier1_critical" in df.columns:
        plt.figure(figsize=(10, 6))
        sub = df.dropna(subset=["isolation_score"]).copy()
        sub["tier1_critical"] = sub["tier1_critical"].map({True: "Tier-1 critical", False: "Not Tier-1"})
        sns.kdeplot(data=sub, x="isolation_score", hue="tier1_critical", fill=True, common_norm=False, palette="Set1")
        plt.title("Isolation forest score distribution by Tier-1 critical status")
        plt.xlabel("Isolation score (numeric)")
        plt.ylabel("Density")
        plt.legend(title="Complaint class", loc="upper right")
        _save("viz_03_isolation_kde_by_tier1.png")

    # --- 4) UMAP scatter: platform colour, thumbs-up size ---
    if {"umap_1", "umap_2", "platform"}.issubset(df.columns):
        plt.figure(figsize=(10, 7))
        sub = df.dropna(subset=["umap_1", "umap_2", "platform"]).copy()
        sub["log_thumbs"] = np.log1p(sub["thumbs_up"].fillna(0))
        sns.scatterplot(
            data=sub.sample(min(4000, len(sub)), random_state=42) if len(sub) > 4000 else sub,
            x="umap_1",
            y="umap_2",
            hue="platform",
            size="log_thumbs",
            sizes=(20, 220),
            alpha=0.65,
            palette="tab10",
        )
        plt.title("UMAP embedding: platform (colour) and log(1+thumbs up) (size)")
        plt.xlabel("UMAP-1 (numeric)")
        plt.ylabel("UMAP-2 (numeric)")
        plt.legend(title="Legend", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        _save("viz_04_umap_platform_thumbs.png")

    # --- 5) Routing confidence by platform (categorical vs numeric) ---
    if "route_confidence" in df.columns and "platform" in df.columns:
        plt.figure(figsize=(10, 6))
        sub = df.dropna(subset=["route_confidence", "platform"])
        sns.boxplot(data=sub, x="platform", y="route_confidence", hue="platform", palette="pastel", dodge=False, legend=False)
        plt.title("Predicted routing confidence by platform")
        plt.xlabel("Platform (categorical)")
        plt.ylabel("Route confidence (numeric)")
        plt.xticks(rotation=25, ha="right")
        _save("viz_05_route_confidence_by_platform.png")

    # --- 6) Severity score by ML severity label (violin) ---
    if "severity_score_ml" in df.columns and "severity_label_ml" in df.columns:
        plt.figure(figsize=(9, 6))
        sub = df.dropna(subset=["severity_score_ml", "severity_label_ml"])
        order = ["Low", "Medium", "High"]
        sub = sub[sub["severity_label_ml"].isin(order)]
        ord_use = [o for o in order if o in sub["severity_label_ml"].unique()]
        sns.violinplot(
            data=sub,
            x="severity_label_ml",
            y="severity_score_ml",
            order=ord_use,
            hue="severity_label_ml",
            palette="muted",
            dodge=False,
            legend=True,
        )
        plt.title("ML severity score distribution by predicted severity label")
        plt.xlabel("Severity label (categorical)")
        plt.ylabel("Severity score (numeric)")
        leg = plt.legend(title="Severity label", loc="upper right")
        if leg:
            leg.set_title("Severity label")
        _save("viz_06_severity_score_violin.png")

    # --- 7) Churn risk by route category ---
    if "churn_risk_score" in df.columns and "route_category" in df.columns:
        plt.figure(figsize=(11, 6))
        sub = df.dropna(subset=["churn_risk_score", "route_category"])
        top_cats = sub["route_category"].value_counts().head(8).index
        sub = sub[sub["route_category"].isin(top_cats)]
        sns.boxplot(data=sub, x="route_category", y="churn_risk_score", hue="route_category", palette="Set2", dodge=False, legend=False)
        plt.title("Churn risk score by route category (top 8 categories)")
        plt.xlabel("Route category (categorical)")
        plt.ylabel("Churn risk score (numeric)")
        plt.xticks(rotation=28, ha="right")
        _save("viz_07_churn_risk_by_route.png")

    # --- 8) Topic drift signals over time (fallback: weekly complaint volume by cluster) ---
    drift_path = DATA_DIR / "topic_drift_report.csv"
    drift_ok = False
    if drift_path.exists():
        drift = pd.read_csv(drift_path, encoding="utf-8-sig")
        if len(drift) and "week" in drift.columns:
            drift_ok = True
            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            fig.suptitle("Weekly topic drift signals (embedding centroid vs topic mix)", fontsize=14)
            if "centroid_shift" in drift.columns:
                sns.lineplot(data=drift, x="week", y="centroid_shift", marker="o", ax=axes[0], color="steelblue", label="Centroid shift")
                axes[0].set_ylabel("Centroid shift")
                axes[0].legend(title="Signal")
            if "js_divergence" in drift.columns:
                sns.lineplot(data=drift, x="week", y="js_divergence", marker="s", ax=axes[1], color="darkorange", label="JS divergence")
                axes[1].set_ylabel("JS divergence")
                axes[1].legend(title="Signal")
            axes[-1].set_xlabel("Week (categorical)")
            for ax in axes:
                for tick in ax.get_xticklabels():
                    tick.set_rotation(40)
                    tick.set_ha("right")
            plt.tight_layout()
            _save("viz_08_weekly_topic_drift_lines.png")
    if not drift_ok:
        spike_path = DATA_DIR / "spike_report.csv"
        if spike_path.exists():
            sp = pd.read_csv(spike_path, encoding="utf-8-sig")
            if len(sp) and "week" in sp.columns and "complaint_count" in sp.columns:
                plt.figure(figsize=(10, 6))
                top_clusters = sp["cluster_name"].value_counts().head(6).index
                sp = sp[sp["cluster_name"].isin(top_clusters)]
                sns.lineplot(data=sp, x="week", y="complaint_count", hue="cluster_name", marker="o", palette="tab10")
                plt.title("Weekly complaint count by cluster (fallback when drift report unavailable)")
                plt.xlabel("Week (categorical)")
                plt.ylabel("Complaint count (numeric)")
                plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.xticks(rotation=35, ha="right")
                _save("viz_08_weekly_topic_drift_lines.png")

    # --- 9) 100% stacked bar: platform composition by cluster ---
    if "platform" in df.columns and "cluster_name" in df.columns:
        plt.figure(figsize=(11, 6))
        sub = df.dropna(subset=["platform", "cluster_name"])
        tab = pd.crosstab(sub["platform"], sub["cluster_name"], normalize="index") * 100
        tab.plot(kind="bar", stacked=True, colormap="tab20", ax=plt.gca(), width=0.82)
        plt.title("Complaint mix by platform (% of rows per cluster)")
        plt.xlabel("Platform (categorical)")
        plt.ylabel("Percentage within platform (numeric)")
        plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.xticks(rotation=20, ha="right")
        _save("viz_09_platform_cluster_stacked_pct.png")

    # --- 10) Review length vs thumbs-up, hue = star bucket ---
    if "review_length" in df.columns and "thumbs_up" in df.columns:
        plt.figure(figsize=(10, 6))
        sub = df.dropna(subset=["review_length", "thumbs_up"]).copy()
        if "star_bucket" in sub.columns:
            hue = "star_bucket"
        elif "score" in sub.columns:
            sub["star_bucket"] = pd.cut(sub["score"].fillna(3), bins=[0, 2, 3, 5], labels=["1–2★", "3★", "4–5★"])
            hue = "star_bucket"
        else:
            hue = None
        samp = sub.sample(min(2500, len(sub)), random_state=42)
        sns.scatterplot(
            data=samp,
            x="review_length",
            y="thumbs_up",
            hue=hue,
            alpha=0.5,
            palette="coolwarm",
        )
        plt.title("Community engagement vs review length (hue = star bucket)")
        plt.xlabel("Review length (numeric)")
        plt.ylabel("Thumbs up count (numeric)")
        if hue:
            plt.legend(title="Star bucket (categorical)", loc="upper right")
        _save("viz_10_length_vs_thumbs_star_bucket.png")

    log.info("Visualization step complete (10 unique charts).")


if __name__ == "__main__":
    main()
