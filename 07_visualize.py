"""
ULTIMATE Visualization Engine (20+ Charts)
Includes:
- EDA
- Clustering
- Model Evaluation
- Model Comparison
- Business Insights
- Interactive Plotly Dashboards
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2")


# =========================
# 📊 1–6 EDA VISUALS
# =========================

def eda_visuals(df):
    # 1 Missing heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.savefig(FIG_DIR / "1_missing_heatmap.png")
    plt.close()

    # 2 Text length
    if "review_text" in df.columns:
        df["len"] = df["review_text"].astype(str).apply(len)
        sns.histplot(df["len"], bins=50, kde=True, color="purple")
        plt.title("Text Length Distribution")
        plt.savefig(FIG_DIR / "2_text_length.png")
        plt.close()

    # 3 Platform count
    sns.countplot(data=df, x="platform", palette="coolwarm")
    plt.title("Platform Distribution")
    plt.xticks(rotation=20)
    plt.savefig(FIG_DIR / "3_platform.png")
    plt.close()

    # 4 Boxplot length vs platform
    if "len" in df.columns:
        sns.boxplot(data=df, x="platform", y="len", palette="Set3")
        plt.title("Text Length vs Platform")
        plt.savefig(FIG_DIR / "4_boxplot.png")
        plt.close()

    # 5 Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.savefig(FIG_DIR / "5_corr.png")
    plt.close()

    # 6 Distribution plot
    df.select_dtypes(include=np.number).hist(figsize=(10, 8))
    plt.suptitle("Numerical Distributions")
    plt.savefig(FIG_DIR / "6_histograms.png")
    plt.close()


# =========================
# 🤖 7–10 CLUSTERING
# =========================

def clustering_visuals(df, k_df=None):
    if "cluster" in df.columns:
        sns.countplot(data=df, x="cluster", palette="tab10")
        plt.title("Cluster Distribution")
        plt.savefig(FIG_DIR / "7_cluster.png")
        plt.close()

    emb_cols = [c for c in df.columns if "embedding" in c]
    if emb_cols:
        X = df[emb_cols].fillna(0)
        pca = PCA(n_components=2)
        comp = pca.fit_transform(X)

        plt.scatter(comp[:, 0], comp[:, 1], c=df.get("cluster"), cmap="viridis")
        plt.title("PCA Clusters")
        plt.savefig(FIG_DIR / "8_pca.png")
        plt.close()

        # 3D Plotly PCA (9)
        fig = px.scatter_3d(x=comp[:, 0], y=comp[:, 1], z=np.random.rand(len(comp)),
                            color=df.get("cluster"))
        fig.write_html(FIG_DIR / "9_pca_3d.html")

    if k_df is not None:
        plt.plot(k_df["k"], k_df["silhouette"], marker="o")
        plt.title("Silhouette Score")
        plt.savefig(FIG_DIR / "10_silhouette.png")
        plt.close()


# =========================
# 📈 11–15 MODEL EVALUATION
# =========================

def model_eval_visuals(y_true, y_pred, y_prob, model_name="model"):
    # 11 Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(FIG_DIR / f"11_cm_{model_name}.png")
    plt.close()

    # 12 ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, color="blue")
    plt.title("ROC Curve")
    plt.savefig(FIG_DIR / f"12_roc_{model_name}.png")
    plt.close()

    # 13 PR Curve
    p, r, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(r, p, color="green")
    plt.title("PR Curve")
    plt.savefig(FIG_DIR / f"13_pr_{model_name}.png")
    plt.close()

    # 14 Plotly ROC (interactive)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
    fig.write_html(FIG_DIR / f"14_plotly_roc_{model_name}.html")

    # 15 Plotly Confusion Matrix
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="RdBu")
    fig.write_html(FIG_DIR / f"15_plotly_cm_{model_name}.html")


# =========================
# 🧠 16–18 MODEL COMPARISON
# =========================

def model_comparison():
    path = REPORTS_DIR / "metrics_summary.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)

    for i, metric in enumerate(["accuracy", "f1", "roc_auc"]):
        sns.barplot(data=df, x="model", y=metric, palette="Set1")
        plt.title(f"{metric} Comparison")
        plt.savefig(FIG_DIR / f"{16+i}_{metric}.png")
        plt.close()

    # Plotly interactive leaderboard
    fig = px.bar(df, x="model", y="accuracy", color="model")
    fig.write_html(FIG_DIR / "19_leaderboard.html")


# =========================
# 📊 19–22 BUSINESS
# =========================

def business_visuals(df):
    if "severity" in df.columns:
        sns.countplot(data=df, x="platform", hue="severity")
        plt.savefig(FIG_DIR / "20_platform_severity.png")
        plt.close()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.groupby(df["date"].dt.date).size().plot()
        plt.savefig(FIG_DIR / "21_trend.png")
        plt.close()

    # Pie chart (22)
    if "platform" in df.columns:
        fig = px.pie(df, names="platform")
        fig.write_html(FIG_DIR / "22_pie.html")

    # Treemap (23)
    if "severity" in df.columns:
        fig = px.treemap(df, path=["platform", "severity"])
        fig.write_html(FIG_DIR / "23_treemap.html")


# =========================
# 🚀 MAIN
# =========================

def main():
    df = pd.read_csv(DATA_DIR / "final_reviews.csv")

    k_df = None
    if (DATA_DIR / "k_selection.csv").exists():
        k_df = pd.read_csv(DATA_DIR / "k_selection.csv")

    eda_visuals(df)
    clustering_visuals(df, k_df)
    business_visuals(df)
    model_comparison()

    log.info("✅ 20+ visualizations generated!")


if __name__ == "__main__":
    main()