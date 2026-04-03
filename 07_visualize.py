"""
Advanced visualization module for Complaint Intelligence Engine.
Covers:
- Dataset EDA
- Clustering insights
- Model evaluation
- Business decision visuals
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def plot_missing_values(df):
    plt.figure(figsize=(10, 5))
    missing = df.isnull().sum().sort_values(ascending=False)
    sns.barplot(x=missing.index, y=missing.values)
    plt.xticks(rotation=45)
    plt.title("Missing Values per Column")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "missing_values.png")
    plt.close()


def plot_text_length(df):
    if "review_text" in df.columns:
        df["text_length"] = df["review_text"].astype(str).apply(len)
        plt.figure(figsize=(10, 5))
        sns.histplot(df["text_length"], bins=50, kde=True)
        plt.title("Distribution of Review Length")
        plt.savefig(FIG_DIR / "text_length_distribution.png")
        plt.close()


def plot_cluster_distribution(df):
    if "cluster" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="cluster")
        plt.title("Cluster Distribution")
        plt.savefig(FIG_DIR / "cluster_distribution.png")
        plt.close()


def plot_pca(df):
    if "embedding_0" in df.columns:
        emb_cols = [col for col in df.columns if "embedding" in col]
        X = df[emb_cols].fillna(0)

        pca = PCA(n_components=2)
        comps = pca.fit_transform(X)

        plt.figure(figsize=(8, 6))
        plt.scatter(comps[:, 0], comps[:, 1], alpha=0.5)
        plt.title("PCA Projection of Embeddings")
        plt.savefig(FIG_DIR / "pca_projection.png")
        plt.close()


def plot_platform_distribution(df):
    plt.figure(figsize=(10, 5))
    order = df["platform"].value_counts().index
    sns.countplot(data=df, x="platform", order=order)
    plt.xticks(rotation=20)
    plt.title("Review Volume by Platform")
    plt.savefig(FIG_DIR / "platform_volume.png")
    plt.close()


def plot_severity_distribution(df):
    if "severity" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="severity")
        plt.title("Severity Distribution")
        plt.savefig(FIG_DIR / "severity_distribution.png")
        plt.close()


def plot_platform_vs_severity(df):
    if "severity" in df.columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x="platform", hue="severity")
        plt.xticks(rotation=20)
        plt.title("Platform vs Severity")
        plt.savefig(FIG_DIR / "platform_vs_severity.png")
        plt.close()


def plot_time_trend(df):
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        trend = df.groupby(df["date"].dt.date).size()

        plt.figure(figsize=(12, 5))
        trend.plot()
        plt.title("Review Volume Over Time")
        plt.savefig(FIG_DIR / "time_trend.png")
        plt.close()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(FIG_DIR / "confusion_matrix.png")
    plt.close()


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(FIG_DIR / "roc_curve.png")
    plt.close()


def plot_precision_recall(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.savefig(FIG_DIR / "precision_recall.png")
    plt.close()


def main():
    final_path = DATA_DIR / "final_reviews.csv"
    ksel_path = DATA_DIR / "k_selection.csv"

    if not final_path.exists():
        raise FileNotFoundError("Run pipeline first")

    df = pd.read_csv(final_path)

    # --- EDA ---
    plot_missing_values(df)
    plot_text_length(df)

    # --- Clustering ---
    plot_cluster_distribution(df)
    plot_pca(df)

    if ksel_path.exists():
        k_df = pd.read_csv(ksel_path)
        plt.figure(figsize=(8, 4))
        plt.plot(k_df["k"], k_df["silhouette"], marker="o")
        plt.title("K Selection via Silhouette")
        plt.savefig(FIG_DIR / "k_selection.png")
        plt.close()

    # --- Business Insights ---
    plot_platform_distribution(df)
    plot_severity_distribution(df)
    plot_platform_vs_severity(df)
    plot_time_trend(df)

    log.info(f"All visualizations saved to {FIG_DIR}")


if __name__ == "__main__":
    main()