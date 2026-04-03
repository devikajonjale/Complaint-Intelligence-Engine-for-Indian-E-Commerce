import logging
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# CONFIG
# -------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (10, 6)

FAST_MODE = True

# -------------------------------
# SAVE FUNCTION
# -------------------------------
def save_plot(name):
    path = FIG_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150 if FAST_MODE else 300)
    log.info(f"Saved: {path}")
    plt.close()

# -------------------------------
# LOAD DATA
# -------------------------------
def load_data():
    candidates = [
        DATA_DIR / "final_reviews_response.csv",
        DATA_DIR / "final_reviews_churn.csv",
        DATA_DIR / "final_reviews_routed.csv",
        DATA_DIR / "final_reviews_scored.csv",
        DATA_DIR / "final_reviews.csv",
    ]

    final_path = next((p for p in candidates if p.exists()), None)

    if final_path is None:
        raise FileNotFoundError("No dataset found in /data")

    df = pd.read_csv(final_path, encoding="utf-8-sig")

    if 'content' in df.columns and 'review_length' not in df.columns:
        df['review_length'] = df['content'].astype(str).apply(len)

    if FAST_MODE:
        df = df.sample(n=min(5000, len(df)), random_state=42)

    return df

df = load_data()
numeric_df = df.select_dtypes(include=[np.number])

palette = sns.color_palette("husl", 10)

# -------------------------------
# CHART 21: Correlation Heatmap
# -------------------------------
if not numeric_df.empty:
    plt.figure()
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Correlation Heatmap of Numerical Features")
    plt.xlabel("Features")
    plt.ylabel("Features")
    save_plot("chart_21_heatmap")

# -------------------------------
# CHART 22–24: Distributions
# -------------------------------
for i, col in enumerate(['score', 'review_length', 'thumbs_up'], start=22):
    if col in df.columns:
        plt.figure()
        sns.histplot(df[col], kde=not FAST_MODE, color=palette[i % 10])
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        save_plot(f"chart_{i}_{col}")

# -------------------------------
# CHART 25: Boxplot
# -------------------------------
if {'kmeans_cluster', 'score'}.issubset(df.columns):
    plt.figure()
    sns.boxplot(x='kmeans_cluster', y='score', data=df, palette='Set2')
    plt.title("Score Distribution across Clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Score")
    save_plot("chart_25_boxplot")

# -------------------------------
# CHART 26: Scatter
# -------------------------------
if {'review_length', 'score'}.issubset(df.columns):
    plt.figure()
    sns.scatterplot(
        x='review_length', y='score',
        hue='score', palette='viridis', data=df
    )
    plt.title("Score vs Review Length")
    plt.xlabel("Review Length")
    plt.ylabel("Score")
    plt.legend(title="Score")
    save_plot("chart_26_scatter")

# -------------------------------
# CHART 27–28: Countplots
# -------------------------------
for i, col in enumerate(['kmeans_cluster', 'hdbscan_cluster'], start=27):
    if col in df.columns:
        plt.figure()
        ax = sns.countplot(x=col, data=df, palette='tab10')
        plt.title(f"{col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Count")

        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom')

        save_plot(f"chart_{i}_{col}")

# -------------------------------
# CHART 29–30: Anomaly Scores
# -------------------------------
for i, col in enumerate(['isolation_score', 'ocsvm_score'], start=29):
    if col in df.columns:
        plt.figure()
        sns.histplot(df[col], kde=not FAST_MODE, color=palette[i % 10])
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        save_plot(f"chart_{i}_{col}")

# -------------------------------
# CHART 31: Pairplot
# -------------------------------
pair_cols = [c for c in ['score', 'review_length', 'thumbs_up'] if c in df.columns]
if len(pair_cols) >= 2:
    sample_df = df[pair_cols].sample(n=min(1000, len(df)), random_state=42)
    g = sns.pairplot(sample_df, palette='husl')
    g.fig.suptitle("Pairwise Relationships", y=1.02)
    g.savefig(FIG_DIR / "chart_31_pairplot.png", dpi=150 if FAST_MODE else 300)
    plt.close()

# -------------------------------
# CHART 32: Violin
# -------------------------------
if 'score' in df.columns:
    plt.figure()
    sns.violinplot(y=df['score'], color='skyblue')
    plt.title("Score Distribution (Violin Plot)")
    plt.ylabel("Score")
    save_plot("chart_32_violin")

# -------------------------------
# CHART 33: UMAP
# -------------------------------
if {'umap_1', 'umap_2', 'kmeans_cluster'}.issubset(df.columns):
    plt.figure()
    sns.scatterplot(
        x='umap_1', y='umap_2',
        hue='kmeans_cluster', palette='tab10', data=df
    )
    plt.title("UMAP Projection of Clusters")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(title="Cluster")
    save_plot("chart_33_umap")

# -------------------------------
# CHART 34: Trend
# -------------------------------
if 'score' in df.columns:
    plt.figure()
    sorted_scores = df['score'].sort_values().reset_index(drop=True)
    plt.plot(sorted_scores, color='black')
    plt.title("Sorted Score Trend")
    plt.xlabel("Index")
    plt.ylabel("Score")
    save_plot("chart_34_trend")

# -------------------------------
# CHART 35: KDE
# -------------------------------
if {'score', 'review_length'}.issubset(df.columns):
    plt.figure()
    sns.kdeplot(df['score'], label='Score')
    sns.kdeplot(df['review_length'], label='Review Length')
    plt.legend()
    plt.title("KDE Comparison")
    plt.xlabel("Value")
    plt.ylabel("Density")
    save_plot("chart_35_kde")

# -------------------------------
# CHART 36: Barplot
# -------------------------------
if {'kmeans_cluster', 'score'}.issubset(df.columns):
    plt.figure()
    cluster_means = df.groupby('kmeans_cluster')['score'].mean().reset_index()
    ax = sns.barplot(x='kmeans_cluster', y='score', data=cluster_means, palette='Set1')

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom')

    plt.title("Average Score per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Average Score")
    save_plot("chart_36_bar")

# -------------------------------
# CHART 37: Boxplot
# -------------------------------
if {'kmeans_cluster', 'review_length'}.issubset(df.columns):
    plt.figure()
    sns.boxplot(x='kmeans_cluster', y='review_length', data=df, palette='pastel')
    plt.title("Review Length vs Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Review Length")
    save_plot("chart_37_box")

# -------------------------------
# CHART 38: Scatter
# -------------------------------
if {'isolation_score', 'ocsvm_score'}.issubset(df.columns):
    plt.figure()
    sns.scatterplot(
        x='isolation_score', y='ocsvm_score',
        color='darkgreen', data=df
    )
    plt.title("Isolation Forest vs OC-SVM Scores")
    plt.xlabel("Isolation Score")
    plt.ylabel("OC-SVM Score")
    save_plot("chart_38_scatter")

# -------------------------------
# CHART 39: UMAP Heatmap
# -------------------------------
umap_cols = [col for col in df.columns if 'umap' in col]
if len(umap_cols) > 1:
    plt.figure()
    sns.heatmap(df[umap_cols].corr(), cmap='viridis', annot=True)
    plt.title("UMAP Feature Correlation")
    plt.xlabel("UMAP Features")
    plt.ylabel("UMAP Features")
    save_plot("chart_39_umap_heatmap")

# -------------------------------
# CHART 40: Pie Chart
# -------------------------------
if 'kmeans_cluster' in df.columns:
    plt.figure()
    df['kmeans_cluster'].value_counts().plot.pie(
        autopct='%1.1f%%', colormap='Set3'
    )
    plt.title("Cluster Distribution")
    plt.ylabel("")
    save_plot("chart_40_pie")