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

palette_husl = sns.color_palette("husl", 10)

# -------------------------------
# CHART 1: Correlation Heatmap
# -------------------------------
if not numeric_df.empty:
    plt.figure()
    sns.heatmap(numeric_df.corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    save_plot("chart_1_heatmap")

# -------------------------------
# CHART 2–4: Distributions
# -------------------------------
for i, (col, color, title) in enumerate([
    ('score', 'purple', "Score Distribution"),
    ('review_length', 'green', "Review Length"),
    ('thumbs_up', 'orange', "Thumbs Up")
], start=2):
    if col in df.columns:
        plt.figure()
        sns.histplot(df[col], kde=not FAST_MODE, color=color)
        plt.title(title)
        save_plot(f"chart_{i}_{col}")

# -------------------------------
# CHART 5: Boxplot
# -------------------------------
if {'kmeans_cluster', 'score'}.issubset(df.columns):
    plt.figure()
    sns.boxplot(x='kmeans_cluster', y='score', data=df, palette='Set2')
    plt.title("Score vs Cluster")
    save_plot("chart_5_boxplot")

# -------------------------------
# CHART 6: Scatter
# -------------------------------
if {'review_length', 'score'}.issubset(df.columns):
    plt.figure()
    sns.scatterplot(
        x='review_length', y='score',
        hue='score', palette='viridis', data=df
    )
    plt.title("Score vs Review Length")
    save_plot("chart_6_scatter")

# -------------------------------
# CHART 7–8: Countplots
# -------------------------------
for i, col in enumerate(['kmeans_cluster', 'hdbscan_cluster'], start=7):
    if col in df.columns:
        plt.figure()
        sns.countplot(x=col, data=df, palette='tab10')
        plt.title(f"{col} Distribution")
        save_plot(f"chart_{i}_{col}")

# -------------------------------
# CHART 9–10: Anomaly Scores
# -------------------------------
for i, col in enumerate(['isolation_score', 'ocsvm_score'], start=9):
    if col in df.columns:
        plt.figure()
        color = palette_husl[i % len(palette_husl)]
        sns.histplot(df[col], kde=not FAST_MODE, color=color)
        plt.title(f"{col} Distribution")
        save_plot(f"chart_{i}_{col}")

# -------------------------------
# CHART 11: Pairplot (optimized)
# -------------------------------
pair_cols = [c for c in ['score', 'review_length', 'thumbs_up'] if c in df.columns]
if len(pair_cols) >= 2:
    sample_df = df[pair_cols].sample(n=min(1000, len(df)), random_state=42)
    g = sns.pairplot(sample_df, palette='husl')
    g.fig.suptitle("Pairplot", y=1.02)
    g.savefig(FIG_DIR / "chart_11_pairplot.png", dpi=150 if FAST_MODE else 300)
    plt.close()

# -------------------------------
# CHART 12: Violin
# -------------------------------
if 'score' in df.columns:
    plt.figure()
    sns.violinplot(y=df['score'], color='skyblue')
    plt.title("Score Distribution")
    save_plot("chart_12_violin")

# -------------------------------
# CHART 13: UMAP
# -------------------------------
if {'umap_1', 'umap_2', 'kmeans_cluster'}.issubset(df.columns):
    plt.figure()
    sns.scatterplot(
        x='umap_1', y='umap_2',
        hue='kmeans_cluster', palette='tab10', data=df
    )
    plt.title("UMAP Projection")
    save_plot("chart_13_umap")

# -------------------------------
# CHART 14: Trend
# -------------------------------
if 'score' in df.columns:
    plt.figure()
    sorted_scores = df['score'].sort_values().reset_index(drop=True)
    plt.plot(sorted_scores, color='black')
    plt.title("Sorted Score Trend")
    save_plot("chart_14_trend")

# -------------------------------
# CHART 15: KDE
# -------------------------------
if {'score', 'review_length'}.issubset(df.columns):
    plt.figure()
    sns.kdeplot(df['score'], label='Score')
    sns.kdeplot(df['review_length'], label='Review Length')
    plt.legend()
    plt.title("KDE Comparison")
    save_plot("chart_15_kde")

# -------------------------------
# CHART 16: Barplot
# -------------------------------
if {'kmeans_cluster', 'score'}.issubset(df.columns):
    plt.figure()
    cluster_means = df.groupby('kmeans_cluster')['score'].mean().reset_index()
    sns.barplot(x='kmeans_cluster', y='score', data=cluster_means, palette='Set1')
    plt.title("Avg Score per Cluster")
    save_plot("chart_16_bar")

# -------------------------------
# CHART 17: Boxplot
# -------------------------------
if {'kmeans_cluster', 'review_length'}.issubset(df.columns):
    plt.figure()
    sns.boxplot(x='kmeans_cluster', y='review_length', data=df, palette='pastel')
    plt.title("Review Length vs Cluster")
    save_plot("chart_17_box")

# -------------------------------
# CHART 18: Scatter
# -------------------------------
if {'isolation_score', 'ocsvm_score'}.issubset(df.columns):
    plt.figure()
    sns.scatterplot(
        x='isolation_score', y='ocsvm_score',
        color='darkgreen', data=df
    )
    plt.title("Isolation vs OC-SVM")
    save_plot("chart_18_scatter")

# -------------------------------
# CHART 19: UMAP Heatmap
# -------------------------------
umap_cols = [col for col in df.columns if 'umap' in col]
if len(umap_cols) > 1:
    plt.figure()
    sns.heatmap(df[umap_cols].corr(), cmap='viridis')
    plt.title("UMAP Correlation")
    save_plot("chart_19_umap_heatmap")

# -------------------------------
# CHART 20: Pie Chart
# -------------------------------
if 'kmeans_cluster' in df.columns:
    plt.figure()
    df['kmeans_cluster'].value_counts().plot.pie(
        autopct='%1.1f%%', colormap='Set3'
    )
    plt.title("Cluster Distribution")
    plt.ylabel("")
    save_plot("chart_20_pie")