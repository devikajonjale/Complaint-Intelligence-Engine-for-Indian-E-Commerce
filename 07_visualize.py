"""
Step 7 — Visualization of the data
Input:  data/final_reviews.csv
Output: figures/
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

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
# LOAD + CLEAN DATA
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

    # Create review_length if missing
    if 'content' in df.columns and 'review_length' not in df.columns:
        df['review_length'] = df['content'].astype(str).apply(len)

    # Convert numeric-like columns properly
    numeric_candidates = [
        'score', 'review_length', 'thumbs_up',
        'kmeans_cluster', 'hdbscan_cluster',
        'isolation_score', 'ocsvm_score',
        'umap_1', 'umap_2'
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=[col for col in numeric_candidates if col in df.columns])

    if FAST_MODE:
        df = df.sample(n=min(5000, len(df)), random_state=42)

    return df

df = load_data()
numeric_df = df.select_dtypes(include=[np.number])

palette = sns.color_palette("husl", 10)

# -------------------------------
# HEATMAP
# -------------------------------
if not numeric_df.empty:
    plt.figure()
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=True)
    plt.title("Correlation Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Features")
    save_plot("new_heatmap")

# -------------------------------
# DISTRIBUTIONS
# -------------------------------
for col in ['score', 'review_length', 'thumbs_up']:
    if col in df.columns:
        plt.figure()
        sns.histplot(df[col], kde=not FAST_MODE, color=palette[0])
        plt.title(f"{col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        save_plot(f"new_{col}_distribution")

# -------------------------------
# BOXPLOT
# -------------------------------
if {'kmeans_cluster', 'score'}.issubset(df.columns):
    plt.figure()
    sns.boxplot(x='kmeans_cluster', y='score', data=df, palette='Set2')
    plt.xlabel("Cluster")
    plt.ylabel("Score")
    plt.title("Score vs Cluster")
    save_plot("new_score_vs_cluster")

# -------------------------------
# SCATTER
# -------------------------------
if {'review_length', 'score'}.issubset(df.columns):
    plt.figure()
    sns.scatterplot(
        x='review_length', y='score',
        hue='score', palette='viridis', data=df
    )
    plt.xlabel("Review Length")
    plt.ylabel("Score")
    plt.legend(title="Score")
    plt.title("Score vs Review Length")
    save_plot("new_score_vs_length")

# -------------------------------
# COUNTPLOTS
# -------------------------------
for col in ['kmeans_cluster', 'hdbscan_cluster']:
    if col in df.columns:
        plt.figure()
        ax = sns.countplot(x=col, data=df, palette='tab10')
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"{col} Distribution")

        for p in ax.patches:
            ax.annotate(int(p.get_height()),
                        (p.get_x() + p.get_width()/2, p.get_height()),
                        ha='center', va='bottom')

        save_plot(f"new_{col}_count")

# -------------------------------
# ANOMALY DISTRIBUTIONS
# -------------------------------
for col in ['isolation_score', 'ocsvm_score']:
    if col in df.columns:
        plt.figure()
        sns.histplot(df[col], kde=not FAST_MODE, color=palette[1])
        plt.title(f"{col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        save_plot(f"new_{col}_distribution")

# -------------------------------
# PAIRPLOT
# -------------------------------
pair_cols = [c for c in ['score', 'review_length', 'thumbs_up'] if c in df.columns]
if len(pair_cols) >= 2:
    sample_df = df[pair_cols].sample(n=min(1000, len(df)), random_state=42)
    g = sns.pairplot(sample_df)
    g.fig.suptitle("Pairwise Relationships", y=1.02)
    g.savefig(FIG_DIR / "new_pairplot.png")
    plt.close()

# -------------------------------
# VIOLIN
# -------------------------------
if 'score' in df.columns:
    plt.figure()
    sns.violinplot(y=df['score'], color='skyblue')
    plt.title("Score Distribution")
    save_plot("new_violin")

# -------------------------------
# UMAP
# -------------------------------
if {'umap_1', 'umap_2', 'kmeans_cluster'}.issubset(df.columns):
    plt.figure()
    sns.scatterplot(
        x='umap_1', y='umap_2',
        hue='kmeans_cluster', palette='tab10', data=df
    )
    plt.title("UMAP Projection")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(title="Cluster")
    save_plot("new_umap")

# -------------------------------
# TREND
# -------------------------------
if 'score' in df.columns:
    plt.figure()
    sorted_scores = df['score'].sort_values().reset_index(drop=True)
    plt.plot(sorted_scores)
    plt.title("Sorted Score Trend")
    plt.xlabel("Index")
    plt.ylabel("Score")
    save_plot("new_trend")

# -------------------------------
# KDE
# -------------------------------
if {'score', 'review_length'}.issubset(df.columns):
    plt.figure()
    sns.kdeplot(df['score'], label='Score')
    sns.kdeplot(df['review_length'], label='Review Length')
    plt.legend()
    plt.title("KDE Comparison")
    save_plot("new_kde")

# -------------------------------
# BARPLOT
# -------------------------------
if {'kmeans_cluster', 'score'}.issubset(df.columns):
    plt.figure()
    means = df.groupby('kmeans_cluster')['score'].mean().reset_index()
    sns.barplot(x='kmeans_cluster', y='score', data=means, palette='Set1')
    plt.title("Average Score per Cluster")
    save_plot("new_bar")

# -------------------------------
# PIE
# -------------------------------
if 'kmeans_cluster' in df.columns:
    plt.figure()
    df['kmeans_cluster'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title("Cluster Distribution")
    plt.ylabel("")
    save_plot("new_pie")