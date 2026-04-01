"""
Step 4 — Dimensionality Reduction
Input:  data/cleaned_reviews.csv, data/embeddings.npy
Output: data/pca_50.npy, data/umap_10.npy, data/tsne_2d.npy
        models/pca_model.pkl, models/umap_model.pkl
        data/variance_report.csv
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def main() -> None:
    emb_path = DATA_DIR / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"{emb_path} not found — run 03_embed.py first")

    embeddings = np.load(emb_path)
    log.info(f"Loaded embeddings: {embeddings.shape}")

    pca = PCA(n_components=50, random_state=42)
    pca_50 = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_
    cum_explained = explained.cumsum()
    log.info(f"PCA-50 cumulative explained variance: {cum_explained[-1]:.4f}")

    var_df = pd.DataFrame(
        {
            "component": np.arange(1, len(explained) + 1),
            "explained_variance": explained,
            "cumulative_explained_variance": cum_explained,
        }
    )
    var_df.to_csv(DATA_DIR / "variance_report.csv", index=False)

    reducer = umap.UMAP(
        n_components=10,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    umap_10 = reducer.fit_transform(pca_50)
    log.info(f"UMAP-10 shape: {umap_10.shape}")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    tsne_2d = tsne.fit_transform(umap_10)
    log.info(f"t-SNE shape: {tsne_2d.shape}")

    np.save(DATA_DIR / "pca_50.npy", pca_50)
    np.save(DATA_DIR / "umap_10.npy", umap_10)
    np.save(DATA_DIR / "tsne_2d.npy", tsne_2d)

    joblib.dump(pca, MODEL_DIR / "pca_model.pkl")
    joblib.dump(reducer, MODEL_DIR / "umap_model.pkl")
    log.info("Saved reduced embeddings and models.")


if __name__ == "__main__":
    main()
