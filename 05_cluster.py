"""
Step 5 — Clustering and Complaint Archetypes
Input:  data/cleaned_reviews.csv, data/umap_10.npy, data/tfidf_matrix.npz
Output: data/clustered_reviews.csv, data/cluster_profiles.csv
        data/k_selection.csv, models/kmeans.pkl
"""

import json
import logging
from pathlib import Path

import hdbscan
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DEFAULT_LABELS = {
    0: "Delivery Failure",
    1: "Refund & Payment Issues",
    2: "Fake/Counterfeit Product",
    3: "App Crash / UX Bug",
    4: "Size/Quality Mismatch",
    5: "Positive Experience",
    6: "Return/Cancellation Problem",
    7: "Other Complaints",
}


def top_terms_for_cluster(tfidf_matrix, labels, cluster_id, vocab, top_n=10):
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        return []
    mean_scores = np.asarray(tfidf_matrix[idx].mean(axis=0)).ravel()
    top_idx = np.argsort(mean_scores)[::-1][:top_n]
    return [vocab[i] for i in top_idx]


def choose_best_k(umap_10: np.ndarray) -> tuple[int, pd.DataFrame]:
    rows = []
    best_k = 6
    best_score = -1.0
    for k in range(3, 13):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(umap_10)
        sil = silhouette_score(umap_10, labels)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        if sil > best_score:
            best_score = sil
            best_k = k
    return best_k, pd.DataFrame(rows)


def main() -> None:
    df_path = DATA_DIR / "cleaned_reviews.csv"
    umap_path = DATA_DIR / "umap_10.npy"
    tfidf_path = DATA_DIR / "tfidf_matrix.npz"
    vocab_path = DATA_DIR / "tfidf_vocab.json"

    for p in [df_path, umap_path, tfidf_path, vocab_path]:
        if not p.exists():
            raise FileNotFoundError(f"{p} missing. Run previous steps first.")

    df = pd.read_csv(df_path, encoding="utf-8-sig")
    umap_10 = np.load(umap_path)
    tfidf = load_npz(tfidf_path)
    vocab = json.loads((DATA_DIR / "tfidf_vocab.json").read_text(encoding="utf-8"))

    best_k, k_eval = choose_best_k(umap_10)
    k_eval.to_csv(DATA_DIR / "k_selection.csv", index=False)
    log.info(f"Selected k={best_k} by max silhouette")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    km_labels = kmeans.fit_predict(umap_10)

    hdb = hdbscan.HDBSCAN(min_cluster_size=30, metric="euclidean")
    hdb_labels = hdb.fit_predict(umap_10)

    ari = adjusted_rand_score(km_labels, hdb_labels)
    log.info(f"KMeans vs HDBSCAN ARI: {ari:.4f}")

    cluster_names = {}
    cluster_profiles = []
    for cid in sorted(np.unique(km_labels)):
        terms = top_terms_for_cluster(tfidf, km_labels, cid, vocab, top_n=10)
        cluster_name = DEFAULT_LABELS.get(int(cid), f"Cluster {cid}")
        cluster_names[int(cid)] = cluster_name
        cluster_profiles.append(
            {
                "cluster_id": int(cid),
                "cluster_name": cluster_name,
                "size": int((km_labels == cid).sum()),
                "top_terms": ", ".join(terms),
            }
        )

    df["kmeans_cluster"] = km_labels
    df["cluster_name"] = df["kmeans_cluster"].map(cluster_names)
    df["hdbscan_cluster"] = hdb_labels
    df["hdbscan_is_noise"] = df["hdbscan_cluster"] == -1

    for i in range(10):
        df[f"umap_{i+1}"] = umap_10[:, i]

    clustered_out = DATA_DIR / "clustered_reviews.csv"
    profiles_out = DATA_DIR / "cluster_profiles.csv"
    df.to_csv(clustered_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(cluster_profiles).to_csv(profiles_out, index=False, encoding="utf-8-sig")

    joblib.dump(kmeans, MODEL_DIR / "kmeans.pkl")
    joblib.dump(hdb, MODEL_DIR / "hdbscan.pkl")
    log.info(f"Saved {clustered_out} and {profiles_out}")


if __name__ == "__main__":
    main()
