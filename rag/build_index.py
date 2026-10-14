"""Build a persistent FAISS index of all reviews.

Reuses the pipeline's precomputed Sentence-BERT vectors (data/embeddings.npy) so the index
is identical to what clustering and anomaly detection saw. Queries are embedded at runtime with
the same model through langchain_huggingface.

Idempotent: a fingerprint of the input files is stored next to the index; if nothing changed the
build is skipped. Use --force to rebuild.

Usage: python rag/build_index.py [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.config import DATA_DIR, INDEX_DIR, get_embeddings, reviews_path  # noqa: E402

MANIFEST = INDEX_DIR / "manifest.json"
ID_MAP = INDEX_DIR / "id_map.csv"


def short_id(i: int) -> str:
    """Short, LLM-friendly citation id. UUIDs are easy for a model to garble."""
    return f"R{i:04d}"


def _fingerprint(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def load_reviews() -> pd.DataFrame:
    """Pipeline output with a short id column. Row order matches embeddings.npy."""
    df = pd.read_csv(reviews_path(), encoding="utf-8-sig")
    df["sid"] = [short_id(i) for i in range(len(df))]
    return df


def _metadata(row: pd.Series) -> dict:
    def clean(v):
        if pd.isna(v):
            return None
        return v.item() if hasattr(v, "item") else v

    return {
        "sid": row["sid"],
        "review_id": row["review_id"],
        "platform": clean(row.get("platform")),
        "rating": clean(row.get("score")),
        "date": str(row.get("at"))[:10],
        "kmeans_cluster": clean(row.get("kmeans_cluster")),
        "cluster_name": clean(row.get("cluster_name")),
        "hdbscan_cluster": clean(row.get("hdbscan_cluster")),
        "severity": clean(row.get("severity")),
        "severity_ml": clean(row.get("severity_label_ml")),
        "is_anomaly": bool(row.get("isolation_anomaly", False)),
        "routing_team": clean(row.get("route_category")),
        "lang": clean(row.get("lang")),
    }


def build(force: bool = False) -> Path:
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_core.documents import Document
    import faiss

    src = reviews_path()
    emb_path = DATA_DIR / "embeddings.npy"
    fp = _fingerprint([src, emb_path])
    if not force and MANIFEST.exists() and json.loads(MANIFEST.read_text())["fingerprint"] == fp:
        print(f"Index up to date at {INDEX_DIR} (skip). Use --force to rebuild.")
        return INDEX_DIR

    df = load_reviews()
    vecs = np.load(emb_path).astype("float32")
    if len(vecs) != len(df):
        raise ValueError(f"embeddings.npy has {len(vecs)} rows but {src.name} has {len(df)}")
    faiss.normalize_L2(vecs)  # cosine similarity via inner product

    docs = [Document(page_content=str(r["content"]), metadata=_metadata(r)) for _, r in df.iterrows()]
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    ids = df["sid"].tolist()
    store = FAISS(
        embedding_function=get_embeddings(),
        index=index,
        docstore=InMemoryDocstore(dict(zip(ids, docs))),
        index_to_docstore_id=dict(enumerate(ids)),
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        normalize_L2=True,
    )
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))
    df[["sid", "review_id"]].to_csv(ID_MAP, index=False)
    MANIFEST.write_text(json.dumps({"fingerprint": fp, "source": src.name, "n_docs": len(docs)}, indent=2))
    print(f"Built FAISS index with {len(docs)} reviews from {src.name} -> {INDEX_DIR}")
    return INDEX_DIR


def load_store():
    """Load the persisted index (building it first if missing)."""
    from langchain_community.vectorstores import FAISS

    if not MANIFEST.exists():
        build()
    return FAISS.load_local(str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    build(force=ap.parse_args().force)
