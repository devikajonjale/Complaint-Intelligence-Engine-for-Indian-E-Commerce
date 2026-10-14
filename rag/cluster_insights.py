"""LLM labels and summaries for every K-Means and HDBSCAN cluster.

For each cluster we take the reviews whose Sentence-BERT vectors are closest to the cluster centroid
(the most representative ones) and ask the LLM for structured output validated by Pydantic.
Results are cached to rag/cache/cluster_insights.json keyed by a hash of the sampled review ids,
so re-running only calls the LLM for clusters whose membership changed.

Usage: python rag/cluster_insights.py [--n 12] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from rag.build_index import load_reviews  # noqa: E402
from rag.config import CACHE_DIR, DATA_DIR, get_llm, robust_invoke  # noqa: E402

CACHE_FILE = CACHE_DIR / "cluster_insights.json"
TEAMS = ["Refund & Payment Issues", "Delivery Failure", "App Crash / UX Bug", "Fake/Counterfeit Product", "Other"]


class ClusterInsight(BaseModel):
    label: str = Field(description="Short cluster label, 2 to 5 words")
    summary: str = Field(description="Two sentence plain-English summary of what these customers say")
    root_causes: list[str] = Field(description="Top 3 likely root causes, each under 12 words")
    owning_team: str = Field(description=f"Exactly one of: {', '.join(TEAMS)}")


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You analyse clusters of customer reviews from Indian e-commerce apps (some in Hindi or Hinglish). "
            "Base everything strictly on the sample reviews and statistics given. Do not use dashes as punctuation.",
        ),
        (
            "human",
            "Cluster statistics: {stats}\n\n{focus}\n\nRepresentative reviews (closest to centroid):\n{reviews}\n\n"
            "Return a short label naming the concrete problem rather than a sentiment, a 2 sentence summary, "
            "the top 3 root causes and the owning team.",
        ),
    ]
)


MIN_COMPLAINTS = 10  # below this a cluster is labelled from all its members instead


def representative_reviews(emb: np.ndarray, mask: np.ndarray, n: int, pick_from: np.ndarray | None = None) -> np.ndarray:
    """Row indices of the n reviews closest (cosine) to the cluster centroid.

    The centroid is always computed from the whole cluster, so it still represents the cluster's
    position. `pick_from` narrows which members may be returned, which lets us describe a cluster
    through its complaints rather than its average review.
    """
    idx = np.where(mask)[0]
    centroid = emb[idx].mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    pool = np.where(mask & pick_from)[0] if pick_from is not None else idx
    return pool[np.argsort(-(emb[pool] @ centroid))[:n]]


def cluster_stats(df, mask) -> dict:
    sub = df[mask]
    return {
        "size": int(mask.sum()),
        "share_pct": round(100 * float(mask.mean()), 1),
        "avg_rating": round(float(sub["score"].mean()), 2),
        "anomalies": int(sub["isolation_anomaly"].sum()),
        "red_severity": int((sub["severity"] == "RED").sum()),
        "top_platforms": sub["platform"].value_counts().head(3).to_dict(),
        "top_routing_team": sub["route_category"].mode().iat[0] if "route_category" in sub else None,
    }


def generate(n: int = 12, force: bool = False, complaints_only: bool = True) -> dict:
    """Label every cluster. With complaints_only (default) the representative reviews are drawn from
    1 and 2 star reviews, because labelling from all members produced uninformative sentiment labels
    ("mostly positive") rather than the actionable complaint themes the dashboard needs."""
    df = load_reviews()
    emb = np.load(DATA_DIR / "embeddings.npy")
    cache = {} if force or not CACHE_FILE.exists() else json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    chain = PROMPT | get_llm(temperature=0.0).with_structured_output(ClusterInsight)
    complaint_rows = (df["score"] <= 2).to_numpy()

    for algo, col in [("kmeans", "kmeans_cluster"), ("hdbscan", "hdbscan_cluster")]:
        for cid in sorted(df[col].unique()):
            if algo == "hdbscan" and cid == -1:
                continue  # HDBSCAN noise is not a cluster
            key = f"{algo}_{cid}"
            mask = (df[col] == cid).to_numpy()
            # Fall back to all members when a cluster has too few complaints to characterise.
            focus = complaints_only and int((mask & complaint_rows).sum()) >= MIN_COMPLAINTS
            rep = representative_reviews(emb, mask, n, complaint_rows if focus else None)
            sig = hashlib.md5((("c" if focus else "a") + ",".join(df["sid"].iloc[rep])).encode()).hexdigest()
            if cache.get(key, {}).get("signature") == sig:
                continue  # cached and membership unchanged
            stats = cluster_stats(df, mask)
            stats["complaints_1_2_star"] = int((mask & complaint_rows).sum())
            reviews = "\n".join(f"[{df['sid'].iat[i]}] {str(df['content'].iat[i])[:300]}" for i in rep)
            insight: ClusterInsight = robust_invoke(
                chain,
                {
                    "stats": json.dumps(stats),
                    "reviews": reviews,
                    "focus": "The reviews below are this cluster's 1 and 2 star complaints."
                    if focus
                    else "This cluster has too few complaints to sample, so these are its typical reviews.",
                },
            )
            cache[key] = {
                "algorithm": algo,
                "cluster_id": int(cid),
                "pipeline_name": df.loc[mask, "cluster_name"].mode().iat[0] if algo == "kmeans" else None,
                "complaint_focused": bool(focus),
                **insight.model_dump(),
                "stats": stats,
                "representative_ids": df["sid"].iloc[rep].tolist(),
                "signature": sig,
            }
            print(f"{key}: {insight.label}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return cache


def load_insights() -> dict:
    return json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--all-reviews", action="store_true", help="label from all members, not just complaints")
    a = ap.parse_args()
    generate(n=a.n, force=a.force, complaints_only=not a.all_reviews)
    print(f"Saved {CACHE_FILE}")
