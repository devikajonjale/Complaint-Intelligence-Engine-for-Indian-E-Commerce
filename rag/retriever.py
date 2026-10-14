"""Retrieval for the RAG layer: dense, hybrid (dense + BM25) and optional cross-encoder reranking.

Why hybrid: dense vectors alone miss exact tokens that matter in these reviews (COD, OTP, UPI,
a brand or product name), and Hinglish spelling varies a lot ("paisa" / "paise" / "pesa"), so a
lexical signal recovers reviews the embedding places far away. Dense and BM25 rankings are merged
with Reciprocal Rank Fusion, which needs no score normalisation across two different scales.

Optional reranking: a multilingual cross-encoder re-scores the fused shortlist. It runs locally on
CPU, so it costs no API calls, and it reads the query and review together instead of comparing two
independent vectors.

Three variants are selectable so they can be measured against each other (see rag/evaluate.py):
    dense   : FAISS cosine only (the original behaviour)
    hybrid  : dense + BM25 fused with RRF            [default]
    rerank  : hybrid shortlist re-scored by the cross-encoder

Metadata filters (platform, severity, anomaly, routing team) are applied to the candidate pool
before any ranking, so every variant searches exactly the same subset.
"""
from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from langchain_core.documents import Document  # noqa: E402

from rag.build_index import load_reviews, load_store  # noqa: E402

# Variant used by the app and CLI unless overridden. "rerank" is the most accurate but adds
# local CPU time per query; "hybrid" is the balanced default.
DEFAULT_VARIANT = os.getenv("RAG_RETRIEVER", "hybrid").split("#")[0].strip() or "hybrid"
VARIANTS = ("dense", "hybrid", "rerank")

# Free, locally run multilingual reranker (no API calls). Downloaded once by sentence-transformers.
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

CANDIDATES = 40  # shortlist size fused/reranked before the top k is returned
RRF_K = 60  # standard RRF damping constant

# Keep Latin words, Devanagari words and digits; drop punctuation and emoji.
TOKEN_RE = re.compile(r"[a-z0-9]+|[ऀ-ॿ]+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


@lru_cache(maxsize=1)
def _corpus():
    """Reviews, their vector store and a BM25 index over the raw review text.

    Row order matches the FAISS index and data/embeddings.npy, so a row position identifies a review
    across all three structures.
    """
    from rank_bm25 import BM25Okapi

    df = load_reviews()
    store = load_store()
    bm25 = BM25Okapi([_tokenize(t) for t in df["content"].astype(str)])
    sids = df["sid"].tolist()
    return df, store, bm25, sids


@lru_cache(maxsize=1)
def _reranker():
    """Cross-encoder, or None when the model is unavailable.

    The weights are a one-off ~470 MB download. If it is missing or fails to load (offline machine,
    no disk space), the "rerank" variant falls back to "hybrid" rather than failing the query.
    """
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(RERANK_MODEL, max_length=512)
    except Exception as e:  # noqa: BLE001 - download, disk and backend errors all land here
        print(f"Reranker unavailable ({type(e).__name__}), falling back to hybrid retrieval.", file=sys.stderr)
        return None


def reranker_available() -> bool:
    """Whether reranking can be offered, WITHOUT triggering the download.

    Checking this cheaply matters on hosted environments such as Streamlit Community Cloud, where a
    surprise 464 MB download would exhaust the container's disk and memory budget. Set
    RAG_ENABLE_RERANK=0 to force the option off regardless of what is cached.
    """
    if os.getenv("RAG_ENABLE_RERANK", "1").split("#")[0].strip() in ("0", "false", "no"):
        return False
    try:
        from huggingface_hub import try_to_load_from_cache

        # Look for the weights themselves: a cached config.json alone does not mean a usable model.
        return isinstance(try_to_load_from_cache(RERANK_MODEL, "model.safetensors"), str)
    except Exception:  # noqa: BLE001 - hub offline or unavailable
        return False


def allowed_mask(df, filters: dict | None) -> np.ndarray | None:
    """Boolean mask of rows passing the metadata filters. None means no filtering.

    Mirrors the FAISS metadata filter so the lexical and dense sides see the same candidate pool.
    """
    if not filters:
        return None
    mask = np.ones(len(df), dtype=bool)
    for field, col in [("platform", "platform"), ("severity", "severity"), ("routing_team", "route_category")]:
        if field in filters and col in df.columns:
            want = filters[field]
            want = want if isinstance(want, (list, tuple, set)) else [want]
            mask &= df[col].isin(list(want)).to_numpy()
    if filters.get("is_anomaly"):
        mask &= df["isolation_anomaly"].to_numpy().astype(bool)
    return mask


def _dense_ranking(store, question: str, filters: dict | None, n: int) -> list[tuple[str, float]]:
    """(sid, cosine similarity) best first. fetch_k is widened when filtering so the filter does
    not simply truncate an already-narrow top-n."""
    hits = store.similarity_search_with_score(
        question, k=n, filter=filters, fetch_k=store.index.ntotal if filters else max(n * 3, 60)
    )
    return [(d.metadata["sid"], float(s)) for d, s in hits]


def _bm25_ranking(bm25, sids, question: str, mask: np.ndarray | None, n: int) -> list[tuple[str, float]]:
    scores = bm25.get_scores(_tokenize(question))
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    top = np.argpartition(-scores, min(n, len(scores) - 1))[:n]
    top = top[np.argsort(-scores[top])]
    return [(sids[i], float(scores[i])) for i in top if np.isfinite(scores[i]) and scores[i] > 0]


def _rrf(rankings: list[list[tuple[str, float]]]) -> list[str]:
    """Reciprocal Rank Fusion: each list contributes 1/(RRF_K + rank) to a document's score."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (sid, _) in enumerate(ranking, start=1):
            fused[sid] = fused.get(sid, 0.0) + 1.0 / (RRF_K + rank)
    return [sid for sid, _ in sorted(fused.items(), key=lambda kv: -kv[1])]


def search(question: str, k: int = 8, filters: dict | None = None, variant: str | None = None):
    """Return [(Document, score)] best first. Score is cosine similarity where available,
    otherwise the reranker or fusion score; it is shown in the UI only as a relative strength."""
    variant = (variant or DEFAULT_VARIANT).lower()
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    df, store, bm25, sids = _corpus()

    dense = _dense_ranking(store, question, filters, CANDIDATES if variant != "dense" else k)
    if variant == "dense":
        return [(store.docstore.search(sid), score) for sid, score in dense[:k]]

    lexical = _bm25_ranking(bm25, sids, question, allowed_mask(df, filters), CANDIDATES)
    order = _rrf([dense, lexical])
    cosine = dict(dense)

    model = _reranker() if variant == "rerank" else None
    if model is not None and order:
        shortlist = order[:CANDIDATES]
        docs = [store.docstore.search(sid) for sid in shortlist]
        scores = model.predict([(question, d.page_content) for d in docs])
        ranked = sorted(zip(shortlist, docs, scores), key=lambda t: -t[2])[:k]
        # Report cosine when the dense side saw the review, so UI scores stay comparable.
        return [(d, cosine.get(sid, float(s))) for sid, d, s in ranked]

    picked = order[:k]
    return [(store.docstore.search(sid), cosine.get(sid, 0.0)) for sid in picked]


def warm_up(variant: str | None = None) -> None:
    """Load the index, embedding model and reranker so the first real query is not slow.

    Model loading was the cause of multi-second latency spikes on the first few queries.
    """
    _corpus()
    search("warm up", k=1, variant=variant)


def as_documents(hits) -> list[Document]:
    return [d for d, _ in hits]
