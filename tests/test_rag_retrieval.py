"""Tests for the RAG retrieval layer. No API key and no LLM calls, so these run in CI."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import retriever  # noqa: E402
from rag.build_index import MANIFEST, load_reviews, short_id  # noqa: E402
from rag.chain import CITE_RE, build_filter, format_docs  # noqa: E402

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="run python rag/build_index.py first")


@pytest.fixture(scope="module")
def corpus():
    return retriever._corpus()


# ── ids and metadata ──────────────────────────────────────────────────────────
def test_short_id_is_zero_padded():
    assert short_id(0) == "R0000"
    assert short_id(42) == "R0042"
    assert CITE_RE.fullmatch(short_id(2240))


def test_index_row_count_matches_reviews_and_embeddings(corpus):
    df, store, _, sids = corpus
    embeddings = np.load(ROOT / "data" / "embeddings.npy")
    assert len(df) == store.index.ntotal == len(embeddings) == len(sids)


def test_every_document_carries_the_metadata_the_ui_reads(corpus):
    _, store, _, sids = corpus
    doc = store.docstore.search(sids[0])
    for field in ("sid", "review_id", "platform", "rating", "date", "severity", "routing_team", "is_anomaly"):
        assert field in doc.metadata


# ── tokenizer ─────────────────────────────────────────────────────────────────
def test_tokenizer_keeps_latin_devanagari_and_digits_and_drops_punctuation():
    assert retriever._tokenize("Refund NOT received!! 20 days") == ["refund", "not", "received", "20", "days"]
    assert retriever._tokenize("पैसा वापस नहीं") == ["पैसा", "वापस", "नहीं"]


def test_tokenizer_drops_emoji_only_text():
    assert retriever._tokenize("😞🙏") == []


# ── metadata filtering ────────────────────────────────────────────────────────
def test_allowed_mask_is_none_without_filters(corpus):
    df, *_ = corpus
    assert retriever.allowed_mask(df, None) is None


def test_allowed_mask_matches_pandas_for_platform_and_severity(corpus):
    df, *_ = corpus
    mask = retriever.allowed_mask(df, {"platform": "Meesho", "severity": ["RED", "AMBER"]})
    expected = (df["platform"] == "Meesho") & df["severity"].isin(["RED", "AMBER"])
    assert mask.sum() == expected.sum() > 0


def test_allowed_mask_anomaly_only_selects_flagged_rows(corpus):
    df, *_ = corpus
    mask = retriever.allowed_mask(df, {"is_anomaly": True})
    assert mask.sum() == int(df["isolation_anomaly"].sum())


@pytest.mark.parametrize("variant", ["dense", "hybrid"])
def test_platform_filter_is_respected_by_every_variant(variant):
    hits = retriever.search("refund not received", k=5, filters={"platform": "Meesho"}, variant=variant)
    assert hits and all(d.metadata["platform"] == "Meesho" for d, _ in hits)


def test_build_filter_omits_empty_values_and_keeps_set_ones():
    assert build_filter() is None
    assert build_filter(platform=[], severity=None) is None
    assert build_filter(platform="Nykaa", anomaly_only=True) == {"platform": "Nykaa", "is_anomaly": True}


# ── ranking ───────────────────────────────────────────────────────────────────
def test_rrf_ranks_a_document_agreed_by_both_rankings_first():
    dense = [("A", 0.9), ("B", 0.8), ("C", 0.7)]
    lexical = [("C", 5.0), ("B", 4.0), ("Z", 1.0)]
    # B is 2nd in both; C is 3rd and 1st; A only appears once.
    assert retriever._rrf([dense, lexical])[0] in {"B", "C"}
    assert set(retriever._rrf([dense, lexical])) == {"A", "B", "C", "Z"}


def test_rrf_is_stable_with_a_single_ranking():
    ranking = [("A", 1.0), ("B", 0.5)]
    assert retriever._rrf([ranking]) == ["A", "B"]


@pytest.mark.parametrize("variant", ["dense", "hybrid"])
def test_search_returns_k_unique_documents(variant):
    hits = retriever.search("delivery was very late", k=6, variant=variant)
    sids = [d.metadata["sid"] for d, _ in hits]
    assert len(hits) == 6
    assert len(set(sids)) == 6


def test_hybrid_finds_an_exact_lexical_match_for_a_rare_token(corpus):
    """BM25's job: a review containing a rare acronym should be reachable even if the embedding is far."""
    df, *_ = corpus
    rare = df[df["content"].astype(str).str.contains(r"\bOTP\b", case=False, regex=True)]
    if rare.empty:
        pytest.skip("corpus has no OTP reviews")
    hits = retriever.search("OTP", k=10, variant="hybrid")
    assert any("otp" in d.page_content.lower() for d, _ in hits)


def test_invalid_variant_is_rejected():
    with pytest.raises(ValueError):
        retriever.search("anything", variant="magic")


# ── prompt context ────────────────────────────────────────────────────────────
def test_format_docs_prefixes_each_review_with_its_citable_id():
    hits = retriever.search("app keeps crashing", k=3, variant="hybrid")
    context = format_docs(hits)
    for doc, _ in hits:
        assert f"[{doc.metadata['sid']}]" in context
    assert len(CITE_RE.findall(context)) >= 3
