"""Tests for citation parsing, metric maths and retry behaviour. Pure logic, no network calls."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

from rag.chain import NO_EVIDENCE, _finalise  # noqa: E402
from rag.config import robust_invoke  # noqa: E402
from rag.evaluate import lat_summary, pct, retrieval_block  # noqa: E402


def _hit(sid: str, text: str = "text"):
    return Document(page_content=text, metadata={"sid": sid}), 0.5


# ── citation handling ─────────────────────────────────────────────────────────
def test_finalise_extracts_cited_ids_in_order_without_duplicates():
    out = _finalise({
        "question": "q",
        "answer": "Refunds are slow [R0002][R0001]. Support ignores users [R0002].",
        "sources": [_hit("R0001"), _hit("R0002")],
    })
    assert out["cited_ids"] == ["R0002", "R0001"]
    assert out["invalid_citations"] == []


def test_finalise_flags_a_citation_that_was_never_retrieved():
    out = _finalise({"question": "q", "answer": "Claim [R9999].", "sources": [_hit("R0001")]})
    assert out["invalid_citations"] == ["R9999"]


def test_finalise_reports_no_citations_for_a_refusal():
    out = _finalise({"question": "q", "answer": NO_EVIDENCE, "sources": [_hit("R0001")]})
    assert out["cited_ids"] == []
    assert out["invalid_citations"] == []


def test_finalise_keeps_source_text_and_score():
    out = _finalise({"question": "q", "answer": "a [R0001]", "sources": [_hit("R0001", "late delivery")]})
    assert out["sources"][0]["text"] == "late delivery"
    assert out["sources"][0]["score"] == pytest.approx(0.5)


# ── metric maths ──────────────────────────────────────────────────────────────
def test_percentile_picks_the_expected_order_statistics():
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert pct(xs, 50) == 5
    assert pct(xs, 95) == 10
    assert pct([7], 95) == 7


def test_latency_summary_shape_and_none_for_empty():
    s = lat_summary([0.1, 0.2, 0.3])
    assert set(s) == {"mean", "p50", "p95"}
    assert s["mean"] == pytest.approx(0.2)
    assert lat_summary([]) is None


def test_retrieval_block_computes_hit_precision_and_recall():
    per_q = [
        # one relevant document, retrieved first
        {"retrieval_scored": True, "rr": 1.0, "hit@1": 1, "hit@3": 1, "hit@5": 1, "hit@8": 1,
         "precision@1": 1.0, "precision@3": 0.333, "precision@5": 0.2, "precision@8": 0.125,
         "recall@1": 1.0, "recall@3": 1.0, "recall@5": 1.0, "recall@8": 1.0},
        # nothing relevant retrieved
        {"retrieval_scored": True, "rr": 0.0, "hit@1": 0, "hit@3": 0, "hit@5": 0, "hit@8": 0,
         "precision@1": 0.0, "precision@3": 0.0, "precision@5": 0.0, "precision@8": 0.0,
         "recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0, "recall@8": 0.0},
    ]
    block = retrieval_block(per_q)
    assert block["hit_rate@1"] == 0.5
    assert block["mrr@8"] == 0.5
    assert block["recall@8"] == 0.5


def test_retrieval_block_is_empty_when_nothing_is_scored():
    assert retrieval_block([{"retrieval_scored": False}]) == {}


# ── retry behaviour ───────────────────────────────────────────────────────────
def test_robust_invoke_returns_the_first_non_empty_result():
    assert robust_invoke(RunnableLambda(lambda _: "answer"), {}) == "answer"


def test_robust_invoke_retries_an_empty_completion_then_succeeds():
    calls = {"n": 0}

    def flaky(_):
        calls["n"] += 1
        return "" if calls["n"] == 1 else "recovered"

    assert robust_invoke(RunnableLambda(flaky), {}, attempts=3, base_delay=0) == "recovered"
    assert calls["n"] == 2


def test_robust_invoke_retries_a_rate_limit_error():
    calls = {"n": 0}

    def rate_limited(_):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("Error code: 429 - rate limit reached")
        return "ok"

    assert robust_invoke(RunnableLambda(rate_limited), {}, attempts=4, base_delay=0) == "ok"
    assert calls["n"] == 3


def test_robust_invoke_does_not_retry_a_permanent_error():
    calls = {"n": 0}

    def bad_request(_):
        calls["n"] += 1
        raise ValueError("model does not exist")

    with pytest.raises(ValueError):
        robust_invoke(RunnableLambda(bad_request), {}, attempts=4, base_delay=0)
    assert calls["n"] == 1  # failed fast instead of burning retries
