"""Streamlit pages for the GenAI layer, kept separate from app.py so existing modules are untouched.

    render_rag_chat()         -> "Ask the Complaints (RAG)"
    render_cluster_insights() -> "LLM Cluster Insights" (plus the executive brief generator)

The chat streams tokens as they arrive, keeps a short history so follow-up questions resolve, and
falls back to showing matching reviews when no LLM is configured or the free tier is exhausted.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from rag.config import RAG_DIR

PLATFORMS = ["Amazon India", "Flipkart", "Meesho", "Myntra", "Nykaa"]
SEVERITIES = ["RED", "AMBER", "GREEN"]
VARIANT_HELP = {
    "hybrid": "Dense vectors plus BM25 keywords, fused. Balanced default.",
    "rerank": "Hybrid shortlist re-scored by a local cross-encoder. Most accurate, slower.",
    "dense": "Sentence-BERT vectors only. Fastest.",
}
HISTORY_TURNS = 3  # how many previous turns are sent to resolve follow-up questions


@st.cache_resource(show_spinner="Loading vector index and retrieval models...")
def _warm(variant: str):
    from rag.retriever import warm_up

    warm_up(variant)
    return True


def _llm_ready() -> tuple[bool, str]:
    try:
        from rag.config import get_llm

        get_llm()
        return True, ""
    except Exception as e:  # missing key or package
        return False, str(e)


def render_rag_chat() -> None:
    from rag.chain import NO_EVIDENCE, build_filter, retrieve, stream_answer

    st.subheader("Ask the Complaints (RAG)")
    st.caption(
        "Ask a question in plain English. The answer is generated only from retrieved reviews and cites review ids "
        f'like [R0042]. When the reviews do not support an answer it replies "{NO_EVIDENCE}"'
    )

    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    platform = c1.multiselect("Platform", PLATFORMS)
    severity = c2.multiselect("Severity", SEVERITIES)
    anomaly_only = c3.checkbox("Critical anomalies only")
    k = c4.slider("Reviews to retrieve", 3, 15, 8)
    c5, c6 = st.columns([2, 2])
    variant = c5.selectbox("Retrieval", ["hybrid", "rerank", "dense"], help="\n".join(f"{k2}: {v}" for k2, v in VARIANT_HELP.items()))
    c6.caption(VARIANT_HELP[variant])

    _warm(variant)
    ok, err = _llm_ready()
    filters = build_filter(platform=platform or None, severity=severity or None, anomaly_only=anomaly_only)

    if not ok:
        # Retrieval still works without an LLM, so show the matching reviews instead of an error page.
        st.warning("No LLM configured, so answers are unavailable. Search still works and shows matching reviews.")
        st.caption(f"Reason: {err}")
        question = st.text_input("Search the reviews")
        if question:
            for doc, score in retrieve(question, k=k, filters=filters, variant=variant):
                _source_line(doc.metadata, score, doc.page_content, cited=False)
        return

    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []
    if st.session_state.rag_history and st.button("Clear chat"):
        st.session_state.rag_history = []
        st.rerun()

    for turn in st.session_state.rag_history:
        _render_turn(turn)

    question = st.chat_input("e.g. Why are Meesho customers not getting refunds?")
    if not question:
        return

    with st.chat_message("user"):
        st.write(question)

    history = [(t["question"], t["answer"]) for t in st.session_state.rag_history[-HISTORY_TURNS:]]
    with st.chat_message("assistant"):
        placeholder = st.empty()
        sources, parts = [], []
        try:
            with st.spinner("Retrieving reviews..."):
                stream = stream_answer(
                    question, k=k, variant=variant, history=history,
                    platform=platform or None, severity=severity or None, anomaly_only=anomaly_only,
                )
                kind, payload = next(stream)
                sources = payload if kind == "sources" else []
            for kind, payload in stream:
                if kind == "token":
                    parts.append(payload)
                    placeholder.markdown("".join(parts))  # stream tokens as they arrive
        except Exception as e:  # rate limit or provider outage
            st.error(f"Answer generation failed: {e}")
            if sources:
                st.info("Showing the retrieved reviews so the question is still useful.")
                for doc, score in sources:
                    _source_line(doc.metadata, score, doc.page_content, cited=False)
            return

        answer = "".join(parts).strip()
        placeholder.markdown(answer)
        turn = _as_turn(question, answer, sources)
        _render_meta_and_sources(turn)

    st.session_state.rag_history.append(turn)


def _as_turn(question: str, answer: str, sources) -> dict:
    from rag.chain import CITE_RE

    retrieved = {d.metadata["sid"] for d, _ in sources}
    cited = list(dict.fromkeys(CITE_RE.findall(answer)))
    return {
        "question": question,
        "answer": answer,
        "cited_ids": cited,
        "invalid_citations": [c for c in cited if c not in retrieved],
        "sources": [{"sid": d.metadata["sid"], "score": float(s), "text": d.page_content, "metadata": d.metadata}
                    for d, s in sources],
    }


def _render_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        _render_meta_and_sources(turn)


def _render_meta_and_sources(turn: dict) -> None:
    meta = f"{len(turn['sources'])} reviews retrieved, {len(turn['cited_ids'])} cited"
    if turn.get("latency_s"):
        meta += f", {turn['latency_s']} s"
    if turn["invalid_citations"]:
        meta += f". Warning: cited ids not in context: {', '.join(turn['invalid_citations'])}"
    st.caption(meta)
    cited = set(turn["cited_ids"])
    with st.expander("Sources"):
        for s in turn["sources"]:
            _source_line(s["metadata"], s["score"], s["text"], cited=s["sid"] in cited)


def _source_line(m: dict, score: float, text: str, cited: bool) -> None:
    st.markdown(
        f"{'**cited** ' if cited else ''}`{m['sid']}` {m['platform']} | {m['rating']} star | {m['date']} | "
        f"severity {m['severity']} | team {m['routing_team']} | anomaly {m['is_anomaly']} | similarity {score:.2f}"
    )
    st.write(text)
    st.caption(f"review_id: {m['review_id']}")
    st.divider()


def render_cluster_insights() -> None:
    from rag.cluster_insights import load_insights

    st.subheader("LLM Cluster Insights")
    st.caption(
        "Each cluster was summarised by an LLM from the reviews closest to its centroid, sampling 1 and 2 star "
        "complaints where a cluster has enough of them. Regenerate with `python rag/cluster_insights.py`."
    )
    insights = load_insights()
    if not insights:
        st.warning("No cached insights. Run `python rag/cluster_insights.py` first.")
    else:
        algo = st.radio("Clustering", ["kmeans", "hdbscan"], horizontal=True,
                        format_func=lambda a: "K-Means" if a == "kmeans" else "HDBSCAN")
        rows = sorted((v for v in insights.values() if v["algorithm"] == algo), key=lambda v: -v["stats"]["size"])
        st.dataframe(
            pd.DataFrame(
                [{"cluster": v["cluster_id"], "LLM label": v["label"], "pipeline name": v.get("pipeline_name"),
                  "size": v["stats"]["size"], "share %": v["stats"]["share_pct"], "avg rating": v["stats"]["avg_rating"],
                  "anomalies": v["stats"]["anomalies"], "owning team": v["owning_team"],
                  "from complaints": v.get("complaint_focused", False)} for v in rows]
            ),
            hide_index=True, width="stretch",
        )
        for v in rows:
            with st.expander(f"Cluster {v['cluster_id']}: {v['label']}  ({v['stats']['size']} reviews)"):
                st.write(v["summary"])
                st.markdown("**Root causes**\n" + "\n".join(f"- {c}" for c in v["root_causes"]))
                st.markdown(f"**Suggested owner:** {v['owning_team']}")
                if not v.get("complaint_focused", True):
                    st.caption("This cluster had too few 1 and 2 star reviews, so it was labelled from all members.")
                st.caption("Representative reviews: " + ", ".join(v["representative_ids"]))

    st.markdown("---")
    st.subheader("Executive Complaint Brief")
    c1, c2 = st.columns(2)
    platform = c1.selectbox("Platform", ["All"] + PLATFORMS)
    days = c2.selectbox("Window", ["All time", 7, 30, 90], format_func=lambda d: d if d == "All time" else f"Last {d} days")
    if st.button("Generate brief"):
        ok, err = _llm_ready()
        if not ok:
            st.error(f"LLM not configured: {err}")
            return
        from rag.report import generate_report

        try:
            with st.spinner("Writing brief..."):
                md = generate_report(None if platform == "All" else platform, None if days == "All time" else days)
        except Exception as e:
            st.error(f"Brief generation failed: {e}")
            return
        st.markdown(md)
        st.download_button("Download markdown", md, file_name="complaint_brief.md")

    results = RAG_DIR / "eval_results.json"
    if results.exists():
        with st.expander("RAG evaluation metrics"):
            st.json(json.loads(results.read_text(encoding="utf-8"))["summary"])
    variants = RAG_DIR / "eval_variants.json"
    if variants.exists():
        with st.expander("Retrieval variant comparison (dense vs hybrid vs rerank)"):
            data = json.loads(variants.read_text(encoding="utf-8"))
            st.dataframe(
                pd.DataFrame({v: d["retrieval"] for v, d in data.items()}).T.reset_index(names="variant"),
                hide_index=True, width="stretch",
            )
