"""Grounded RAG Q&A over complaint reviews (LangChain LCEL).

    retriever (hybrid dense + BM25, optional rerank, metadata filtered)
        ->  format context with [R####] ids  ->  grounded prompt  ->  LLM  ->  answer
    The output dict carries the answer, the retrieved source reviews and the ids the model cited.

Supports conversational follow-ups: a short history is condensed into a standalone question before
retrieval, so "what about Flipkart?" still retrieves the right reviews.

Usage: python rag/chain.py "Why are Meesho users asking for refunds?" --platform Meesho
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.runnables import RunnableLambda, RunnablePassthrough  # noqa: E402

from rag.config import get_llm, robust_invoke  # noqa: E402
from rag.retriever import search, warm_up  # noqa: E402

NO_EVIDENCE = "Not enough evidence in the retrieved reviews to answer this."

# The refusal rule is deliberately narrow: an earlier version refused whenever the reviews did not
# answer the question completely, which rejected questions that had partial but real evidence.
SYSTEM_PROMPT = f"""You are a complaint analyst for Indian e-commerce apps.
Answer ONLY using the customer reviews in the CONTEXT. Each review starts with its id in square brackets, e.g. [R0042].
Rules:
1. Every factual claim must cite at least one review id in square brackets, e.g. "Refunds are delayed [R0042][R0107]".
2. Cite only ids that appear in the CONTEXT. Never invent ids, numbers, policies or facts.
3. Reviews may be in Hindi or Hinglish; answer in English.
4. Partial evidence is still an answer. If only some reviews touch the question, answer from those and
   add one short sentence saying the evidence is limited. Do not refuse just because the match is partial,
   because the reviews are few, or because they do not explain the underlying cause.
5. Reply exactly "{NO_EVIDENCE}" ONLY when no review in the CONTEXT relates to the question's topic at all,
   for example when the question asks about company financials, staff or policy documents that customer
   reviews cannot contain.
6. Be concise: at most 6 sentences or bullet points."""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}")]
)

# Rewrites a follow-up into a standalone question using the previous turns.
CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the follow-up question as a standalone question in English, resolving pronouns and "
            "implied subjects from the chat history. Keep it short. Output only the rewritten question, "
            "with no preamble. If it is already standalone, output it unchanged.",
        ),
        ("human", "Chat history:\n{history}\n\nFollow-up question: {question}"),
    ]
)

CITE_RE = re.compile(r"R\d{4}")


def build_filter(platform=None, severity=None, anomaly_only: bool = False, routing_team=None) -> dict | None:
    """Metadata filter for retrieval. List values mean 'any of'."""
    f = {}
    if platform:
        f["platform"] = platform
    if severity:
        f["severity"] = severity
    if routing_team:
        f["routing_team"] = routing_team
    if anomaly_only:
        f["is_anomaly"] = True
    return f or None


def retrieve(question: str, k: int = 8, filters: dict | None = None, variant: str | None = None):
    """Top-k reviews as [(Document, score)]. Metadata filters are applied before ranking."""
    return search(question, k=k, filters=filters, variant=variant)


def format_docs(docs_scores) -> str:
    lines = []
    for doc, _ in docs_scores:
        m = doc.metadata
        lines.append(
            f"[{m['sid']}] (platform={m['platform']}, rating={m['rating']}, date={m['date']}, "
            f"severity={m['severity']}, team={m['routing_team']}) {doc.page_content.strip()}"
        )
    return "\n".join(lines)


def condense(question: str, history: list[tuple[str, str]] | None) -> str:
    """Turn a follow-up plus history into a standalone question. No history means no LLM call."""
    if not history:
        return question
    text = "\n".join(f"Q: {q}\nA: {a[:300]}" for q, a in history[-3:])
    rewritten = robust_invoke(CONDENSE_PROMPT | get_llm(0.0) | StrOutputParser(), {"history": text, "question": question}) or ""
    return rewritten.strip() or question


def _invoke_with_retry(chain, payload) -> str:
    """Empty completions and free-tier rate limits are retried in rag.config.robust_invoke."""
    return (robust_invoke(chain, payload) or "").strip()


def make_chain(k: int = 8, filters: dict | None = None, variant: str | None = None):
    """LCEL chain: question -> {answer, sources, cited_ids}."""
    answer_chain = (
        {"context": lambda x: format_docs(x["sources"]), "question": lambda x: x["question"]}
        | PROMPT
        | get_llm(temperature=0.0)
        | StrOutputParser()
    )
    return (
        {
            "question": RunnablePassthrough(),
            "sources": RunnableLambda(lambda q: retrieve(q, k=k, filters=filters, variant=variant)),
        }
        | RunnablePassthrough.assign(answer=lambda x: _invoke_with_retry(answer_chain, x))
        | RunnableLambda(_finalise)
    )


def _finalise(x: dict) -> dict:
    retrieved = {d.metadata["sid"] for d, _ in x["sources"]}
    cited = list(dict.fromkeys(CITE_RE.findall(x["answer"])))
    return {
        "question": x["question"],
        "answer": x["answer"].strip(),
        "cited_ids": cited,
        "invalid_citations": [c for c in cited if c not in retrieved],
        "sources": [
            {"sid": d.metadata["sid"], "score": float(s), "text": d.page_content, "metadata": d.metadata}
            for d, s in x["sources"]
        ],
    }


def ask(question: str, k: int = 8, variant: str | None = None, history=None, **filter_kwargs) -> dict:
    """Answer one question. Measures end-to-end latency and keeps the original wording for display."""
    t0 = time.perf_counter()
    standalone = condense(question, history)
    out = make_chain(k=k, filters=build_filter(**filter_kwargs), variant=variant).invoke(standalone)
    out["question"] = question
    out["standalone_question"] = standalone
    out["latency_s"] = round(time.perf_counter() - t0, 3)
    return out


def stream_answer(question: str, k: int = 8, variant: str | None = None, history=None, **filter_kwargs):
    """Yield ("sources", hits) once, then answer text chunks, so a UI can render tokens as they arrive."""
    standalone = condense(question, history)
    hits = retrieve(standalone, k=k, filters=build_filter(**filter_kwargs), variant=variant)
    yield "sources", hits
    chain = PROMPT | get_llm(temperature=0.0) | StrOutputParser()
    for chunk in chain.stream({"context": format_docs(hits), "question": standalone}):
        if chunk:
            yield "token", chunk


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--platform")
    ap.add_argument("--severity", help="RED / AMBER / GREEN")
    ap.add_argument("--anomaly-only", action="store_true")
    ap.add_argument("--variant", choices=["dense", "hybrid", "rerank"], help="retrieval strategy")
    ap.add_argument("-k", type=int, default=8)
    a = ap.parse_args()
    warm_up(a.variant)
    res = ask(a.question, k=a.k, variant=a.variant, platform=a.platform, severity=a.severity,
              anomaly_only=a.anomaly_only)
    print(res["answer"], "\n")
    for s in res["sources"]:
        print(f"  {s['sid']} [{s['metadata']['platform']}, {s['metadata']['severity']}] {s['text'][:110]}")
    print(f"\nlatency: {res['latency_s']}s")
