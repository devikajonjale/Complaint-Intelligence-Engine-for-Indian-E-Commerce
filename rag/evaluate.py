"""Evaluate the RAG system: retrieval quality, answer quality and latency.

Ground truth for retrieval (two modes):
  judged  : pooled relevance labels from rag/eval_labels.json (see rag/label_eval_set.py). Preferred.
  keyword : a review is relevant if its text matches the question's keyword regex within the
            question's metadata filters. Needs no API key, but understates quality because a review
            can answer a question without containing its keywords.
Unanswerable questions (out of corpus) test whether the model refuses instead.

Metrics
  retrieval : hit rate@k, MRR@k, precision@k, recall@k (k = 1, 3, 5, 8)   [no LLM needed]
  generation: LLM-as-judge faithfulness and answer relevance (1 to 5), citation validity,
              refusal accuracy on unanswerable questions
  latency   : retrieval latency and end-to-end latency per query (mean, p50, p95)

Usage
  python rag/evaluate.py                      # full evaluation, default retrieval variant
  python rag/evaluate.py --retrieval-only     # no API key needed
  python rag/evaluate.py --compare            # retrieval metrics for dense vs hybrid vs rerank
  python rag/evaluate.py --variant rerank
Writes rag/eval_results.json (and rag/eval_variants.json for --compare) and prints a summary table.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from rag.build_index import load_reviews  # noqa: E402
from rag.chain import NO_EVIDENCE, ask, build_filter, format_docs, retrieve  # noqa: E402
from rag.config import ANSWER_MODEL, JUDGE_MODEL, RAG_DIR, get_judge_llm, robust_invoke  # noqa: E402
from rag.label_eval_set import load_labels  # noqa: E402
from rag.retriever import DEFAULT_VARIANT, VARIANTS, warm_up  # noqa: E402

EVAL_SET = RAG_DIR / "eval_set.json"
RESULTS = RAG_DIR / "eval_results.json"
VARIANT_RESULTS = RAG_DIR / "eval_variants.json"
KS = [1, 3, 5, 8]
ANSWER_K = 8


class Judgement(BaseModel):
    faithfulness: int = Field(description="1 to 5. 5 = every claim in the ANSWER is directly supported by the CONTEXT; 1 = mostly unsupported or invented")
    answer_relevance: int = Field(description="1 to 5. 5 = the ANSWER directly and fully addresses the QUESTION; 1 = off topic")
    reasoning: str = Field(description="One sentence justification")


JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a strict evaluator of retrieval augmented answers. Judge only against the CONTEXT provided."),
        ("human", "QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nScore faithfulness and answer relevance."),
    ]
)


def keyword_relevant_ids(df, q) -> set[str]:
    """Keyword ground truth inside the question's metadata filters."""
    f = q.get("filters", {})
    sub = df
    if "platform" in f:
        sub = sub[sub["platform"] == f["platform"]]
    if f.get("anomaly_only"):
        sub = sub[sub["isolation_anomaly"]]
    hit = sub["content"].astype(str).str.contains(q["keywords"], case=False, regex=True)
    return set(sub.loc[hit, "sid"])


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def lat_summary(xs):
    return {"mean": round(statistics.mean(xs), 3), "p50": round(pct(xs, 50), 3), "p95": round(pct(xs, 95), 3)} if xs else None


def score_retrieval(questions, df, labels, variant: str) -> tuple[list[dict], list[float]]:
    """Rank once per question and score it against whichever ground truth is available."""
    per_q, lat = [], []
    for q in questions:
        filters = build_filter(**q.get("filters", {}))
        rec = {"id": q["id"], "question": q["question"], "unanswerable": q.get("unanswerable", False)}

        t0 = time.perf_counter()
        hits = retrieve(q["question"], k=max(KS), filters=filters, variant=variant)
        rec["retrieval_latency_s"] = round(time.perf_counter() - t0, 4)
        lat.append(rec["retrieval_latency_s"])
        ranked = [d.metadata["sid"] for d, _ in hits]
        rec["retrieved"] = ranked

        judged = set(labels.get(q["id"], {}).get("relevant_ids", []))
        if judged:
            rel, mode = judged, "judged"
        elif not rec["unanswerable"] and q["keywords"] != ".":
            # keyword "." would match every review in scope, so it carries no signal
            rel, mode = keyword_relevant_ids(df, q), "keyword"
        else:
            rel, mode = set(), None

        rec["ground_truth"] = mode
        rec["retrieval_scored"] = bool(rel)
        if rel:
            rec["n_relevant"] = len(rel)
            first = next((i + 1 for i, s in enumerate(ranked) if s in rel), None)
            for k in KS:
                rec[f"hit@{k}"] = int(first is not None and first <= k)
                rec[f"precision@{k}"] = round(sum(s in rel for s in ranked[:k]) / k, 3)
                rec[f"recall@{k}"] = round(sum(s in rel for s in ranked[:k]) / len(rel), 3)
            rec["rr"] = round(1 / first, 4) if first and first <= max(KS) else 0.0
        per_q.append(rec)
    return per_q, lat


def retrieval_block(per_q: list[dict]) -> dict:
    scored = [r for r in per_q if r["retrieval_scored"]]
    if not scored:
        return {}
    return (
        {f"hit_rate@{k}": round(statistics.mean(r[f"hit@{k}"] for r in scored), 3) for k in KS}
        | {f"precision@{k}": round(statistics.mean(r[f"precision@{k}"] for r in scored), 3) for k in KS}
        | {f"recall@{k}": round(statistics.mean(r[f"recall@{k}"] for r in scored), 3) for k in KS}
        | {f"mrr@{max(KS)}": round(statistics.mean(r["rr"] for r in scored), 3)}
    )


def run(retrieval_only: bool = False, variant: str | None = None) -> dict:
    variant = variant or DEFAULT_VARIANT
    questions = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    df = load_reviews()
    labels = load_labels()
    warm_up(variant)  # load index, embedding model and reranker outside the timed region

    per_q, ret_lat = score_retrieval(questions, df, labels, variant)

    if not retrieval_only:
        for q, rec in zip(questions, per_q):
            filters = build_filter(**q.get("filters", {}))
            res = ask(q["question"], k=ANSWER_K, variant=variant, **q.get("filters", {}))
            rec.update(answer=res["answer"], cited_ids=res["cited_ids"],
                       invalid_citations=res["invalid_citations"], e2e_latency_s=res["latency_s"])
            refused = NO_EVIDENCE.lower().rstrip(".") in res["answer"].lower()
            rec["refused"] = refused
            if not rec["unanswerable"] and not refused:
                ctx = retrieve(q["question"], k=ANSWER_K, filters=filters, variant=variant)
                j: Judgement | None = robust_invoke(
                    JUDGE_PROMPT | get_judge_llm().with_structured_output(Judgement),
                    {"question": q["question"], "context": format_docs(ctx), "answer": res["answer"]},
                )
                if j:
                    rec.update(faithfulness=j.faithfulness, answer_relevance=j.answer_relevance,
                               judge_reason=j.reasoning)
            print(f"{q['id']} done", flush=True)

    return summarise(per_q, ret_lat, retrieval_only, variant)


def summarise(per_q, ret_lat, retrieval_only, variant) -> dict:
    scored = [r for r in per_q if r["retrieval_scored"]]
    modes = {r["ground_truth"] for r in scored}
    summary = {
        "n_questions": len(per_q),
        "n_retrieval_scored": len(scored),
        "n_unanswerable": sum(r["unanswerable"] for r in per_q),
        "corpus_size": len(load_reviews()),
        "retriever_variant": variant,
        "ground_truth": "judged" if modes == {"judged"} else "+".join(sorted(m for m in modes if m)),
        "retrieval": retrieval_block(per_q),
        "retrieval_latency_s": lat_summary(ret_lat),
    }
    if not retrieval_only:
        ans = [r for r in per_q if not r["unanswerable"]]
        judged = [r for r in ans if "faithfulness" in r]
        una = [r for r in per_q if r["unanswerable"]]
        cited = [r for r in judged if r["cited_ids"]]
        summary["generation"] = {
            "n_judged": len(judged),
            "faithfulness_mean_1to5": round(statistics.mean(r["faithfulness"] for r in judged), 2) if judged else None,
            "answer_relevance_mean_1to5": round(statistics.mean(r["answer_relevance"] for r in judged), 2) if judged else None,
            "faithful_rate_score_ge4": round(sum(r["faithfulness"] >= 4 for r in judged) / len(judged), 3) if judged else None,
            "answers_with_citations_rate": round(len(cited) / len(judged), 3) if judged else None,
            "citation_validity_rate": round(
                sum(len(r["cited_ids"]) - len(r["invalid_citations"]) for r in judged) / max(1, sum(len(r["cited_ids"]) for r in judged)), 3),
            "false_refusal_rate_answerable": round(sum(r["refused"] for r in ans) / len(ans), 3),
            "refusal_accuracy_unanswerable": round(sum(r["refused"] for r in una) / len(una), 3) if una else None,
        }
        summary["e2e_latency_s"] = lat_summary([r["e2e_latency_s"] for r in per_q])
        summary["answer_model"] = ANSWER_MODEL
        summary["judge_model"] = JUDGE_MODEL
    out = {"summary": summary, "per_question": per_q}
    RESULTS.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print_table(summary)
    return out


def compare() -> dict:
    """Retrieval-only metrics for every variant, printed side by side. No answer generation."""
    questions = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    df, labels, out = load_reviews(), load_labels(), {}
    for variant in VARIANTS:
        warm_up(variant)
        per_q, lat = score_retrieval(questions, df, labels, variant)
        out[variant] = {"retrieval": retrieval_block(per_q), "retrieval_latency_s": lat_summary(lat),
                        "n_scored": sum(r["retrieval_scored"] for r in per_q)}
        print(f"{variant} done", flush=True)
    VARIANT_RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")

    rows = ["hit_rate@1", "hit_rate@3", "hit_rate@5", "hit_rate@8", "mrr@8", "precision@5", "recall@8"]
    w = max(len(r) for r in rows) + 2
    print("\n" + "metric".ljust(w) + "".join(v.ljust(12) for v in VARIANTS))
    print("-" * (w + 12 * len(VARIANTS)))
    for r in rows:
        print(r.ljust(w) + "".join(str(out[v]["retrieval"].get(r, "-")).ljust(12) for v in VARIANTS))
    print("latency p50 (s)".ljust(w) + "".join(str(out[v]["retrieval_latency_s"]["p50"]).ljust(12) for v in VARIANTS))
    print(f"\nSaved {VARIANT_RESULTS}")
    return out


def print_table(s: dict) -> None:
    rows = [
        ("Corpus size (reviews)", s["corpus_size"]),
        ("Retriever variant", s["retriever_variant"]),
        ("Ground truth", s["ground_truth"]),
        ("Questions (retrieval scored / unanswerable)", f"{s['n_retrieval_scored']} / {s['n_unanswerable']}"),
    ]
    rows += list(s["retrieval"].items())
    rows += [("Retrieval latency p50 / p95 (s)", f"{s['retrieval_latency_s']['p50']} / {s['retrieval_latency_s']['p95']}")]
    rows += list(s.get("generation", {}).items())
    if s.get("e2e_latency_s"):
        rows.append(("End-to-end latency mean / p95 (s)", f"{s['e2e_latency_s']['mean']} / {s['e2e_latency_s']['p95']}"))
    w = max(len(str(r[0])) for r in rows)
    print("\n" + "-" * (w + 14))
    for name, val in rows:
        print(f"{str(name).ljust(w)} | {val}")
    print("-" * (w + 14) + f"\nSaved {RESULTS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--compare", action="store_true", help="retrieval metrics for every variant")
    ap.add_argument("--variant", choices=list(VARIANTS))
    a = ap.parse_args()
    compare() if a.compare else run(retrieval_only=a.retrieval_only, variant=a.variant)
