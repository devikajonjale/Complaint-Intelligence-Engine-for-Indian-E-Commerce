"""Build judge-labelled relevance ground truth for the retrieval evaluation (pooled labelling).

Keyword matching understates retrieval quality: a review can answer a question without containing any
of its keywords, especially in Hindi or Hinglish. This script uses the standard TREC-style pooling
approach instead:

  1. Pool candidates per question from every retrieval variant (dense, hybrid, rerank), so the labels
     are not biased towards one retriever.
  2. Ask the judge model, in one call per question, which pooled reviews actually answer the question.
  3. Save the labels to rag/eval_labels.json.

rag/evaluate.py prefers these labels when the file exists and falls back to keywords otherwise.
Labels are cached per question and only regenerated when the candidate pool changes.

Usage: python rag/label_eval_set.py [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from rag.chain import build_filter  # noqa: E402
from rag.config import RAG_DIR, get_judge_llm, robust_invoke  # noqa: E402
from rag.retriever import VARIANTS, search, warm_up  # noqa: E402

EVAL_SET = RAG_DIR / "eval_set.json"
LABELS = RAG_DIR / "eval_labels.json"
POOL_DEPTH = 12  # per variant, so the pool is at most 36 reviews before de-duplication


class RelevanceLabels(BaseModel):
    relevant_ids: list[str] = Field(
        description="Ids (e.g. R0042) of the reviews that provide evidence for the question. Empty list if none do."
    )


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You label customer reviews for relevance to an analyst's question. A review is relevant if it "
            "provides direct evidence an analyst would cite when answering, even partially, and even if the "
            "wording differs or the review is in Hindi or Hinglish. A review that merely mentions the same "
            "platform, or is generic praise or abuse unrelated to the question's topic, is NOT relevant. "
            "Return only the ids of relevant reviews.",
        ),
        ("human", "QUESTION: {question}\n\nCANDIDATE REVIEWS:\n{candidates}"),
    ]
)


def pool(question: str, filters: dict | None) -> dict[str, str]:
    """Union of the top results from every variant: {sid: review text}."""
    pooled: dict[str, str] = {}
    for variant in VARIANTS:
        for doc, _ in search(question, k=POOL_DEPTH, filters=filters, variant=variant):
            pooled.setdefault(doc.metadata["sid"], doc.page_content)
    return pooled


def run(force: bool = False) -> dict:
    questions = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    labels = {} if force or not LABELS.exists() else json.loads(LABELS.read_text(encoding="utf-8"))
    chain = PROMPT | get_judge_llm().with_structured_output(RelevanceLabels)
    warm_up("rerank")

    for q in questions:
        if q.get("unanswerable"):
            continue  # nothing in the corpus should be relevant
        candidates = pool(q["question"], build_filter(**q.get("filters", {})))
        sig = hashlib.md5(",".join(sorted(candidates)).encode()).hexdigest()
        if labels.get(q["id"], {}).get("signature") == sig:
            continue
        listing = "\n".join(f"[{sid}] {text[:400]}" for sid, text in candidates.items())
        result: RelevanceLabels | None = robust_invoke(chain, {"question": q["question"], "candidates": listing})
        kept = [s for s in (result.relevant_ids if result else []) if s in candidates]  # drop hallucinated ids
        labels[q["id"]] = {"question": q["question"], "relevant_ids": kept, "pool_size": len(candidates),
                           "signature": sig}
        print(f"{q['id']}: {len(kept)}/{len(candidates)} judged relevant", flush=True)

    LABELS.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {LABELS}")
    return labels


def load_labels() -> dict:
    return json.loads(LABELS.read_text(encoding="utf-8")) if LABELS.exists() else {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run(force=ap.parse_args().force)
