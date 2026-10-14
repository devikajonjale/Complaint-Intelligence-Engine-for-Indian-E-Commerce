"""Shared configuration for the GenAI layer: paths, embedding model and LLM factory."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAG_DIR = ROOT / "rag"
INDEX_DIR = RAG_DIR / "index"
CACHE_DIR = RAG_DIR / "cache"

# Same model as 03_embed.py so query vectors live in the same space as the stored review vectors.
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

load_dotenv(ROOT / ".env")

# Most enriched pipeline output first; mirrors load_data() in app.py.
REVIEW_CANDIDATES = [
    "final_reviews_response.csv",
    "final_reviews_churn.csv",
    "final_reviews_routed.csv",
    "final_reviews_scored.csv",
    "final_reviews.csv",
]


def reviews_path() -> Path:
    for name in REVIEW_CANDIDATES:
        p = DATA_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError("No pipeline output found in data/. Run run_pipeline.py first.")


def get_embeddings():
    """LangChain wrapper around the pipeline's Sentence-BERT model (normalised, like 03_embed.py)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )


# Pinned models. Groq only (free tier, no card needed). Both are open-weight models hosted on Groq.
# They are fixed in code, not read from .env, so a stray setting cannot switch to a paid model or provider.
ANSWER_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "openai/gpt-oss-120b"


def _make_llm(model: str, temperature: float):
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add a free Groq key.")
    from langchain_groq import ChatGroq

    return ChatGroq(model=model, temperature=temperature, max_retries=3)


def get_llm(temperature: float = 0.0):
    """LLM used for answers, cluster labels and reports."""
    return _make_llm(ANSWER_MODEL, temperature)


def get_judge_llm():
    """Larger pinned model for LLM-as-judge, so answers are not graded by the model that wrote them."""
    return _make_llm(JUDGE_MODEL, 0.0)


# Transient failures seen with these reasoning models on the free tier:
#   - an empty completion (nothing generated, or structured output called no tool)
#   - HTTP 429 when the free tier rate limit is reached
# Both are retried with exponential backoff rather than crashing a dashboard page or an eval run.
_RETRYABLE = ("tool_use_failed", "rate limit", "429", "too many requests", "service unavailable", "timeout")


def robust_invoke(chain, payload, attempts: int = 4, base_delay: float = 2.0):
    """Invoke a chain, retrying empty results and transient provider errors. Returns None if all fail."""
    import time

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            out = chain.invoke(payload)
            if out is None or (isinstance(out, str) and not out.strip()):
                raise RuntimeError("empty completion")
            return out
        except Exception as e:  # noqa: BLE001 - provider raises several unrelated error types
            last_error = e
            message = str(e).lower()
            retryable = "empty completion" in message or any(t in message for t in _RETRYABLE)
            if not retryable or attempt == attempts - 1:
                break
            time.sleep(base_delay * (2**attempt))
    if last_error:
        raise last_error
    return None
