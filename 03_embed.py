"""
Step 3 — Multilingual Sentence-BERT Embeddings
Input:  data/cleaned_reviews.csv
Output: data/embeddings.npy
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def validate_semantic_alignment(df: pd.DataFrame, embeddings: np.ndarray) -> None:
    """Print quick semantic sanity check between complaint and positive groups."""
    complaint_idx = df.index[df["star_bucket"] == "complaint"].tolist()[:20]
    positive_idx = df.index[df["star_bucket"] == "positive"].tolist()[:20]

    if len(complaint_idx) < 5 or len(positive_idx) < 5:
        log.warning("Not enough complaint/positive samples for semantic validation.")
        return

    c_vec = embeddings[complaint_idx]
    p_vec = embeddings[positive_idx]

    c2c = cosine_similarity(c_vec, c_vec).mean()
    p2p = cosine_similarity(p_vec, p_vec).mean()
    c2p = cosine_similarity(c_vec, p_vec).mean()

    log.info("Semantic validation:")
    log.info(f"  Complaint-Complaint mean cosine: {c2c:.3f}")
    log.info(f"  Positive-Positive mean cosine : {p2p:.3f}")
    log.info(f"  Complaint-Positive mean cosine: {c2p:.3f}")


def main() -> None:
    in_path = DATA_DIR / "cleaned_reviews.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found — run 02_preprocess.py first")

    df = pd.read_csv(in_path, encoding="utf-8-sig")
    texts = df["text_for_model"].fillna("").astype(str).tolist()
    log.info(f"Loaded {len(texts)} reviews for embeddings")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    log.info(f"Embeddings shape: {embeddings.shape}")

    validate_semantic_alignment(df, embeddings)

    out_path = DATA_DIR / "embeddings.npy"
    np.save(out_path, embeddings)
    log.info(f"Saved embeddings to {out_path}")


if __name__ == "__main__":
    main()
