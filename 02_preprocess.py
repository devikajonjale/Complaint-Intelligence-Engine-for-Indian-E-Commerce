"""
Step 2 — Text Preprocessing, Cleaning & Feature Engineering
Complaint Intelligence Engine | MSc Data Science, NMIMS NSoMASA

Input:  data/raw_reviews.csv
Output: data/cleaned_reviews.csv
        data/tfidf_matrix.npz
        data/tfidf_vocab.json

Install:
    pip install pandas numpy scikit-learn langdetect scipy
"""

import json
import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from langdetect import detect, LangDetectException
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Hinglish tokens — common Hindi words written in Roman script
# Used to flag "English" reviews that are actually Hinglish
HINGLISH_TOKENS = {
    "nahi", "hai", "tha", "the", "kya", "aur", "bhi", "bahut", "bohot",
    "achha", "acha", "accha", "mujhe", "mera", "meri", "mere", "humara",
    "unka", "unki", "unhone", "karo", "karta", "karti", "karte", "agar",
    "toh", "to", "phir", "lekin", "par", "woh", "yeh", "ye", "jo", "jab",
    "kab", "kaise", "kyun", "sirf", "sab", "sabhi", "paise", "paisa",
    "baar", "bar", "gaya", "gayi", "gaye", "mila", "mili", "mile",
    "diya", "diya", "liya", "liya", "kiya", "kiya", "hua", "hui", "hue",
    "raha", "rahi", "rahe", "wala", "wali", "waale", "bilkul", "sahi",
    "galat", "bura", "buri", "bure", "acchi", "bekaar", "bekar",
}


# ── Text Cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Light cleaning that preserves Hinglish signal.
    - Removes URLs, HTML tags, repeated punctuation
    - Lowercases
    - Strips leading/trailing whitespace
    - Does NOT remove Hindi-origin words
    - Does NOT transliterate
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
    text = re.sub(r"<[^>]+>", " ", text)                  # HTML tags
    text = re.sub(r"[^\w\s\u0900-\u097F.,!?'-]", " ", text)  # keep Devanagari
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)            # "veryyyy" → "veryy"
    text = re.sub(r"\s+", " ", text)                       # collapse whitespace
    return text.strip()


# ── Language Detection ─────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Returns ISO 639-1 code or 'unknown'."""
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def is_hinglish(text: str, lang: str) -> bool:
    """
    A review is Hinglish if langdetect says English but it contains
    at least 2 known Hinglish tokens.
    """
    if lang != "en":
        return False
    tokens = set(text.lower().split())
    return len(tokens & HINGLISH_TOKENS) >= 2


# ── Feature Engineering ────────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Adding derived features ...")

    # -- Text length
    df["review_length"] = df["cleaned_content"].str.split().str.len()

    # -- Language detection (on cleaned content)
    log.info("  Detecting languages (may take 30-60 sec) ...")
    df["detected_lang"] = df["cleaned_content"].apply(detect_language)

    # -- Hinglish flag
    df["is_hinglish"] = df.apply(
        lambda r: is_hinglish(r["cleaned_content"], r["detected_lang"]), axis=1
    )

    # -- Star bucket (complaint vs mixed vs positive)
    def star_bucket(score):
        if pd.isna(score):
            return "unknown"
        if score <= 2:
            return "complaint"
        if score == 3:
            return "mixed"
        return "positive"

    df["star_bucket"] = df["score"].apply(star_bucket)

    # -- High engagement flag (proxy for "important" review)
    df["high_engagement"] = df["thumbs_up"] > 10

    # -- Week number (for temporal spike detection later)
    df["week"] = df["at"].dt.to_period("W").astype(str)
    df["year_month"] = df["at"].dt.to_period("M").astype(str)

    # -- Combined text field used for TF-IDF and embeddings
    # For Google Play: use content only
    # For Reddit: first 300 chars to avoid very long posts dominating
    df["text_for_model"] = df.apply(
        lambda r: r["cleaned_content"][:300] if r["source"] == "reddit"
        else r["cleaned_content"],
        axis=1
    )

    log.info(f"  Hinglish reviews: {df['is_hinglish'].sum()} ({df['is_hinglish'].mean()*100:.1f}%)")
    log.info(f"  Language distribution:\n{df['detected_lang'].value_counts().head(8).to_string()}")
    log.info(f"  Star bucket distribution:\n{df['star_bucket'].value_counts().to_string()}")

    return df


# ── TF-IDF Matrix ──────────────────────────────────────────────────────────────

def build_tfidf(df: pd.DataFrame) -> tuple:
    """
    Build a TF-IDF matrix on cleaned review text.
    This is used ONLY for extracting cluster keywords later —
    not for clustering itself (clustering uses SBERT embeddings).
    """
    log.info("Building TF-IDF matrix ...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),       # unigrams + bigrams
        min_df=3,                 # ignore terms in fewer than 3 reviews
        max_df=0.85,              # ignore terms in more than 85% of reviews
        sublinear_tf=True,        # log(1+tf) scaling
        strip_accents="unicode",
    )
    tfidf_matrix = vectorizer.fit_transform(df["text_for_model"].fillna(""))
    vocab = vectorizer.get_feature_names_out().tolist()
    log.info(f"TF-IDF matrix: {tfidf_matrix.shape[0]} docs x {tfidf_matrix.shape[1]} features")
    return tfidf_matrix, vocab, vectorizer


# ── Quality Checks ─────────────────────────────────────────────────────────────

def quality_report(df: pd.DataFrame):
    log.info("=" * 50)
    log.info("QUALITY REPORT")
    log.info("=" * 50)
    log.info(f"Total rows: {len(df)}")
    log.info(f"Columns: {list(df.columns)}")
    log.info(f"Null counts:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")
    log.info(f"Platform counts:\n{df['platform'].value_counts().to_string()}")
    log.info(f"Source counts:\n{df['source'].value_counts().to_string()}")
    log.info(f"Review length stats:\n{df['review_length'].describe().to_string()}")
    log.info("=" * 50)

    # Sample complaints for manual inspection
    complaints = df[df["star_bucket"] == "complaint"].head(5)
    log.info("Sample complaints:")
    for _, row in complaints.iterrows():
        log.info(f"  [{row['platform']}] {row['cleaned_content'][:120]} ...")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Step 2: Text Preprocessing & Feature Engineering")
    log.info("=" * 60)

    # Load raw data
    raw_path = DATA_DIR / "raw_reviews.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found — run 01_ingest.py first")

    df = pd.read_csv(raw_path, encoding="utf-8-sig", parse_dates=["at"])
    log.info(f"Loaded: {len(df)} rows from {raw_path}")

    # -- Step 2a: Clean text
    log.info("Cleaning text ...")
    df["cleaned_content"] = df["content"].apply(clean_text)

    # -- Step 2b: Drop empty rows after cleaning
    before = len(df)
    df = df[df["cleaned_content"].str.len() >= 10].copy()
    log.info(f"Dropped {before - len(df)} rows with content < 10 chars after cleaning")

    # -- Step 2c: Drop duplicates on cleaned content
    before = len(df)
    df = df.drop_duplicates(subset=["cleaned_content"], keep="first")
    log.info(f"Dropped {before - len(df)} duplicate cleaned reviews")

    # -- Step 2d: Feature engineering
    df = add_features(df)

    # -- Step 2e: Final column ordering
    cols = [
        "id", "review_id", "platform", "source", "lang", "detected_lang",
        "is_hinglish", "content", "cleaned_content", "text_for_model",
        "score", "star_bucket", "thumbs_up", "high_engagement",
        "review_length", "at", "week", "year_month", "app_version"
    ]
    # Keep only columns that exist
    cols = [c for c in cols if c in df.columns]
    df = df[cols].reset_index(drop=True)

    # Quality check
    quality_report(df)

    # -- Save cleaned CSV
    out_path = DATA_DIR / "cleaned_reviews.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"Saved cleaned reviews: {out_path}  ({len(df)} rows)")

    # -- Build and save TF-IDF
    tfidf_matrix, vocab, _ = build_tfidf(df)
    save_npz(DATA_DIR / "tfidf_matrix.npz", tfidf_matrix)
    with open(DATA_DIR / "tfidf_vocab.json", "w") as f:
        json.dump(vocab, f)
    log.info(f"Saved TF-IDF matrix: data/tfidf_matrix.npz")
    log.info(f"Saved TF-IDF vocab: data/tfidf_vocab.json  ({len(vocab)} terms)")

    log.info("Step 2 complete.")


if __name__ == "__main__":
    main()
