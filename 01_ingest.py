"""
Step 1 — Live Data Ingestion
Pulls reviews from 5 Indian e-commerce apps via Google Play Store
and supplementary posts from Reddit using PRAW.

Input:  N/A
Outputs: data/raw_reviews.csv
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, reviews
import praw
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

APPS = {
    "Myntra":       "com.myntra.android",
    "Meesho":       "com.meesho.supply",
    "Nykaa":        "com.fsn.nykaa",
    "Flipkart":     "com.flipkart.android",
    "Amazon India": "in.amazon.mShop.android.shopping",
}

REVIEWS_PER_APP   = 500   # per language
LANGS             = ["en", "hi"]
REDDIT_LIMIT      = 200
REDDIT_SUBREDDITS = ["IndianShopping", "Meesho", "india"]
REDDIT_KEYWORDS   = ["delivery", "refund", "fake", "return", "cancel", "fraud", "quality",
                     "damaged", "wrong product", "not received", "complaint", "scam"]

OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

# ── Google Play Ingestion ──────────────────────────────────────────────────────

def pull_play_reviews(app_name: str, app_id: str, lang: str) -> list[dict]:
    """Pull up to REVIEWS_PER_APP newest reviews for one app in one language."""
    log.info(f"  Pulling {app_name} ({lang}) ...")
    try:
        result, _ = reviews(
            app_id,
            lang=lang,
            country="in",
            sort=Sort.NEWEST,
            count=REVIEWS_PER_APP,
        )
        records = []
        for r in result:
            records.append({
                "review_id":    r.get("reviewId", ""),
                "platform":     app_name,
                "source":       "google_play",
                "lang":         lang,
                "content":      (r.get("content") or "").strip(),
                "score":        r.get("score"),
                "thumbs_up":    r.get("thumbsUpCount", 0),
                "at":           r.get("at"),
                "app_version":  r.get("appVersion", ""),
            })
        log.info(f"    -> {len(records)} reviews")
        return records
    except Exception as e:
        log.warning(f"    Failed for {app_name} ({lang}): {e}")
        return []


def collect_play_reviews() -> pd.DataFrame:
    all_records = []
    for app_name, app_id in APPS.items():
        for lang in LANGS:
            records = pull_play_reviews(app_name, app_id, lang)
            all_records.extend(records)
            time.sleep(1.5)   # be polite to the Play Store endpoint
    df = pd.DataFrame(all_records)
    log.info(f"Google Play total: {len(df)} rows across {df['platform'].nunique()} platforms")
    return df


# ── Reddit Ingestion ───────────────────────────────────────────────────────────

def is_relevant(text: str) -> bool:
    """Return True if the post/comment text contains at least one keyword."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in REDDIT_KEYWORDS)


def collect_reddit_posts() -> pd.DataFrame:
    """
    Requires either:
      - REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET in .env (OAuth script app), or
      - No credentials for read-only access to public subreddits
    """
    client_id     = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    user_agent    = "complaint_intelligence_engine:v1.0 (by /u/your_username)"

    if not client_id:
        log.warning("No Reddit credentials found — using unauthenticated read-only access (rate limited)")
        reddit = praw.Reddit(
            client_id="anonymous",
            client_secret="anonymous",
            user_agent=user_agent,
        )
        reddit.read_only = True
    else:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )

    records = []
    for sub_name in REDDIT_SUBREDDITS:
        log.info(f"  Pulling r/{sub_name} ...")
        try:
            subreddit = reddit.subreddit(sub_name)
            count = 0
            for post in subreddit.hot(limit=REDDIT_LIMIT * 2):  # over-fetch then filter
                text = f"{post.title} {post.selftext}".strip()
                if not is_relevant(text):
                    continue
                records.append({
                    "review_id":   post.id,
                    "platform":    detect_platform_from_text(text),
                    "source":      "reddit",
                    "lang":        "en",
                    "content":     text[:2000],  # cap at 2000 chars
                    "score":       None,
                    "thumbs_up":   post.score,
                    "at":          datetime.fromtimestamp(post.created_utc),
                    "app_version": "",
                })
                count += 1
                if count >= REDDIT_LIMIT:
                    break
            log.info(f"    -> {count} relevant posts from r/{sub_name}")
            time.sleep(1.0)
        except Exception as e:
            log.warning(f"    Failed for r/{sub_name}: {e}")

    df = pd.DataFrame(records)
    log.info(f"Reddit total: {len(df)} rows")
    return df


def detect_platform_from_text(text: str) -> str:
    """Heuristic: assign a platform label based on app mentions in the post."""
    text_lower = text.lower()
    platform_keywords = {
        "Myntra":       ["myntra"],
        "Meesho":       ["meesho"],
        "Nykaa":        ["nykaa"],
        "Flipkart":     ["flipkart"],
        "Amazon India": ["amazon"],
    }
    for platform, keywords in platform_keywords.items():
        if any(kw in text_lower for kw in keywords):
            return platform
    return "Unknown"


# ── Merge, Deduplicate, Basic Clean ───────────────────────────────────────────

def merge_and_clean(df_play: pd.DataFrame, df_reddit: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([df_play, df_reddit], ignore_index=True)

    # Drop exact content duplicates (same review submitted twice)
    df = df.drop_duplicates(subset=["content"], keep="first")

    # Drop reviews shorter than 10 chars (meaningless: "ok", ".", "5 star")
    df = df[df["content"].str.len() >= 10].copy()

    # Normalise timestamp
    df["at"] = pd.to_datetime(df["at"], errors="coerce", utc=True)

    # Fill missing scores for Reddit rows with NaN explicitly
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # Add a unique integer ID for convenience
    df = df.reset_index(drop=True)
    df.insert(0, "id", df.index)

    log.info(f"After merge + clean: {len(df)} rows, {df['platform'].nunique()} platforms")
    log.info(f"Platform breakdown:\n{df['platform'].value_counts().to_string()}")
    log.info(f"Source breakdown:\n{df['source'].value_counts().to_string()}")

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Step 1: Live Data Ingestion")
    log.info("=" * 60)

    log.info("Collecting Google Play reviews ...")
    df_play = collect_play_reviews()

    log.info("Collecting Reddit posts ...")
    df_reddit = collect_reddit_posts()

    log.info("Merging and cleaning ...")
    df = merge_and_clean(df_play, df_reddit)

    out_path = OUT_DIR / "raw_reviews.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"Saved: {out_path}  ({len(df)} rows, {df.shape[1]} columns)")
    log.info(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
