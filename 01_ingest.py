"""
Step 1 - Live Data Ingestion
Pulls reviews from 5 Indian e-commerce apps via the Google Play Store.

Input:  N/A
Outputs: data/raw_reviews.csv
"""

import time
import logging
from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, reviews

import warnings
warnings.filterwarnings("ignore")

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


# ── Merge, Deduplicate, Basic Clean ───────────────────────────────────────────

def merge_and_clean(df_play: pd.DataFrame) -> pd.DataFrame:
    df = df_play.copy()

    # Drop exact content duplicates (same review submitted twice)
    df = df.drop_duplicates(subset=["content"], keep="first")

    # Drop reviews shorter than 10 chars (meaningless: "ok", ".", "5 star")
    df = df[df["content"].str.len() >= 10].copy()

    # Normalise timestamp
    df["at"] = pd.to_datetime(df["at"], errors="coerce", utc=True)

    # Keep score numeric; a missing star rating stays NaN rather than becoming 0
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

    log.info("Merging and cleaning ...")
    df = merge_and_clean(df_play)

    out_path = OUT_DIR / "raw_reviews.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"Saved: {out_path}  ({len(df)} rows, {df.shape[1]} columns)")
    log.info(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
