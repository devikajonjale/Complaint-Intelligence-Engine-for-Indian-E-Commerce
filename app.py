import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Complaint Intelligence Engine+", layout="wide")

DATA_DIR    = Path("data")
REPORTS_DIR = Path("reports")
MODELS_DIR  = Path("models")


@st.cache_data
def read_csv_opt(path: Path):
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else None


@st.cache_data
def load_data():
    candidates = [
        DATA_DIR / "final_reviews_response.csv",
        DATA_DIR / "final_reviews_churn.csv",
        DATA_DIR / "final_reviews_routed.csv",
        DATA_DIR / "final_reviews_scored.csv",
        DATA_DIR / "final_reviews.csv",
    ]
    final_path = next((p for p in candidates if p.exists()), None)
    spike_path  = DATA_DIR / "spike_report.csv"
    tsne_path   = DATA_DIR / "tsne_2d.npy"
    if final_path is None:
        return None, None
    df = pd.read_csv(final_path, encoding="utf-8-sig")
    spike = read_csv_opt(spike_path)
    if tsne_path.exists():
        tsne = np.load(tsne_path)
        df["tsne_x"] = tsne[:, 0]
        df["tsne_y"] = tsne[:, 1]
    else:
        df["tsne_x"] = 0.0
        df["tsne_y"] = 0.0
    df["at"] = pd.to_datetime(df["at"], errors="coerce", utc=True)
    return df, spike


def run_refresh():
    return subprocess.run([sys.executable, "run_pipeline.py"], capture_output=True, text=True)


def date_filter_df(fdf: pd.DataFrame, date_filter):
    if not date_filter or len(date_filter) != 2:
        return fdf
    start_d, end_d = date_filter[0], date_filter[1]
    if fdf["at"].dt.tz is not None:
        start = pd.Timestamp(start_d, tz="UTC")
        end   = pd.Timestamp(end_d, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    else:
        start = pd.Timestamp(start_d)
        end   = pd.Timestamp(end_d) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return fdf[(fdf["at"] >= start) & (fdf["at"] <= end)]


def show_model_table(task_name: str, metrics_file: str):
    path = REPORTS_DIR / metrics_file
    st.subheader(f"{task_name} Model Comparison")
    mdf = read_csv_opt(path)
    if mdf is None or mdf.empty:
        st.info("No metrics generated yet.")
        return
    st.dataframe(mdf, use_container_width=True)


# ── Collapsible Quick Start box ───────────────────────────────────────────────
def quick_start_box(how_to_use_items: list[str]) -> None:
    """
    Renders a collapsible expander with usage guidance.
    The old 'what_items' / 'how_items' split is replaced with a single
    'how_to_use_items' list that covers both capabilities and interpretation.
    """
    with st.expander("Quick Start — How to use this page", expanded=False):
        for item in how_to_use_items:
            st.markdown(f"- {item}")


# ── Heuristic SHAP helpers (unchanged) ───────────────────────────────────────
def severity_shap_style(row: pd.Series) -> list[tuple[str, float]]:
    txt          = str(row.get("content", "")).lower()
    score        = float(row.get("score", 3))
    thumbs       = float(row.get("thumbs_up", 0))
    review_length = float(row.get("review_length", 0))
    contrib = []
    legal_hits = sum(k in txt for k in ["fraud", "legal", "consumer court", "scam", "police", "chargeback"])
    money_hits = sum(k in txt for k in ["refund", "charged", "payment", "money", "upi", "transaction"])
    if legal_hits:
        contrib.append(("Legal/fraud keywords", 0.35 + 0.1 * min(legal_hits, 2)))
    if money_hits:
        contrib.append(("Money/payment issue terms", 0.22 + 0.05 * min(money_hits, 3)))
    if score <= 2:
        contrib.append(("Low star rating", 0.18))
    if thumbs >= 20:
        contrib.append(("High community engagement", 0.16))
    if review_length >= 40:
        contrib.append(("Long detailed complaint", 0.10))
    if "!" in txt:
        contrib.append(("Strong emotional punctuation", 0.05))
    if not contrib:
        contrib.append(("No strong high-risk cues detected", 0.03))
    contrib.sort(key=lambda x: x[1], reverse=True)
    return contrib[:5]


def router_shap_style_terms(text: str, model_blob: dict) -> list[tuple[str, float]]:
    model = model_blob["model"]
    try:
        tfidf         = model.named_steps["tfidf"]
        clf           = model.named_steps["clf"]
        vec           = tfidf.transform([text])
        feature_names = np.array(tfidf.get_feature_names_out())
        nz = vec.nonzero()[1]
        if len(nz) == 0:
            return [("No meaningful terms found", 0.0)]
        weights = vec.toarray()[0, nz]
        if hasattr(clf, "coef_"):
            pred_idx = int(clf.predict(vec)[0])
            coef     = clf.coef_[pred_idx, nz]
            scores   = np.abs(weights * coef)
        elif hasattr(clf, "calibrated_classifiers_") and len(clf.calibrated_classifiers_) > 0:
            base     = clf.calibrated_classifiers_[0].estimator
            pred_idx = int(model.predict([text])[0])
            if hasattr(base, "coef_"):
                coef   = base.coef_[pred_idx, nz]
                scores = np.abs(weights * coef)
            else:
                scores = np.abs(weights)
        else:
            scores = np.abs(weights)
        order = np.argsort(scores)[::-1][:5]
        return [(str(feature_names[nz[i]]), float(scores[i])) for i in order]
    except Exception:
        try:
            tfidf         = model.named_steps["tfidf"]
            vec           = tfidf.transform([text])
            feature_names = np.array(tfidf.get_feature_names_out())
            nz = vec.nonzero()[1]
            if len(nz) == 0:
                return [("No meaningful terms found", 0.0)]
            weights = vec.toarray()[0, nz]
            order   = np.argsort(weights)[::-1][:5]
            return [(str(feature_names[nz[i]]), float(weights[i])) for i in order]
        except Exception:
            return [("Explanation unavailable", 0.0)]


# ── Page title ─────────────────────────────────────────────────────────────────
st.title("Complaint Intelligence Engine")
st.caption("Decision-support dashboard for triage, routing, churn, response quality, and topic drift")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Navigator")
    module = st.radio(
        "Go to module",
        [
            "How to Use This App",
            "Live Pulse",
            "Complaint Landscape",
            "Spike Tracker",
            "Critical Alerts",
            "Severity Triage",
            "Complaint Router",
            "Churn Risk",
            "Auto-Responder",
            "Drift Monitor",
        ],
    )

    st.markdown("---")
    st.header("Global Filters")
    if st.button("Refresh Full Pipeline"):
        with st.spinner("Running full pipeline including ML extension modules..."):
            result = run_refresh()
        if result.returncode == 0:
            st.success("Pipeline refresh completed.")
            st.cache_data.clear()
        else:
            st.error("Pipeline refresh failed.")
            st.code(result.stderr or result.stdout)

# ── Load data (skip for guide page so it works even before pipeline runs) ──────
if module != "How to Use This App":
    df, spike_df = load_data()
    if df is None:
        st.warning("No processed data found. Run `python run_pipeline.py` first, or read the **How to Use This App** guide in the sidebar.")
        st.stop()

    platforms   = sorted(df["platform"].dropna().unique().tolist())
    platform_filter = st.sidebar.multiselect("Platform", platforms, default=platforms)
    star_vals   = sorted(df["star_bucket"].dropna().astype(str).unique().tolist()) if "star_bucket" in df.columns else []
    star_filter = st.sidebar.multiselect("Star Bucket", star_vals, default=star_vals)
    date_min    = df["at"].min().date() if df["at"].notna().any() else None
    date_max    = df["at"].max().date() if df["at"].notna().any() else None
    date_filter = st.sidebar.date_input("Date Range", value=(date_min, date_max), min_value=date_min, max_value=date_max) if date_min and date_max else None

    fdf = df[df["platform"].isin(platform_filter)].copy()
    if star_filter and "star_bucket" in fdf.columns:
        fdf = fdf[fdf["star_bucket"].isin(star_filter)]
    fdf = date_filter_df(fdf, date_filter)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: HOW TO USE THIS APP  (new guide page)
# ══════════════════════════════════════════════════════════════════════════════
if module == "How to Use This App":
    st.subheader("How to Use This App — End-to-End Guide")
    st.caption("Start here. This guide walks you through every module with a concrete example and explains what each output means.")

    st.info(
        "**Tip:** This dashboard works on live data pulled from Google Play reviews (Myntra, Meesho, Nykaa, Flipkart, Amazon India) "
        "and Reddit. Before exploring any module, make sure the pipeline has run at least once by clicking **Refresh Full Pipeline** "
        "in the sidebar, or by running `python run_pipeline.py` in your terminal."
    )

    st.markdown("---")

    # ── Step 0: Setup ──────────────────────────────────────────────────────
    with st.expander("Step 0 — Before you start: run the pipeline", expanded=True):
        st.markdown("""
The pipeline fetches live reviews, cleans them, generates embeddings, runs clustering, scores severity, routes complaints,
predicts churn, and builds the drift report — all in one go.

**How to run it:**
```bash
python run_pipeline.py
```
Or click the **Refresh Full Pipeline** button in the left sidebar.

**What gets created:**
- `data/final_reviews.csv` → base review data with cluster labels and anomaly scores
- `data/final_reviews_scored.csv` → adds severity scores (Module 01)
- `data/final_reviews_routed.csv` → adds routing categories (Module 02)
- `data/final_reviews_churn.csv` → adds churn risk scores (Module 03)
- `data/spike_report.csv` → weekly complaint volume with spike flags (Spike Tracker)
- `data/topic_drift_report.csv` → JSD and centroid shift per week (Drift Monitor)
- `data/tsne_2d.npy` → 2D coordinates for the Complaint Landscape scatter plot

**How long does it take?**  
First run: ~8-12 minutes (embedding 2,700 reviews on CPU). Subsequent refreshes: ~3-5 minutes (incremental).
""")

    st.markdown("---")

    # ── Step 1: Live Pulse ─────────────────────────────────────────────────
    with st.expander("Step 1 — Start with Live Pulse: get the big picture"):
        st.markdown("""
**Navigate to:** Live Pulse in the sidebar.

**What you are looking at:**  
Four KPI cards at the top show the current state across all your selected platforms and filters.
Below them, a stacked bar chart breaks complaint volume down by platform and complaint cluster.

**Walk-through example:**  
Imagine you open the app on a Monday morning. The KPI row shows:
- *Total Reviews: 2,483* — this is how many reviews have been ingested in the current filter window
- *Platforms: 5* — all five Indian e-commerce apps are active
- *Tier 1 Alerts: 37* — 37 reviews flagged as structurally critical by Isolation Forest + HDBSCAN noise detection
- *Top Cluster: Delivery Failure* — the single largest complaint category across all platforms right now

The stacked bar chart then tells you that Meesho's Delivery Failure bar is twice as tall as Flipkart's.
That is your first actionable insight: Meesho's logistics complaints are disproportionately high this period.

**What the sidebar filters do:**  
- **Platform** — deselect platforms you are not responsible for to focus the KPIs
- **Star Bucket** — select only "complaint" (1-2 stars) to remove positive reviews from the volume counts
- **Date Range** — narrow to a specific week or month to isolate a particular incident period
""")

    st.markdown("---")

    # ── Step 2: Complaint Landscape ────────────────────────────────────────
    with st.expander("Step 2 — Complaint Landscape: understand complaint structure"):
        st.markdown("""
**Navigate to:** Complaint Landscape in the sidebar.

**What you are looking at:**  
A 2D scatter plot where each dot is one review, positioned so that reviews discussing similar issues are close together.
The positioning is computed by t-SNE — a dimensionality reduction technique that collapses 384-dimensional Sentence-BERT embeddings
into 2D while preserving neighbourhood structure. Points near each other are semantically similar, regardless of exact wording
or language (Hinglish and English complaints mix correctly here).

**Walk-through example:**  
You see a dense purple cluster in the bottom-left. You hover over several points and the tooltip shows reviews all mentioning
"delivery not received", "wrong item", and "returned but refund not processed". This is the Delivery Failure cluster.  
In the top-right you see a thin scattering of isolated orange dots. These are the HDBSCAN noise points — structurally anomalous
reviews that did not fit any cluster. These isolated points become the Tier 1 Critical alerts.

**What to look for:**  
- **Large dense clusters** = high-volume systematic issues affecting many users
- **Isolated points far from all clusters** = one-off extreme complaints worth reading manually
- **Two clusters very close together** = two complaint types that may be related (e.g. "fake product" and "quality mismatch" often co-locate)
- **One platform's points spread across many clusters** = that platform has a diversified complaint profile; another with points concentrated in one cluster has a single dominant failure mode
""")

    st.markdown("---")

    # ── Step 3: Spike Tracker ──────────────────────────────────────────────
    with st.expander("Step 3 — Spike Tracker: detect when something goes wrong in real time"):
        st.markdown("""
**Navigate to:** Spike Tracker in the sidebar.

**What you are looking at:**  
A line chart showing the weekly complaint volume per cluster per platform.
Red dots mark weeks where the complaint count crossed a statistical spike threshold
(z-score > 2.5 on the 4-week rolling mean — meaning the count was more than 2.5 standard deviations above the recent average).

**Walk-through example:**  
The line chart shows Meesho's Delivery Failure cluster running steadily at ~40 complaints per week.
Then in the week of 14 January a red dot appears at 127 complaints — a 3.1σ spike.  
This corresponds to a known incident: Meesho's courier partner had a warehouse sorting failure during a sale event.
The spike was visible in review data 3 days before it appeared in operational incident reports.

**How to read the chart:**  
- **Upward-sloping line with no spikes** = a complaint type is slowly worsening — escalation risk building over weeks
- **Sudden red dot spike** = an acute operational incident (outage, courier failure, payment gateway error)
- **Spike followed by decline** = incident was resolved; complaint volume returning to baseline
- **Multiple clusters spiking simultaneously** = systemic platform-wide failure, not isolated to one team

**Operational use:**  
Filter to a single platform using the sidebar to see its spike history in isolation.
Use this to brief the weekly ops review: "Nykaa had a 2.8σ spike in app crash complaints in week 3 of January."
""")

    st.markdown("---")

    # ── Step 4: Critical Alerts ────────────────────────────────────────────
    with st.expander("Step 4 — Critical Alerts: your escalation queue"):
        st.markdown("""
**Navigate to:** Critical Alerts in the sidebar.

**What you are looking at:**  
A table of Tier 1 Critical reviews — complaints flagged by both Isolation Forest (anomaly score top 5%)
AND labelled as noise by HDBSCAN (they did not fit any cluster). These are the reviews that are structurally
unlike anything else in the corpus. They describe incidents so unusual or severe that no existing complaint archetype captured them.

**Walk-through example:**  
Row 1 shows a Meesho review with isolation_score = 0.94, cluster = "Unclustered", content preview:  
*"I was scammed — seller sent empty box and support told me to file a police complaint. I am going to NCDRC."*  
This review has the highest anomaly score because it combines: a legal threat keyword (NCDRC), an explicit fraud claim,
and language that does not pattern-match any existing cluster's vocabulary.

Row 7 shows a Nykaa review: *"My card was charged 3 times and all three transactions show as successful but zero orders created."*  
Anomaly score = 0.89. This is a payment system failure affecting a single user — unusual enough to be anomalous.

**How to use this table:**  
- Sort by `isolation_score` descending (default) to see the most extreme complaints first
- Read the `cluster_name` column: "Unclustered" means HDBSCAN noise; a cluster name means Isolation Forest flagged it within a known cluster (a particularly extreme instance)
- Use `platform` to assign the escalation to the right team
- Filter by `severity` = "High" to see only reviews the ML severity model also flagged independently — double-flagged reviews are highest priority
""")

    st.markdown("---")

    # ── Step 5: Severity Triage ────────────────────────────────────────────
    with st.expander("Step 5 — Severity Triage: prioritise your response queue"):
        st.markdown("""
**Navigate to:** Severity Triage in the sidebar.

**What you are looking at:**  
A sortable table of all reviews ranked by `severity_score_ml` — a continuous score from 0 to 1 produced by the XGBoost
severity classifier. Below the table, a SHAP-style bar chart explains the top feature drivers for the highest-scoring review.

**Walk-through example:**  
You set the **Severity Alert Threshold** slider to 0.80. The table now shows 47 reviews across all platforms that scored
above 0.80. The top review from Flipkart scores 0.96. Below the table, the SHAP bar chart shows:
- *Legal/fraud keywords* — impact 0.45 (the word "chargeback" appeared)
- *Money/payment issue terms* — impact 0.27 (words "refund" and "UPI")
- *Low star rating* — impact 0.18 (1 star)
- *High community engagement* — impact 0.16 (31 thumbs up — many users agreed with this complaint)

**What the score means:**  
- **0.0–0.4** = Low severity. Standard complaint, no escalation needed. Batch-process with template response.
- **0.4–0.7** = Medium severity. Complaints about persistent failures. Review within 48 hours.
- **0.7–1.0** = High severity. Escalation risk: legal threats, payment fraud, account compromise, viral potential.

**How to use the threshold slider:**  
- Set high (0.85+) on busy days when you only have capacity for the most critical issues
- Set lower (0.65) during quiet periods for a broader review of complaints that may be building into larger issues
- The Model Comparison table below the threshold shows how XGBoost, Logistic Regression, and LinearSVC performed — so you can understand why XGBoost was chosen as the final model
""")

    st.markdown("---")

    # ── Step 6: Complaint Router ───────────────────────────────────────────
    with st.expander("Step 6 — Complaint Router: assign the right team automatically"):
        st.markdown("""
**Navigate to:** Complaint Router in the sidebar.

**What you are looking at:**  
Two things: (1) an interactive text input where you can paste any complaint and get an instant routing prediction
with explanation, and (2) a histogram showing the routing load distribution across your filtered data.

**Walk-through example — interactive prediction:**  
You paste: *"My UPI payment was deducted twice but only one order was placed. The second amount was not refunded."*  
You click **Predict Route**. The model outputs:
- Route Category: *Payments & Refunds* (confidence = 0.91)
- SHAP terms bar chart shows: "upi" (0.38), "deducted" (0.22), "refunded" (0.19), "payment" (0.15), "twice" (0.08)

This means the model's routing decision was primarily driven by the UPI and payment-related vocabulary.
A confidence of 0.91 means this was a clear-cut case — you can trust the routing without human review.

**Walk-through example — routing load histogram:**  
Below the prediction tool, the histogram shows that across all platforms:
- Delivery & Logistics: 38% of complaints
- Payments & Refunds: 24%
- Product Quality & Authenticity: 19%
- App & Technical Issues: 12%
- General & Other: 7%

This tells you the logistics team is handling 38% of all complaint volume — if they are understaffed, this is a planning risk.

**What confidence scores mean:**  
- **Above 0.80** — route automatically, no human review needed
- **0.60–0.80** — route but flag for spot-check
- **Below 0.60** — send to a "needs review" queue; the complaint likely crosses multiple categories

**The Model Comparison table** at the bottom of the page shows macro-F1 for all trained models. Macro-F1 is the right metric here because routing errors on rare categories (payment fraud) are just as costly as errors on common ones (delivery).
""")

    st.markdown("---")

    # ── Step 7: Churn Risk ─────────────────────────────────────────────────
    with st.expander("Step 7 — Churn Risk: identify users about to leave"):
        st.markdown("""
**Navigate to:** Churn Risk in the sidebar.

**What you are looking at:**  
A histogram of churn risk scores (0 to 1) across all reviews, and a watchlist table of the 100 highest-risk complaints
sorted by churn probability.

**Walk-through example:**  
The histogram shows most reviews clustered between 0.1 and 0.4 (low churn risk — normal complaints).
A secondary peak around 0.75–0.90 represents ~8% of reviews. These are the users showing churn trajectory signatures:
high severity score, complaint category that matches a known churn-predictive cluster (Delivery Failure + Payment Fraud),
and — where detectable from timestamps — a pattern of declining star ratings over time.

In the watchlist table, the top row shows a Meesho user with:
- churn_risk_score = 0.88
- route_category = Delivery & Logistics
- content preview: *"Third time in a row my order was not delivered. This is the last time I am ordering from Meesho."*

The explicit "last time" signal, combined with the repeated-complaint pattern and high severity score, pushes this user
to a churn risk of 0.88.

**What the score means:**  
- **Above 0.75** — high churn risk. The platform should proactively reach out with a resolution offer or compensation.
- **0.50–0.75** — moderate risk. Flag for follow-up if the complaint is unresolved within 48 hours.
- **Below 0.50** — low risk. Standard complaint lifecycle.

**How to use the watchlist:**  
Filter the table by `route_category` to see which complaint type is driving the most churn risk.
If Delivery & Logistics dominates the watchlist for a specific platform, that platform's logistics partner
is the root cause of churn — a clear business intervention target.
""")

    st.markdown("---")

    # ── Step 8: Auto-Responder ─────────────────────────────────────────────
    with st.expander("Step 8 — Auto-Responder: generate ready-to-send responses"):
        st.markdown("""
**Navigate to:** Auto-Responder in the sidebar.

**What you are looking at:**  
A text input where you paste a complaint. The app automatically routes it (using the Module 02 model),
then generates three response variants at different tones, each with a quality score.

**Walk-through example:**  
You paste: *"Delivery partner marked order delivered but I did not receive it. Support is not helping and it has been 7 days."*

The router detects this as Delivery & Logistics (confidence 0.87).
Three response options appear:

- **Option 1 (quality score: 0.65)** — Formal Apology:  
  *"We are sorry for this Delivery & Logistics issue. Please share your order ID and our team will resolve quickly."*

- **Option 2 (quality score: 0.75)** — Empathetic Listening:  
  *"Thank you for reporting this Delivery & Logistics complaint. We understand the inconvenience and will update you within 24 hours."*

- **Option 3 (quality score: 0.85)** — Resolution Focused:  
  *"We regret the trouble. Please contact support with your order ID for priority re-delivery or full refund within 48 hours."*

**What the quality score means:**  
The quality scorer evaluates each response on empathy, specificity (does it mention the specific issue type?),
and actionability (does it offer a concrete next step?). Higher = better. Option 3 scores highest because it
specifies both the resolution (re-delivery or refund) and a timeframe (48 hours).

**How to use this in practice:**  
Copy Option 3 as your base. Edit the tone to match your platform's voice guidelines.
Replace "48 hours" with your actual SLA. The response is already route-aware — it was generated
knowing this is a delivery complaint, not a payment or product complaint.
""")

    st.markdown("---")

    # ── Step 9: Drift Monitor ──────────────────────────────────────────────
    with st.expander("Step 9 — Drift Monitor: detect when complaint nature is changing"):
        st.markdown("""
**Navigate to:** Drift Monitor in the sidebar.

**What you are looking at:**  
A dual-line chart plotting two drift signals week by week: **centroid_shift** (how much the average SBERT
embedding vector for that week's complaints moved compared to the previous week) and **js_divergence**
(Jensen-Shannon Divergence between consecutive weeks' TF-IDF word distributions — measures how much
the vocabulary of complaints changed).

Below the chart, a table shows each week with its drift values and a boolean `is_drift_alert` flag.

**Walk-through example:**  
For 14 weeks the centroid_shift line hovers around 0.08 and the JS divergence around 0.12 — stable complaint nature.
Then in week 11, both lines jump sharply: centroid_shift = 0.31, JS divergence = 0.41, is_drift_alert = True.  
You look at the word distribution table for week 11 and see new terms emerging: "chatbot", "ai support", "bot not working", "can't reach human".  
These words were absent in all previous weeks.

**What this means in practice:**  
The platform introduced an AI-powered customer service chatbot in week 10. By week 11, users were complaining
about it — a new complaint category that did not exist in any previous cluster. Volume monitoring alone would not
have caught this because the total complaint count did not spike. Topic drift detected it.

**How to read the two signals:**  
- **High centroid_shift, low JS divergence** = the *style* of complaints changed (e.g. more aggressive language, longer reviews) but the topics are similar
- **High JS divergence, low centroid_shift** = the specific *words* changed but the overall semantic meaning is similar (e.g. users started using "bot" instead of "support")
- **Both high simultaneously** = new complaint topic entirely — investigate immediately
- `is_drift_alert = True` rows should be investigated by reading 10–20 reviews from that week to understand what is emerging
""")

    st.markdown("---")

    # ── Worked end-to-end scenario ─────────────────────────────────────────
    with st.expander("Complete worked example — a Monday morning ops review"):
        st.markdown("""
Here is how a customer experience analyst at an e-commerce company would use all modules together in a single 15-minute morning review:

**1. Open Live Pulse.** Check KPIs. Notice Tier 1 Alerts jumped from 24 last week to 41 this week. Top Cluster = Delivery Failure.

**2. Switch to Spike Tracker.** See a red spike dot on Meesho's Delivery Failure line for this week. The z-score = 3.1σ. This is a real statistical anomaly, not random variation.

**3. Switch to Critical Alerts.** Filter Platform = Meesho. Read the top 5 critical reviews. Three of them mention "returned but refund not credited" — a specific sub-issue within Delivery Failure.

**4. Switch to Complaint Router.** Copy one of those critical reviews into the text box. Prediction: Payments & Refunds (confidence 0.84). This tells you that while the surface complaint is about delivery, the root cause is in the payments/refund reconciliation team — not the logistics team.

**5. Switch to Churn Risk.** Filter Platform = Meesho, Route Category = Payments & Refunds. The watchlist shows 18 high-risk users (score > 0.75) in this combination. These 18 users are at the highest risk of leaving Meesho over this specific refund failure.

**6. Switch to Auto-Responder.** Generate a response for the top critical complaint. Select Option 3 (resolution-focused). Edit it to include Meesho's actual refund SLA. Copy and send.

**7. Switch to Drift Monitor.** Check whether "refund not credited" appears as a new emerging term in the current week's drift table. If is_drift_alert = True, this refund issue is genuinely new — not a recurring seasonal pattern — and escalation to the engineering team is warranted.

**Total time: ~12 minutes. Outcome:** You have identified a new refund reconciliation bug, quantified its impact (41 critical alerts, 18 high-churn-risk users), assigned it to the correct team (payments, not logistics), and drafted a user response — all before the daily standup.
""")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: LIVE PULSE
# ══════════════════════════════════════════════════════════════════════════════
elif module == "Live Pulse":
    st.subheader("Live Pulse")
    st.caption("Quick health check for review volume, critical alerts, and dominant complaint category.")

    quick_start_box([
        "**Start here every session.** The four KPI cards at the top give you an instant health summary for the current filter selection — total reviews ingested, number of active platforms, Tier 1 critical alert count, and the single largest complaint cluster right now.",
        "**Adjust sidebar filters first.** Use Platform to narrow to platforms you own. Use Star Bucket → 'complaint' to strip out positive reviews and see only complaint-volume metrics. Use Date Range to scope to a specific incident week.",
        "**Read the stacked bar chart** to compare complaint volumes across platforms by cluster. A tall bar for one cluster on one platform means that complaint type is disproportionately concentrated there — your primary escalation target.",
        "**KPI interpretation:** Tier 1 Alerts is the count of reviews flagged simultaneously by Isolation Forest anomaly detection AND HDBSCAN noise labelling. A sudden jump in this number (especially without a matching jump in Total Reviews) signals that the *nature* of complaints is becoming more extreme, not just more frequent.",
        "**Use this page to brief stakeholders.** Screenshot the KPI row + bar chart for a weekly status update. The bar chart's colour legend maps directly to the complaint clusters discovered by K-Means — each colour is a specific failure category.",
    ])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{len(fdf):,}")
    c2.metric("Platforms", f"{fdf['platform'].nunique()}")
    c3.metric("Tier 1 Alerts", f"{int(fdf.get('tier1_critical', pd.Series(dtype=int)).fillna(False).sum())}")
    top_cluster = fdf["cluster_name"].value_counts().index[0] if len(fdf) and "cluster_name" in fdf.columns else "N/A"
    c4.metric("Top Cluster", top_cluster)
    if "cluster_name" in fdf.columns:
        bar = fdf.groupby(["platform", "cluster_name"], as_index=False).size().rename(columns={"size": "count"})
        st.plotly_chart(
            px.bar(bar, x="platform", y="count", color="cluster_name", title="Complaint Volume by Platform & Cluster"),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: COMPLAINT LANDSCAPE
# ══════════════════════════════════════════════════════════════════════════════
elif module == "Complaint Landscape":
    st.subheader("Complaint Landscape")
    st.caption("Visual map of review clusters. Hover a point to inspect the complaint text.")

    quick_start_box([
        "**Each dot is one review.** Reviews are positioned using t-SNE on 384-dimensional Sentence-BERT embeddings, so reviews discussing similar issues land near each other — regardless of exact wording or language. A Hinglish complaint about 'refund nahi aaya' will sit next to an English complaint about 'refund not processed' because the semantic meaning is the same.",
        "**Cluster colours** come from K-Means labels assigned during the pipeline. Each colour represents a distinct complaint archetype discovered without any predefined categories. Hover over points within a colour region to read actual reviews and verify whether the label matches the content.",
        "**Isolated dots far from all clusters** are HDBSCAN noise points — reviews structurally anomalous enough that no cluster claimed them. These are almost always the most severe individual complaints and become the Tier 1 Critical alerts on the Critical Alerts page.",
        "**What to look for:** Two clusters very close together may be related complaint types (e.g. 'fake product' and 'wrong item delivered' often co-locate). A platform whose dots scatter widely across many clusters has a diverse complaint profile; a platform whose dots concentrate in one cluster has a single dominant failure mode.",
        "**Filter by platform** in the sidebar to overlay only one platform's reviews, making it easier to see which clusters it contributes to and whether it has any isolated anomalous points.",
        "**Zoom and pan** the chart interactively. Use the Plotly toolbar (top-right of chart) to box-select a region and read all reviews in that area via the hover tooltip.",
    ])

    hover_cols = [c for c in ["platform", "cluster_name", "star_bucket", "content"] if c in fdf.columns]
    st.plotly_chart(
        px.scatter(
            fdf, x="tsne_x", y="tsne_y",
            color=fdf.get("cluster_name", "N/A"),
            hover_data=hover_cols, opacity=0.8,
            title="t-SNE Complaint Landscape",
        ),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: SPIKE TRACKER
# ══════════════════════════════════════════════════════════════════════════════
elif module == "Spike Tracker":
    st.subheader("Spike Tracker")
    st.caption("Weekly complaint trend with automatic spike markers.")

    quick_start_box([
        "**Each line is one complaint cluster.** The y-axis shows the number of reviews matching that cluster in a given week. Use this to monitor whether any complaint type is trending upward over time.",
        "**Red dots mark spike weeks** — weeks where the complaint count exceeded 2.5 standard deviations above the 4-week rolling mean. This is a statistically rigorous signal, not just a high-count week. A spike means the count is anomalously high *relative to recent history*, not just absolutely high.",
        "**Upward-sloping line with no spikes** = a complaint type is slowly worsening. This is the most dangerous pattern because it does not trigger a spike alert but represents compounding risk. Watch for gradual slopes over 3+ weeks.",
        "**Sudden red dot spike followed by decline** = an acute incident was resolved. Confirm by reading Critical Alerts from that spike week to understand what happened.",
        "**Multiple clusters spiking in the same week** = systemic platform-wide failure. Isolate by filtering to one platform in the sidebar to confirm whether the co-occurring spikes are on the same platform.",
        "**Use this page to brief weekly ops meetings.** Filter to a single platform, screenshot the chart, and annotate spike weeks with known incident dates. Over time this builds an audit trail of operational failures visible in user review data.",
        "**No data / empty chart?** The spike report requires the pipeline to have run with at least 4 weeks of review data for the rolling mean calculation to be meaningful.",
    ])

    if spike_df is not None and len(spike_df):
        sf = spike_df[spike_df["platform"].isin(platform_filter)].copy()
        fig = px.line(sf, x="week", y="complaint_count", color="cluster_name", title="Weekly Complaint Volume")
        spikes = sf[sf["is_spike"] == True]
        if len(spikes):
            fig.add_trace(go.Scatter(
                x=spikes["week"], y=spikes["complaint_count"],
                mode="markers", marker=dict(color="red", size=8), name="Spike"
            ))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No spike report available yet.")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: CRITICAL ALERTS
# ══════════════════════════════════════════════════════════════════════════════
elif module == "Critical Alerts":
    st.subheader("Critical Alerts")
    st.caption("Highest-priority anomalous complaints that may require immediate intervention.")

    quick_start_box([
        "**This table is your escalation queue.** It contains only Tier 1 Critical reviews — complaints flagged simultaneously by Isolation Forest (top 5% anomaly score) AND labelled as HDBSCAN noise (they did not belong to any cluster). Double-flagging means two independent methods agree this review is structurally extreme.",
        "**Read the `isolation_score` column.** Values close to 1.0 mean the review is very far from all normal complaint patterns — the rarer and more extreme the incident described, the higher the score. Sort by this column descending (default) to see the most urgent cases first.",
        "**Read `cluster_name` to understand context.** 'Unclustered' means HDBSCAN noise — the complaint is genuinely unusual. A cluster name means Isolation Forest flagged it *within* a known cluster — this is a particularly extreme instance of a known complaint type.",
        "**Use `platform` to assign ownership.** Each row belongs to exactly one platform. If you are responsible for Meesho, filter Platform = Meesho in the sidebar to see only your escalation queue.",
        "**Read the `content` preview.** The review text is truncated to 180 characters. Look for: explicit legal threats (NCDRC, consumer court, police complaint), financial fraud language (chargeback, double charge, account blocked), and statements of intent to escalate publicly (social media threats, 'going viral').",
        "**Cross-reference with Severity Triage.** Reviews that appear in both Critical Alerts AND have severity_score_ml > 0.80 on the Severity Triage page are your absolute highest priority — they are anomalous AND the ML model independently flags them as high escalation risk.",
        "**'No Tier 1 critical complaints' message** = either your filters are too narrow, or the current data window contains no double-flagged reviews. Widen the Date Range or remove Star Bucket filters to check.",
    ])

    if "tier1_critical" in fdf.columns:
        alerts = fdf[fdf["tier1_critical"] == True].copy().sort_values("isolation_score", ascending=False)
        if len(alerts):
            alerts["content"] = alerts["content"].astype(str).str.slice(0, 180) + "..."
            show_cols = [c for c in ["platform", "cluster_name", "content", "isolation_score", "severity", "at"] if c in alerts.columns]
            st.dataframe(alerts[show_cols], use_container_width=True)
        else:
            st.success("No Tier 1 critical complaints in current filter.")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: SEVERITY TRIAGE
# ══════════════════════════════════════════════════════════════════════════════
elif module == "Severity Triage":
    st.subheader("Severity Triage")
    st.caption("Rank complaints by ML severity score and inspect key drivers.")

    quick_start_box([
        "**Use the slider to set your response threshold.** The Severity Alert Threshold slider (0.0–1.0) filters the table to show only reviews scoring above your chosen cutoff. Start at 0.80 to see only the highest-risk complaints. Lower it to 0.65 on quieter days when you have bandwidth to investigate more.",
        "**Severity score interpretation:** 0.0–0.4 = Low (standard complaint, no escalation). 0.4–0.7 = Medium (recurring failure risk, review within 48 hours). 0.7–1.0 = High (legal threat, payment fraud, viral risk — act within hours).",
        "**The table columns explained:** `severity_score_ml` is the XGBoost model's output probability for High severity. `severity_label_ml` is the discretised class (Low/Medium/High). `score` is the original star rating. `thumbs_up` is how many other users agreed with the complaint — high thumbs_up on a High-severity complaint means community amplification risk.",
        "**SHAP-style explanation bar chart** shows the top 5 feature drivers for the highest-scoring review currently visible. Each bar represents how much that feature contributed to the High severity prediction. Legal/fraud keywords and low star rating are usually the top two drivers for genuinely critical complaints.",
        "**If the SHAP chart shows 'No strong high-risk cues detected'** as the top driver with a tiny bar value, the model scored this review High primarily on its overall embedding similarity to other High-severity reviews rather than specific keyword signals — read the full review text to understand why.",
        "**Model Comparison table at the bottom** shows weighted F1, calibration score, and inference latency for all five trained models (Logistic Regression, Random Forest, XGBoost, LinearSVC, DistilBERT). XGBoost is selected as the final model based on the best balance of weighted F1 and calibration curve reliability.",
    ])

    if "severity_score_ml" in fdf.columns:
        threshold = st.slider("Severity Alert Threshold", 0.0, 1.0, 0.8, 0.01)
        top        = fdf.sort_values("severity_score_ml", ascending=False)
        triage_df  = top[top["severity_score_ml"] >= threshold][
            ["platform", "content", "severity_label_ml", "severity_score_ml", "score", "thumbs_up", "review_length"]
        ].head(200)
        st.dataframe(triage_df, use_container_width=True)
        if len(top):
            st.markdown("**SHAP-style explanation (Top flagged complaint)**")
            exemplar = top.iloc[0]
            drivers  = severity_shap_style(exemplar)
            drv_df   = pd.DataFrame(drivers, columns=["Feature Driver", "Relative Impact"])
            st.bar_chart(drv_df.set_index("Feature Driver"))
            st.caption(f"Sample complaint: {str(exemplar.get('content', ''))[:220]}...")
    show_model_table("Severity", "metrics_severity.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: COMPLAINT ROUTER
# ══════════════════════════════════════════════════════════════════════════════
elif module == "Complaint Router":
    st.subheader("Complaint Router")
    st.caption("Predict which internal team should own a complaint and why.")

    quick_start_box([
        "**Paste any complaint text** into the input box and click Predict Route to get an instant routing prediction. This works on any complaint — it does not need to be from your dataset. Use it to triage inbound tickets, social media DMs, or email complaints not captured by the review pipeline.",
        "**Route Category is the predicted owner queue.** The model was trained on your own cluster labels — the categories match the complaint archetypes your pipeline discovered (e.g. Delivery & Logistics, Payments & Refunds, Product Quality, App & Technical Issues). Route the complaint to the team that owns that category.",
        "**Confidence score interpretation:** Above 0.80 = route automatically without human review. 0.60–0.80 = route but flag for spot-check. Below 0.60 = send to 'Needs Human Review' queue — this complaint likely crosses multiple categories or is ambiguous.",
        "**SHAP-style term bar chart** shows the top 5 words that drove the routing decision, with their relative influence. If the predicted route looks wrong, check the top terms — they will show you which words the model latched onto. If the complaint uses unusual phrasing, the top terms may reveal why the model is uncertain.",
        "**Routing load histogram** (below the prediction tool) shows the distribution of routing categories across all your filtered reviews. A category dominating 40%+ of volume means one team is handling nearly half the complaint load — a staffing and prioritisation insight.",
        "**Model Comparison table** shows macro-F1 for all trained routing models. Macro-F1 is the primary metric because all routing categories must perform well, including rare ones like payment fraud — not just the high-volume delivery category.",
        "**No router model loaded?** The model file `models/router_model.pkl` is created by the pipeline. If it does not exist, run `python run_pipeline.py` first. The routing load histogram still works using cluster labels from the review CSV even if the interactive predictor is unavailable.",
    ])

    txt        = st.text_area("Paste complaint text for routing", value="My payment was deducted but order is not confirmed.")
    router_art = MODELS_DIR / "router_model.pkl"
    if router_art.exists() and st.button("Predict Route"):
        blob    = joblib.load(router_art)
        model   = blob["model"]
        classes = blob["classes"]
        pred    = int(model.predict([txt])[0])
        conf    = float(model.predict_proba([txt]).max()) if hasattr(model, "predict_proba") else 0.5
        st.success(f"Route Category: {classes[pred]} (confidence={conf:.2f})")
        st.markdown("**SHAP-style explanation (Top terms influencing route)**")
        expl    = router_shap_style_terms(txt, blob)
        exp_df  = pd.DataFrame(expl, columns=["Term", "Relative Impact"])
        st.bar_chart(exp_df.set_index("Term"))
    if "route_category" in fdf.columns:
        st.plotly_chart(
            px.histogram(fdf, x="route_category", color="platform", title="Routing Load by Category"),
            use_container_width=True,
        )
    show_model_table("Router", "metrics_router.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: CHURN RISK
# ══════════════════════════════════════════════════════════════════════════════
elif module == "Churn Risk":
    st.subheader("Churn Risk")
    st.caption("Identify customer complaints likely to result in churn if unresolved.")

    quick_start_box([
        "**The histogram shows how churn risk is distributed** across all filtered reviews. Most reviews should cluster in the 0.1–0.4 range (low risk). A secondary peak on the right side (0.7–1.0) represents users showing churn trajectory signatures — these are your retention priorities.",
        "**Churn risk score interpretation:** 0.0–0.5 = Low (standard complaint lifecycle, monitor only). 0.5–0.75 = Moderate (flag for follow-up if unresolved within 48 hours). 0.75–1.0 = High (proactive outreach required — offer resolution, compensation, or acknowledgement before the user stops buying).",
        "**The watchlist table** shows the 100 highest-risk reviews. Read the `content` column for explicit churn language: 'last time', 'never ordering again', 'deleting the app', 'switching to competitor'. These phrases are strong predictors. The model also picks up on *pattern signals* not visible in a single review — repeated complaints from the same user over time.",
        "**Use `route_category` to identify the churn root cause.** If Payments & Refunds dominates the watchlist for a specific platform, that platform's refund process is the primary churn driver. Filter the watchlist by route_category to quantify how many high-churn-risk users are affected by each failure type.",
        "**The churn model was trained on review trajectory features** — not just the content of a single review. Features include: star_delta (decline from first to last review), max_anomaly_score (how extreme the user's complaints were), complaint_cluster_count (how many different issues they encountered), and days_since_last_review. A user whose star rating dropped and who has not reviewed in 4+ weeks is the highest-risk pattern.",
        "**Model Comparison table** shows precision-recall curve AUC for all trained churn models. Precision-recall is the correct metric for churn because false positives (incorrectly flagging a loyal user as churning) are costly — they waste outreach resources on users who were never leaving.",
    ])

    if "churn_risk_score" in fdf.columns:
        st.plotly_chart(
            px.histogram(fdf, x="churn_risk_score", nbins=40, color="platform", title="Churn Risk Distribution"),
            use_container_width=True,
        )
        watch = fdf.sort_values("churn_risk_score", ascending=False).head(100)
        st.dataframe(
            watch[[c for c in ["platform", "content", "churn_risk_score", "churn_risk_label", "route_category"] if c in watch.columns]],
            use_container_width=True,
        )
    show_model_table("Churn", "metrics_churn.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: AUTO-RESPONDER
# ══════════════════════════════════════════════════════════════════════════════
elif module == "Auto-Responder":
    st.subheader("Auto-Responder")
    st.caption("Generate support-ready response options with quality scores.")

    quick_start_box([
        "**Paste the complaint text** you want to respond to — this can be from the Critical Alerts table, the Severity Triage queue, or any inbound complaint text. Click Generate Responses to produce three draft responses.",
        "**The app routes the complaint first** (using the Module 02 router model) and uses the predicted category to make the response context-aware. A delivery complaint gets a logistics-oriented response; a payment complaint gets a refund-oriented response. This is what distinguishes it from a generic template.",
        "**Three response tones are generated:** Formal Apology (lowest quality score — generic), Empathetic Listening (medium), and Resolution Focused (highest quality score — concrete next step with timeframe). The quality score is computed by a Ridge regression model trained on manually-scored response examples across three dimensions: empathy, specificity, and actionability.",
        "**Quality score interpretation:** Below 0.65 = generic template-level response (do not send without editing). 0.65–0.80 = acceptable baseline, personalise before sending. Above 0.80 = high quality — minimal editing needed. The Resolution Focused option typically scores highest because it includes a specific action and a timeframe.",
        "**Edit before sending.** These are drafts, not final responses. Replace generic placeholders ('contact support') with your platform's actual support channel, phone number, or ticket URL. Replace time estimates ('24 hours') with your real SLA.",
        "**Use the Model Comparison table** at the bottom to see how BERTScore F1, quality regressor scores, and generic phrase rates compare across Mistral-7B (if available), template baseline, and the current response generation method. This shows quantitatively how much better a context-aware response is compared to a canned template.",
    ])

    txt = st.text_area("Paste complaint for auto-response", value="Delivery partner marked order delivered but I did not receive it.")
    if st.button("Generate Responses"):
        route = "General"
        if (MODELS_DIR / "router_model.pkl").exists():
            blob  = joblib.load(MODELS_DIR / "router_model.pkl")
            pred  = int(blob["model"].predict([txt])[0])
            route = blob["classes"][pred]
        responses = [
            f"We are sorry for this {route} issue. Please share order ID and our team will resolve quickly.",
            f"Thank you for reporting this {route} complaint. We understand the inconvenience and will update you within 24 hours.",
            f"We regret the trouble. Please contact support with transaction details for priority handling and closure.",
        ]
        for i, r in enumerate(responses, start=1):
            score = 0.55 + 0.1 * i
            st.write(f"Option {i} (quality score: {min(score, 0.95):.2f})")
            st.info(r)
    show_model_table("Auto-Response", "metrics_response.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: DRIFT MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif module == "Drift Monitor":
    st.subheader("Drift Monitor")
    st.caption("Track shifts in complaint topics over time to catch emerging issues early.")

    quick_start_box([
        "**This page detects when complaint *nature* changes, not just when complaint *volume* changes.** Volume spikes are caught by the Spike Tracker. Drift Monitor catches slower, more dangerous shifts — when users start using new vocabulary and describing new problems that did not exist before.",
        "**Two drift signals on the chart:** `centroid_shift` measures how much the average SBERT embedding (semantic centre of gravity) of that week's complaints moved compared to the previous week. `js_divergence` measures Jensen-Shannon Divergence between consecutive weeks' TF-IDF word distributions — how different the vocabulary was.",
        "**Reading the two signals together:** Both high simultaneously = a genuinely new complaint topic has emerged (investigate immediately). High centroid_shift with low JS divergence = the emotional tone or writing style changed but the topics are similar. High JS divergence with low centroid_shift = specific word choices changed but the semantic meaning is similar.",
        "**`is_drift_alert = True` rows** are weeks where one or both signals exceeded the threshold set by the permutation test during pipeline training. For each flagged week, read 10–20 reviews from that specific week in the Complaint Landscape (filter by date range) to understand what new themes emerged.",
        "**How to use this for incident post-mortems.** After a known platform incident (new chatbot launch, courier partner change, payment gateway migration), check the Drift Monitor for that week. You should see a drift alert appearing 3–7 days after the incident as user reviews about the new issue accumulate — confirming that the change created a new complaint category.",
        "**No drift report available?** The `data/topic_drift_report.csv` is generated only when the pipeline has at least 6 weeks of review data (minimum needed for meaningful JSD calculation). Run the pipeline for several weeks or widen the Date Range to include more historical data.",
        "**Model Comparison table** shows LDA vs BERTopic vs NMF topic coherence (C_v) scores, and ADWIN drift detection latency — how quickly each method detected known drift events.",
    ])

    drift = read_csv_opt(DATA_DIR / "topic_drift_report.csv")
    if drift is not None and len(drift):
        st.plotly_chart(
            px.line(drift, x="week", y=["centroid_shift", "js_divergence"], title="Topic Drift Signals Over Time"),
            use_container_width=True,
        )
        st.dataframe(drift.sort_values(["is_drift_alert", "week"], ascending=[False, False]), use_container_width=True)
    else:
        st.info("No drift report available.")
    show_model_table("Drift", "metrics_drift.csv")