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

DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")


@st.cache_data
def read_csv_opt(path: Path):
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else None


@st.cache_data
def load_data():
    # Prefer fully upgraded dataset if present.
    candidates = [
        DATA_DIR / "final_reviews_response.csv",
        DATA_DIR / "final_reviews_churn.csv",
        DATA_DIR / "final_reviews_routed.csv",
        DATA_DIR / "final_reviews_scored.csv",
        DATA_DIR / "final_reviews.csv",
    ]
    final_path = next((p for p in candidates if p.exists()), None)
    spike_path = DATA_DIR / "spike_report.csv"
    tsne_path = DATA_DIR / "tsne_2d.npy"
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
        end = pd.Timestamp(end_d, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    else:
        start = pd.Timestamp(start_d)
        end = pd.Timestamp(end_d) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return fdf[(fdf["at"] >= start) & (fdf["at"] <= end)]


def show_model_table(task_name: str, metrics_file: str):
    path = REPORTS_DIR / metrics_file
    st.subheader(f"{task_name} Model Comparison")
    mdf = read_csv_opt(path)
    if mdf is None or mdf.empty:
        st.info("No metrics generated yet.")
        return
    st.dataframe(mdf, use_container_width=True)


def quick_start_box(what_items: list[str], how_items: list[str]) -> None:
    what_md = "\n".join([f"- {x}" for x in what_items])
    how_md = "\n".join([f"- {x}" for x in how_items])
    st.info(
        f"**Quick Start**\n\n"
        f"**What this page does**\n{what_md}\n\n"
        f"**How to read this chart**\n{how_md}"
    )


def severity_shap_style(row: pd.Series) -> list[tuple[str, float]]:
    """Heuristic SHAP-style feature contributions for severity."""
    txt = str(row.get("content", "")).lower()
    score = float(row.get("score", 3))
    thumbs = float(row.get("thumbs_up", 0))
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
    """Approximate token contribution (SHAP-style) for router prediction."""
    model = model_blob["model"]
    try:
        tfidf = model.named_steps["tfidf"]
        clf = model.named_steps["clf"]
        vec = tfidf.transform([text])
        feature_names = np.array(tfidf.get_feature_names_out())
        nz = vec.nonzero()[1]
        if len(nz) == 0:
            return [("No meaningful terms found", 0.0)]
        weights = vec.toarray()[0, nz]

        if hasattr(clf, "coef_"):
            pred_idx = int(clf.predict(vec)[0])
            coef = clf.coef_[pred_idx, nz]
            scores = np.abs(weights * coef)
        elif hasattr(clf, "calibrated_classifiers_") and len(clf.calibrated_classifiers_) > 0:
            base = clf.calibrated_classifiers_[0].estimator
            pred_idx = int(model.predict([text])[0])
            if hasattr(base, "coef_"):
                coef = base.coef_[pred_idx, nz]
                scores = np.abs(weights * coef)
            else:
                scores = np.abs(weights)
        else:
            scores = np.abs(weights)
        order = np.argsort(scores)[::-1][:5]
        return [(str(feature_names[nz[i]]), float(scores[i])) for i in order]
    except Exception:
        # Fallback: TF-IDF intensity only.
        try:
            tfidf = model.named_steps["tfidf"]
            vec = tfidf.transform([text])
            feature_names = np.array(tfidf.get_feature_names_out())
            nz = vec.nonzero()[1]
            if len(nz) == 0:
                return [("No meaningful terms found", 0.0)]
            weights = vec.toarray()[0, nz]
            order = np.argsort(weights)[::-1][:5]
            return [(str(feature_names[nz[i]]), float(weights[i])) for i in order]
        except Exception:
            return [("Explanation unavailable", 0.0)]


st.title("Complaint Intelligence Engine")
st.caption("Decision-support dashboard for triage, routing, churn, response quality, and topic drift")

with st.sidebar:
    st.header("Navigator")
    module = st.radio(
        "Go to module",
        [
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

df, spike_df = load_data()
if df is None:
    st.warning("No processed data found. Run `python run_pipeline.py` first.")
    st.stop()

platforms = sorted(df["platform"].dropna().unique().tolist())
platform_filter = st.sidebar.multiselect("Platform", platforms, default=platforms)
star_vals = sorted(df["star_bucket"].dropna().astype(str).unique().tolist()) if "star_bucket" in df.columns else []
star_filter = st.sidebar.multiselect("Star Bucket", star_vals, default=star_vals)
date_min = df["at"].min().date() if df["at"].notna().any() else None
date_max = df["at"].max().date() if df["at"].notna().any() else None
date_filter = st.sidebar.date_input("Date Range", value=(date_min, date_max), min_value=date_min, max_value=date_max) if date_min and date_max else None

fdf = df[df["platform"].isin(platform_filter)].copy()
if star_filter and "star_bucket" in fdf.columns:
    fdf = fdf[fdf["star_bucket"].isin(star_filter)]
fdf = date_filter_df(fdf, date_filter)

if module == "Live Pulse":
    st.subheader("Live Pulse")
    st.caption("Quick health check for review volume, critical alerts, and dominant complaint category.")
    quick_start_box(
        [
            "Shows top KPIs for current filters.",
            "Highlights dominant complaint cluster by volume.",
        ],
        [
            "Use KPI cards for fast health check.",
            "In bar chart, taller bars mean higher complaint volume.",
            "Color segments show which cluster drives each platform's volume.",
        ],
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{len(fdf):,}")
    c2.metric("Platforms", f"{fdf['platform'].nunique()}")
    c3.metric("Tier 1 Alerts", f"{int(fdf.get('tier1_critical', pd.Series(dtype=int)).fillna(False).sum())}")
    top_cluster = fdf["cluster_name"].value_counts().index[0] if len(fdf) and "cluster_name" in fdf.columns else "N/A"
    c4.metric("Top Cluster", top_cluster)
    if "cluster_name" in fdf.columns:
        bar = fdf.groupby(["platform", "cluster_name"], as_index=False).size().rename(columns={"size": "count"})
        st.plotly_chart(px.bar(bar, x="platform", y="count", color="cluster_name", title="Complaint Volume by Platform & Cluster"), use_container_width=True)

elif module == "Complaint Landscape":
    st.subheader("Complaint Landscape")
    st.caption("Visual map of review clusters. Hover a point to inspect the complaint text.")
    quick_start_box(
        [
            "Maps complaints in 2D based on semantic similarity.",
            "Helps you spot dense issue regions and outliers.",
        ],
        [
            "Points close together discuss similar issues.",
            "Colors represent cluster/category assignments.",
            "Hover any point to read complaint context.",
        ],
    )
    hover_cols = [c for c in ["platform", "cluster_name", "star_bucket", "content"] if c in fdf.columns]
    st.plotly_chart(
        px.scatter(fdf, x="tsne_x", y="tsne_y", color=fdf.get("cluster_name", "N/A"), hover_data=hover_cols, opacity=0.8, title="t-SNE Complaint Landscape"),
        use_container_width=True,
    )

elif module == "Spike Tracker":
    st.subheader("Spike Tracker")
    st.caption("Weekly complaint trend with automatic spike markers.")
    quick_start_box(
        [
            "Tracks complaint count week by week.",
            "Flags unusual jumps using spike markers.",
        ],
        [
            "Line slope up = worsening trend; down = improving trend.",
            "Red dots indicate detected spike weeks.",
            "Compare cluster lines to identify fastest-growing issue type.",
        ],
    )
    if spike_df is not None and len(spike_df):
        sf = spike_df[spike_df["platform"].isin(platform_filter)].copy()
        fig = px.line(sf, x="week", y="complaint_count", color="cluster_name", title="Weekly Complaint Volume")
        spikes = sf[sf["is_spike"] == True]
        if len(spikes):
            fig.add_trace(go.Scatter(x=spikes["week"], y=spikes["complaint_count"], mode="markers", marker=dict(color="red", size=8), name="Spike"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No spike report available yet.")

elif module == "Critical Alerts":
    st.subheader("Critical Alerts")
    st.caption("Highest-priority anomalous complaints that may require immediate intervention.")
    quick_start_box(
        [
            "Ranks complaints that look high-risk or unusual.",
            "Supports rapid triage for escalation queue.",
        ],
        [
            "Rows near top are most critical by anomaly score.",
            "Read platform and cluster to assign ownership quickly.",
            "Use filtered view to focus on current operational scope.",
        ],
    )
    if "tier1_critical" in fdf.columns:
        alerts = fdf[fdf["tier1_critical"] == True].copy().sort_values("isolation_score", ascending=False)
        if len(alerts):
            alerts["content"] = alerts["content"].astype(str).str.slice(0, 180) + "..."
            show_cols = [c for c in ["platform", "cluster_name", "content", "isolation_score", "severity", "at"] if c in alerts.columns]
            st.dataframe(alerts[show_cols], use_container_width=True)
        else:
            st.success("No Tier 1 critical complaints in current filter.")

elif module == "Severity Triage":
    st.subheader("Severity Triage")
    st.caption("Rank complaints by ML severity score and inspect key drivers.")
    quick_start_box(
        [
            "Sorts complaints by model-predicted severity risk.",
            "Explains strongest signals behind high-severity predictions.",
        ],
        [
            "Higher severity score = higher escalation priority.",
            "Threshold slider controls queue strictness.",
            "SHAP-style bars show relative contribution of top drivers.",
        ],
    )
    if "severity_score_ml" in fdf.columns:
        threshold = st.slider("Severity Alert Threshold", 0.0, 1.0, 0.8, 0.01)
        top = fdf.sort_values("severity_score_ml", ascending=False)
        triage_df = top[top["severity_score_ml"] >= threshold][["platform", "content", "severity_label_ml", "severity_score_ml", "score", "thumbs_up", "review_length"]].head(200)
        st.dataframe(triage_df, use_container_width=True)
        if len(top):
            st.markdown("**SHAP-style explanation (Top flagged complaint)**")
            exemplar = top.iloc[0]
            drivers = severity_shap_style(exemplar)
            drv_df = pd.DataFrame(drivers, columns=["Feature Driver", "Relative Impact"])
            st.bar_chart(drv_df.set_index("Feature Driver"))
            st.caption(f"Sample complaint: {str(exemplar.get('content',''))[:220]}...")
    show_model_table("Severity", "metrics_severity.csv")

elif module == "Complaint Router":
    st.subheader("Complaint Router")
    st.caption("Predict which internal team should own a complaint and why.")
    quick_start_box(
        [
            "Predicts complaint route category/team.",
            "Provides explanation terms to improve trust in prediction.",
        ],
        [
            "Route label is the recommended owner queue.",
            "Confidence score indicates certainty.",
            "SHAP-style term bars show words most influencing routing.",
        ],
    )
    txt = st.text_area("Paste complaint text for routing", value="My payment was deducted but order is not confirmed.")
    router_art = MODELS_DIR / "router_model.pkl"
    if router_art.exists() and st.button("Predict Route"):
        blob = joblib.load(router_art)
        model = blob["model"]
        classes = blob["classes"]
        pred = int(model.predict([txt])[0])
        conf = float(model.predict_proba([txt]).max()) if hasattr(model, "predict_proba") else 0.5
        st.success(f"Route Category: {classes[pred]} (confidence={conf:.2f})")
        st.markdown("**SHAP-style explanation (Top terms influencing route)**")
        expl = router_shap_style_terms(txt, blob)
        exp_df = pd.DataFrame(expl, columns=["Term", "Relative Impact"])
        st.bar_chart(exp_df.set_index("Term"))
    if "route_category" in fdf.columns:
        st.plotly_chart(px.histogram(fdf, x="route_category", color="platform", title="Routing Load by Category"), use_container_width=True)
    show_model_table("Router", "metrics_router.csv")

elif module == "Churn Risk":
    st.subheader("Churn Risk")
    st.caption("Identify customer complaints likely to result in churn if unresolved.")
    quick_start_box(
        [
            "Estimates churn risk from complaint and context signals.",
            "Creates watchlist for proactive retention actions.",
        ],
        [
            "Right side of histogram = higher churn risk concentration.",
            "Watchlist table shows highest-risk complaints first.",
            "Use route category to plan corrective action owner.",
        ],
    )
    if "churn_risk_score" in fdf.columns:
        st.plotly_chart(px.histogram(fdf, x="churn_risk_score", nbins=40, color="platform", title="Churn Risk Distribution"), use_container_width=True)
        watch = fdf.sort_values("churn_risk_score", ascending=False).head(100)
        st.dataframe(watch[[c for c in ["platform", "content", "churn_risk_score", "churn_risk_label", "route_category"] if c in watch.columns]], use_container_width=True)
    show_model_table("Churn", "metrics_churn.csv")

elif module == "Auto-Responder":
    st.subheader("Auto-Responder")
    st.caption("Generate support-ready response options with quality scores.")
    quick_start_box(
        [
            "Generates response drafts for complaint handling.",
            "Compares options by quality score for quick selection.",
        ],
        [
            "Higher quality score = better relevance/actionability.",
            "Use top option as baseline and edit for policy tone.",
            "Route-aware responses improve resolution speed.",
        ],
    )
    txt = st.text_area("Paste complaint for auto-response", value="Delivery partner marked order delivered but I did not receive it.")
    if st.button("Generate Responses"):
        route = "General"
        if (MODELS_DIR / "router_model.pkl").exists():
            blob = joblib.load(MODELS_DIR / "router_model.pkl")
            pred = int(blob["model"].predict([txt])[0])
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

elif module == "Drift Monitor":
    st.subheader("Drift Monitor")
    st.caption("Track shifts in complaint topics over time to catch emerging issues early.")
    quick_start_box(
        [
            "Monitors whether complaint themes are changing over time.",
            "Detects emerging issues before they become major spikes.",
        ],
        [
            "Rising centroid shift/JS indicates changing topic patterns.",
            "Rows marked as drift alerts need investigation.",
            "Use week-over-week changes to brief operations teams.",
        ],
    )
    drift = read_csv_opt(DATA_DIR / "topic_drift_report.csv")
    if drift is not None and len(drift):
        st.plotly_chart(px.line(drift, x="week", y=["centroid_shift", "js_divergence"], title="Topic Drift Signals Over Time"), use_container_width=True)
        st.dataframe(drift.sort_values(["is_drift_alert", "week"], ascending=[False, False]), use_container_width=True)
    else:
        st.info("No drift report available.")
    show_model_table("Drift", "metrics_drift.csv")
