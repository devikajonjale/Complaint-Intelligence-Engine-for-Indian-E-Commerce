import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Complaint Intelligence Engine", layout="wide")

DATA_DIR = Path("data")


@st.cache_data
def load_data():
    final_path = DATA_DIR / "final_reviews.csv"
    spike_path = DATA_DIR / "spike_report.csv"
    tsne_path = DATA_DIR / "tsne_2d.npy"
    if not final_path.exists() or not spike_path.exists():
        return None, None
    df = pd.read_csv(final_path, encoding="utf-8-sig")
    spike = pd.read_csv(spike_path, encoding="utf-8-sig")
    if tsne_path.exists():
        import numpy as np

        tsne = np.load(tsne_path)
        df["tsne_x"] = tsne[:, 0]
        df["tsne_y"] = tsne[:, 1]
    else:
        df["tsne_x"] = 0
        df["tsne_y"] = 0
    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    return df, spike


def run_refresh():
    cmd = [sys.executable, "run_pipeline.py"]
    return subprocess.run(cmd, capture_output=True, text=True)


st.title("Complaint Intelligence Engine")
st.caption("Unsupervised NLP-driven complaint discovery for Indian e-commerce")

with st.sidebar:
    st.header("Controls")
    if st.button("Refresh Full Pipeline"):
        with st.spinner("Running full data and ML pipeline..."):
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
star_filter = st.sidebar.multiselect("Star Bucket", sorted(df["star_bucket"].dropna().unique()), default=sorted(df["star_bucket"].dropna().unique()))
date_min = df["at"].min().date() if df["at"].notna().any() else None
date_max = df["at"].max().date() if df["at"].notna().any() else None
if date_min and date_max:
    date_filter = st.sidebar.date_input("Date Range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
else:
    date_filter = None

fdf = df[df["platform"].isin(platform_filter) & df["star_bucket"].isin(star_filter)].copy()
if date_filter and len(date_filter) == 2:
    start, end = pd.to_datetime(date_filter[0]), pd.to_datetime(date_filter[1])
    fdf = fdf[(fdf["at"] >= start) & (fdf["at"] <= end)]

tab1, tab2, tab3, tab4 = st.tabs(["Live Pulse", "Complaint Landscape", "Spike Tracker", "Critical Alerts"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{len(fdf):,}")
    c2.metric("Platforms Monitored", f"{fdf['platform'].nunique()}")
    c3.metric("Tier 1 Critical Alerts", f"{int(fdf['tier1_critical'].sum())}")
    top_cluster = fdf["cluster_name"].value_counts().index[0] if len(fdf) else "N/A"
    c4.metric("Top Complaint Cluster", top_cluster)

    bar = (
        fdf.groupby(["platform", "cluster_name"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    fig = px.bar(bar, x="platform", y="count", color="cluster_name", title="Complaint Volume by Platform & Cluster")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    hover_cols = ["platform", "cluster_name", "star_bucket", "content"]
    fig = px.scatter(
        fdf,
        x="tsne_x",
        y="tsne_y",
        color="cluster_name",
        hover_data=hover_cols,
        opacity=0.8,
        title="t-SNE Complaint Landscape",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    if spike_df is not None and len(spike_df):
        sf = spike_df[spike_df["platform"].isin(platform_filter)].copy()
        fig = px.line(sf, x="week", y="complaint_count", color="cluster_name", title="Weekly Complaint Volume")
        spikes = sf[sf["is_spike"] == True]
        if len(spikes):
            fig.add_trace(
                go.Scatter(
                    x=spikes["week"],
                    y=spikes["complaint_count"],
                    mode="markers",
                    marker=dict(color="red", size=8),
                    name="Spike",
                )
            )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No spike report available yet.")

with tab4:
    alerts = fdf[fdf["tier1_critical"] == True].copy().sort_values("isolation_score", ascending=False)
    show_cols = ["platform", "cluster_name", "content", "isolation_score", "severity", "at"]
    if len(alerts):
        alerts["content"] = alerts["content"].astype(str).str.slice(0, 180) + "..."
        st.dataframe(alerts[show_cols], use_container_width=True)
    else:
        st.success("No Tier 1 critical complaints in current filter.")
