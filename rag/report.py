"""Executive complaint report (markdown) for non-technical stakeholders.

Facts (counts, shares, ratings) are computed in pandas; the LLM only turns those facts plus a sample
of critical anomalies into readable prose and is told not to add numbers of its own.

Usage: python rag/report.py [--platform Meesho] [--days 7]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from rag.build_index import load_reviews  # noqa: E402
from rag.cluster_insights import load_insights  # noqa: E402
from rag.config import RAG_DIR, get_llm, robust_invoke  # noqa: E402

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write concise executive complaint briefs for e-commerce leaders who are not technical. "
            "Use ONLY the numbers in FACTS and never invent figures. Cite example reviews by their [R####] id. "
            "Never use em dashes. Output markdown with exactly these sections: "
            "## Headline, ## Key complaint themes, ## Critical incidents, ## Recommended actions (name an owner team per action).",
        ),
        ("human", "Scope: {scope}\n\nFACTS:\n{facts}\n\nCRITICAL ANOMALY SAMPLES:\n{anomalies}"),
    ]
)


def compute_facts(df: pd.DataFrame) -> dict:
    """All numbers in the report come from here, not from the LLM."""
    labels = {v["cluster_id"]: v["label"] for v in load_insights().values() if v["algorithm"] == "kmeans"}
    themes = (
        df.groupby("kmeans_cluster")
        .agg(reviews=("sid", "size"), avg_rating=("score", "mean"), anomalies=("isolation_anomaly", "sum"),
             pipeline_name=("cluster_name", lambda s: s.mode().iat[0]))
        .reset_index()
    )
    themes["theme"] = [labels.get(int(c), n) for c, n in zip(themes["kmeans_cluster"], themes["pipeline_name"])]
    themes["share_pct"] = (100 * themes["reviews"] / len(df)).round(1)
    return {
        "total_reviews": len(df),
        "avg_rating": round(float(df["score"].mean()), 2),
        "pct_1_or_2_star": round(100 * float((df["score"] <= 2).mean()), 1),
        "critical_anomalies": int(df["isolation_anomaly"].sum()),
        "severity_counts": df["severity"].value_counts().to_dict(),
        "by_platform": df["platform"].value_counts().to_dict(),
        "by_routing_team": df["route_category"].value_counts().to_dict(),
        "themes": themes[["theme", "reviews", "share_pct", "avg_rating", "anomalies"]].round(2).to_dict("records"),
    }


def generate_report(platform: str | None = None, days: int | None = None, save: bool = True) -> str:
    df = load_reviews()
    df["at"] = pd.to_datetime(df["at"], errors="coerce", utc=True)
    scope = []
    if platform:
        df = df[df["platform"] == platform]
        scope.append(platform)
    if days:
        end = df["at"].max()
        df = df[df["at"] >= end - pd.Timedelta(days=days)]
        scope.append(f"last {days} days to {end.date()}")
    if df.empty:
        return "No reviews in this scope."

    facts = compute_facts(df)
    crit = df[df["isolation_anomaly"]].sort_values("isolation_score").head(12)  # most anomalous first
    anomalies = "\n".join(f"[{r.sid}] ({r.platform}, {r.score} star) {str(r.content)[:220]}" for r in crit.itertuples())
    inputs = {
        "scope": ", ".join(scope) or "all platforms, all dates",
        "facts": json.dumps(facts, indent=1, default=str),
        "anomalies": anomalies or "none",
    }
    chain = PROMPT | get_llm(temperature=0.2) | StrOutputParser()
    body = (robust_invoke(chain, inputs) or "").strip()
    # House style: no em or en dashes in stakeholder docs. A dash used as a separator becomes a
    # colon; anywhere else it becomes a comma. The non-breaking hyphen is normalised to a plain one.
    for dash in (chr(0x2014), chr(0x2013)):  # em dash, en dash
        body = body.replace(f" {dash} ", ": ").replace(dash, ", ")
    body = body.replace(chr(0x2011), "-")  # non-breaking hyphen
    md = f"# Complaint Intelligence Brief: {', '.join(scope) or 'All platforms'}\n\n{body}"
    if save:
        out = RAG_DIR / "reports"
        out.mkdir(exist_ok=True)
        (out / f"brief_{(platform or 'all').replace(' ', '_')}_{days or 'all'}d.md").write_text(md, encoding="utf-8")
    return md


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform")
    ap.add_argument("--days", type=int)
    a = ap.parse_args()
    print(generate_report(a.platform, a.days))
