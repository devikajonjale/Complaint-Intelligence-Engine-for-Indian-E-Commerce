"""Module 11: Auto-response generator and quality scoring."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from ml_utils import DATA_DIR, MODELS_DIR, REPORTS_DIR, ensure_dirs, update_selected_model

import warnings
warnings.filterwarnings("ignore")

EMPATHY_TOKENS = {"sorry", "apologize", "understand", "regret"}
ACTION_TOKENS = {"share", "provide", "check", "resolve", "contact", "update", "refund"}


def gen_template_v1(route: str, complaint: str) -> str:
    return (
        f"We are sorry for this experience regarding {route}. "
        f"Please share your order ID via in-app support so our team can resolve this quickly."
    )


def gen_template_v2(route: str, complaint: str) -> str:
    return (
        f"Thank you for reporting this {route} issue. We understand the inconvenience. "
        "Our specialist team will review the case and provide an update within 24 hours."
    )


def gen_template_v3(route: str, complaint: str) -> str:
    return (
        f"We regret the trouble. For {route} complaints, please contact support with transaction/order details. "
        "If this involves payment, we will prioritize verification and refund checks."
    )


def quality_score(complaint: str, response: str) -> float:
    c_words = set(re.findall(r"[a-z]+", complaint.lower()))
    r_words = set(re.findall(r"[a-z]+", response.lower()))
    overlap = len(c_words & r_words) / max(len(c_words), 1)
    empathy = float(any(tok in r_words for tok in EMPATHY_TOKENS))
    action = float(any(tok in r_words for tok in ACTION_TOKENS))
    toxicity_penalty = 1.0 if any(bad in response.lower() for bad in {"fault", "blame", "ignore"}) else 0.0
    return float(0.45 * overlap + 0.25 * empathy + 0.3 * action - 0.4 * toxicity_penalty)


def main() -> None:
    ensure_dirs()
    path = DATA_DIR / "final_reviews_churn.csv"
    df = pd.read_csv(path if path.exists() else DATA_DIR / "final_reviews_routed.csv", encoding="utf-8-sig")
    df["route_category"] = df.get("route_category", df.get("cluster_name", "General")).fillna("General")

    generators = {
        "template_v1": gen_template_v1,
        "template_v2": gen_template_v2,
        "template_v3": gen_template_v3,
    }
    sample = df.sample(min(300, len(df)), random_state=42).copy()
    rows = []
    for name, fn in generators.items():
        scores = []
        for _, r in sample.iterrows():
            resp = fn(str(r["route_category"]), str(r["content"]))
            scores.append(quality_score(str(r["content"]), resp))
        rows.append({"generator": name, "avg_quality_score": float(np.mean(scores)), "p90_quality_score": float(np.percentile(scores, 90))})

    metrics = pd.DataFrame(rows).sort_values("avg_quality_score", ascending=False)
    metrics.to_csv(REPORTS_DIR / "metrics_response.csv", index=False, encoding="utf-8-sig")
    best_name = metrics.iloc[0]["generator"]

    fn = generators[best_name]
    out = df.copy()
    out["suggested_response"] = out.apply(lambda r: fn(str(r["route_category"]), str(r["content"])), axis=1)
    out["response_quality_score"] = out.apply(lambda r: quality_score(str(r["content"]), str(r["suggested_response"])), axis=1)
    out.to_csv(DATA_DIR / "final_reviews_response.csv", index=False, encoding="utf-8-sig")

    (MODELS_DIR / "response_generator.json").write_text(
        json.dumps({"selected_generator": best_name, "generators": list(generators.keys())}, indent=2),
        encoding="utf-8",
    )
    update_selected_model("response_generation", best_name, float(metrics.iloc[0]["avg_quality_score"]), {"metrics_file": "reports/metrics_response.csv"})
    print(f"Response module complete. Selected generator: {best_name}")


if __name__ == "__main__":
    main()
