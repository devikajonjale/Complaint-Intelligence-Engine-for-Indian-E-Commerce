"""
Optional visualization artifact generator.
Creates saved PNG charts for reports/presentations.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def main() -> None:
    final_path = DATA_DIR / "final_reviews.csv"
    ksel_path = DATA_DIR / "k_selection.csv"
    if not final_path.exists():
        raise FileNotFoundError("Run pipeline first to generate final_reviews.csv")

    df = pd.read_csv(final_path, encoding="utf-8-sig")
    sns.set_theme(style="whitegrid")

    if ksel_path.exists():
        k_df = pd.read_csv(ksel_path)
        plt.figure(figsize=(8, 4))
        plt.plot(k_df["k"], k_df["silhouette"], marker="o")
        plt.title("K Selection via Silhouette")
        plt.xlabel("k")
        plt.ylabel("Silhouette")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "k_selection_silhouette.png", dpi=150)
        plt.close()

    plt.figure(figsize=(10, 5))
    order = df["platform"].value_counts().index
    sns.countplot(data=df, x="platform", order=order)
    plt.xticks(rotation=20)
    plt.title("Review Volume by Platform")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "platform_volume.png", dpi=150)
    plt.close()

    if "tier1_critical" in df.columns:
        crit = (
            df.groupby("platform", as_index=False)["tier1_critical"]
            .sum()
            .sort_values("tier1_critical", ascending=False)
        )
        plt.figure(figsize=(10, 5))
        sns.barplot(data=crit, x="platform", y="tier1_critical")
        plt.xticks(rotation=20)
        plt.title("Tier 1 Critical Alerts by Platform")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "tier1_by_platform.png", dpi=150)
        plt.close()

    log.info(f"Saved visualizations under {FIG_DIR}")


if __name__ == "__main__":
    main()
