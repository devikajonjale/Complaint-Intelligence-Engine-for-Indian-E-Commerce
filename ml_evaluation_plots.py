"""Seaborn/matplotlib figures for ML benchmarking (darkgrid, titled, legend)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


def _apply_style():
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="darkgrid")
    plt.rcParams["figure.figsize"] = (10, 5.5)
    plt.rcParams["axes.titlesize"] = 13


def save_model_comparison_bars(
    metrics_df: pd.DataFrame,
    metric_cols: list[str],
    out_path: Path,
    title: str,
    hue_legend_title: str = "Metric",
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _apply_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = metrics_df.copy()
    if "model" not in df.columns:
        return
    melt = df.melt(id_vars=["model"], value_vars=[c for c in metric_cols if c in df.columns], var_name="metric", value_name="value")
    plt.figure()
    ax = sns.barplot(data=melt, x="model", y="value", hue="metric", palette="Set2")
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    leg = ax.legend(title=hue_legend_title)
    if leg:
        leg.get_title().set_text(hue_legend_title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_heatmap(
    cm: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _apply_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="mako",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Count"},
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_binary_metrics_bars(metrics_df: pd.DataFrame, out_path: Path, title: str) -> None:
    preferred = [
        "f1",
        "precision",
        "recall",
        "balanced_accuracy",
        "pr_auc",
        "roc_auc",
        "specificity",
        "brier_score",
        "cv_pr_auc_mean",
        "cv_roc_auc_mean",
    ]
    cols = [c for c in preferred if c in metrics_df.columns]
    if not cols:
        return
    save_model_comparison_bars(metrics_df, cols, out_path, title, hue_legend_title="Metric")


def save_pr_curve_plot(y_true: np.ndarray, y_score: np.ndarray, out_path: Path, title: str) -> None:
    from sklearn.metrics import precision_recall_curve

    import matplotlib.pyplot as plt
    import seaborn as sns

    _apply_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label="PR curve", color="coral", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_lineplot_drift(drift_df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _apply_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if drift_df.empty or "week" not in drift_df.columns:
        return
    d = drift_df.copy()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(title, fontsize=14)
    if "centroid_shift" in d.columns:
        sns.lineplot(data=d, x="week", y="centroid_shift", marker="o", ax=axes[0], color="steelblue", label="Centroid shift")
        axes[0].set_ylabel("Centroid shift (L2)")
        axes[0].legend(title="Metric", loc="upper right")
    if "js_divergence" in d.columns:
        sns.lineplot(data=d, x="week", y="js_divergence", marker="s", ax=axes[1], color="darkorange", label="JS divergence")
        axes[1].set_ylabel("Jensen–Shannon divergence")
        axes[1].legend(title="Metric", loc="upper right")
    axes[-1].set_xlabel("Week")
    for ax in axes:
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_generator_quality_charts(long_df: pd.DataFrame, summary_df: pd.DataFrame, out_bar: Path, out_box: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _apply_style()
    out_bar = Path(out_bar)
    out_box = Path(out_box)
    out_bar.parent.mkdir(parents=True, exist_ok=True)
    if not summary_df.empty and "generator" in summary_df.columns and "avg_quality_score" in summary_df.columns:
        plt.figure(figsize=(8, 4))
        sns.barplot(data=summary_df, y="generator", x="avg_quality_score", hue="generator", palette="crest", dodge=False, legend=True)
        plt.title("Auto-response generators — mean quality score")
        plt.xlabel("Average quality score (numeric)")
        plt.ylabel("Generator (categorical)")
        leg = plt.legend(title="Generator", bbox_to_anchor=(1.02, 1), loc="upper left")
        if leg:
            leg.set_title("Generator")
        plt.tight_layout()
        plt.savefig(out_bar, dpi=150, bbox_inches="tight")
        plt.close()
    if not long_df.empty and {"generator", "quality_score"}.issubset(long_df.columns):
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=long_df, x="generator", y="quality_score", hue="generator", palette="flare", dodge=False, legend=True)
        plt.title("Per-response quality score distribution by generator")
        plt.xlabel("Generator (categorical)")
        plt.ylabel("Quality score (numeric)")
        plt.legend(title="Generator", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(out_box, dpi=150, bbox_inches="tight")
        plt.close()


def save_boxplot_by_category(
    df: pd.DataFrame,
    x: str,
    y: str,
    out_path: Path,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _apply_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if x not in df.columns or y not in df.columns:
        return
    sub = df[[x, y]].dropna()
    if sub.empty:
        return
    plt.figure(figsize=(11, 5.5))
    sns.boxplot(data=sub, x=x, y=y, hue=x, palette="pastel", dodge=False, legend=False)
    plt.title(title)
    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel or y)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
