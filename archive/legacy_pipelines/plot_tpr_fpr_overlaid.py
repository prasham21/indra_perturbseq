"""Overlaid TPR/FPR plots across 1-hop, <=2-hop, and 3-hop with separate FPR panels."""
from __future__ import annotations

import argparse

import logging

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

LE2HOP_CSV = "tpr_fpr_le2hop_with_outliers.csv"
TITLE_SUFFIX = "(with outliers)"

HOP3 = pd.DataFrame([
    {"threshold": 0.5, "TPR_overall": 0.6500, "FPR_overall": 0.1900},
    {"threshold": 0.2, "TPR_overall": 0.7000, "FPR_overall": 0.2050},
    {"threshold": 0.1, "TPR_overall": 0.7400, "FPR_overall": 0.2120},
    {"threshold": 0.05, "TPR_overall": 0.8000, "FPR_overall": 0.2180},
    {"threshold": 0.02, "TPR_overall": 0.8300, "FPR_overall": 0.2230},
    {"threshold": 0.01, "TPR_overall": 0.8600, "FPR_overall": 0.2260},
    {"threshold": 0.005, "TPR_overall": 0.8800, "FPR_overall": 0.2280},
    {"threshold": 0.001, "TPR_overall": 0.9200, "FPR_overall": 0.2310},
    {"threshold": 0.0005, "TPR_overall": 0.9300, "FPR_overall": 0.2320},
    {"threshold": 0.0001, "TPR_overall": 0.9500, "FPR_overall": 0.2330},
]).sort_values("threshold")


def _load_stats_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    needed = {"threshold", "TPR_overall", "FPR_overall"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}. Has: {df.columns.tolist()}")

    df = df.copy()
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    df["TPR_overall"] = pd.to_numeric(df["TPR_overall"], errors="coerce")
    df["FPR_overall"] = pd.to_numeric(df["FPR_overall"], errors="coerce")
    df = df.dropna(subset=["threshold", "TPR_overall", "FPR_overall"])
    return df.sort_values("threshold")


def _plot_overlay(dfs, metric_col, ylabel, title, out_png):
    plt.figure(figsize=(7.5, 4.8))

    for label, df, style, color in dfs:
        plt.plot(
            df["threshold"].values,
            df[metric_col].values,
            style,
            color=color,
            marker="o",
            markersize=6,
            linewidth=2,
            alpha=0.9,
            label=label,
        )

    plt.xlabel("P-value threshold")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    logger.info("Wrote: %s", out_png)


def _plot_single(df, metric_col, ylabel, title, out_png, style, color):
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(
        df["threshold"].values,
        df[metric_col].values,
        style,
        color=color,
        marker="o",
        markersize=6,
        linewidth=2,
        alpha=0.9,
    )
    plt.xlabel("P-value threshold")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    logger.info("Wrote: %s", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onehop-csv", default="tpr_fpr_1hop_with_outliers.csv", help="Path for args.onehop_csv.")
    ap.add_argument("--out-prefix", default="outliers_overlay", help="Path for args.out_prefix.")
    args = ap.parse_args()

    one = _load_stats_csv(args.onehop_csv)
    le2 = _load_stats_csv(LE2HOP_CSV)

    dfs_tpr = [
        ("1-hop", one, "-", "#1f77b4"),
        ("<=2-hop (1+2 combined)", le2, "-", "#ff7f0e"),
        ("3-hop", HOP3, "--", "#2ca02c"),
    ]
    _plot_overlay(
        dfs=dfs_tpr,
        metric_col="TPR_overall",
        ylabel="TPR (TP / total positives)",
        title=f"TPR vs p-value threshold {TITLE_SUFFIX}",
        out_png=f"{args.out_prefix}__TPR.png",
    )

    _plot_single(
        df=one,
        metric_col="FPR_overall",
        ylabel="FPR (FP / total negatives)",
        title=f"FPR vs p-value threshold {TITLE_SUFFIX} -- 1-hop",
        out_png=f"{args.out_prefix}__FPR__1hop.png",
        style="-",
        color="#1f77b4",
    )

    _plot_single(
        df=le2,
        metric_col="FPR_overall",
        ylabel="FPR (FP / total negatives)",
        title=f"FPR vs p-value threshold {TITLE_SUFFIX} -- <=2-hop (1+2 combined)",
        out_png=f"{args.out_prefix}__FPR__le2hop.png",
        style="-",
        color="#ff7f0e",
    )

    _plot_single(
        df=HOP3,
        metric_col="FPR_overall",
        ylabel="FPR (FP / total negatives)",
        title=f"FPR vs p-value threshold {TITLE_SUFFIX} -- 3-hop",
        out_png=f"{args.out_prefix}__FPR__3hop.png",
        style="--",
        color="#2ca02c",
    )


if __name__ == "__main__":
    main()
