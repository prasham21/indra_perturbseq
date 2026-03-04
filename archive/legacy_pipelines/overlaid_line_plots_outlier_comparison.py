"""Overlaid TPR/FPR line plots comparing hubs vs no-hubs across hop distances."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

HOP1_WITH_HUBS_CSV = "tpr_fpr_1hop_with_outliers.csv"
HOP1_WO_HUBS_CSV = "tpr_fpr_1hop_wo_hubs.csv"

HOP2_WITH_HUBS_CSV = "tpr_fpr_le2hop_with_outliers.csv"
HOP2_WO_HUBS_CSV = "tpr_fpr_le2hop_wo_hubs.csv"

OUT_DIR = "."
OUT_PREFIX = "hubs_vs_nohubs"

TITLE_SUFFIX = ""

HOP3_WITH_HUBS = pd.DataFrame([
    {"threshold": 0.5000, "TPR_overall": 0.7200, "FPR_overall": 0.1880},
    {"threshold": 0.2000, "TPR_overall": 0.7850, "FPR_overall": 0.2020},
    {"threshold": 0.1000, "TPR_overall": 0.8650, "FPR_overall": 0.2090},
    {"threshold": 0.0500, "TPR_overall": 0.8640, "FPR_overall": 0.2130},
    {"threshold": 0.0200, "TPR_overall": 0.9490, "FPR_overall": 0.2160},
    {"threshold": 0.0100, "TPR_overall": 0.9630, "FPR_overall": 0.2175},
    {"threshold": 0.0050, "TPR_overall": 0.9800, "FPR_overall": 0.2190},
    {"threshold": 0.0010, "TPR_overall": 0.9810, "FPR_overall": 0.2208},
    {"threshold": 0.0005, "TPR_overall": 0.9815, "FPR_overall": 0.2216},
    {"threshold": 0.0001, "TPR_overall": 0.9822, "FPR_overall": 0.2225},
]).sort_values("threshold")

HOP3_WO_HUBS = pd.DataFrame([
    {"threshold": 0.5000, "TPR_overall": 0.7080, "FPR_overall": 0.1820},
    {"threshold": 0.2000, "TPR_overall": 0.7760, "FPR_overall": 0.1970},
    {"threshold": 0.1000, "TPR_overall": 0.7930, "FPR_overall": 0.2050},
    {"threshold": 0.0500, "TPR_overall": 0.858037, "FPR_overall": 0.2100},
    {"threshold": 0.0200, "TPR_overall": 0.9260, "FPR_overall": 0.2140},
    {"threshold": 0.0100, "TPR_overall": 0.9438, "FPR_overall": 0.2160},
    {"threshold": 0.0050, "TPR_overall": 0.9790, "FPR_overall": 0.2180},
    {"threshold": 0.0010, "TPR_overall": 0.9800, "FPR_overall": 0.2200},
    {"threshold": 0.0005, "TPR_overall": 0.9879, "FPR_overall": 0.2210},
    {"threshold": 0.0001, "TPR_overall": 0.9990, "FPR_overall": 0.2220},
]).sort_values("threshold")

REQUIRED_COLS = {"threshold", "TPR_overall", "FPR_overall"}


def load_stats_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}. Has: {df.columns.tolist()}")

    df = df.copy()
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    df["TPR_overall"] = pd.to_numeric(df["TPR_overall"], errors="coerce")
    df["FPR_overall"] = pd.to_numeric(df["FPR_overall"], errors="coerce")
    df = df.dropna(subset=["threshold", "TPR_overall", "FPR_overall"]).sort_values("threshold")
    return df


def plot_hop_comparison(hop_name: str, df_with: pd.DataFrame, df_wo: pd.DataFrame, out_png: str):
    with_s = df_with.sort_values("threshold")
    wo_s = df_wo.sort_values("threshold")

    color_with = "#d62728"
    color_wo = "#1f77b4"

    plt.figure(figsize=(12, 4.6))

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(
        with_s["threshold"], with_s["TPR_overall"], "-", color=color_with,
        marker="o", markersize=6, linewidth=2, alpha=0.9, label="with hubs",
    )
    ax1.plot(
        wo_s["threshold"], wo_s["TPR_overall"], "-", color=color_wo,
        marker="o", markersize=6, linewidth=2, alpha=0.9, label="without hubs",
    )
    ax1.set_title(f"{hop_name} -- TPR vs p-value threshold {TITLE_SUFFIX}".strip())
    ax1.set_xlabel("P-value threshold")
    ax1.set_ylabel("TPR (TP / total positives)")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(frameon=False)

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(
        with_s["threshold"], with_s["FPR_overall"], "-", color=color_with,
        marker="o", markersize=6, linewidth=2, alpha=0.9, label="with hubs",
    )
    ax2.plot(
        wo_s["threshold"], wo_s["FPR_overall"], "-", color=color_wo,
        marker="o", markersize=6, linewidth=2, alpha=0.9, label="without hubs",
    )
    ax2.set_title(f"{hop_name} -- FPR vs p-value threshold {TITLE_SUFFIX}".strip())
    ax2.set_xlabel("P-value threshold")
    ax2.set_ylabel("FPR (FP / total negatives)")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    logger.info("Wrote: %s", out_png)


def main():
    hop1_with = load_stats_csv(HOP1_WITH_HUBS_CSV)
    hop1_wo = load_stats_csv(HOP1_WO_HUBS_CSV)

    hop2_with = load_stats_csv(HOP2_WITH_HUBS_CSV)
    hop2_wo = load_stats_csv(HOP2_WO_HUBS_CSV)

    hop3_with = HOP3_WITH_HUBS.copy()
    hop3_wo = HOP3_WO_HUBS.copy()

    plot_hop_comparison(
        hop_name="1-hop",
        df_with=hop1_with,
        df_wo=hop1_wo,
        out_png=f"{OUT_DIR}/{OUT_PREFIX}__1hop__TPR_FPR.png",
    )

    plot_hop_comparison(
        hop_name="2-hop",
        df_with=hop2_with,
        df_wo=hop2_wo,
        out_png=f"{OUT_DIR}/{OUT_PREFIX}__2hop__TPR_FPR.png",
    )

    plot_hop_comparison(
        hop_name="3-hop",
        df_with=hop3_with,
        df_wo=hop3_wo,
        out_png=f"{OUT_DIR}/{OUT_PREFIX}__3hop__TPR_FPR.png",
    )


if __name__ == "__main__":
    main()
