#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# EDIT THESE PATHS
# =========================
ONEHOP_CSV = "/Users/prashammarfatia/Downloads/tpr_fpr_1hop_with_outliers.csv"
LE2HOP_CSV = "/Users/prashammarfatia/Downloads/tpr_fpr_le2hop_with_outliers.csv"
OUT_PREFIX = "/Users/prashammarfatia/Downloads/outliers_overlay"
TITLE_SUFFIX = "(with outliers)"

# =========================
# 3-hop values (from your chat table; labeled as "3-hop")
# =========================
HOP3 = pd.DataFrame([
    {"threshold": 0.5,    "TPR_overall": 0.6500, "FPR_overall": 0.1900},
    {"threshold": 0.2,    "TPR_overall": 0.7000, "FPR_overall": 0.2050},
    {"threshold": 0.1,    "TPR_overall": 0.7400, "FPR_overall": 0.2120},
    {"threshold": 0.05,   "TPR_overall": 0.8000, "FPR_overall": 0.2180},
    {"threshold": 0.02,   "TPR_overall": 0.8300, "FPR_overall": 0.2230},
    {"threshold": 0.01,   "TPR_overall": 0.8600, "FPR_overall": 0.2260},
    {"threshold": 0.005,  "TPR_overall": 0.8800, "FPR_overall": 0.2280},
    {"threshold": 0.001,  "TPR_overall": 0.9200, "FPR_overall": 0.2310},
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
    print(f"Wrote: {out_png}")


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
    print(f"Wrote: {out_png}")


def main():
    one = _load_stats_csv(ONEHOP_CSV)
    le2 = _load_stats_csv(LE2HOP_CSV)

    # --- TPR: keep overlaid (manager only asked to split FPR) ---
    dfs_tpr = [
        ("1-hop", one, "-", "#1f77b4"),                 # blue
        ("≤2-hop (1+2 combined)", le2, "-", "#ff7f0e"), # orange
        ("3-hop", HOP3, "--", "#2ca02c"),               # green dashed
    ]
    _plot_overlay(
        dfs=dfs_tpr,
        metric_col="TPR_overall",
        ylabel="TPR (TP / total positives)",
        title=f"TPR vs p-value threshold {TITLE_SUFFIX}",
        out_png=f"{OUT_PREFIX}__TPR.png",
    )

    # --- FPR: separate plots per hop (to avoid squished overlay) ---
    _plot_single(
        df=one,
        metric_col="FPR_overall",
        ylabel="FPR (FP / total negatives)",
        title=f"FPR vs p-value threshold {TITLE_SUFFIX} — 1-hop",
        out_png=f"{OUT_PREFIX}__FPR__1hop.png",
        style="-",
        color="#1f77b4",
    )

    _plot_single(
        df=le2,
        metric_col="FPR_overall",
        ylabel="FPR (FP / total negatives)",
        title=f"FPR vs p-value threshold {TITLE_SUFFIX} — ≤2-hop (1+2 combined)",
        out_png=f"{OUT_PREFIX}__FPR__le2hop.png",
        style="-",
        color="#ff7f0e",
    )

    _plot_single(
        df=HOP3,
        metric_col="FPR_overall",
        ylabel="FPR (FP / total negatives)",
        title=f"FPR vs p-value threshold {TITLE_SUFFIX} — 3-hop",
        out_png=f"{OUT_PREFIX}__FPR__3hop.png",
        style="--",
        color="#2ca02c",
    )


if __name__ == "__main__":
    main()