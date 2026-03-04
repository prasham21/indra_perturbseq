#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop1",   required=True, help="indra_1hop_NETWORK_EXPORT_main.csv")
    ap.add_argument("--hop2",   required=True, help="2hop_network_export_main.csv")
    ap.add_argument("--hop3",   required=True, help="3hop_network_export_raw.csv")
    ap.add_argument("--tv",     required=True, help="target_validation_expanded.csv")
    ap.add_argument("--de-dir", required=True, help="de_results_per_gene directory")
    ap.add_argument("--out",    required=True, help="output PNG path")
    ap.add_argument("--trim-top-pct", type=float, default=10.0,
                    help="Remove top N%% (smallest pvals) per group before plotting (default: 10)")
    args = ap.parse_args()

    # ── Load hop pairs ────────────────────────────────────────────────────
    def load_pairs(path):
        d = pd.read_csv(path, usecols=["source", "target"]).astype(str)
        d["source"] = d["source"].str.strip().str.upper()
        d["target"] = d["target"].str.strip().str.upper()
        return set(map(tuple, d.values))

    h1 = load_pairs(args.hop1)
    h2 = load_pairs(args.hop2)
    h3 = load_pairs(args.hop3)
    h2_only = h2 - h1
    h3_only = h3 - h1 - h2

    # ── Karen-filtered sources ────────────────────────────────────────────
    tv = pd.read_csv(args.tv, usecols=["Gene", "Karen_Flag"])
    sources = set(tv.loc[tv["Karen_Flag"] == "Use_for_analysis", "Gene"].astype(str).str.strip())
    sources -= {"TP53", "CDKN1A"}

    # ── DEG pairs (pvals < 0.05) ──────────────────────────────────────────
    rows = []
    for s in sorted(sources):
        f = os.path.join(args.de_dir, f"{s}_vs_control.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f, usecols=["names", "logfoldchanges", "pvals"]).dropna()
        d = d[d["pvals"] < 0.05]
        for _, r in d.iterrows():
            rows.append([s.upper(), str(r["names"]).strip().upper(),
                         float(r["logfoldchanges"]), float(r["pvals"])])

    df = pd.DataFrame(rows, columns=["source", "target", "logfc", "pval"])
    df = df.loc[df.groupby(["source", "target"])["pval"].idxmin()].copy()
    df["neglogp"] = -np.log10(df["pval"].clip(lower=1e-300))

    # ── Hop labels ────────────────────────────────────────────────────────
    def hop_label(r):
        p = (r.source, r.target)
        if p in h1:       return "1-hop"
        if p in h2_only:  return "2-hop"
        if p in h3_only:  return "3-hop"
        return "3+"

    df["hop"] = df.apply(hop_label, axis=1)

    # ── Remove top N% per group (except 1-hop) ────────────────────────────
    trim_pct = args.trim_top_pct
    cutoff_quantile = 1.0 - trim_pct / 100.0

    groups_to_trim = ["2-hop", "3-hop", "3+"]
    trimmed_parts = []

    hop1_df = df[df["hop"] == "1-hop"].copy()   # 1-hop untouched
    trimmed_parts.append(hop1_df)

    for grp in groups_to_trim:
        sub = df[df["hop"] == grp].copy()
        if len(sub) == 0:
            continue
        threshold = sub["neglogp"].quantile(cutoff_quantile)
        before = len(sub)
        sub = sub[sub["neglogp"] <= threshold]
        after = len(sub)
        print(f"{grp}: removed {before - after} / {before} rows (top {trim_pct}% pval outliers)")
        trimmed_parts.append(sub)

    df_plot = pd.concat(trimmed_parts, ignore_index=True)

    # ── Plot setup ────────────────────────────────────────────────────────
    order  = ["1-hop", "2-hop", "3-hop", "3+"]
    colors = {"1-hop": "#2563EB", "2-hop": "#059669", "3-hop": "#D97706", "3+": "#DC2626"}

    x, y = df_plot["logfc"].values, df_plot["neglogp"].values
    xmin, xmax = np.percentile(x, [0.5, 99.5])
    ymax = np.percentile(y, 99.5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, hop in enumerate(order):
        ax = axes[i]
        hb = ax.hexbin(x, y, gridsize=50, cmap="YlOrRd", mincnt=1, alpha=0.85)
        d = df_plot[df_plot["hop"] == hop]
        ax.scatter(
            d["logfc"], d["neglogp"], s=20, c=colors[hop], alpha=0.8,
            edgecolors="black", linewidths=0.3, label=f"{hop} (n={len(d):,})"
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Log Fold Change")
        ax.set_ylabel("-log10(P-value)")
        ax.set_title(f"Highlighting {hop} (top {trim_pct}% outliers removed from 2-hop/3-hop/3+)")
        ax.axvline(0, ls="--", lw=1, c="gray")
        ax.axhline(-np.log10(0.05), ls="--", lw=1, c="gray")
        ax.legend(loc="upper right", fontsize=8)
        plt.colorbar(hb, ax=ax, label="Count")

    plt.suptitle(f"FC vs P-value landscape (top {trim_pct}% outliers removed from 2/3-hop/3+)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()