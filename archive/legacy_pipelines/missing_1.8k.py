"""P-value boxplots by pathway type (1-hop, 2-hop, 3-hop, unexplained)."""
from __future__ import annotations

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

def make_group(df, label):
    p = pd.to_numeric(df["pval"], errors="coerce").dropna()
    return pd.DataFrame({"pval": p, "pathway_type": label})




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indra-1hop-network-export-main", default="indra_1hop_NETWORK_EXPORT_main.csv", help="Path: indra_1hop_NETWORK_EXPORT_main.csv")
    ap.add_argument("--2hop-network-export-main", default="2hop_network_export_main.csv", help="Path: 2hop_network_export_main.csv")
    ap.add_argument("--3hop-network-export-raw", default="3hop_network_export_raw.csv", help="Path: 3hop_network_export_raw.csv")
    ap.add_argument("--target-validation-expanded", default="target_validation_expanded.csv", help="Path: target_validation_expanded.csv")
    ap.add_argument("--de-results-per-gene", default="de_results_per_gene", help="Path: de_results_per_gene")
    args = ap.parse_args()

    HOP1 = "indra_1hop_NETWORK_EXPORT_main.csv"
    HOP2 = "2hop_network_export_main.csv"
    HOP3 = "3hop_network_export_raw.csv"
    TV = "target_validation_expanded.csv"
    DE_DIR = "de_results_per_gene"

    OUT_WITH_FLIERS = "pvalue_boxplot_network_export_with_outliers.png"
    OUT_WITHOUT_FLIERS = "pvalue_boxplot_network_export_no_outliers.png"

    PVAL_CUTOFF = 0.05

    hop1 = pd.read_csv(HOP1)
    hop2 = pd.read_csv(HOP2)
    hop3 = pd.read_csv(HOP3, low_memory=False)

    for df in [hop1, hop2, hop3]:
        df["pval"] = pd.to_numeric(df["pval"], errors="coerce")

    hop1 = hop1.sort_values("pval").drop_duplicates(["source", "target"], keep="first")
    hop2 = hop2.sort_values("pval").drop_duplicates(["source", "target"], keep="first")
    hop3 = hop3.sort_values("pval").drop_duplicates(["source", "target"], keep="first")

    pairs1 = set(zip(hop1["source"], hop1["target"]))
    hop2 = hop2[~hop2.apply(lambda r: (r["source"], r["target"]) in pairs1, axis=1)]
    pairs12 = pairs1 | set(zip(hop2["source"], hop2["target"]))
    hop3 = hop3[~hop3.apply(lambda r: (r["source"], r["target"]) in pairs12, axis=1)]

    all_explained = pairs12 | set(zip(hop3["source"], hop3["target"]))

    tv = pd.read_csv(TV, low_memory=False)
    allowed = set(tv[tv["analysis_flag"] == "Use_for_analysis"]["Gene"].astype(str).str.strip())
    pathway_sources = set(hop1["source"]) | set(hop2["source"]) | set(hop3["source"])

    unexp_rows = []
    for src in pathway_sources:
        if src not in allowed:
            continue
        f = os.path.join(DE_DIR, f"{src}_vs_control.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        df["pvals"] = pd.to_numeric(df["pvals"], errors="coerce")
        df = df[df["pvals"] < PVAL_CUTOFF].copy()
        df["source"] = src
        df = df[~df.apply(lambda r: (r["source"], r["names"]) in all_explained, axis=1)]
        unexp_rows.append(df[["pvals"]].rename(columns={"pvals": "pval"}))

    unexp = pd.concat(unexp_rows, ignore_index=True) if unexp_rows else pd.DataFrame(columns=["pval"])


    combined = pd.concat([
        make_group(hop1, "1-hop"),
        make_group(hop2, "2-hop"),
        make_group(hop3, "3-hop"),
        make_group(unexp, "unexplained"),
    ], ignore_index=True)

    combined["neg_log10_pval"] = -np.log10(combined["pval"].clip(lower=1e-50))

    pathway_order = ["1-hop", "2-hop", "3-hop", "unexplained"]
    palette = {"1-hop": "blue", "2-hop": "green", "3-hop": "orange", "unexplained": "red"}
    counts = combined["pathway_type"].value_counts()
    labels = [f"{p}\n(n={counts.get(p, 0):,})" for p in pathway_order]

    for showfliers, outpath in [(True, OUT_WITH_FLIERS), (False, OUT_WITHOUT_FLIERS)]:
        plt.figure(figsize=(10, 7))
        sns.boxplot(
            data=combined,
            x="pathway_type",
            y="neg_log10_pval",
            order=pathway_order,
            palette=palette,
            showfliers=showfliers,
        )
        plt.xticks(ticks=range(len(pathway_order)), labels=labels, fontsize=10)
        plt.axhline(-np.log10(0.05), color="orange", linestyle="--", alpha=0.7, label="p=0.05")
        plt.axhline(-np.log10(0.01), color="red", linestyle="--", alpha=0.7, label="p=0.01")
        plt.xlabel("Pathway Type")
        plt.ylabel("-log10(P-value)")
        title_suffix = "with outliers" if showfliers else "no outliers"
        plt.title(f"P-value Distributions by Pathway Type ({title_suffix})")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved: %s", outpath)



if __name__ == "__main__":
    main()
