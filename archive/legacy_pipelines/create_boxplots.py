#!/usr/bin/env python3

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")



# CONFIGURATION
PATH_1HOP = "/Users/prashammarfatia/Downloads/indra_1hop_no_v2.csv"
PATH_2HOP = "/Users/prashammarfatia/Downloads/indra_2hop_all_perturbations.xlsx"
PATH_3HOP = "/Users/prashammarfatia/Downloads/indra_3hop_optimized_all_perturbations_combined.csv"

PATH_TARGET_VALIDATION = "/Users/prashammarfatia/Downloads/target_validation_expanded.csv"
DE_RESULTS_FOLDER = "/Users/prashammarfatia/Downloads/de_results_per_gene"

BELIEF_CUTOFF = 0.7
DE_PVAL_CUTOFF = 0.05
EXCLUDE_SOURCES = {"TP53", "CDKN1A"}

OUTPUT_PVAL_PLOT = "pvalue_distributions_boxplot.png"
OUTPUT_LFC_PLOT = "logfoldchange_distributions_boxplot.png"
OUTPUT_FINAL_DATA = "final_dataset_for_boxplots.csv"


def load_and_filter_pathway_datasets():
    """Load 1-hop, 2-hop, 3-hop and apply belief cutoff."""
    print("Loading and filtering pathway datasets...")

    # 1-hop
    df_1hop = pd.read_csv(PATH_1HOP)
    df_1hop = df_1hop[df_1hop["belief"] >= BELIEF_CUTOFF].copy()
    df_1hop["pathway_type"] = "1-hop"

    # 2-hop
    df_2hop = pd.read_excel(PATH_2HOP)
    df_2hop["belief"] = (df_2hop["belief_1"] + df_2hop["belief_2"]) / 2
    df_2hop = df_2hop[df_2hop["belief"] >= BELIEF_CUTOFF].copy()
    df_2hop["pathway_type"] = "2-hop"

    # 3-hop
    df_3hop = pd.read_csv(PATH_3HOP)
    df_3hop["belief"] = (df_3hop["belief_1"] + df_3hop["belief_2"] + df_3hop["belief_3"]) / 3
    df_3hop = df_3hop[df_3hop["belief"] >= BELIEF_CUTOFF].copy()
    df_3hop["pathway_type"] = "3-hop"

    print(f"  1-hop (filtered): {len(df_1hop):,}")
    print(f"  2-hop (filtered): {len(df_2hop):,}")
    print(f"  3-hop (filtered): {len(df_3hop):,}")

    return df_1hop, df_2hop, df_3hop


def create_priority_combined_dataset(df_1hop, df_2hop, df_3hop):
    """
    Priority logic:
    1-hop > 2-hop > 3-hop
    tie-break by higher belief
    """
    all_rows = []

    for _, r in df_1hop.iterrows():
        all_rows.append({
            "source": r["source"],
            "target": r["target"],
            "belief": r["belief"],
            "logfoldchange": r["logfoldchange"],
            "pval": r["pval"],
            "pathway_type": "1-hop",
            "priority": 1
        })

    for _, r in df_2hop.iterrows():
        all_rows.append({
            "source": r["source"],
            "target": r["target"],
            "belief": r["belief"],
            "logfoldchange": r["logfoldchange"],
            "pval": r["pval"],
            "pathway_type": "2-hop",
            "priority": 2
        })

    for _, r in df_3hop.iterrows():
        all_rows.append({
            "source": r["source"],
            "target": r["target"],
            "belief": r["belief"],
            "logfoldchange": r["logfoldchange"],
            "pval": r["pval"],
            "pathway_type": "3-hop",
            "priority": 3
        })

    combined = pd.DataFrame(all_rows)

    def select_best(group):
        return group.sort_values(["priority", "belief"], ascending=[True, False]).iloc[0]

    best = combined.groupby(["source", "target"], as_index=False, group_keys=False).apply(select_best).reset_index(drop=True)

    print(f"\nAfter priority logic: {len(best):,} unique source-target pairs")
    print(best["pathway_type"].value_counts().to_string())

    return best


def add_unexplained_targets(combined_df):
    """
    Add unexplained = significant DE targets not already explained by 1/2/3-hop.
    """
    print("\nAdding unexplained targets...")

    tv = pd.read_csv(PATH_TARGET_VALIDATION)
    allowed_sources = set(
        tv.query("Karen_Flag == 'Use_for_analysis'")["Gene"]
        .astype(str).str.strip().tolist()
    )
    allowed_sources = {g for g in allowed_sources if g not in EXCLUDE_SOURCES}

    explained_pairs = set(zip(combined_df["source"], combined_df["target"]))
    pathway_sources = combined_df["source"].dropna().unique()

    unexplained_rows = []

    for source in pathway_sources:
        if source not in allowed_sources:
            continue

        deg_file = Path(DE_RESULTS_FOLDER) / f"{source}_vs_control.csv"
        if not deg_file.exists():
            continue

        try:
            deg_df = pd.read_csv(deg_file)
        except Exception:
            continue

        required_cols = {"names", "pvals", "logfoldchanges"}
        if not required_cols.issubset(deg_df.columns):
            continue

        sig = deg_df[deg_df["pvals"] < DE_PVAL_CUTOFF].copy()

        for _, row in sig.iterrows():
            target = row["names"]

            if (source, target) in explained_pairs:
                continue

            unexplained_rows.append({
                "source": source,
                "target": target,
                "belief": 0.5,
                "logfoldchange": row["logfoldchanges"],
                "pval": row["pvals"],
                "pathway_type": "unexplained",
                "priority": 4,
                "explained": False
            })

    explained_df = combined_df.copy()
    explained_df["explained"] = True

    if unexplained_rows:
        unexplained_df = pd.DataFrame(unexplained_rows)
        final_df = pd.concat([explained_df, unexplained_df], ignore_index=True)
    else:
        final_df = explained_df

    print(f"Final dataset size: {len(final_df):,}")
    print(final_df["pathway_type"].value_counts().to_string())

    return final_df


def _common_plot_setup():
    pathway_order = ["1-hop", "2-hop", "3-hop", "unexplained"]
    palette = {
        "1-hop": "blue",
        "2-hop": "green",
        "3-hop": "orange",
        "unexplained": "red"
    }
    return pathway_order, palette


def plot_pvalue_boxplot(final_df):
    """
    Plot boxplot of -log10(p-value) by pathway type.
    """
    pathway_order, palette = _common_plot_setup()

    df_plot = final_df.copy()
    df_plot["neg_log10_pval"] = -np.log10(df_plot["pval"].clip(lower=1e-50))

    box_df = df_plot[df_plot["pathway_type"].isin(pathway_order)][["pathway_type", "neg_log10_pval"]].dropna().copy()
    counts = box_df.groupby("pathway_type").size().reindex(pathway_order, fill_value=0)

    plt.figure(figsize=(10, 7))

    if len(box_df) == 0:
        plt.text(0.5, 0.5, "No p-value data available", ha="center", va="center", transform=plt.gca().transAxes)
    else:
        sns.boxplot(
            data=box_df,
            x="pathway_type",
            y="neg_log10_pval",
            order=pathway_order,
            palette=palette,
            showfliers=True
        )

    plt.xticks(
        ticks=range(len(pathway_order)),
        labels=[f"{p}\n(n={counts[p]:,})" for p in pathway_order],
        fontsize=10
    )
    plt.axhline(y=-np.log10(0.05), color="orange", linestyle="--", alpha=0.7, label="p=0.05")
    plt.axhline(y=-np.log10(0.01), color="red", linestyle="--", alpha=0.7, label="p=0.01")
    plt.xlabel("Pathway Type")
    plt.ylabel("-log10(P-value)")
    plt.title("P-value Distributions by Pathway Type (Box Plot)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_PVAL_PLOT, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {OUTPUT_PVAL_PLOT}")


def plot_logfoldchange_boxplot(final_df):
    """
    Plot boxplot of logfoldchange by pathway type.
    """
    pathway_order, palette = _common_plot_setup()

    box_df = final_df[final_df["pathway_type"].isin(pathway_order)][["pathway_type", "logfoldchange"]].dropna().copy()
    counts = box_df.groupby("pathway_type").size().reindex(pathway_order, fill_value=0)

    plt.figure(figsize=(10, 7))

    if len(box_df) == 0:
        plt.text(0.5, 0.5, "No logfoldchange data available", ha="center", va="center", transform=plt.gca().transAxes)
    else:
        sns.boxplot(
            data=box_df,
            x="pathway_type",
            y="logfoldchange",
            order=pathway_order,
            palette=palette,
            showfliers=True
        )

    plt.xticks(
        ticks=range(len(pathway_order)),
        labels=[f"{p}\n(n={counts[p]:,})" for p in pathway_order],
        fontsize=10
    )
    plt.axhline(y=0.0, color="black", linestyle="--", alpha=0.7, label="logFC=0")
    plt.xlabel("Pathway Type")
    plt.ylabel("Log Fold Change")
    plt.title("Log Fold Change Distributions by Pathway Type (Box Plot)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_LFC_PLOT, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {OUTPUT_LFC_PLOT}")


def main():
    df_1hop, df_2hop, df_3hop = load_and_filter_pathway_datasets()
    combined_df = create_priority_combined_dataset(df_1hop, df_2hop, df_3hop)
    final_df = add_unexplained_targets(combined_df)

    final_df.to_csv(OUTPUT_FINAL_DATA, index=False)
    print(f"Saved final data: {OUTPUT_FINAL_DATA}")

    # Two total output plots
    plot_pvalue_boxplot(final_df)
    plot_logfoldchange_boxplot(final_df)

    print("\nDone. Generated 2 plots:")
    print(f"  1) {OUTPUT_PVAL_PLOT}")
    print(f"  2) {OUTPUT_LFC_PLOT}")


if __name__ == "__main__":
    main()