"""2-hop pathway coverage analysis: per-perturbation explained descendant statistics."""
from __future__ import annotations

import argparse
import logging
import os
import time
import warnings
from typing import Dict, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
plt.style.use("default")

logger = logging.getLogger(__name__)


def read_pathways_xlsx(xlsx_path: str) -> pd.DataFrame:
    """Read pathway Excel with first-row headers. Return unique ['source','target'] pairs."""
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    missing = [c for c in ["source", "target"] if c not in df.columns]
    if missing:
        raise ValueError(f"Pathway file missing required columns: {missing}")
    pairs = df[["source", "target"]].dropna().drop_duplicates().reset_index(drop=True)
    return pairs


def read_valid_genes(tv_csv: str) -> pd.Series:
    """Return Series of genes with analysis_flag == 'Use_for_analysis'."""
    tv = pd.read_csv(tv_csv)
    if "analysis_flag" not in tv.columns or "Gene" not in tv.columns:
        raise ValueError("target_validation_expanded.csv must have columns 'Gene' and 'analysis_flag'.")
    genes = tv.loc[tv["analysis_flag"] == "Use_for_analysis", "Gene"].dropna().drop_duplicates()
    if genes.empty:
        raise ValueError("No genes with analysis_flag == 'Use_for_analysis'.")
    return genes


def read_deg_significant(csv_path: str) -> pd.DataFrame:
    """Read a single DEG CSV and return significant rows (FDR or raw p < 0.05)."""
    df = pd.read_csv(csv_path, low_memory=False)

    name_col = next((c for c in ["names", "gene", "gene_names", "target"] if c in df.columns), None)
    p_col = next((c for c in ["pvals", "pval", "p_value", "p_val"] if c in df.columns), None)
    fdr_col = next((c for c in ["pvals_adj", "padj", "qval", "fdr", "p_adj"] if c in df.columns), None)

    if name_col is None:
        raise ValueError("DEG file missing a gene/target name column (e.g., 'names').")
    if p_col is None and fdr_col is None:
        raise ValueError("DEG file missing both raw p-value and FDR columns.")

    if fdr_col is not None:
        sig = df[(df[fdr_col].astype(float) < 0.05)]
        sig = sig[[name_col]].dropna().drop_duplicates()
    else:
        sig = df[(df[p_col].astype(float) < 0.05)]
        sig = sig[[name_col]].dropna().drop_duplicates()

    sig.columns = ["target"]
    return sig


def load_all_descendants(valid_genes: pd.Series, deg_dir: str) -> Tuple[Dict[str, Set[str]], Set[Tuple[str, str]]]:
    """Load significant descendants for each valid gene."""
    per_gene: Dict[str, Set[str]] = {}
    global_pairs: Set[Tuple[str, str]] = set()

    logger.info("Loading significant descendants from: %s", deg_dir)
    for i, gene in enumerate(valid_genes, start=1):
        csv_path = os.path.join(deg_dir, f"{gene}_vs_control.csv")
        try:
            sig = read_deg_significant(csv_path)
            targets = set(sig["target"].astype(str))
            per_gene[gene] = targets
            for t in targets:
                global_pairs.add((gene, t))
            if i % 25 == 0:
                logger.info("  processed %4d genes", i)
        except FileNotFoundError:
            per_gene[gene] = set()
        except Exception as e:
            logger.warning("  %s: %s", gene, e)
            per_gene[gene] = set()

    logger.info("Collected %s unique (source,target) descendant pairs", f"{len(global_pairs):,}")
    return per_gene, global_pairs


def compute_coverage_2hop(pairs_pathway: pd.DataFrame,
                          per_gene_desc: Dict[str, Set[str]]) -> pd.DataFrame:
    """Compute per-perturbation coverage (percent explained) for 2-hop."""
    by_src = (pairs_pathway
              .dropna()
              .groupby("source")["target"]
              .apply(lambda s: set(s.astype(str)))
              .to_dict())

    rows = []
    for gene, descendants in per_gene_desc.items():
        total = len(descendants)
        if total == 0:
            explained = 0
            pct = np.nan
        else:
            pathway_targets = by_src.get(gene, set())
            explained = len(descendants & pathway_targets)
            pct = explained / total * 100.0
        rows.append({
            "gene": gene,
            "deg_count": total,
            "explained_unique_targets": explained,
            "percent_explained": pct
        })

    df_cov = pd.DataFrame(rows).sort_values("percent_explained", ascending=False)
    return df_cov


def compute_overall_coverage(pairs_pathway: pd.DataFrame,
                             global_desc_pairs: Set[Tuple[str, str]]) -> float:
    """Overall coverage across all perturbations."""
    explained_pairs = set(map(tuple, pairs_pathway[["source", "target"]].astype(str).itertuples(index=False)))
    denom = len(global_desc_pairs)
    if denom == 0:
        return float("nan")
    return len(global_desc_pairs & explained_pairs) / denom * 100.0


def hist_with_annotations(ax, data, bins, title, xlabel, annotate_mean=True):
    """Draw a histogram with mean line and count labels on bars."""
    n, bins_out, patches = ax.hist(data, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)
    if annotate_mean and len(data) > 0 and np.isfinite(np.nanmean(data)):
        ax.axvline(np.nanmean(data), color='orange', linestyle='--', linewidth=2,
                   label=f"Mean: {np.nanmean(data):.1f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of Perturbations")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if len(patches) <= 100:
        for rect, count in zip(patches, n):
            if count > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                        f"{int(count)}", ha="center", va="bottom", fontsize=8, rotation=0)


def main():
    parser = argparse.ArgumentParser(description="2-hop pathway coverage analysis")
    parser.add_argument("--pathway-xlsx", required=True, help="Path to 2-hop pathway Excel")
    parser.add_argument("--validation-csv", required=True, help="Path to target_validation_expanded.csv")
    parser.add_argument("--deg-folder", required=True, help="Path to de_results_per_gene folder")
    parser.add_argument("--output-dir", default="./out_2hop_coverage", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("LOADING PATHWAY PAIRS (2-HOP)")
    pairs_pathway = read_pathways_xlsx(args.pathway_xlsx)
    logger.info("Loaded %s pathway rows, %s unique pairs",
                f"{len(pairs_pathway):,}", f"{pairs_pathway.drop_duplicates().shape[0]:,}")

    logger.info("SELECTING VALID PERTURBATIONS")
    valid_genes = read_valid_genes(args.validation_csv)
    logger.info("Valid perturbations (analysis_flag=Use_for_analysis): %s", f"{len(valid_genes):,}")

    logger.info("LOADING SIGNIFICANT DESCENDANTS PER GENE")
    per_gene_desc, global_desc_pairs = load_all_descendants(valid_genes, args.deg_folder)

    logger.info("COMPUTING COVERAGE (PER PERTURBATION & OVERALL)")
    df_cov = compute_coverage_2hop(pairs_pathway, per_gene_desc)
    overall = compute_overall_coverage(pairs_pathway, global_desc_pairs)
    logger.info("Overall coverage (unique explained/descendant pairs): %.2f%%", overall)

    out_csv = os.path.join(args.output_dir, "coverage_2hop_per_perturbation.csv")
    df_cov.to_csv(out_csv, index=False)
    logger.info("Saved per-perturbation coverage to %s", out_csv)

    explained_counts = df_cov["explained_unique_targets"].fillna(0).astype(float).values

    plt.figure(figsize=(11, 7))
    ax = plt.gca()
    hist_with_annotations(
        ax=ax, data=explained_counts, bins=100,
        title="Explained descendants per perturbation (2-hop) - 100 bins",
        xlabel="# explained unique targets"
    )
    if (df_cov["gene"] == "TP53").any():
        tp53_val = float(df_cov.loc[df_cov["gene"] == "TP53", "explained_unique_targets"].iloc[0])
        ax.axvline(tp53_val, color="red", linestyle="--", linewidth=2, label=f"TP53: {int(tp53_val)}")
        ax.legend()
    out_png1 = os.path.join(args.output_dir, "hist_explained_counts_2hop_100bins.png")
    plt.tight_layout()
    plt.savefig(out_png1, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_png1)

    pct_vals = df_cov["percent_explained"].values
    plt.figure(figsize=(11, 7))
    ax = plt.gca()
    hist_with_annotations(
        ax=ax, data=pct_vals[~np.isnan(pct_vals)], bins=100,
        title="Percent of descendants explained per perturbation (2-hop) - 100 bins",
        xlabel="% explained"
    )
    if (df_cov["gene"] == "TP53").any():
        tp53_pct = float(df_cov.loc[df_cov["gene"] == "TP53", "percent_explained"].iloc[0])
        ax.axvline(tp53_pct, color="red", linestyle="--", linewidth=2, label=f"TP53: {tp53_pct:.1f}%")
        ax.legend()
    out_png2 = os.path.join(args.output_dir, "hist_percent_explained_2hop_100bins.png")
    plt.tight_layout()
    plt.savefig(out_png2, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_png2)

    if (df_cov["gene"] == "TP53").any():
        pct_all = df_cov["percent_explained"].dropna().values
        pct_no_tp53 = df_cov.loc[df_cov["gene"] != "TP53", "percent_explained"].dropna().values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
        hist_with_annotations(
            ax=ax1, data=pct_all, bins=100,
            title="TP53 INCLUDED - % explained (2-hop)", xlabel="% explained")
        ax1.grid(True, alpha=0.3)
        hist_with_annotations(
            ax=ax2, data=pct_no_tp53, bins=100,
            title="TP53 EXCLUDED - % explained (2-hop)", xlabel="% explained")
        ax2.grid(True, alpha=0.3)

        out_png3 = os.path.join(args.output_dir, "hist_percent_explained_2hop_tp53_included_vs_excluded.png")
        plt.tight_layout()
        plt.savefig(out_png3, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", out_png3)

    logger.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
