import os
import time
import warnings
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.style.use("default")

# =======================
# --- CONFIG: EDIT ME ---
# =======================
PATHWAY_XLSX = "/Users/prashammarfatia/Downloads/indra_2hop_all_perturbations.xlsx"
TV_CSV = "/Users/prashammarfatia/Downloads/target_validation_expanded.csv"
DE_RESULTS_DIR = "/Users/prashammarfatia/Downloads/de_results_per_gene"
OUT_DIR = "./out_2hop_coverage"


# =======================
# --- Helper functions ---
# =======================
def read_pathways_xlsx(xlsx_path: str) -> pd.DataFrame:
    """Read pathway Excel with first-row headers. Return unique ['source','target'] pairs."""
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    # basic validation
    missing = [c for c in ["source", "target"] if c not in df.columns]
    if missing:
        raise ValueError(f"Pathway file missing required columns: {missing}")
    pairs = df[["source", "target"]].dropna().drop_duplicates().reset_index(drop=True)
    return pairs


def read_valid_genes(tv_csv: str) -> pd.Series:
    """Return Series of genes with Karen_Flag == 'Use_for_analysis'."""
    tv = pd.read_csv(tv_csv)
    if "Karen_Flag" not in tv.columns or "Gene" not in tv.columns:
        raise ValueError("target_validation_expanded.csv must have columns 'Gene' and 'Karen_Flag'.")
    genes = tv.loc[tv["Karen_Flag"] == "Use_for_analysis", "Gene"].dropna().drop_duplicates()
    if genes.empty:
        raise ValueError("No genes with Karen_Flag == 'Use_for_analysis'.")
    return genes


def read_deg_significant(csv_path: str) -> pd.DataFrame:
    """
    Read a single DEG CSV and return rows considered significant.
    Uses FDR if an adjusted column exists, else pvals.
    Recognizes common column names for robustness.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # normalize likely columns
    name_col = next((c for c in ["names", "gene", "gene_names", "target"] if c in df.columns), None)
    p_col = next((c for c in ["pvals", "pval", "p_value", "p_val"] if c in df.columns), None)
    fdr_col = next((c for c in ["pvals_adj", "padj", "qval", "fdr", "p_adj"] if c in df.columns), None)

    if name_col is None:
        raise ValueError("DEG file missing a gene/target name column (e.g., 'names').")
    if p_col is None and fdr_col is None:
        raise ValueError("DEG file missing both raw p-value and FDR columns.")

    # use FDR if present, else raw p
    if fdr_col is not None:
        sig = df[(df[fdr_col].astype(float) < 0.05)]
        sig = sig[[name_col]].dropna().drop_duplicates()
    else:
        sig = df[(df[p_col].astype(float) < 0.05)]
        sig = sig[[name_col]].dropna().drop_duplicates()

    sig.columns = ["target"]  # standardize
    return sig


def load_all_descendants(valid_genes: pd.Series, de_dir: str) -> Tuple[Dict[str, Set[str]], Set[Tuple[str, str]]]:
    """
    Load significant descendants for each valid gene.
    Returns:
      - dict: gene -> set(targets)
      - global set of (gene, target) pairs
    """
    per_gene: Dict[str, Set[str]] = {}
    global_pairs: Set[Tuple[str, str]] = set()

    print(f"\nLoading significant descendants from: {de_dir}")
    for i, gene in enumerate(valid_genes, start=1):
        csv_path = os.path.join(de_dir, f"{gene}_vs_control.csv")
        try:
            sig = read_deg_significant(csv_path)
            targets = set(sig["target"].astype(str))
            per_gene[gene] = targets
            for t in targets:
                global_pairs.add((gene, t))
            if i % 25 == 0:
                print(f"  - processed {i:4d} genes…")
        except FileNotFoundError:
            # missing file → treat as no descendants
            per_gene[gene] = set()
        except Exception as e:
            # on parse error, treat as empty but log
            print(f"  ! {gene}: {e}")
            per_gene[gene] = set()

    print(f"Done. Collected {len(global_pairs):,} unique (source,target) descendant pairs.")
    return per_gene, global_pairs


def compute_coverage_2hop(pairs_pathway: pd.DataFrame,
                          per_gene_desc: Dict[str, Set[str]]) -> pd.DataFrame:
    """
    Compute per-perturbation coverage (% explained) for 2-hop.
    Coverage per gene = |targets with pathway| / |significant descendants| * 100
    Also returns a DataFrame with counts.
    """
    # pathway targets per source (unique)
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
            pct = np.nan  # or 0.0 if you prefer
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
    """
    Overall coverage across all perturbations:
    (# unique explained (source,target)) / (# unique descendant (source,target))
    """
    explained_pairs = set(map(tuple, pairs_pathway[["source", "target"]].astype(str).itertuples(index=False)))
    denom = len(global_desc_pairs)
    if denom == 0:
        return float("nan")
    return len(global_desc_pairs & explained_pairs) / denom * 100.0


def hist_with_annotations(ax, data, bins, title, xlabel, annotate_mean=True):
    """Draw a 100-bin histogram with mean line + count labels on bars (for small N)."""
    n, bins, patches = ax.hist(data, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)
    if annotate_mean and len(data) > 0 and np.isfinite(np.nanmean(data)):
        ax.axvline(np.nanmean(data), color='orange', linestyle='--', linewidth=2,
                   label=f"Mean: {np.nanmean(data):.1f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of Perturbations")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Optional: annotate each bar with its height (only if few bars visible)
    if len(patches) <= 100:  # you asked for per-bar counts; keep it reasonable
        for rect, count in zip(patches, n):
            if count > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                        f"{int(count)}", ha="center", va="bottom", fontsize=8, rotation=0)


# ================
# --- Main run  ---
# ================
def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 80)
    print("LOADING PATHWAY PAIRS (2-HOP)")
    print("=" * 80)
    pairs_pathway = read_pathways_xlsx(PATHWAY_XLSX)  # standardized to ['source','target']
    print(f"Loaded {len(pairs_pathway):,} pathway rows → {pairs_pathway.drop_duplicates().shape[0]:,} unique pairs.")

    print("\n" + "=" * 80)
    print("SELECTING VALID PERTURBATIONS")
    print("=" * 80)
    valid_genes = read_valid_genes(TV_CSV)
    print(f"Valid perturbations (Karen_Flag=Use_for_analysis): {len(valid_genes):,}")

    print("\n" + "=" * 80)
    print("LOADING SIGNIFICANT DESCENDANTS PER GENE")
    print("=" * 80)
    per_gene_desc, global_desc_pairs = load_all_descendants(valid_genes, DE_RESULTS_DIR)

    print("\n" + "=" * 80)
    print("COMPUTING COVERAGE (PER PERTURBATION & OVERALL)")
    print("=" * 80)
    df_cov = compute_coverage_2hop(pairs_pathway, per_gene_desc)
    overall = compute_overall_coverage(pairs_pathway, global_desc_pairs)
    print(f"Overall coverage (unique explained pairs / unique descendant pairs): {overall:.2f}%")

    # Save per-perturbation table
    out_csv = os.path.join(OUT_DIR, "coverage_2hop_per_perturbation.csv")
    df_cov.to_csv(out_csv, index=False)
    print(f"Saved per-perturbation coverage → {out_csv}")

    # ==========================
    # --- Plotting (100 bins) ---
    # ==========================
    # 1) Histogram: explained counts
    explained_counts = df_cov["explained_unique_targets"].fillna(0).astype(float).values

    plt.figure(figsize=(11, 7))
    ax = plt.gca()
    hist_with_annotations(
        ax=ax,
        data=explained_counts,
        bins=100,
        title="Explained descendants per perturbation (2-hop) — 100 bins",
        xlabel="# explained unique targets"
    )
    # mark TP53 if present
    if (df_cov["gene"] == "TP53").any():
        tp53_val = float(df_cov.loc[df_cov["gene"] == "TP53", "explained_unique_targets"].iloc[0])
        ax.axvline(tp53_val, color="red", linestyle="--", linewidth=2, label=f"TP53: {int(tp53_val)}")
        ax.legend()
    out_png1 = os.path.join(OUT_DIR, "hist_explained_counts_2hop_100bins.png")
    plt.tight_layout()
    plt.savefig(out_png1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_png1}")

    # 2) Histogram: percent explained
    pct_vals = df_cov["percent_explained"].values
    plt.figure(figsize=(11, 7))
    ax = plt.gca()
    hist_with_annotations(
        ax=ax,
        data=pct_vals[~np.isnan(pct_vals)],
        bins=100,
        title="Percent of descendants explained per perturbation (2-hop) — 100 bins",
        xlabel="% explained"
    )
    # mark TP53 if present
    if (df_cov["gene"] == "TP53").any():
        tp53_pct = float(df_cov.loc[df_cov["gene"] == "TP53", "percent_explained"].iloc[0])
        ax.axvline(tp53_pct, color="red", linestyle="--", linewidth=2, label=f"TP53: {tp53_pct:.1f}%")
        ax.legend()
    out_png2 = os.path.join(OUT_DIR, "hist_percent_explained_2hop_100bins.png")
    plt.tight_layout()
    plt.savefig(out_png2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_png2}")

    # 3) Side-by-side: TP53 included vs excluded (percent explained)
    if (df_cov["gene"] == "TP53").any():
        pct_all = df_cov["percent_explained"].dropna().values
        pct_no_tp53 = df_cov.loc[df_cov["gene"] != "TP53", "percent_explained"].dropna().values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

        # included
        hist_with_annotations(
            ax=ax1, data=pct_all, bins=100,
            title="TP53 INCLUDED — % explained (2-hop)",
            xlabel="% explained"
        )
        ax1.grid(True, alpha=0.3)

        # excluded
        hist_with_annotations(
            ax=ax2, data=pct_no_tp53, bins=100,
            title="TP53 EXCLUDED — % explained (2-hop)",
            xlabel="% explained"
        )
        ax2.grid(True, alpha=0.3)

        out_png3 = os.path.join(OUT_DIR, "hist_percent_explained_2hop_tp53_included_vs_excluded.png")
        plt.tight_layout()
        plt.savefig(out_png3, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved → {out_png3}")

    print("\nDone in {:.1f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
