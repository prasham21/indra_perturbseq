"""Legacy script: ROC curve creation for 2-hop INDRA validation."""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import logging


logger = logging.getLogger(__name__)
plt.rcParams.update({
    "figure.figsize": (6, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2,
    "savefig.dpi": 300,
})

N_BINS = 20

FIXED_P_THRESHOLDS = np.array([0.5, 0.2, 0.1, 0.05, 0.02,
                               0.01, 0.005, 0.001, 0.0005, 0.0001])

INDRA_SOURCE_COL = "source"
INDRA_TARGET_COL = "target"

TV_SOURCE_COL = "Gene"
FILTER_COLUMN = "analysis_flag"
TV_FLAG_KEEP = "Use_for_analysis"


def extract_source_name_from_deg_path(path: str) -> str:
    """From '<SOURCE>_vs_control.csv' -> 'SOURCE'."""
    base = os.path.basename(path)
    return base.replace("_vs_control.csv", "")


def get_filtered_sources(target_validation_path: str) -> list[str]:
    """Return approved perturbation genes."""
    tv_df = pd.read_csv(target_validation_path)

    if TV_SOURCE_COL not in tv_df.columns:
        raise ValueError(
            f"TV_SOURCE_COL='{TV_SOURCE_COL}' not found in {target_validation_path}. "
            f"Available columns: {tv_df.columns.tolist()}"
        )
    if FILTER_COLUMN not in tv_df.columns:
        raise ValueError(
            f"FILTER_COLUMN='{FILTER_COLUMN}' not found in {target_validation_path}. "
            f"Available columns: {tv_df.columns.tolist()}"
        )

    keep_df = tv_df[tv_df[FILTER_COLUMN] == TV_FLAG_KEEP].copy()
    sources = keep_df[TV_SOURCE_COL].astype(str).tolist()
    sources_unique = list(dict.fromkeys(sources))  # preserve order, drop dups

    logger.info("From target_validation_expanded: %s sources with %s == '%s'", len(sources_unique), FILTER_COLUMN, TV_FLAG_KEEP)
    return sources_unique


def build_data_matrix(endo_path: str,
                      deg_glob: str,
                      target_validation_path: str):
    """Build p-value data matrix from DE result files."""
    filtered_sources = get_filtered_sources(target_validation_path)

    endo_df = pd.read_csv(endo_path)
    if "gene" not in endo_df.columns:
        raise ValueError(f"'gene' column not found in {endo_path}")
    endo_genes = [str(g) for g in endo_df["gene"].astype(str).tolist()]

    all_deg_paths = glob.glob(deg_glob)
    if not all_deg_paths:
        raise FileNotFoundError(f"No DEG files found for pattern: {deg_glob}")

    source_to_path = {}
    for path in all_deg_paths:
        src = extract_source_name_from_deg_path(path)
        source_to_path[src] = path

    selected_sources = []
    selected_paths = []
    for src in filtered_sources:
        if src in source_to_path:
            selected_sources.append(src)
            selected_paths.append(source_to_path[src])
        else:
            logger.warning("Source '%s' has no matching DEG file (%s_vs_control.csv)", src, src)

    if not selected_sources:
        raise RuntimeError(
            "No overlap between approved sources and DEG files.\n"
            "Check TV_SOURCE_COL and DEG filenames."
        )

    logger.info("Using %s sources that have both analysis_flag and DEG files.", len(selected_sources))

    example_df = pd.read_csv(selected_paths[0])
    if "names" not in example_df.columns:
        raise ValueError(f"'names' column not found in DEG file: {selected_paths[0]}")
    if "pvals" not in example_df.columns:
        raise ValueError(f"'pvals' column not found in DEG file: {selected_paths[0]}")

    gene_column = "names"
    pval_col = "pvals"

    deg_genes_universe = set(example_df[gene_column].astype(str))
    logger.info("DEG-universe genes from example: %d", len(deg_genes_universe))

    gene_list = [g for g in endo_genes if g in deg_genes_universe]
    logger.info(
        "Genes kept after intersection: %d (%.2f%% coverage)",
        len(gene_list), len(gene_list) / len(endo_genes) * 100,
    )

    gene_to_idx = {g: j for j, g in enumerate(gene_list)}

    n_sources = len(selected_sources)
    n_genes = len(gene_list)

    logger.info("Number of sources: %d", n_sources)
    logger.info("Number of genes (matrix columns): %d", n_genes)

    data_matrix = np.full((n_sources, n_genes), np.nan, dtype=float)

    for i, (src_name, path) in enumerate(zip(selected_sources, selected_paths)):
        df = pd.read_csv(path)

        if gene_column not in df.columns or pval_col not in df.columns:
            raise ValueError(
                f"Expected columns '{gene_column}' and '{pval_col}' in file: {path}"
            )

        df[gene_column] = df[gene_column].astype(str)
        gene_to_pval = dict(zip(df[gene_column], pd.to_numeric(df[pval_col], errors="coerce")))

        for g, j in gene_to_idx.items():
            if g in gene_to_pval:
                data_matrix[i, j] = gene_to_pval[g]

    return data_matrix, gene_list, selected_sources


def build_indra_matrix_2hop(indra_path: str,
                            gene_list: list[str],
                            source_list: list[str]) -> np.ndarray:
    """
    Build INDRA child indicator matrix with shape (n_sources, n_genes).
    2-hop input file is assumed DEDUP: one row per (source,target).
    Uses only source/target columns; intermediate is ignored for ROC.
    """
    indra_df = pd.read_csv(indra_path, low_memory=False)

    if INDRA_SOURCE_COL not in indra_df.columns:
        raise ValueError(
            f"INDRA source column '{INDRA_SOURCE_COL}' not found. "
            f"Columns: {indra_df.columns.tolist()}"
        )
    if INDRA_TARGET_COL not in indra_df.columns:
        raise ValueError(
            f"INDRA target column '{INDRA_TARGET_COL}' not found. "
            f"Columns: {indra_df.columns.tolist()}"
        )

    logger.info("Using INDRA source column: %s", INDRA_SOURCE_COL)
    logger.info("Using INDRA target column: %s", INDRA_TARGET_COL)

    indra_df[INDRA_SOURCE_COL] = indra_df[INDRA_SOURCE_COL].astype(str)
    indra_df[INDRA_TARGET_COL] = indra_df[INDRA_TARGET_COL].astype(str)

    src_set = set(source_list)
    gene_set = set(gene_list)

    indra_df = indra_df[
        indra_df[INDRA_SOURCE_COL].isin(src_set)
        & indra_df[INDRA_TARGET_COL].isin(gene_set)
    ]
    logger.info("INDRA rows after restricting to our sources & genes: %s", len(indra_df))

    src_to_idx = {s: i for i, s in enumerate(source_list)}
    gene_to_idx = {g: j for j, g in enumerate(gene_list)}

    n_sources = len(source_list)
    n_genes = len(gene_list)
    indra_matrix = np.zeros((n_sources, n_genes), dtype=int)

    for _, row in indra_df.iterrows():
        s = str(row[INDRA_SOURCE_COL])
        g = str(row[INDRA_TARGET_COL])
        i = src_to_idx.get(s)
        j = gene_to_idx.get(g)
        if i is not None and j is not None:
            indra_matrix[i, j] = 1

    return indra_matrix


def compute_tp_fp_fn_for_mask(emp_child: np.ndarray,
                              indra_child: np.ndarray):
    """
    Given:
      emp_child   - boolean array for empirical children (B)
      indra_child - boolean array for INDRA children (A)

    Return:
      TPR, FPR, FNR (NaN if denominators empty)
    """
    tp_mask = emp_child & indra_child
    fn_mask = emp_child & (~indra_child)
    fp_mask = indra_child & (~emp_child)

    tp = tp_mask.sum()
    fn = fn_mask.sum()
    fp = fp_mask.sum()

    n_emp = emp_child.sum()     # |B|
    n_indra = indra_child.sum() # |A|

    if n_emp > 0:
        tpr = tp / n_emp
        fnr = fn / n_emp
    else:
        tpr = np.nan
        fnr = np.nan

    if n_indra > 0:
        fpr = fp / n_indra
    else:
        fpr = np.nan

    return tpr, fpr, fnr


def compute_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal AUC for curve."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 2:
        return np.nan
    order = np.argsort(x_clean)
    x_sorted = x_clean[order]
    y_sorted = y_clean[order]
    return np.trapezoid(y_sorted, x_sorted)


def mean_curve_over_sources(tpr_matrix: np.ndarray,
                            fpr_matrix: np.ndarray):
    """Average TPR and FPR across sources (ignoring NaNs)."""
    tpr_mean = np.nanmean(tpr_matrix, axis=0)
    fpr_mean = np.nanmean(fpr_matrix, axis=0)
    return tpr_mean, fpr_mean


def roc_from_pvalue_quantiles(data_matrix: np.ndarray,
                              indra_matrix: np.ndarray,
                              n_bins: int = 20):
    """
    ROC variant #1:
    - For each source, take all p-values
    - For quantile fractions f = 0.05, 0.10, ..., 1.0
      threshold t_f = quantile(pvals, f)
      empirical children = p <= t_f   (top f most significant)
    - Compute TPR, FPR per bin and average across sources
    """
    assert data_matrix.shape == indra_matrix.shape
    n_sources, _ = data_matrix.shape
    fractions = np.linspace(0.05, 1.0, n_bins)

    indra_bool = indra_matrix.astype(bool)

    tpr_all = np.full((n_sources, n_bins), np.nan)
    fpr_all = np.full((n_sources, n_bins), np.nan)

    for i in range(n_sources):
        pvals = data_matrix[i, :]
        indra_child = indra_bool[i, :]

        if np.all(np.isnan(pvals)):
            continue

        for k, f in enumerate(fractions):
            t = np.nanquantile(pvals, f)
            emp_child = (pvals <= t)

            tpr, fpr, _ = compute_tp_fp_fn_for_mask(emp_child, indra_child)
            tpr_all[i, k] = tpr
            fpr_all[i, k] = fpr

    tpr_mean, fpr_mean = mean_curve_over_sources(tpr_all, fpr_all)
    return fpr_mean, tpr_mean, fractions


def roc_from_neglog_quantiles(data_matrix: np.ndarray,
                              indra_matrix: np.ndarray,
                              n_bins: int = 20):
    """
    ROC variant #2:
    - Transform p-values: -log10(p)
    - For fractions f = 0.05..1.0, threshold T_f = quantile(neglog, 1 - f)
      empirical children = neglog >= T_f (top f most significant in log-space)
    """
    assert data_matrix.shape == indra_matrix.shape
    n_sources, _ = data_matrix.shape
    fractions = np.linspace(0.05, 1.0, n_bins)

    p_clipped = np.clip(data_matrix, 1e-300, 1.0)
    neglog = -np.log10(p_clipped)

    indra_bool = indra_matrix.astype(bool)

    tpr_all = np.full((n_sources, n_bins), np.nan)
    fpr_all = np.full((n_sources, n_bins), np.nan)

    for i in range(n_sources):
        nl = neglog[i, :]
        indra_child = indra_bool[i, :]

        if np.all(np.isnan(nl)):
            continue

        for k, f in enumerate(fractions):
            T = np.nanquantile(nl, 1.0 - f)
            emp_child = (nl >= T)

            tpr, fpr, _ = compute_tp_fp_fn_for_mask(emp_child, indra_child)
            tpr_all[i, k] = tpr
            fpr_all[i, k] = fpr

    tpr_mean, fpr_mean = mean_curve_over_sources(tpr_all, fpr_all)
    return fpr_mean, tpr_mean, fractions


def roc_from_fixed_p_thresholds(data_matrix: np.ndarray,
                                indra_matrix: np.ndarray,
                                thresholds: np.ndarray):
    """
    ROC variant #3:
    - For each fixed p-value threshold t in thresholds:
         empirical children = p <= t
    - Compute TPR / FPR per threshold and average over sources.
    """
    assert data_matrix.shape == indra_matrix.shape
    n_sources, _ = data_matrix.shape
    indra_bool = indra_matrix.astype(bool)
    n_bins = len(thresholds)

    tpr_all = np.full((n_sources, n_bins), np.nan)
    fpr_all = np.full((n_sources, n_bins), np.nan)

    for i in range(n_sources):
        pvals = data_matrix[i, :]
        indra_child = indra_bool[i, :]

        if np.all(np.isnan(pvals)):
            continue

        for k, t in enumerate(thresholds):
            emp_child = (pvals <= t)
            tpr, fpr, _ = compute_tp_fp_fn_for_mask(emp_child, indra_child)
            tpr_all[i, k] = tpr
            fpr_all[i, k] = fpr

    tpr_mean, fpr_mean = mean_curve_over_sources(tpr_all, fpr_all)
    return fpr_mean, tpr_mean, thresholds


def plot_roc_curve(fpr: np.ndarray,
                   tpr: np.ndarray,
                   label: str):
    mask = ~(np.isnan(fpr) | np.isnan(tpr))
    x = fpr[mask]
    y = tpr[mask]

    if len(x) == 0:
        return np.nan

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    auc = compute_auc(x, y)
    plt.plot(
        x,
        y,
        marker="o",
        markersize=4,
        linestyle="-",
        label=f"{label} (AUC = {auc:.3f})",
    )
    return auc


def export_raw_tp_fp_fn(
    data_matrix,
    indra_matrix,
    gene_list,
    source_list,
    bin_thresholds,
    outfile="raw_tp_fp_fn_2hop.csv",
):
    """
    Export raw TP, FP, FN for each source and each fixed p-value threshold.
    Uses the same nonstandard rates as your 1-hop ROC script:
      - TPR = TP / |B|   (B = empirical children)
      - FPR = FP / |A|   (A = INDRA children)
    """
    rows = []

    for si, source in enumerate(source_list):
        A_vec = indra_matrix[si]
        A_set = set([i for i, v in enumerate(A_vec) if v == 1])
        size_A = len(A_set)

        pvals = data_matrix[si]

        for bi, thr in enumerate(bin_thresholds):
            B_set = set([i for i, p in enumerate(pvals) if p <= thr])
            size_B = len(B_set)

            TP = len(A_set & B_set)
            FP = len(A_set - B_set)
            FN = len(B_set - A_set)

            TPR = TP / size_B if size_B > 0 else 0
            FPR = FP / size_A if size_A > 0 else 0

            rows.append({
                "source": source,
                "bin_id": bi + 1,
                "threshold": float(thr),
                "TP": int(TP),
                "FP": int(FP),
                "FN": int(FN),
                "size_A": int(size_A),
                "size_B": int(size_B),
                "TPR": float(TPR),
                "FPR": float(FPR),
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(outfile, index=False)
    logger.info("\nSaved raw TP/FP/FN to: %s", outfile)


def main():
    ap = argparse.ArgumentParser(description="Create ROC curves for 2-hop INDRA validation.")
    ap.add_argument("--endo-path", required=True, help="Endothelial gene list CSV.")
    ap.add_argument("--deg-glob", required=True, help="Glob pattern for DE result CSVs.")
    ap.add_argument("--indra-2hop-path", required=True, help="INDRA 2-hop dedup dataset CSV.")
    ap.add_argument("--target-validation-path", required=True, help="Target validation CSV.")
    args = ap.parse_args()

    data_matrix, gene_list, source_list = build_data_matrix(
        args.endo_path,
        args.deg_glob,
        args.target_validation_path,
    )

    indra_matrix = build_indra_matrix_2hop(
        args.indra_2hop_path,
        gene_list=gene_list,
        source_list=source_list,
    )

    logger.info("data_matrix shape: %s", data_matrix.shape)
    logger.info("indra_matrix shape: %s", indra_matrix.shape)

    fpr_qp, tpr_qp, _ = roc_from_pvalue_quantiles(
        data_matrix, indra_matrix, n_bins=N_BINS
    )

    fpr_ql, tpr_ql, _ = roc_from_neglog_quantiles(
        data_matrix, indra_matrix, n_bins=N_BINS
    )

    fpr_fp, tpr_fp, thr_fp = roc_from_fixed_p_thresholds(
        data_matrix, indra_matrix, FIXED_P_THRESHOLDS
    )

    export_raw_tp_fp_fn(
        data_matrix=data_matrix,
        indra_matrix=indra_matrix,
        gene_list=gene_list,
        source_list=source_list,
        bin_thresholds=FIXED_P_THRESHOLDS,
        outfile="raw_tp_fp_fn_2hop.csv",
    )

    plt.figure()
    auc1 = plot_roc_curve(fpr_qp, tpr_qp, label="Quantile p-value thresholds (2-hop)")
    plt.plot([0, 1], [0, 1], "r--", linewidth=1.4, label="Chance")
    plt.xlabel("False Positive Rate (|A\\B| / |A|)")
    plt.ylabel("True Positive Rate (|A∩B| / |B|)")
    plt.title(f"ROC: Quantile-Based p-Value Thresholds (2-hop)\nAUC = {auc1:.3f}")
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_2hop_quantile_p.png")

    plt.figure()
    auc2 = plot_roc_curve(fpr_ql, tpr_ql, label="Quantile -log10(p) thresholds (2-hop)")
    plt.plot([0, 1], [0, 1], "r--", linewidth=1.4, label="Chance")
    plt.xlabel("False Positive Rate (|A\\B| / |A|)")
    plt.ylabel("True Positive Rate (|A∩B| / |B|)")
    plt.title(f"ROC: Quantile-Based -log10(p) Thresholds (2-hop)\nAUC = {auc2:.3f}")
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_2hop_quantile_neglog_p.png")

    plt.figure()
    auc3 = plot_roc_curve(fpr_fp, tpr_fp, label="Fixed p-value thresholds (2-hop)")
    plt.plot([0, 1], [0, 1], "r--", linewidth=1.4, label="Chance")
    plt.xlabel("False Positive Rate (|A\\B| / |A|)")
    plt.ylabel("True Positive Rate (|A∩B| / |B|)")
    plt.title(f"ROC: Fixed p-Value Thresholds (2-hop)\nAUC = {auc3:.3f}")
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_2hop_fixed_p_thresholds.png")

    plt.figure()
    plot_roc_curve(fpr_qp, tpr_qp, label="Quantile p-value thresholds (2-hop)")
    plot_roc_curve(fpr_ql, tpr_ql, label="Quantile -log10(p) thresholds (2-hop)")
    plot_roc_curve(fpr_fp, tpr_fp, label="Fixed p-value thresholds (2-hop)")
    plt.plot([0, 1], [0, 1], "r--", linewidth=1.4, label="Chance")
    plt.xlabel("False Positive Rate (|A\\B| / |A|)")
    plt.ylabel("True Positive Rate (|A∩B| / |B|)")
    plt.title("ROC Comparison of Thresholding Strategies (2-hop)")
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_2hop_all_strategies_combined.png")

    plt.show()

    logger.info("AUC summary (2-hop): Quantile(p)=%.3f, Quantile(-log10p)=%.3f, Fixed=%.3f",
                auc1, auc2, auc3)


if __name__ == "__main__":
    main()
