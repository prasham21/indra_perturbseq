import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Global matplotlib style for publication-quality figures ---
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

# =========================
# 0. PATHS + CONSTANTS
# =========================

# Edit these if your paths differ
ENDO_PATH = "/Users/prashammarfatia/Downloads/endothelial_present_plus_manual.csv"
DEG_GLOB = "/Users/prashammarfatia/Downloads/de_results_per_gene/*_vs_control.csv"
INDRA_1HOP_PATH = "/Users/prashammarfatia/Downloads/indra_1hop_all_perturbations_endothelial_dataset.csv"
TARGET_VALIDATION_PATH = "/Users/prashammarfatia/Downloads/target_validation_expanded.csv"

# Number of cumulative quantile bins for ROC #1 and ROC #2
N_BINS = 20

# Fixed raw p-value thresholds for ROC #3
FIXED_P_THRESHOLDS = np.array([0.5, 0.2, 0.1, 0.05, 0.02,
                               0.01, 0.005, 0.001, 0.0005, 0.0001])

# INDRA column names (you confirmed these)
INDRA_SOURCE_COL = "source"  # perturbation / source gene
INDRA_TARGET_COL = "target"  # downstream / child gene
INDRA_STMT_TYPE_COL = "stmt_type"  # statement type (IncreaseAmount / DecreaseAmount)

# target_validation_expanded column names  ----> ADJUST THESE TO MATCH YOUR FILE
TV_SOURCE_COL = "Gene"  # e.g. "perturbation", "KD_gene", "source_gene"
TV_FLAG_COL = "Karen_Flag"
TV_FLAG_KEEP = "Use_for_analysis"  # value indicating we should keep this perturbation


# =========================
# 1. BUILD DATA MATRIX (USING TARGET_VALIDATION)
# =========================

def extract_source_name_from_deg_path(path: str) -> str:
    """From '<SOURCE>_vs_control.csv' → 'SOURCE'."""
    base = os.path.basename(path)
    return base.replace("_vs_control.csv", "")


def get_karen_sources(tv_path: str) -> list[str]:
    """
    Load target_validation_expanded.csv and return the list of perturbation
    genes that Karen marked to use (via Karen_Flag).
    """
    tv_df = pd.read_csv(tv_path)

    if TV_SOURCE_COL not in tv_df.columns:
        raise ValueError(
            f"TV_SOURCE_COL='{TV_SOURCE_COL}' not found in {tv_path}. "
            f"Available columns: {tv_df.columns.tolist()}"
        )
    if TV_FLAG_COL not in tv_df.columns:
        raise ValueError(
            f"TV_FLAG_COL='{TV_FLAG_COL}' not found in {tv_path}. "
            f"Available columns: {tv_df.columns.tolist()}"
        )

    keep_df = tv_df[tv_df[TV_FLAG_COL] == TV_FLAG_KEEP].copy()
    sources = keep_df[TV_SOURCE_COL].astype(str).tolist()
    sources_unique = list(dict.fromkeys(sources))  # preserve order, drop dups

    print(f"From target_validation_expanded: {len(sources_unique)} sources with {TV_FLAG_COL} == '{TV_FLAG_KEEP}'")
    return sources_unique


def build_data_matrix(endo_path: str,
                      deg_glob: str,
                      tv_path: str):
    """
    Build:
      - gene_list: ordered list of endothelial genes that appear in DEG files
      - source_list: ordered list of perturbation genes selected via target_validation_expanded
      - data_matrix: shape (n_sources, n_genes), p-values

    Uses:
      - endothelial_present_plus_manual.csv → column 'gene'
      - target_validation_expanded.csv → TV_SOURCE_COL, TV_FLAG_COL
      - DEG files → columns 'names' (gene), 'pvals' (p-value)
    """
    # 0) Get Karen-approved sources
    karen_sources = get_karen_sources(tv_path)
    karen_source_set = set(karen_sources)

    # 1) Load endothelial genes
    endo_df = pd.read_csv(endo_path)
    if "gene" not in endo_df.columns:
        raise ValueError(f"'gene' column not found in {endo_path}")
    endo_genes = [str(g) for g in endo_df["gene"].astype(str).tolist()]

    # 2) Collect ALL DEG files and map source -> path
    all_deg_paths = glob.glob(deg_glob)
    if not all_deg_paths:
        raise FileNotFoundError(f"No DEG files found for pattern: {deg_glob}")

    source_to_path = {}
    for path in all_deg_paths:
        src = extract_source_name_from_deg_path(path)
        # if duplicates exist, last one wins; adjust if you need more logic
        source_to_path[src] = path

    # 3) Restrict sources to those that both:
    #    - Karen wants (from target_validation)
    #    - Have a corresponding DEG file
    selected_sources = []
    selected_paths = []
    for src in karen_sources:
        if src in source_to_path:
            selected_sources.append(src)
            selected_paths.append(source_to_path[src])
        else:
            # can print a warning to see missing ones, if any
            print(f"WARNING: Karen source '{src}' has no matching DEG file (<{src}>_vs_control.csv)")

    if not selected_sources:
        raise RuntimeError(
            "No overlap between Karen-approved sources and DEG files.\n"
            "Check TV_SOURCE_COL and DEG filenames."
        )

    print(f"Using {len(selected_sources)} sources that have both Karen_Flag and DEG files.")

    # 4) Use the first selected DEG file to define DEG gene universe
    example_df = pd.read_csv(selected_paths[0])
    if "names" not in example_df.columns:
        raise ValueError(f"'names' column not found in DEG file: {selected_paths[0]}")
    if "pvals" not in example_df.columns:
        raise ValueError(f"'pvals' column not found in DEG file: {selected_paths[0]}")

    gene_col = "names"
    pval_col = "pvals"

    deg_genes_universe = set(example_df[gene_col].astype(str))
    print(f"DEG-universe genes from example: {len(deg_genes_universe)}")

    # 5) Restrict to genes present in both lists (keep endothelial order)
    gene_list = [g for g in endo_genes if g in deg_genes_universe]
    print(
        f"Genes kept after intersection: {len(gene_list)} "
        f"({len(gene_list) / len(endo_genes) * 100:.2f}% coverage)"
    )

    gene_to_idx = {g: j for j, g in enumerate(gene_list)}

    n_sources = len(selected_sources)
    n_genes = len(gene_list)

    print(f"Number of sources (Karen-approved): {n_sources}")
    print(f"Number of genes (matrix columns): {n_genes}")

    # 6) Initialize data matrix with NaNs
    data_matrix = np.full((n_sources, n_genes), np.nan, dtype=float)

    # 7) Fill matrix row by row
    for i, (src_name, path) in enumerate(zip(selected_sources, selected_paths)):
        df = pd.read_csv(path)

        if gene_col not in df.columns or pval_col not in df.columns:
            raise ValueError(
                f"Expected columns '{gene_col}' and '{pval_col}' in file: {path}"
            )

        df[gene_col] = df[gene_col].astype(str)
        gene_to_pval = dict(zip(df[gene_col], df[pval_col].astype(float)))

        for g, j in gene_to_idx.items():
            if g in gene_to_pval:
                data_matrix[i, j] = gene_to_pval[g]
            # else remains NaN

    # source_list returned here is exactly the Karen-approved, DEG-present list
    return data_matrix, gene_list, selected_sources


# =========================
# 2. BUILD INDRA MATRIX
# =========================

def build_indra_matrix(indra_path: str,
                       gene_list: list[str],
                       source_list: list[str]) -> np.ndarray:
    """
    Build INDRA child indicator matrix with shape (n_sources, n_genes).

    Uses:
      - INDRA_SOURCE_COL as the perturbation/source column
      - INDRA_TARGET_COL as the child/target gene column
      - INDRA_STMT_TYPE_COL to filter to IncreaseAmount / DecreaseAmount

    Only keeps rows whose source is in source_list and target in gene_list.
    """
    indra_df = pd.read_csv(indra_path)

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

    print(f"Using INDRA source column: {INDRA_SOURCE_COL}")
    print(f"Using INDRA target column: {INDRA_TARGET_COL}")

    if INDRA_STMT_TYPE_COL is not None:
        if INDRA_STMT_TYPE_COL not in indra_df.columns:
            raise ValueError(
                f"INDRA stmt_type column '{INDRA_STMT_TYPE_COL}' not found. "
                f"Columns: {indra_df.columns.tolist()}"
            )
        # Filter to IncreaseAmount / DecreaseAmount
        mask = indra_df[INDRA_STMT_TYPE_COL].isin(["IncreaseAmount", "DecreaseAmount"])
        indra_df = indra_df[mask].copy()
        print(f"Filtered INDRA rows to IncreaseAmount/DecreaseAmount: {len(indra_df)}")
    else:
        print("No stmt_type filtering (INDRA_STMT_TYPE_COL is None). Using all rows.")

    # Normalize names to strings
    indra_df[INDRA_SOURCE_COL] = indra_df[INDRA_SOURCE_COL].astype(str)
    indra_df[INDRA_TARGET_COL] = indra_df[INDRA_TARGET_COL].astype(str)

    # Restrict to our sources and genes
    src_set = set(source_list)
    gene_set = set(gene_list)

    indra_df = indra_df[
        indra_df[INDRA_SOURCE_COL].isin(src_set)
        & indra_df[INDRA_TARGET_COL].isin(gene_set)
        ]
    print(f"INDRA rows after restricting to our sources & genes: {len(indra_df)}")

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


# =========================
# 3. CORE METRICS HELPERS
# =========================

def compute_tp_fp_fn_for_mask(emp_child: np.ndarray,
                              indra_child: np.ndarray):
    """
    Given:
      emp_child   - boolean array for empirical children
      indra_child - boolean array for INDRA children

    Return:
      TPR, FPR, FNR  (or NaN if denominators empty)
    """
    tp_mask = emp_child & indra_child
    fn_mask = emp_child & (~indra_child)
    fp_mask = indra_child & (~emp_child)

    tp = tp_mask.sum()
    fn = fn_mask.sum()
    fp = fp_mask.sum()

    n_emp = emp_child.sum()
    n_indra = indra_child.sum()

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
    """Trapezoidal AUC for ROC curve."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 2:
        return np.nan
    order = np.argsort(x_clean)
    x_sorted = x_clean[order]
    y_sorted = y_clean[order]
    # use trapezoid instead of deprecated trapz
    return np.trapezoid(y_sorted, x_sorted)


def mean_curve_over_sources(tpr_matrix: np.ndarray,
                            fpr_matrix: np.ndarray):
    """Average TPR and FPR across sources (ignoring NaNs)."""
    tpr_mean = np.nanmean(tpr_matrix, axis=0)
    fpr_mean = np.nanmean(fpr_matrix, axis=0)
    return tpr_mean, fpr_mean


# =========================
# 4. ROC #1 – P-VALUE QUANTILE BINS
# =========================

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


# =========================
# 5. ROC #2 – NEG LOG P-VALUE QUANTILE BINS
# =========================

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


# =========================
# 6. ROC #3 – FIXED RAW P-VALUE THRESHOLDS
# =========================

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


# =========================
# 7. PLOTTING HELPER
# =========================

def plot_roc_curve(fpr: np.ndarray,
                   tpr: np.ndarray,
                   label: str,
                   color=None):
    """
    Plot a single ROC curve on the current axis and return its AUC.
    Uses professional defaults: thicker line, small markers.
    """
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
    outfile="raw_tp_fp_fn.csv",
):
    """
    Export raw TP, FP, FN for each source and each bin threshold.
    bin_thresholds should be a list of raw p-value thresholds (0 to 1).
    """

    rows = []

    for si, source in enumerate(source_list):

        # INDRA children for this source (vector of 0/1)
        A_vec = indra_matrix[si]
        A_set = set([i for i, v in enumerate(A_vec) if v == 1])
        size_A = len(A_set)

        # p-values for this source (vector)
        pvals = data_matrix[si]

        for bi, thr in enumerate(bin_thresholds):
            # empirical children: pval <= threshold
            B_set = set([i for i, p in enumerate(pvals) if p <= thr])
            size_B = len(B_set)

            # intersections
            TP = len(A_set & B_set)
            FP = len(A_set - B_set)
            FN = len(B_set - A_set)

            # avoid division-by-zero
            TPR = TP / size_B if size_B > 0 else 0
            FPR = FP / size_A if size_A > 0 else 0

            rows.append({
                "source": source,
                "bin_id": bi + 1,
                "threshold": thr,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "size_A": size_A,
                "size_B": size_B,
                "TPR": TPR,
                "FPR": FPR,
            })

    # write CSV
    df_out = pd.DataFrame(rows)
    df_out.to_csv(outfile, index=False)
    print(f"\nSaved raw TP/FP/FN to: {outfile}")


# =========================
# 8. MAIN
# =========================
def main():
    # ======================================================
    # 1) BUILD DATA MATRIX (P-values) FROM DE FILES
    # ======================================================
    data_matrix, gene_list, source_list = build_data_matrix(
        ENDO_PATH,
        DEG_GLOB,
        TARGET_VALIDATION_PATH,
    )

    # ======================================================
    # 2) BUILD INDRA MATRIX (1-hop inc/dec children)
    # ======================================================
    indra_matrix = build_indra_matrix(
        INDRA_1HOP_PATH,
        gene_list=gene_list,
        source_list=source_list,
    )

    print("data_matrix shape:", data_matrix.shape)
    print("indra_matrix shape:", indra_matrix.shape)

    # ======================================================
    # 3) COMPUTE ROC CURVES
    # Quantile(p), Quantile(-log10 p), Fixed thresholds
    # ======================================================

    # --- ROC 1: Quantile(p) ---
    fpr_qp, tpr_qp, frac_qp = roc_from_pvalue_quantiles(
        data_matrix, indra_matrix, n_bins=N_BINS
    )

    # --- ROC 2: Quantile(-log10 p) ---
    fpr_ql, tpr_ql, frac_ql = roc_from_neglog_quantiles(
        data_matrix, indra_matrix, n_bins=N_BINS
    )

    # --- ROC 3: Fixed raw p-value thresholds ---
    fpr_fp, tpr_fp, thr_fp = roc_from_fixed_p_thresholds(
        data_matrix, indra_matrix, FIXED_P_THRESHOLDS
    )

    # ======================================================
    # 4) EXPORT RAW TP–FP–FN DATA (THE CSV KAREN ASKED FOR)
    # ======================================================

    # Export only using fixed thresholds (most interpretable)
    export_raw_tp_fp_fn(
        data_matrix=data_matrix,
        indra_matrix=indra_matrix,
        gene_list=gene_list,
        source_list=source_list,
        bin_thresholds=FIXED_P_THRESHOLDS,
        outfile="raw_tp_fp_fn.csv",
    )

    print("\nSaved raw TP/FP/FN CSV for Karen: raw_tp_fp_fn.csv")

    # ======================================================
    # 5) PLOTTING — THREE INDIVIDUAL ROC CURVES
    # ======================================================

    # ---------- FIGURE 1: Quantile(p) ----------
    plt.figure()
    auc1 = plot_roc_curve(fpr_qp, tpr_qp, label="Quantile p-value thresholds")
    plt.plot([0, 1], [0, 1], "r--", linewidth=1.4, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"ROC Curve: Quantile-Based p-Value Thresholds (1-Hop Increase/Decrease)\n"
        f"AUC = {auc1:.3f}"
    )
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_quantile_p.png")

    # ---------- FIGURE 2: Quantile(-log10 p) ----------
    plt.figure()
    auc2 = plot_roc_curve(fpr_ql, tpr_ql, label="Quantile -log10(p) thresholds")
    plt.plot([0, 1], [0, 1], "r--", linewidth=1.4, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"ROC Curve: Quantile-Based -log10(p) Thresholds (1-Hop Increase/Decrease)\n"
        f"AUC = {auc2:.3f}"
    )
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_quantile_neglog_p.png")

    # ---------- FIGURE 3: Fixed Raw p-Thresh -->
    plt.figure()
    auc3 = plot_roc_curve(fpr_fp, tpr_fp, label="Fixed p-value thresholds")
    plt.plot([0, 1], [0, 1], "r--", linewidth=1.4, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"ROC Curve: Fixed p-Value Thresholds (1-Hop Increase/Decrease)\n"
        f"AUC = {auc3:.3f}"
    )
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_fixed_p_thresholds.png")

    # ======================================================
    # 6) COMBINED COMPARISON FIGURE
    # ======================================================
    plt.figure()
    plot_roc_curve(fpr_qp, tpr_qp, label="Quantile p-value thresholds")
    plot_roc_curve(fpr_ql, tpr_ql, label="Quantile -log10(p) thresholds")
    plot_roc_curve(fpr_fp, tpr_fp, label="Fixed p-value thresholds")
    plt.plot([0, 1], [0, 1], "r--", linewidth=1.4, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison of Thresholding Strategies (1-Hop Increase/Decrease)")
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_all_strategies_combined.png")

    plt.show()

    # ======================================================
    # 7) PRINT SUMMARY
    # ======================================================
    print("\nAUC summary:")
    print(f"  Quantile(p):                {auc1:.3f}")
    print(f"  Quantile(-log10 p):         {auc2:.3f}")
    print(f"  Fixed p-value thresholds:   {auc3:.3f}")
    print("\nROC PNGs saved:")
    print("  - roc_quantile_p.png")
    print("  - roc_quantile_neglog_p.png")
    print("  - roc_fixed_p_thresholds.png")
    print("  - roc_all_strategies_combined.png")
    print("  - raw_tp_fp_fn.csv  (for Karen)")


if __name__ == "__main__":
    main()
