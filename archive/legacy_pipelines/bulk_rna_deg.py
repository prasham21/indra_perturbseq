#!/usr/bin/env python3
"""
Bulk RNA-seq DEG analysis for 6 source genes (ITGB1BP1, CCM2, PDCD10, MAP3K3, MAP2K5, KLF2).

Method : limma-trend on log2(TPM+1) via InMoose
Fallback: Welch t-test + BH correction (scipy/statsmodels) if InMoose is not installed

Each KD condition (2 replicates) is compared against 4 negative control KD samples.

Output: one CSV per gene saved to OUT_DIR/<GENE>_vs_control.csv
Columns: names, logfoldchanges, pvals, pvals_adj
(format matches what run_1hop_network_export.py, 2hop and 3hop scripts already expect)

Install dependencies:
    pip install inmoose            # primary method
    pip install scipy statsmodels  # fallback method
"""

import os
import numpy as np
import pandas as pd

# ── CONFIG ─────────────────────────────────────────────────────────────────────
TPM_FILE = "/Users/prashammarfatia/Downloads/GSE210522_All_bulk_RNAseq_TPM_by_Symbol.txt"
OUT_DIR  = "/Users/prashammarfatia/Downloads/de_results_bulk_6genes"
# ───────────────────────────────────────────────────────────────────────────────

NEG_CTRL_COLS = [
    "TeloHAEC_B6_neg_ctrl_KD",
    "TeloHAEC_D2_neg_ctrl_KD",
    "TeloHAEC_D8_neg_ctrl_KD",
    "TeloHAEC_F4_neg_ctrl_KD",
]

GENE_TO_KD_COLS = {
    "MAP2K5":   ["TeloHAEC_A11_MAP2K5_KD",  "TeloHAEC_B5_MAP2K5_KD"],
    "MAP3K3":   ["TeloHAEC_A4_MAP3K3_KD",   "TeloHAEC_C3_MAP3K3_KD"],
    "CCM2":     ["TeloHAEC_C2_CCM2_KD",     "TeloHAEC_F6_CCM2_KD"],
    "ITGB1BP1": ["TeloHAEC_C7_ITGB1BP1_KD", "TeloHAEC_G8_ITGB1BP1_KD"],
    "KLF2":     ["TeloHAEC_C9_KLF2_KD",     "TeloHAEC_G10_KLF2_KD"],
    "PDCD10":   ["TeloHAEC_D11_PDCD10_KD",  "TeloHAEC_G1_PDCD10_KD"],
}


def run_limma_trend(expr_df: pd.DataFrame, n_ctrl: int, n_kd: int) -> pd.DataFrame:
    from inmoose.limma import lmFit, eBayes, topTable

    n_samples = n_ctrl + n_kd

    # Use numpy array — InMoose assigns integer column indices (0=Intercept, 1=KD effect)
    design = np.column_stack([
        np.ones(n_samples),                            # col 0: intercept
        np.array([0] * n_ctrl + [1] * n_kd, dtype=float),  # col 1: KD effect
    ])

    fit = lmFit(expr_df, design)
    fit = eBayes(fit, trend=True)

    # Inspect what InMoose actually named the coefficients, then pick the 2nd one (KD effect)
    # Column 0 is always the intercept; column 1 is always the group effect
    coef_label = fit.coefficients.columns[1]
    print(f"  Using coefficient: {coef_label!r}")  # helps debug future issues

    result = topTable(
        fit,
        coef=coef_label,
        number=np.inf,
        adjust_method="fdr_bh",
    )

    return pd.DataFrame(
        {
            "names":          result.index,
            "logfoldchanges": result["log2FoldChange"].values,
            "pvals":          result["pvalue"].values,
            "pvals_adj":      result["adj_pvalue"].values,
        }
    )


def run_ttest_fallback(expr_df: pd.DataFrame, n_ctrl: int) -> pd.DataFrame:
    """
    Welch t-test + BH multiple-testing correction.
    Used automatically if InMoose is not installed.
    Less powerful than limma for n=2 replicates but guaranteed to run.
    """
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests

    mat  = expr_df.values          # (n_genes, n_samples)
    ctrl = mat[:, :n_ctrl]
    kd   = mat[:, n_ctrl:]

    lfc            = kd.mean(axis=1) - ctrl.mean(axis=1)
    _, pvals       = ttest_ind(kd, ctrl, axis=1, equal_var=False)
    pvals          = np.where(np.isnan(pvals), 1.0, pvals)  # NaN → 1.0 for constant genes
    _, padj, _, _  = multipletests(pvals, method="fdr_bh")

    return pd.DataFrame(
        {
            "names":          expr_df.index,
            "logfoldchanges": lfc,
            "pvals":          pvals,
            "pvals_adj":      padj,
        }
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load TPM ───────────────────────────────────────────────────────────────
    print("Loading TPM matrix...")
    tpm = pd.read_csv(TPM_FILE, sep="\t", index_col="Symbol")
    tpm = tpm.apply(pd.to_numeric, errors="coerce")
    print(f"  Loaded: {tpm.shape[0]:,} genes x {tpm.shape[1]} samples")

    # log2(TPM + 1) — pseudo-count of 1 avoids log(0)
    log_tpm = np.log2(tpm + 1)

    # ── Process each of the 6 genes ───────────────────────────────────────────
    for gene, kd_cols in GENE_TO_KD_COLS.items():
        print(f"\n── {gene} ──")

        all_cols = NEG_CTRL_COLS + kd_cols        # ctrl first, then KD
        sub      = log_tpm[all_cols].copy()

        # Drop genes where any sample is NaN (e.g. symbol only in one annotation set)
        before  = len(sub)
        sub     = sub.dropna()
        dropped = before - len(sub)
        if dropped:
            print(f"  Note: dropped {dropped:,} genes with NaN in selected samples")

        n_ctrl = len(NEG_CTRL_COLS)   # 4
        n_kd   = len(kd_cols)         # 2
        print(f"  Samples: {n_ctrl} ctrl + {n_kd} KD  |  Genes: {len(sub):,}")

        # ── Run DEG ───────────────────────────────────────────────────────────
        try:
            out    = run_limma_trend(sub, n_ctrl, n_kd)
            method = "limma-trend via InMoose"
        except ImportError:
            print("  [!] InMoose not found — falling back to Welch t-test + BH")
            print("      To install: pip install inmoose")
            out    = run_ttest_fallback(sub, n_ctrl)
            method = "Welch t-test + BH (fallback)"

        # ── Save ──────────────────────────────────────────────────────────────
        out_path = os.path.join(OUT_DIR, f"{gene}_vs_control.csv")
        out.to_csv(out_path, index=False)

        # ── Summary ───────────────────────────────────────────────────────────
        n_up   = ((out["pvals"] < 0.05) & (out["logfoldchanges"] > 0)).sum()
        n_down = ((out["pvals"] < 0.05) & (out["logfoldchanges"] < 0)).sum()
        n_fdr  = (out["pvals_adj"] < 0.05).sum()
        top5   = (
            out.nsmallest(5, "pvals")
            [["names", "logfoldchanges", "pvals"]]
            .to_string(index=False)
        )

        print(f"  Method   : {method}")
        print(f"  p < 0.05 : {n_up + n_down:,} total  ({n_up:,} up, {n_down:,} down)")
        print(f"  FDR<0.05 : {n_fdr:,} genes")
        print(f"  Top 5 by p-value:\n{top5}")
        print(f"  Saved -> {out_path}")

    print(f"\n\nDONE. Files written to: {OUT_DIR}")
    print("Next step: run run_1hop_network_export.py with --de-dir pointing to this folder.")


if __name__ == "__main__":
    main()