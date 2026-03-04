"""Bulk RNA-seq DEG analysis for 6 source genes via limma-trend or t-test fallback."""
from __future__ import annotations

import argparse

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


NEG_CTRL_COLS = [
    "TeloHAEC_B6_neg_ctrl_KD",
    "TeloHAEC_D2_neg_ctrl_KD",
    "TeloHAEC_D8_neg_ctrl_KD",
    "TeloHAEC_F4_neg_ctrl_KD",
]

GENE_TO_KD_COLS = {
    "MAP2K5": ["TeloHAEC_A11_MAP2K5_KD", "TeloHAEC_B5_MAP2K5_KD"],
    "MAP3K3": ["TeloHAEC_A4_MAP3K3_KD", "TeloHAEC_C3_MAP3K3_KD"],
    "CCM2": ["TeloHAEC_C2_CCM2_KD", "TeloHAEC_F6_CCM2_KD"],
    "ITGB1BP1": ["TeloHAEC_C7_ITGB1BP1_KD", "TeloHAEC_G8_ITGB1BP1_KD"],
    "KLF2": ["TeloHAEC_C9_KLF2_KD", "TeloHAEC_G10_KLF2_KD"],
    "PDCD10": ["TeloHAEC_D11_PDCD10_KD", "TeloHAEC_G1_PDCD10_KD"],
}


def run_limma_trend(expr_df: pd.DataFrame, n_ctrl: int, n_kd: int) -> pd.DataFrame:
    from inmoose.limma import lmFit, eBayes, topTable

    n_samples = n_ctrl + n_kd

    design = np.column_stack([
        np.ones(n_samples),
        np.array([0] * n_ctrl + [1] * n_kd, dtype=float),
    ])

    fit = lmFit(expr_df, design)
    fit = eBayes(fit, trend=True)

    coef_label = fit.coefficients.columns[1]
    logger.debug("Using coefficient: %r", coef_label)

    result = topTable(
        fit,
        coef=coef_label,
        number=np.inf,
        adjust_method="fdr_bh",
    )

    return pd.DataFrame({
        "names": result.index,
        "logfoldchanges": result["log2FoldChange"].values,
        "pvals": result["pvalue"].values,
        "pvals_adj": result["adj_pvalue"].values,
    })


def run_ttest_fallback(expr_df: pd.DataFrame, n_ctrl: int) -> pd.DataFrame:
    """Welch t-test + BH correction fallback when InMoose is unavailable."""
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests

    mat = expr_df.values
    ctrl = mat[:, :n_ctrl]
    kd = mat[:, n_ctrl:]

    lfc = kd.mean(axis=1) - ctrl.mean(axis=1)
    _, pvals = ttest_ind(kd, ctrl, axis=1, equal_var=False)
    pvals = np.where(np.isnan(pvals), 1.0, pvals)
    _, padj, _, _ = multipletests(pvals, method="fdr_bh")

    return pd.DataFrame({
        "names": expr_df.index,
        "logfoldchanges": lfc,
        "pvals": pvals,
        "pvals_adj": padj,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpm-file", default="GSE210522_All_bulk_RNAseq_TPM_by_Symbol.txt", help="Path for args.tpm_file.")
    ap.add_argument("--out-dir", default="de_results_bulk_6genes", help="Path for args.out_dir.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    logger.info("Loading TPM matrix...")
    tpm = pd.read_csv(args.tpm_file, sep="\t", index_col="Symbol")
    tpm = tpm.apply(pd.to_numeric, errors="coerce")
    logger.info("Loaded: %d genes x %d samples", tpm.shape[0], tpm.shape[1])

    log_tpm = np.log2(tpm + 1)

    for gene, kd_cols in GENE_TO_KD_COLS.items():
        logger.info("Processing %s", gene)

        all_cols = NEG_CTRL_COLS + kd_cols
        sub = log_tpm[all_cols].copy()

        before = len(sub)
        sub = sub.dropna()
        dropped = before - len(sub)
        if dropped:
            logger.info("  Dropped %d genes with NaN in selected samples", dropped)

        n_ctrl = len(NEG_CTRL_COLS)
        n_kd = len(kd_cols)
        logger.info("  Samples: %d ctrl + %d KD  |  Genes: %d", n_ctrl, n_kd, len(sub))

        try:
            out = run_limma_trend(sub, n_ctrl, n_kd)
            method = "limma-trend via InMoose"
        except ImportError:
            logger.warning("  InMoose not found -- falling back to Welch t-test + BH")
            out = run_ttest_fallback(sub, n_ctrl)
            method = "Welch t-test + BH (fallback)"

        out_path = os.path.join(args.out_dir, f"{gene}_vs_control.csv")
        out.to_csv(out_path, index=False)

        n_up = ((out["pvals"] < 0.05) & (out["logfoldchanges"] > 0)).sum()
        n_down = ((out["pvals"] < 0.05) & (out["logfoldchanges"] < 0)).sum()
        n_fdr = (out["pvals_adj"] < 0.05).sum()
        top5 = (
            out.nsmallest(5, "pvals")[["names", "logfoldchanges", "pvals"]]
            .to_string(index=False)
        )

        logger.info("  Method   : %s", method)
        logger.info("  p < 0.05 : %d total  (%d up, %d down)", n_up + n_down, n_up, n_down)
        logger.info("  FDR<0.05 : %d genes", n_fdr)
        logger.info("  Top 5 by p-value:\n%s", top5)
        logger.info("  Saved -> %s", out_path)

    logger.info("Done. Files written to: %s", args.out_dir)


if __name__ == "__main__":
    main()
