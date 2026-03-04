"""Compute scRNA DEGs for selected perturbations vs control using Scanpy Wilcoxon."""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

logger = logging.getLogger(__name__)


def parse_args():
    ap = argparse.ArgumentParser(
        description="scRNA DEGs for selected perturbations vs control (Scanpy Wilcoxon).",
    )
    ap.add_argument("--adata", required=True, help="Path to AnnData .h5ad (post-QC)")
    ap.add_argument("--out-dir", required=True, help="Output folder for <GENE>_vs_control.csv")
    ap.add_argument(
        "--genes",
        nargs="+",
        required=True,
        help="Perturbation genes to run DE for (e.g. ITGB1BP1 CCM2 ...)",
    )
    ap.add_argument(
        "--perturb-col",
        default="Gene",
        help="adata.obs column containing perturbation labels",
    )
    ap.add_argument(
        "--controls",
        nargs="+",
        default=["negative-control", "safe-targeting"],
        help="Control labels in adata.obs[perturb-col]",
    )
    ap.add_argument(
        "--strip-suffix",
        default="-TSS2",
        help="If set, removes this suffix from perturbation labels",
    )
    ap.add_argument(
        "--min-cells",
        type=int,
        default=1,
        help="Minimum cells required for a perturbation to run DE (controls always kept)",
    )
    ap.add_argument("--endo-list", required=True, help="CSV of endothelial gene universe")
    ap.add_argument("--endo-col", default="gene", help="Column name in endo list CSV")
    ap.add_argument(
        "--top-variable-fraction",
        type=float,
        default=0.0,
        help="If >0, keep top fraction most variable genes within endothelial set (e.g. 0.5).",
    )
    ap.add_argument(
        "--normalize-log1p",
        action="store_true",
        help="Run normalize_total + log1p before DE (only if .X is raw counts)",
    )
    ap.add_argument("--target-sum", type=float, default=1e4)
    ap.add_argument(
        "--symbol-col",
        default=None,
        help="If var_names are Ensembl IDs, provide adata.var column with gene symbols (e.g. gene_name).",
    )
    return ap.parse_args()


def load_endo_set(path: str, col: str) -> set[str]:
    df = pd.read_csv(path, low_memory=False)
    if col not in df.columns:
        raise ValueError(f"Endo list missing column '{col}'. Columns: {df.columns.tolist()}")
    return set(df[col].astype(str).str.strip().replace("", pd.NA).dropna().tolist())


def compute_gene_variance(X):
    if sparse.issparse(X):
        mean_sq = np.array(X.power(2).mean(axis=0)).ravel()
        mean = np.array(X.mean(axis=0)).ravel()
        return mean_sq - mean ** 2
    return np.array(X).var(axis=0)


def write_empty_deg(path: str):
    cols = ["names", "scores", "logfoldchanges", "pvals", "pvals_adj", "pct_nz_group"]
    pd.DataFrame({c: [] for c in cols}).to_csv(path, index=False)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    endo_set = load_endo_set(args.endo_list, args.endo_col)

    adata = sc.read_h5ad(args.adata)

    if args.symbol_col is not None:
        if args.symbol_col not in adata.var.columns:
            raise ValueError(
                f"--symbol-col '{args.symbol_col}' not in adata.var. "
                f"Columns: {adata.var.columns.tolist()}"
            )
        adata = adata.copy()
        adata.var_names = adata.var[args.symbol_col].astype(str).str.strip().values

    if args.perturb_col not in adata.obs.columns:
        raise ValueError(
            f"adata.obs missing '{args.perturb_col}'. Columns: {adata.obs.columns.tolist()}"
        )

    labels = adata.obs[args.perturb_col].astype(str).str.strip()
    if args.strip_suffix:
        labels = labels.str.replace(args.strip_suffix, "", regex=False)
    adata.obs["Gene_Clean"] = labels

    controls = {str(x).strip() for x in args.controls}
    requested = [str(x).strip() for x in args.genes if str(x).strip()]
    keep_groups = set(requested) | controls

    adata = adata[adata.obs["Gene_Clean"].isin(keep_groups)].copy()

    adata.obs["Gene_Group"] = adata.obs["Gene_Clean"].where(
        ~adata.obs["Gene_Clean"].isin(controls), "control"
    )

    counts = adata.obs["Gene_Clean"].value_counts().to_dict()
    runnable = [g for g in requested if counts.get(g, 0) >= args.min_cells]

    if not runnable:
        for g in requested:
            write_empty_deg(os.path.join(args.out_dir, f"{g}_vs_control.csv"))
        logger.info("No perturbations met --min-cells. Wrote empty DEG CSVs.")
        return

    if args.normalize_log1p:
        sc.pp.normalize_total(adata, target_sum=args.target_sum)
        sc.pp.log1p(adata)

    endo_mask = adata.var_names.astype(str).isin(endo_set)
    if int(endo_mask.sum()) == 0:
        raise RuntimeError(
            "No overlap between adata.var_names and endothelial gene list. "
            "If var_names are Ensembl IDs, pass --symbol-col (e.g. gene_name)."
        )
    adata = adata[:, endo_mask].copy()

    if args.top_variable_fraction and args.top_variable_fraction > 0:
        if not (0.0 < args.top_variable_fraction <= 1.0):
            raise ValueError("--top-variable-fraction must be in (0,1].")
        var = compute_gene_variance(adata.X)
        cutoff = np.quantile(var, 1.0 - args.top_variable_fraction)
        adata = adata[:, var >= cutoff].copy()

    logger.info("Running DE for %d perturbations vs control (wilcoxon)...", len(runnable))
    sc.tl.rank_genes_groups(
        adata,
        groupby="Gene_Group",
        reference="control",
        method="wilcoxon",
        pts=True,
    )
    logger.info("DE complete.")

    groups_done = set(adata.uns["rank_genes_groups"]["names"].dtype.names)

    for g in requested:
        out_path = os.path.join(args.out_dir, f"{g}_vs_control.csv")
        if g not in groups_done:
            write_empty_deg(out_path)
            continue
        df = sc.get.rank_genes_groups_df(adata, group=g)
        df.to_csv(out_path, index=False)

    logger.info("Done. Wrote per-gene DEGs to: %s", args.out_dir)


if __name__ == "__main__":
    main()
