"""Calculate TP/FP/FN/TN and related metrics for bulk RNA validation genes."""
from __future__ import annotations

import argparse

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)



GENES = ["ITGB1BP1", "CCM2", "PDCD10", "MAP3K3", "MAP2K5", "KLF2"]
FDR_THRESHOLD = 0.05
ALLOWED_HOPS = {1, 2}


def load_endo_set(path: str) -> set[str]:
    df = pd.read_csv(path, low_memory=False)
    if "gene" not in df.columns:
        raise ValueError(f"Endothelial list must have 'gene' column. Got: {df.columns.tolist()}")
    s = df["gene"].astype(str).str.strip()
    return set(x for x in s.tolist() if x)


def load_predicted_targets(indra_csv: str, endo_set: set[str]) -> tuple[set[str], pd.DataFrame]:
    indra = pd.read_csv(indra_csv, low_memory=False)
    if "target" not in indra.columns or "hop" not in indra.columns:
        raise ValueError(
            f"INDRA file must include 'target' and 'hop'. Columns: {indra.columns.tolist()}"
        )

    indra["target"] = indra["target"].astype(str).str.strip()
    indra["hop"] = pd.to_numeric(indra["hop"], errors="coerce")

    indra = indra[indra["hop"].isin(ALLOWED_HOPS)].copy()
    indra = indra[indra["target"].isin(endo_set)].copy()

    minhop = (
        indra.dropna(subset=["hop"])
        .groupby("target", as_index=False)["hop"]
        .min()
        .rename(columns={"hop": "min_hop"})
    )
    predicted = set(minhop["target"].tolist())
    return predicted, minhop


def load_deg(deg_csv: str) -> pd.DataFrame:
    deg = pd.read_csv(deg_csv, low_memory=False)
    required = {"names", "logfoldchanges", "pvals", "pvals_adj"}
    missing = required - set(deg.columns)
    if missing:
        raise ValueError(
            f"DEG file missing columns {sorted(missing)}. Columns: {deg.columns.tolist()}"
        )

    deg = deg.copy()
    deg["names"] = deg["names"].astype(str).str.strip()
    for c in ["logfoldchanges", "pvals", "pvals_adj"]:
        deg[c] = pd.to_numeric(deg[c], errors="coerce")
    return deg


def safe_rate(num: int, den: int) -> float:
    return (num / den) if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indra-dir", default="validation_6genes_hops", help="Path for args.indra_dir.")
    ap.add_argument("--deg-dir", default="de_results_bulk_6genes", help="Path for args.deg_dir.")
    ap.add_argument("--endo-list", default="endothelial_present_plus_manual.csv", help="Path for args.endo_list.")
    ap.add_argument("--out-dir", default="validation_6genes_comparison_hops12", help="Path for args.out_dir.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    endo_set = load_endo_set(args.endo_list)

    summary_rows = []

    for g in GENES:
        indra_path = os.path.join(args.indra_dir, f"{g}_all_hops.csv")
        deg_path = os.path.join(args.deg_dir, f"{g}_vs_control.csv")

        if not os.path.exists(indra_path):
            raise FileNotFoundError(f"Missing INDRA file: {indra_path}")
        if not os.path.exists(deg_path):
            raise FileNotFoundError(f"Missing DEG file: {deg_path}")

        predicted, minhop_df = load_predicted_targets(indra_path, endo_set)
        deg = load_deg(deg_path)

        universe = set(deg["names"].tolist())
        universe = {x for x in universe if x in endo_set}

        empirical_sig = set(deg.loc[deg["pvals_adj"] < FDR_THRESHOLD, "names"].tolist())
        empirical_sig = {x for x in empirical_sig if x in endo_set}

        tp = predicted & empirical_sig
        fp = predicted - empirical_sig
        fn = empirical_sig - predicted
        tn = universe - (predicted | empirical_sig)

        tpr = safe_rate(len(tp), len(tp) + len(fn))
        fnr = safe_rate(len(fn), len(tp) + len(fn))
        fpr = safe_rate(len(fp), len(fp) + len(tn))
        precision = safe_rate(len(tp), len(tp) + len(fp))

        deg_for_join = deg.rename(columns={"names": "target"})

        out = minhop_df.copy()
        out["source"] = g

        out = out.merge(
            deg_for_join[["target", "logfoldchanges", "pvals", "pvals_adj"]],
            on="target",
            how="left",
        )

        out["is_empirical_sig_fdr005"] = out["pvals_adj"] < FDR_THRESHOLD
        out["classification"] = out["is_empirical_sig_fdr005"].map({True: "TP", False: "FP"})

        fn_df = deg_for_join.loc[
            (deg_for_join["pvals_adj"] < FDR_THRESHOLD)
            & (~deg_for_join["target"].isin(predicted))
            & (deg_for_join["target"].isin(endo_set)),
            ["target", "logfoldchanges", "pvals", "pvals_adj"],
        ].copy()

        fn_df["source"] = g
        fn_df["min_hop"] = pd.NA
        fn_df["is_empirical_sig_fdr005"] = True
        fn_df["classification"] = "FN"

        final = pd.concat([out, fn_df], ignore_index=True)

        desired_cols = [
            "source", "target", "min_hop",
            "logfoldchanges", "pvals", "pvals_adj",
            "is_empirical_sig_fdr005", "classification",
        ]
        extra_cols = [c for c in final.columns if c not in desired_cols]
        final = final[desired_cols + extra_cols]

        final_out_path = os.path.join(
            args.out_dir, f"{g}_descendants_with_empirical_stats_frd005_hops_1_2.csv"
        )
        final.to_csv(final_out_path, index=False)

        summary_rows.append({
            "gene": g,
            "hops_used": "1+2",
            "fdr_threshold": FDR_THRESHOLD,
            "universe_size": len(universe),
            "n_predicted_targets": len(predicted),
            "n_empirical_sig_fdr": len(empirical_sig),
            "TP": len(tp),
            "FP": len(fp),
            "FN": len(fn),
            "TN": len(tn),
            "TPR": tpr,
            "FNR": fnr,
            "FPR": fpr,
            "precision": precision,
            "per_gene_output_csv": final_out_path,
        })

        logger.info("[%s] wrote: %s", g, final_out_path)

    summary = pd.DataFrame(summary_rows).sort_values("gene")
    summary_out = os.path.join(args.out_dir, "validation_6genes_metrics_fdr005_hops1_2.csv")
    summary.to_csv(summary_out, index=False)

    logger.info("Wrote summary: %s", summary_out)
    logger.debug("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
