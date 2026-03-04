"""Compute FPR from false-positive path results relative to DEG negative pairs."""
from __future__ import annotations

import argparse
import logging
import os

import pandas as pd
from indra.databases import hgnc_client

logger = logging.getLogger(__name__)

def normalize_hgnc_symbol(symbol: str):
    if symbol is None:
        return None
    s = str(symbol).strip()
    if not s:
        return None
    hid = hgnc_client.get_current_hgnc_id(s.upper())
    if not hid:
        return s.upper()
    if isinstance(hid, (list, tuple, set)):
        hid = sorted(list(hid))[0] if hid else None
        if not hid:
            return s.upper()
    name = hgnc_client.get_hgnc_name(hid) if hid else None
    return name or s.upper()


def pick_sig_column(df: pd.DataFrame) -> str:
    for c in ["pvals", "pval", "p_value", "p_val", "pvals_adj", "padj", "qval", "fdr", "p_adj"]:
        if c in df.columns:
            return c
    raise ValueError(f"No p-value column found. Columns: {df.columns.tolist()}")


def load_pairs(csv_path: str):
    df = pd.read_csv(csv_path, low_memory=False)
    if not {"source", "target"}.issubset(df.columns):
        raise ValueError(f"{csv_path} missing source/target columns. Has: {df.columns.tolist()}")
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df = df[df["source"] != df["target"]].copy()
    return set(zip(df["source"], df["target"]))


def total_negative_pairs_tested():
    genes_df = pd.read_csv(GENES_CSV, low_memory=False)
    if FILTER_COLUMN in genes_df.columns:
        genes_df = genes_df[genes_df[FILTER_COLUMN] == FILTER_VALUE].copy()
    sources_raw = [str(x).strip() for x in genes_df[GENE_COL].dropna().tolist() if str(x).strip()]

    total_neg = 0
    per_source = []

    for raw_src in sources_raw:
        src = normalize_hgnc_symbol(raw_src)
        if not src:
            continue

        deg_path = os.path.join(DE_DIR, f"{raw_src}_vs_control.csv")
        if not os.path.exists(deg_path):
            continue

        d = pd.read_csv(deg_path, low_memory=False)
        if "names" not in d.columns:
            continue

        pcol = pick_sig_column(d)
        d[pcol] = pd.to_numeric(d[pcol], errors="coerce")

        all_set = set()
        for t in d["names"].dropna().astype(str).tolist():
            tn = normalize_hgnc_symbol(t)
            if tn:
                all_set.add(tn)

        pos_set = set()
        dd = d[d[pcol] < P_THRESHOLD].copy()
        for t in dd["names"].dropna().astype(str).tolist():
            tn = normalize_hgnc_symbol(t)
            if tn:
                pos_set.add(tn)

        all_set.discard(src)
        pos_set.discard(src)

        neg_set = all_set - pos_set
        nneg = len(neg_set)
        total_neg += nneg

        per_source.append((src, len(all_set), len(pos_set), nneg))

    return total_neg, pd.DataFrame(per_source, columns=["source", "universe_targets", "positive_targets", "negative_targets"])


def add(hop, fp_pairs):
    rows.append({
        "dataset": "real",
        "hop": hop,
        "fp_unique_pairs": len(fp_pairs),
        "total_negative_pairs_tested": TOTAL_NEG,
        "fpr": (len(fp_pairs) / TOTAL_NEG) if TOTAL_NEG else 0.0,
    })




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-1hop-paths-p005", default="fp_1hop_paths_p005.csv", help="Path: fp_1hop_paths_p005.csv")
    ap.add_argument("--fp-2hop-paths-p005", default="fp_2hop_paths_p005.csv", help="Path: fp_2hop_paths_p005.csv")
    ap.add_argument("--target-validation-expanded", default="target_validation_expanded.csv", help="Path: target_validation_expanded.csv")
    ap.add_argument("--de-results-per-gene", default="de_results_per_gene/", help="Path: de_results_per_gene/")
    ap.add_argument("--real-fp-fpr-summary-p005", default="real_fp_fpr_summary_p005.csv", help="Path: real_fp_fpr_summary_p005.csv")
    args = ap.parse_args()

    REAL_FP_1HOP = "fp_1hop_paths_p005.csv"
    REAL_FP_2HOP = "fp_2hop_paths_p005.csv"

    GENES_CSV = "target_validation_expanded.csv"
    DE_DIR = "de_results_per_gene/"
    P_THRESHOLD = 0.05

    FILTER_COLUMN = "analysis_flag"
    FILTER_VALUE = "Use_for_analysis"
    GENE_COL = "Gene"

    OUT_SUMMARY = "real_fp_fpr_summary_p005.csv"


    logger.info("Loading REAL FP result CSVs...")
    real1 = load_pairs(REAL_FP_1HOP)
    real2 = load_pairs(REAL_FP_2HOP)
    real_u = real1 | real2

    logger.info("Computing total negative pairs tested from DEG control files...")
    TOTAL_NEG, per_src_df = total_negative_pairs_tested()
    logger.info("Total negative pairs tested (summed over sources): %d", TOTAL_NEG)

    rows = []


    add("1hop", real1)
    add("2hop", real2)
    add("<=2hop_union", real_u)

    out = pd.DataFrame(rows)
    logger.info("SUMMARY\n%s", out.to_string(index=False))

    out.to_csv(OUT_SUMMARY, index=False)
    logger.info("Wrote: %s", OUT_SUMMARY)

    per_src_out = OUT_SUMMARY.replace(".csv", "_per_source_neg_counts.csv")
    per_src_df.to_csv(per_src_out, index=False)
    logger.info("Wrote: %s", per_src_out)



if __name__ == "__main__":
    main()
