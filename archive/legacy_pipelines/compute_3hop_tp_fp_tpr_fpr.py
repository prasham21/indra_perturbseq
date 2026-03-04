"""Compute TP/FP/TN/FN and TPR/FPR for 3-hop results across p-value thresholds."""
from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from indra.databases import hgnc_client

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001]


def normalize_hgnc_symbol(symbol: str):
    if symbol is None or (isinstance(symbol, float) and pd.isna(symbol)):
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


def pick_pcol(df: pd.DataFrame) -> str:
    for c in ["pvals", "pval", "p_value", "p_val", "pvals_adj", "padj", "qval", "fdr", "p_adj"]:
        if c in df.columns:
            return c
    raise ValueError(f"No p-value column found. Columns: {df.columns.tolist()}")


def load_filtered_sources(target_validation_path: str, source_col: str, flag_col: str, flag_value: str):
    tv = pd.read_csv(target_validation_path, low_memory=False)
    for c in (source_col, flag_col):
        if c not in tv.columns:
            raise ValueError(f"Missing column '{c}' in {target_validation_path}. Columns: {tv.columns.tolist()}")

    tv = tv[tv[flag_col] == flag_value].copy()
    raw = [str(x).strip() for x in tv[source_col].dropna().tolist() if str(x).strip()]

    out = []
    seen = set()
    for r in raw:
        n = normalize_hgnc_symbol(r)
        if n and n not in seen:
            out.append(n)
            seen.add(n)
    return out


def load_explained_pairs(path_csv: str, source_col: str, target_col: str):
    df = pd.read_csv(path_csv, low_memory=False)
    if not {source_col, target_col}.issubset(df.columns):
        raise ValueError(
            f"{path_csv} missing {source_col}/{target_col}. Has: {df.columns.tolist()}"
        )

    s = df[source_col].astype(str).str.strip().map(normalize_hgnc_symbol)
    t = df[target_col].astype(str).str.strip().map(normalize_hgnc_symbol)
    dd = pd.DataFrame({"source": s, "target": t}).dropna()
    dd = dd[dd["source"] != dd["target"]].copy()
    return set(zip(dd["source"].tolist(), dd["target"].tolist()))


def deg_path_for_source(deg_dir: str, src: str) -> str:
    return os.path.join(deg_dir, f"{src}_vs_control.csv")


def build_pmap_for_source(deg_csv: str, src: str) -> dict:
    d = pd.read_csv(deg_csv, low_memory=False)
    if "names" not in d.columns:
        raise ValueError(f"{deg_csv} missing 'names' column. Columns: {d.columns.tolist()}")

    pcol = pick_pcol(d)
    d[pcol] = pd.to_numeric(d[pcol], errors="coerce")

    pmap = {}
    for _, r in d.iterrows():
        p = r[pcol]
        if pd.isna(p):
            continue

        t = normalize_hgnc_symbol(r["names"])
        if not t or t == src:
            continue

        p = float(p)
        if (t not in pmap) or (p < pmap[t]):
            pmap[t] = p
    return pmap


def main():
    ap = argparse.ArgumentParser(
        description="Compute TP/FP/TN/FN + TPR/FPR for 3-hop path CSV.",
    )
    ap.add_argument("--paths-csv", required=True, help="3-hop path CSV")
    ap.add_argument("--target-validation", required=True, help="target_validation_expanded.csv")
    ap.add_argument("--deg-dir", required=True, help="Folder with <SOURCE>_vs_control.csv files")
    ap.add_argument("--out-csv", required=True, help="Output per-threshold stats CSV")

    ap.add_argument("--source-column", default="Gene")
    ap.add_argument("--filter-column", default="analysis_flag")
    ap.add_argument("--filter-value", default="Use_for_analysis")
    ap.add_argument("--path-source-col", default="source")
    ap.add_argument("--path-target-col", default="target")
    ap.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)

    args = ap.parse_args()
    thresholds = np.array(args.thresholds, dtype=float)

    logger.info("Loading filtered sources...")
    sources = load_filtered_sources(
        target_validation_path=args.target_validation,
        source_col=args.source_column,
        flag_col=args.filter_column,
        flag_value=args.filter_value,
    )
    logger.info("Filtered sources (unique, normalized): %d", len(sources))

    logger.info("Loading explained pairs from 3-hop CSV...")
    explained_pairs = load_explained_pairs(
        args.paths_csv, args.path_source_col, args.path_target_col,
    )
    logger.info("Explained unique pairs (3-hop): %d", len(explained_pairs))

    expl_by_src = defaultdict(set)
    for s, t in explained_pairs:
        expl_by_src[s].add(t)

    logger.info("Caching per-source control p-values (from DEG files)...")
    cache = {}
    missing = 0
    for src in sources:
        f = deg_path_for_source(args.deg_dir, src)
        if not os.path.exists(f):
            missing += 1
            continue
        pmap = build_pmap_for_source(f, src)
        if not pmap:
            missing += 1
            continue
        pvals_all = np.fromiter(pmap.values(), dtype=float)
        cache[src] = (pvals_all, pmap)
    logger.info("Cached sources: %d | missing/unusable DEG: %d", len(cache), missing)

    rows = []
    for thr in thresholds:
        tp_all = fp_all = positives_all = negatives_all = 0
        tpr_list = []
        fpr_list = []

        for src, (pvals_all, pmap) in cache.items():
            pos_size = int(np.sum(pvals_all <= thr))
            neg_size = int(np.sum(pvals_all > thr))

            explained_targets = expl_by_src.get(src, set())
            tp = fp = 0
            for t in explained_targets:
                p = pmap.get(t)
                if p is None:
                    continue
                if p <= thr:
                    tp += 1
                else:
                    fp += 1

            tp_all += tp
            fp_all += fp
            positives_all += pos_size
            negatives_all += neg_size

            tpr_list.append((tp / pos_size) if pos_size else np.nan)
            fpr_list.append((fp / neg_size) if neg_size else np.nan)

        fn_all = positives_all - tp_all
        tn_all = negatives_all - fp_all

        tpr_overall = (tp_all / positives_all) if positives_all else np.nan
        fpr_overall = (fp_all / negatives_all) if negatives_all else np.nan
        tpr_avg = float(np.nanmean(tpr_list)) if len(tpr_list) else np.nan
        fpr_avg = float(np.nanmean(fpr_list)) if len(fpr_list) else np.nan

        row = {
            "threshold": float(thr),
            "TP_total": int(tp_all),
            "FP_total": int(fp_all),
            "TN_total": int(tn_all),
            "FN_total": int(fn_all),
            "positives_total": int(positives_all),
            "negatives_total": int(negatives_all),
            "TPR_overall": float(tpr_overall) if not np.isnan(tpr_overall) else np.nan,
            "FPR_overall": float(fpr_overall) if not np.isnan(fpr_overall) else np.nan,
            "TPR_per_source_avg": tpr_avg,
            "FPR_per_source_avg": fpr_avg,
        }
        rows.append(row)

        logger.info(
            "thr %g | TP=%d FP=%d TN=%d FN=%d | TPR=%.4f FPR=%.4f",
            thr, row["TP_total"], row["FP_total"], row["TN_total"], row["FN_total"],
            row["TPR_overall"], row["FPR_overall"],
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    logger.info("Wrote CSV: %s", args.out_csv)
    logger.debug("Per-threshold summary:\n%s", out_df.to_string(index=False))


if __name__ == "__main__":
    main()
