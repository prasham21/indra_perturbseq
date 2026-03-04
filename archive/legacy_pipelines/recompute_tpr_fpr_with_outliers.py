#!/usr/bin/env python3
"""
Compute TP/FP + TPR/FPR across p-value thresholds using:

- Sources = target_validation_expanded.csv filtered to Karen_Flag == Use_for_analysis
- Denominators from each source's <SOURCE>_vs_control.csv (names + pval column), self removed
- Explained pairs = union of (source,target) from one or more INDRA path CSVs (TP/FP/outlier), self removed
- For each threshold t:
    positives_total(t) = sum_s |{x: p_sx <= t}|
    negatives_total(t) = sum_s |{x: p_sx >  t}|
    TP_total(t) = sum_s |{x in explained(s): p_sx <= t}|
    FP_total(t) = sum_s |{x in explained(s): p_sx >  t}|
    TPR_overall(t) = TP_total / positives_total
    FPR_overall(t) = FP_total / negatives_total

Prints per-threshold stats. Optional output CSV.
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from indra.databases import hgnc_client

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


def load_karen_sources(tv_path: str, source_col: str, flag_col: str, flag_value: str):
    tv = pd.read_csv(tv_path, low_memory=False)
    for c in (source_col, flag_col):
        if c not in tv.columns:
            raise ValueError(f"Missing column '{c}' in {tv_path}. Columns: {tv.columns.tolist()}")

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


def load_explained_pairs(paths: list[str], source_col: str, target_col: str):
    explained = set()
    for p in paths:
        df = pd.read_csv(p, low_memory=False)
        if not {source_col, target_col}.issubset(df.columns):
            raise ValueError(f"{p} missing {source_col}/{target_col}. Has: {df.columns.tolist()}")

        s = df[source_col].astype(str).str.strip().map(normalize_hgnc_symbol)
        t = df[target_col].astype(str).str.strip().map(normalize_hgnc_symbol)
        dd = pd.DataFrame({"source": s, "target": t}).dropna()
        dd = dd[dd["source"] != dd["target"]].copy()
        explained |= set(zip(dd["source"].tolist(), dd["target"].tolist()))
    return explained


def deg_path_for_source(de_dir: str, src: str) -> str:
    return os.path.join(de_dir, f"{src}_vs_control.csv")


def build_pmap_for_source(deg_csv: str, src: str) -> dict[str, float]:
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
    ap = argparse.ArgumentParser(description="TP/FP + TPR/FPR over p-value thresholds (Karen sources).")
    ap.add_argument("--tv-path", required=True, help="target_validation_expanded.csv")
    ap.add_argument("--tv-source-col", default="Gene")
    ap.add_argument("--karen-flag-col", default="Karen_Flag")
    ap.add_argument("--karen-flag-value", default="Use_for_analysis")

    ap.add_argument("--de-dir", required=True, help="Folder containing <SOURCE>_vs_control.csv files")
    ap.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="One or more INDRA path CSVs to union (TP/FP/outliers). Must have source/target columns.",
    )
    ap.add_argument("--path-source-col", default="source")
    ap.add_argument("--path-target-col", default="target")

    ap.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    ap.add_argument("--print-table", action="store_true")
    ap.add_argument("--out-csv", default="", help="If set, write per-threshold summary CSV to this path.")

    args = ap.parse_args()
    thresholds = np.array(args.thresholds, dtype=float)

    print("Loading Karen-flagged sources...")
    sources = load_karen_sources(
        tv_path=args.tv_path,
        source_col=args.tv_source_col,
        flag_col=args.karen_flag_col,
        flag_value=args.karen_flag_value,
    )
    print(f"Karen sources (unique, normalized): {len(sources)}")

    print("Loading explained pairs (union across path CSVs)...")
    explained_pairs = load_explained_pairs(args.paths, args.path_source_col, args.path_target_col)
    print(f"Explained unique pairs: {len(explained_pairs):,}")

    expl_by_src = defaultdict(set)
    for s, t in explained_pairs:
        expl_by_src[s].add(t)

    print("Caching per-source control p-values (from DEG files)...")
    cache = {}
    missing = 0
    for src in sources:
        f = deg_path_for_source(args.de_dir, src)
        if not os.path.exists(f):
            missing += 1
            continue
        pmap = build_pmap_for_source(f, src)
        if not pmap:
            missing += 1
            continue
        pvals_all = np.fromiter(pmap.values(), dtype=float)
        cache[src] = (pvals_all, pmap)

    print(f"Cached sources: {len(cache)} | missing/unusable DEG: {missing}")
    print()

    rows = []
    for thr in thresholds:
        TP_all = FP_all = POS_all = NEG_all = 0
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

            TP_all += tp
            FP_all += fp
            POS_all += pos_size
            NEG_all += neg_size

            tpr_list.append((tp / pos_size) if pos_size else np.nan)
            fpr_list.append((fp / neg_size) if neg_size else np.nan)

        tpr_overall = (TP_all / POS_all) if POS_all else np.nan
        fpr_overall = (FP_all / NEG_all) if NEG_all else np.nan
        tpr_avg = float(np.nanmean(tpr_list)) if len(tpr_list) else np.nan
        fpr_avg = float(np.nanmean(fpr_list)) if len(fpr_list) else np.nan

        row = {
            "threshold": float(thr),
            "TP_total": int(TP_all),
            "FP_total": int(FP_all),
            "positives_total": int(POS_all),
            "negatives_total": int(NEG_all),
            "TPR_overall": float(tpr_overall) if not np.isnan(tpr_overall) else np.nan,
            "FPR_overall": float(fpr_overall) if not np.isnan(fpr_overall) else np.nan,
            "TPR_per_source_avg": tpr_avg,
            "FPR_per_source_avg": fpr_avg,
        }
        rows.append(row)

        print(
            f"thr {thr:g} | "
            f"Overall: TPR={row['TPR_overall']:.4f}, FPR={row['FPR_overall']:.4f} | "
            f"Per-source avg: TPR={row['TPR_per_source_avg']:.4f}, FPR={row['FPR_per_source_avg']:.4f}"
        )

    out_df = pd.DataFrame(rows)

    if args.out_csv:
        out_df.to_csv(args.out_csv, index=False)
        print(f"\nWrote CSV: {args.out_csv}")

    if args.print_table:
        print("\nPer-threshold summary:")
        print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()