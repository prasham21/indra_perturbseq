"""Legacy script: inspect target genes."""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)

TV_PATH = "target_validation_expanded.csv"
DEG_DIR = "de_results_per_gene"
OUT_CSV = "target_parent_counts.csv"

PVAL_THRESHOLD = 0.05

IGNORE = {"TP53", "CDK1NA", "CDKN1A"}



def main():
    ap = argparse.ArgumentParser()

    tv = pd.read_csv(TV_PATH)
    sources = (
        tv.loc[tv["analysis_flag"] == "Use_for_analysis", "Gene"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    parents = defaultdict(set)
    missing_deg = 0

    for src in sources:
        deg_path = os.path.join(DEG_DIR, f"{src}_vs_control.csv")
        if not os.path.exists(deg_path):
            missing_deg += 1
            continue

        deg = pd.read_csv(deg_path, usecols=["names", "pvals"])
        deg = deg.dropna(subset=["names", "pvals"])
        deg["pvals"] = pd.to_numeric(deg["pvals"], errors="coerce")
        deg = deg[deg["pvals"] < PVAL_THRESHOLD]

        for tgt in deg["names"].astype(str).str.strip().unique():
            if tgt and tgt.upper() not in IGNORE:
                parents[tgt].add(src)

    out = pd.DataFrame(
        {
            "target": list(parents.keys()),
            "n_parents": [len(v) for v in parents.values()],
            "parents": [";".join(sorted(v)) for v in parents.values()],
        }
    ).sort_values(["n_parents", "target"], ascending=[False, True])

    out.to_csv(OUT_CSV, index=False)

    arr = out["n_parents"].to_numpy()
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    iqr = q3 - q1
    hi_cut = q3 + 1.5 * iqr

    logger.info(f"Sources used: {len(sources)} (missing DEG files: {missing_deg})")
    logger.info(f"Targets: {len(out)}")
    logger.info(f"n_parents min/median/mean/max: {arr.min()} / {np.median(arr):.1f} / {arr.mean():.2f} / {arr.max()}")
    logger.info(f"High-outlier cutoff (Q3+1.5*IQR): {hi_cut:.1f}")
    logger.info(f"Targets with 1 parent: {int((out['n_parents'] == 1).sum())}")
    logger.info(f"Targets with 2 parents: {int((out['n_parents'] == 2).sum())}")

    logger.info("Top 20 most parents:")
    logger.info(out.head(20)[["target", "n_parents"]].to_string(index=False))

    high_outliers = out[out["n_parents"] > hi_cut]
    logger.info(f"High outliers: {len(high_outliers)}")
    if len(high_outliers) > 0:
        logger.info(high_outliers.head(30)[["target", "n_parents"]].to_string(index=False))

    logger.info("Bottom 20 least parents:")
    logger.info(out.tail(20)[["target", "n_parents"]].to_string(index=False))

    logger.info("CSV inspection")
    df = pd.read_csv(OUT_CSV)
    logger.info(f"shape: {df.shape}")
    logger.info(f"columns: {df.columns.tolist()}")
    logger.info(f"dtypes:\n{df.dtypes}")
    logger.info(f"missing:\n{df.isna().sum()}")
    logger.info(f"head:\n{df.head(5).to_string(index=False)}")

    lens = df["parents"].fillna("").apply(lambda s: 0 if s.strip() == "" else len([x for x in s.split(";") if x.strip()]))
    bad = df[lens != df["n_parents"]]
    logger.info(f"rows where len(parents)!=n_parents: {len(bad)}")



if __name__ == "__main__":
    main()
