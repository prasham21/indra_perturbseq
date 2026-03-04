"""Legacy script: top150_filtered."""
from __future__ import annotations

import argparse
import pandas as pd

import logging


logger = logging.getLogger(__name__)
INPUT_FILE = "1hop_mesh_filtered__.csv"
OUTPUT_FILE = "1hop_top150_unique_pairs.csv"



def main():
    ap = argparse.ArgumentParser(description="Filter top 150 unique source-target pairs.")
    ap.add_argument("--input", default=INPUT_FILE, help="Input MeSH-filtered CSV.")
    ap.add_argument("--output", default=OUTPUT_FILE, help="Output top 150 CSV.")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    logger.info("Loaded %d rows from %s", len(df), args.input)

    df_unique = df.sort_values("logfoldchange", ascending=False).drop_duplicates(
        subset=["source", "target"], keep="first"
    )
    logger.info("After deduplication: %d unique (source, target) pairs", len(df_unique))

    df_top150 = df_unique.sort_values("logfoldchange", ascending=False).head(150)

    df_top150.to_csv(args.output, index=False)
    logger.info("Top 150 unique source-target pairs saved to: %s", args.output)
    logger.info("Total unique pairs analyzed: %d", len(df_unique))
    logger.info("Top 150 logfoldchange range: %.4f to %.4f", df_top150['logfoldchange'].min(), df_top150['logfoldchange'].max())



if __name__ == "__main__":
    main()
