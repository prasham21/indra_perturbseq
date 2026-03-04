"""Legacy script: map_mesh_terms."""
from __future__ import annotations

import argparse
import pandas as pd
import re

import logging


logger = logging.getLogger(__name__)
REFERENCE_FILE = "comprehensive_mesh_list_EXPANDED_.csv"
HOP_FILE = "indra_2hop_with_mesh_terms.csv"
OUTPUT_FILE = "_new_2hop_mesh_filtered.csv"

def filter_mesh_terms(text, valid_ids):
    """Keep only valid MeSH terms present in the reference list."""
    if pd.isna(text):
        return ""
    pairs = re.findall(r'[^,]+?\(D\d{5,10}\)', text)
    filtered = []
    for p in pairs:
        match = re.search(r'\((D\d{5,10})\)', p)
        if match and match.group(1) in valid_ids:
            filtered.append(p.strip())
    return ", ".join(filtered)



def main():
    ap = argparse.ArgumentParser(description="Filter MeSH terms against reference list.")
    ap.add_argument("--reference", default=REFERENCE_FILE, help="Reference MeSH list CSV.")
    ap.add_argument("--input", default=HOP_FILE, help="Input hop CSV with MeSH annotations.")
    ap.add_argument("--output", default=OUTPUT_FILE, help="Output filtered CSV.")
    args = ap.parse_args()

    ref_df = pd.read_csv(args.reference, encoding='utf-8-sig', on_bad_lines='skip')

    ref_df["mesh_id"] = (
        ref_df["mesh_id"]
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.replace(r"[^A-Za-z0-9]", "", regex=True)
    )

    valid_ids = set(ref_df["mesh_id"][ref_df["mesh_id"].str.startswith("D")])
    logger.info("Loaded %d valid MeSH IDs from reference list", len(valid_ids))

    hop_df = pd.read_csv(args.input)
    logger.info("Loaded %d rows from %s", len(hop_df), args.input)

    mesh_columns = ["Annotated MeSH terms hop1", "Annotated MeSH terms hop2"]

    for target_col in mesh_columns:
        if target_col in hop_df.columns:
            logger.info("Filtering column: %s", target_col)
            hop_df[target_col] = hop_df[target_col].apply(lambda t: filter_mesh_terms(t, valid_ids))
        else:
            logger.info("Column '%s' not found, skipping.", target_col)

    drop_cols = [
        c for c in hop_df.columns
        if any(s in c for s in [".1", ".2", ".3", "(copy)", "n_kept", "n_total", "n_dropped"])
    ]
    if drop_cols:
        hop_df.drop(columns=drop_cols, inplace=True)
        logger.info(" Dropped extra columns: %s", drop_cols)
    else:
        logger.info(" No extra or duplicate columns found to drop.")

    hop_df.to_csv(args.output, index=False)
    logger.info("Clean filtered 2-hop file saved to: %s", args.output)



if __name__ == "__main__":
    main()
