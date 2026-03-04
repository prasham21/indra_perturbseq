"""Legacy script: combine multi-hop INDRA results into standardized top-100 CSV."""
from __future__ import annotations

import argparse
import os

import pandas as pd

import logging


logger = logging.getLogger(__name__)

STANDARD_COLUMNS = [
    "source", "target", "hop_number",
    "intermediate_1", "intermediate_2", "intermediate_3",
    "stmt_type_1", "edge1_evidence_text", "edge1_pmids",
    "stmt_type_2", "edge2_evidence_text", "edge2_pmids",
    "stmt_type_3", "edge3_evidence_text", "edge3_pmids",
    "stmt_type_4", "edge4_evidence_text", "edge4_pmids",
    "logfoldchange", "pval",
    "belief_1", "belief_2", "belief_3", "belief_4",
    "evidence_1", "evidence_2", "evidence_3", "evidence_4",
]

COLUMN_MAPPING = {
    "intermediate_1": "intermediate_1",
    "intermediate_2": "intermediate_2",
    "intermediate_3": "intermediate_3",
    "stmt_type_1": "stmt_type_1",
    "stmt_type_2": "stmt_type_2",
    "stmt_type_3": "stmt_type_3",
    "stmt_type_4": "stmt_type_4",
    "edge1_evidence_text": "edge1_evidence_text",
    "edge2_evidence_text": "edge2_evidence_text",
    "edge3_evidence_text": "edge3_evidence_text",
    "edge4_evidence_text": "edge4_evidence_text",
    "edge1_pmids": "edge1_pmids",
    "edge2_pmids": "edge2_pmids",
    "edge3_pmids": "edge3_pmids",
    "edge4_pmids": "edge4_pmids",
    "belief_1": "belief_1",
    "belief_2": "belief_2",
    "belief_3": "belief_3",
    "belief_4": "belief_4",
    "evidence_1": "evidence_1",
    "evidence_2": "evidence_2",
    "evidence_3": "evidence_3",
    "evidence_4": "evidence_4",
}


def load_and_standardize(hop_files: dict[str, str]) -> pd.DataFrame:
    """Load hop CSVs, standardize columns, and concatenate."""
    dfs = []
    for hop, path in hop_files.items():
        if not os.path.exists(path):
            logger.warning("Missing file for %s: %s", hop, path)
            continue

        df = pd.read_csv(path)
        df["hop_number"] = hop

        missing_cols = [c for c in ["source", "target", "logfoldchange", "pval"] if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{hop} file missing columns: {missing_cols}")

        standardized_df = pd.DataFrame()
        standardized_df["source"] = df["source"]
        standardized_df["target"] = df["target"]
        standardized_df["hop_number"] = df["hop_number"]
        standardized_df["logfoldchange"] = df["logfoldchange"]
        standardized_df["pval"] = df["pval"]

        for col in STANDARD_COLUMNS:
            if col not in standardized_df.columns:
                standardized_df[col] = ""

        for orig_col, std_col in COLUMN_MAPPING.items():
            if orig_col in df.columns:
                standardized_df[std_col] = df[orig_col]

        dfs.append(standardized_df)

    if not dfs:
        raise FileNotFoundError("No valid hop files found!")

    return pd.concat(dfs, ignore_index=True, sort=False)


def main():
    ap = argparse.ArgumentParser(description="Combine multi-hop INDRA results into top-100 CSV.")
    ap.add_argument("--hop1", default="", help="1-hop CSV path.")
    ap.add_argument("--hop2", default="", help="2-hop CSV path.")
    ap.add_argument("--hop3", default="", help="3-hop CSV path.")
    ap.add_argument("--hop4", default="", help="4-hop CSV path.")
    ap.add_argument("--output", required=True, help="Output CSV path.")
    ap.add_argument("--duplicates-output", default="", help="Optional path for removed duplicates.")
    args = ap.parse_args()

    hop_files = {}
    if args.hop1:
        hop_files["1hop"] = args.hop1
    if args.hop2:
        hop_files["2hop"] = args.hop2
    if args.hop3:
        hop_files["3hop"] = args.hop3
    if args.hop4:
        hop_files["4hop"] = args.hop4

    if not hop_files:
        raise ValueError("At least one --hop* argument is required.")

    combined = load_and_standardize(hop_files)
    logger.info("Loaded %d rows across all hops.", len(combined))

    combined["logfoldchange"] = pd.to_numeric(combined["logfoldchange"], errors="coerce")
    combined["pval"] = pd.to_numeric(combined["pval"], errors="coerce")

    combined["abs_logfc"] = combined["logfoldchange"].abs()
    hop_priority = {"1hop": 1, "2hop": 2, "3hop": 3, "4hop": 4}
    combined["hop_rank"] = combined["hop_number"].map(hop_priority)

    dupes = combined[combined.duplicated(subset=["source", "target"], keep=False)]
    if not dupes.empty and args.duplicates_output:
        dupes_output = dupes[STANDARD_COLUMNS]
        dupes_output.to_csv(args.duplicates_output, index=False)
        logger.info("Duplicates saved to: %s (%d rows)", args.duplicates_output, len(dupes))

    top_fc = (
        combined.sort_values(["abs_logfc", "hop_rank"], ascending=[False, True])
        .drop_duplicates(subset=["source", "target"], keep="first")
        .head(100)
    )
    logger.info("Top 100 by |logFC|: %d", len(top_fc))

    top_pval = (
        combined.sort_values(["pval", "hop_rank"], ascending=[True, True])
        .drop_duplicates(subset=["source", "target"], keep="first")
        .head(100)
    )
    logger.info("Top 100 by p-value: %d", len(top_pval))

    final = (
        pd.concat([top_fc, top_pval], ignore_index=True, sort=False)
        .sort_values(["hop_rank", "pval"], ascending=[True, True])
        .drop_duplicates(subset=["source", "target"], keep="first")
    )

    final_output = final[STANDARD_COLUMNS]
    final_output.to_csv(args.output, index=False)
    logger.info("Saved standardized output: %s", args.output)
    logger.info("Rows: %d", len(final_output))
    logger.info("Hops represented: %s", final['hop_number'].value_counts().to_dict())


if __name__ == "__main__":
    main()
