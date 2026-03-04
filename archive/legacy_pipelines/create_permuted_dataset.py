"""Legacy script: create permuted source-target dataset for null model evaluation."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import logging


logger = logging.getLogger(__name__)

EXCLUDED_GENES = ["TP53", "CDKN1A"]


def main():
    ap = argparse.ArgumentParser(description="Create permuted source-target dataset.")
    ap.add_argument("--target-validation", required=True, help="Target validation CSV.")
    ap.add_argument("--de-folder", required=True, help="Folder with DE result CSVs.")
    ap.add_argument("--output", required=True, help="Output path for permuted dataset CSV.")
    ap.add_argument("--summary-output", default="", help="Optional path for summary text file.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = ap.parse_args()

    perturb_df = pd.read_csv(args.target_validation)
    perturb_df = perturb_df[perturb_df['analysis_flag'] == "Use_for_analysis"]
    perturb_df = perturb_df[~perturb_df["Gene"].isin(EXCLUDED_GENES)]

    source_genes = perturb_df["Gene"].tolist()
    logger.info("%d source genes loaded (excluded: %s)", len(source_genes), EXCLUDED_GENES)

    all_targets = []
    for source_gene in source_genes:
        csv_path = os.path.join(args.de_folder, f"{source_gene}_vs_control.csv")
        if not os.path.exists(csv_path):
            logger.warning("Missing DE file for %s (skipping)", source_gene)
            continue

        df = pd.read_csv(csv_path)
        df = df[df["pvals"] < 0.05]

        logger.info("  %s: %d targets with pval < 0.05", source_gene, len(df))

        for _, row in df.iterrows():
            all_targets.append({
                "original_source": source_gene,
                "target": row["names"],
                "logfoldchange": row["logfoldchanges"],
                "pval": row["pvals"]
            })

    targets_df = pd.DataFrame(all_targets)
    logger.info("Total: %d source-target pairs loaded", len(targets_df))
    logger.info("Unique sources with targets: %d", targets_df['original_source'].nunique())
    logger.info("Unique targets: %d", targets_df['target'].nunique())

    np.random.seed(args.seed)

    available_sources = source_genes.copy()
    shuffled_sources = [np.random.choice(available_sources) for _ in range(len(targets_df))]
    targets_df["permuted_source"] = shuffled_sources

    true_pairs = set(zip(targets_df["original_source"], targets_df["target"]))
    targets_df["is_true_pair"] = targets_df.apply(
        lambda row: (row["permuted_source"], row["target"]) in true_pairs, axis=1
    )

    before_removal = len(targets_df)
    targets_df_filtered = targets_df[~targets_df["is_true_pair"]].copy()
    after_removal = len(targets_df_filtered)

    logger.info("Removed %d accidental true matches", before_removal - after_removal)

    permuted_pairs = targets_df_filtered[["permuted_source", "target", "logfoldchange", "pval", "original_source"]].copy()
    permuted_pairs.columns = ["source", "target", "logfoldchange", "pval", "original_source"]
    permuted_pairs.to_csv(args.output, index=False)

    logger.info("Saved master permuted dataset: %s", args.output)
    logger.info("Rows: %d", len(permuted_pairs))
    logger.info("Unique permuted sources: %d", permuted_pairs['source'].nunique())
    logger.info("Unique targets: %d", permuted_pairs['target'].nunique())

    if args.summary_output:
        with open(args.summary_output, "w") as f:
            f.write("PERMUTATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total permuted pairs: {len(permuted_pairs)}\n")
            f.write(f"Unique permuted sources: {permuted_pairs['source'].nunique()}\n")
            f.write(f"Unique targets: {permuted_pairs['target'].nunique()}\n")
            f.write(f"Accidental true matches removed: {before_removal - after_removal}\n")
            f.write(f"\nExcluded genes (outliers): {', '.join(EXCLUDED_GENES)}\n")
            f.write(f"\nSource genes used for permutation:\n")
            for gene in source_genes:
                f.write(f"  - {gene}\n")
        logger.info("Saved summary: %s", args.summary_output)

    grouped = permuted_pairs.groupby("source")
    logger.info("Permuted sources distribution:")
    for source in sorted(grouped.groups.keys()):
        count = len(grouped.get_group(source))
        logger.info("  %s: %d targets", source, count)


if __name__ == "__main__":
    main()
