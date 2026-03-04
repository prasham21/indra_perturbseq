"""3-hop pathway coverage calculation for completed genes."""
from __future__ import annotations

import argparse
import logging
import os

import pandas as pd
from indra_cogex.analysis.source_targets_explanation import get_valid_gene_ids

logger = logging.getLogger(__name__)


def calculate_3hop_coverage(results_file, deg_folder):
    """Calculate 3-hop pathway coverage for completed genes only."""
    logger.info("Calculating 3-hop pathway coverage for completed genes only")

    if not os.path.exists(results_file):
        logger.error("Results file '%s' not found", results_file)
        return

    results_df = pd.read_csv(results_file)
    logger.info("Loaded 3-hop results: %d pathways", len(results_df))

    results_pairs = results_df[['source', 'target']].drop_duplicates()
    found_pairs = len(results_pairs)
    logger.info("Unique source-target pairs found in 3-hop results: %d", found_pairs)

    completed_sources = sorted(results_df['source'].unique())
    logger.info("Completed source genes: %d", len(completed_sources))
    logger.info("First 10 completed genes: %s...", ', '.join(completed_sources[:10]))

    total_possible_pairs = 0
    gene_pair_counts = {}

    logger.info("Calculating total possible pairs for %d completed genes", len(completed_sources))

    for i, source_gene in enumerate(completed_sources):
        logger.info("Processing %d/%d: %s", i + 1, len(completed_sources), source_gene)

        deg_path = os.path.join(deg_folder, f"{source_gene}_vs_control.csv")

        if os.path.exists(deg_path):
            try:
                df = pd.read_csv(deg_path)
                df = df[df["pvals"] < 0.05]

                gene_symbols = df["names"].dropna().unique().tolist()
                converted = get_valid_gene_ids(gene_symbols)
                valid_targets = len([v for v in converted if v])

                total_possible_pairs += valid_targets
                gene_pair_counts[source_gene] = valid_targets
                logger.info("  %s: %d possible targets", source_gene, valid_targets)

            except Exception as e:
                logger.warning("  Error processing %s: %s", source_gene, e)
                gene_pair_counts[source_gene] = 0
        else:
            logger.info("  DEG file not found for %s", source_gene)
            gene_pair_counts[source_gene] = 0

    if total_possible_pairs > 0:
        coverage_percent = (found_pairs / total_possible_pairs) * 100

        logger.info("3-HOP PATHWAY COVERAGE ANALYSIS (COMPLETED GENES ONLY)")
        logger.info("Analysis scope: First %d completed perturbations", len(completed_sources))
        logger.info("Total possible source-target pairs: %s", f"{total_possible_pairs:,}")
        logger.info("Source-target pairs with 3-hop pathways: %s", f"{found_pairs:,}")
        logger.info("Coverage percentage: %.2f%%", coverage_percent)

        avg_pathways_per_pair = len(results_df) / found_pairs if found_pairs > 0 else 0
        logger.info("Total 3-hop pathways found: %s", f"{len(results_df):,}")
        logger.info("Average pathways per source-target pair: %.1f", avg_pathways_per_pair)

        top_genes = sorted(gene_pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("Top 5 genes by possible targets:")
        for gene, count in top_genes:
            source_results = results_df[results_df['source'] == gene]
            actual_pairs = len(source_results[['source', 'target']].drop_duplicates())
            gene_coverage = (actual_pairs / count * 100) if count > 0 else 0
            logger.info("  %s: %d/%d pairs (%.1f%% coverage)", gene, actual_pairs, count, gene_coverage)

        bottom_genes = sorted(gene_pair_counts.items(), key=lambda x: x[1])[:5]
        logger.info("Bottom 5 genes by possible targets:")
        for gene, count in bottom_genes:
            source_results = results_df[results_df['source'] == gene]
            actual_pairs = len(source_results[['source', 'target']].drop_duplicates())
            gene_coverage = (actual_pairs / count * 100) if count > 0 else 0
            logger.info("  %s: %d/%d pairs (%.1f%% coverage)", gene, actual_pairs, count, gene_coverage)
    else:
        logger.warning("No possible pairs found - check data files")


def main():
    parser = argparse.ArgumentParser(description="3-hop pathway coverage statistics")
    parser.add_argument("--results-file", default="partial_3hop_results.csv",
                        help="Path to partial_3hop_results.csv")
    parser.add_argument("--deg-folder", required=True,
                        help="Path to de_results_per_gene folder")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    calculate_3hop_coverage(args.results_file, args.deg_folder)


if __name__ == "__main__":
    main()
