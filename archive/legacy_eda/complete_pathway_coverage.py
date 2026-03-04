"""Complete pathway coverage analysis across 1-hop, 2-hop, and 3-hop pathways."""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Complete pathway coverage analysis")
    parser.add_argument("--hop1-csv", required=True, help="Path to 1-hop CSV (e.g. indra_1hop_no_v2.csv)")
    parser.add_argument("--hop2-xlsx", required=True, help="Path to 2-hop Excel file")
    parser.add_argument("--hop3-csv", required=True, help="Path to 3-hop CSV")
    parser.add_argument("--validation-csv", required=True, help="Path to target_validation_expanded.csv")
    parser.add_argument("--deg-folder", required=True, help="Path to de_results_per_gene folder")
    parser.add_argument("--output-dir", default=".", help="Output directory for CSVs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Deep Statistical Coverage Analysis for 1-hop, 2-hop, 3-hop Pathways")

    logger.info("Loading pathway datasets")

    df_1hop_combined = pd.read_csv(args.hop1_csv)
    df_2hop_combined = pd.read_excel(args.hop2_xlsx)
    df_3hop = pd.read_csv(args.hop3_csv)

    logger.info("1-hop total pathways: %s", f"{len(df_1hop_combined):,}")
    logger.info("2-hop total pathways: %s", f"{len(df_2hop_combined):,}")
    logger.info("3-hop total pathways: %s", f"{len(df_3hop):,}")

    pairs_1hop = set(zip(df_1hop_combined['source'], df_1hop_combined['target']))
    pairs_2hop = set(zip(df_2hop_combined['source'], df_2hop_combined['target']))
    pairs_3hop = set(zip(df_3hop['source'], df_3hop['target']))

    logger.info("Unique source-target pairs (unfiltered): 1-hop=%s, 2-hop=%s, 3-hop=%s",
                f"{len(pairs_1hop):,}", f"{len(pairs_2hop):,}", f"{len(pairs_3hop):,}")

    target_validation_path = args.validation_csv
    deg_folder = args.deg_folder

    allowed_sources = set(
        pd.read_csv(target_validation_path)
        .query("analysis_flag == 'Use_for_analysis'")["Gene"]
        .astype(str).str.strip()
        .tolist()
    )
    EXCLUDE_SOURCES = {"TP53", "CDKN1A"}
    allowed_sources = {g for g in allowed_sources if g not in EXCLUDE_SOURCES}

    experimental_pairs = set()
    perturbation_stats = {}

    for src in sorted(allowed_sources):
        deg_path = Path(deg_folder) / f"{src}_vs_control.csv"
        if not deg_path.exists():
            continue
        df = pd.read_csv(deg_path)
        if not {"names", "pvals"}.issubset(df.columns):
            continue
        sig = df[df["pvals"] < 0.05].copy()
        targets = set(sig["names"].dropna().astype(str))
        for tgt in targets:
            experimental_pairs.add((src, tgt))
        perturbation_stats[src] = {"total_targets": len(targets), "targets": targets}

    total_experimental_pairs = len(experimental_pairs)
    logger.info("Total experimental pairs (allowed perturbations only): %s", f"{total_experimental_pairs:,}")
    logger.info("Perturbations with valid DEG: %s", f"{len(perturbation_stats):,}")

    pairs_1hop = {(s, t) for (s, t) in pairs_1hop if s in allowed_sources}
    pairs_2hop = {(s, t) for (s, t) in pairs_2hop if s in allowed_sources}
    pairs_3hop = {(s, t) for (s, t) in pairs_3hop if s in allowed_sources}

    logger.info("Filtered pairs: 1-hop=%s, 2-hop=%s, 3-hop=%s",
                f"{len(pairs_1hop):,}", f"{len(pairs_2hop):,}", f"{len(pairs_3hop):,}")

    coverage_1hop = (len(pairs_1hop.intersection(experimental_pairs)) / total_experimental_pairs) * 100
    coverage_2hop = (len(pairs_2hop.intersection(experimental_pairs)) / total_experimental_pairs) * 100
    coverage_3hop = (len(pairs_3hop.intersection(experimental_pairs)) / total_experimental_pairs) * 100

    logger.info("Individual Coverage: 1-hop=%.2f%%, 2-hop=%.2f%%, 3-hop=%.2f%%",
                coverage_1hop, coverage_2hop, coverage_3hop)

    explained_1hop_only = pairs_1hop.intersection(experimental_pairs)
    explained_2hop_only = (pairs_2hop - pairs_1hop).intersection(experimental_pairs)
    explained_3hop_only = (pairs_3hop - pairs_1hop - pairs_2hop).intersection(experimental_pairs)

    all_explained = pairs_1hop.union(pairs_2hop).union(pairs_3hop).intersection(experimental_pairs)
    unexplained = experimental_pairs - all_explained

    logger.info("Exclusive Coverage:")
    logger.info("  1-hop only: %s pairs (%.2f%%)",
                f"{len(explained_1hop_only):,}", len(explained_1hop_only) / total_experimental_pairs * 100)
    logger.info("  2-hop only: %s pairs (%.2f%%)",
                f"{len(explained_2hop_only):,}", len(explained_2hop_only) / total_experimental_pairs * 100)
    logger.info("  3-hop only: %s pairs (%.2f%%)",
                f"{len(explained_3hop_only):,}", len(explained_3hop_only) / total_experimental_pairs * 100)
    logger.info("  Total explained: %s pairs (%.2f%%)",
                f"{len(all_explained):,}", len(all_explained) / total_experimental_pairs * 100)
    logger.info("  Unexplained: %s pairs (%.2f%%)",
                f"{len(unexplained):,}", len(unexplained) / total_experimental_pairs * 100)

    perturbation_coverage = []
    for gene_name, stats in perturbation_stats.items():
        targets = stats['targets']
        total_targets = len(targets)
        if total_targets == 0:
            continue
        gene_pairs = {(gene_name, target) for target in targets}

        explained_1hop_excl = len(gene_pairs.intersection(pairs_1hop))
        explained_2hop_excl = len(gene_pairs.intersection(pairs_2hop - pairs_1hop))
        explained_3hop_excl = len(gene_pairs.intersection(pairs_3hop - pairs_1hop - pairs_2hop))
        total_explained = explained_1hop_excl + explained_2hop_excl + explained_3hop_excl

        perturbation_coverage.append({
            'perturbation': gene_name, 'total_targets': total_targets,
            'explained_1hop': len(gene_pairs.intersection(pairs_1hop)),
            'explained_2hop': len(gene_pairs.intersection(pairs_2hop)),
            'explained_3hop': len(gene_pairs.intersection(pairs_3hop)),
            'explained_1hop_excl': explained_1hop_excl,
            'explained_2hop_excl': explained_2hop_excl,
            'explained_3hop_excl': explained_3hop_excl,
            'total_explained': total_explained,
            'coverage_1hop': (len(gene_pairs.intersection(pairs_1hop)) / total_targets) * 100,
            'coverage_2hop': (len(gene_pairs.intersection(pairs_2hop)) / total_targets) * 100,
            'coverage_3hop': (len(gene_pairs.intersection(pairs_3hop)) / total_targets) * 100,
            'coverage_total': (total_explained / total_targets) * 100,
            'unexplained': total_targets - total_explained
        })

    coverage_df = pd.DataFrame(perturbation_coverage)
    logger.info("Mean total coverage: %.2f%%", coverage_df['coverage_total'].mean())
    logger.info("Mean 1-hop coverage: %.2f%%", coverage_df['coverage_1hop'].mean())
    logger.info("Mean 2-hop coverage: %.2f%%", coverage_df['coverage_2hop'].mean())
    logger.info("Mean 3-hop coverage: %.2f%%", coverage_df['coverage_3hop'].mean())

    logger.info("Top 10 perturbations by total coverage:")
    logger.info("\n%s", coverage_df.nlargest(10, 'coverage_total')[
        ['perturbation', 'total_targets', 'coverage_total', 'coverage_1hop',
         'coverage_2hop', 'coverage_3hop']
    ].to_string(index=False, float_format='%.1f'))

    summary_data = {
        'Coverage_Type': [
            'Individual_1hop', 'Individual_2hop', 'Individual_3hop',
            'Exclusive_1hop_only', 'Exclusive_2hop_only', 'Exclusive_3hop_only',
            'Total_Explained', 'Unexplained'
        ],
        'Source_Target_Pairs': [
            len(pairs_1hop.intersection(experimental_pairs)),
            len(pairs_2hop.intersection(experimental_pairs)),
            len(pairs_3hop.intersection(experimental_pairs)),
            len(explained_1hop_only), len(explained_2hop_only), len(explained_3hop_only),
            len(all_explained), len(unexplained)
        ],
        'Percentage': [
            coverage_1hop, coverage_2hop, coverage_3hop,
            (len(explained_1hop_only) / total_experimental_pairs) * 100,
            (len(explained_2hop_only) / total_experimental_pairs) * 100,
            (len(explained_3hop_only) / total_experimental_pairs) * 100,
            (len(all_explained) / total_experimental_pairs) * 100,
            (len(unexplained) / total_experimental_pairs) * 100
        ]
    }
    summary_table = pd.DataFrame(summary_data)
    logger.info("Coverage summary:\n%s", summary_table.to_string(index=False, float_format='%.2f'))

    summary_table.to_csv(f'{args.output_dir}/pathway_coverage_summary.csv', index=False)
    coverage_df.to_csv(f'{args.output_dir}/per_perturbation_coverage.csv', index=False)

    detailed_pairs = []
    for pair_type, pairs_list in [
        ('explained_1hop_only', explained_1hop_only),
        ('explained_2hop_only', explained_2hop_only),
        ('explained_3hop_only', explained_3hop_only),
        ('unexplained_pairs', unexplained),
    ]:
        for source, target in pairs_list:
            detailed_pairs.append({'source': source, 'target': target, 'classification': pair_type})

    detailed_df = pd.DataFrame(detailed_pairs)
    detailed_df.to_csv(f'{args.output_dir}/detailed_pair_classification.csv', index=False)

    logger.info("Coverage analysis complete")
    logger.info("Combined coverage: %.1f%%", len(all_explained) / total_experimental_pairs * 100)
    logger.info("Mean per-perturbation coverage: %.1f%%", coverage_df['coverage_total'].mean())


if __name__ == "__main__":
    main()
