"""Legacy script: compare_original_vs_permuted."""
from __future__ import annotations

import argparse
import pandas as pd

import logging


logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indra-1hop-with-statements", default="indra_1hop_with_statements_.csv", help="Path: indra_1hop_with_statements_.csv")
    ap.add_argument("--indra-2hop-all-perturbation", default="indra_2hop_all_perturbations.csv", help="Path: indra_2hop_all_perturbations.csv")
    ap.add_argument("--indra-1hop-permuted", default="indra_1hop_PERMUTED.csv", help="Path: indra_1hop_PERMUTED.csv")
    ap.add_argument("--indra-2hop-permuted", default="indra_2hop_PERMUTED.csv", help="Path: indra_2hop_PERMUTED.csv")
    ap.add_argument("--master-permuted-source-target-pair", default="MASTER_permuted_source_target_pairs.csv", help="Path: MASTER_permuted_source_target_pairs.csv")
    ap.add_argument("--target-validation-expanded", default="target_validation_expanded.csv", help="Path: target_validation_expanded.csv")
    ap.add_argument("--de-results-per-gene", default="de_results_per_gene/", help="Path: de_results_per_gene/")
    args = ap.parse_args()

    logger.info("=" * 80)
    logger.info(" VERIFYING 2-HOP PAIRS (Removing Self-Loops)")
    logger.info("=" * 80)

    real_1hop = pd.read_csv("indra_1hop_with_statements_.csv")
    real_2hop = pd.read_csv("indra_2hop_all_perturbations.csv")
    permuted_1hop = pd.read_csv("indra_1hop_PERMUTED.csv")
    permuted_2hop = pd.read_csv("indra_2hop_PERMUTED.csv")
    master_permuted = pd.read_csv("MASTER_permuted_source_target_pairs.csv")

    logger.info("\n Before removing self-loops:")
    logger.info("Real 1-hop rows: %d", len(real_1hop))
    logger.info("Real 2-hop rows: %d", len(real_2hop))
    logger.info("Permuted 1-hop rows: %d", len(permuted_1hop))
    logger.info("Permuted 2-hop rows: %d", len(permuted_2hop))
    logger.info("Master permuted pairs: %d", len(master_permuted))

    real_1hop_clean = real_1hop[real_1hop['source'] != real_1hop['target']].copy()
    real_2hop_clean = real_2hop[real_2hop['source'] != real_2hop['target']].copy()
    permuted_1hop_clean = permuted_1hop[permuted_1hop['source'] != permuted_1hop['target']].copy()
    permuted_2hop_clean = permuted_2hop[permuted_2hop['source'] != permuted_2hop['target']].copy()
    master_permuted_clean = master_permuted[master_permuted['source'] != master_permuted['target']].copy()

    logger.info("\n  Self-loops removed:")
    logger.info("Real 1-hop: %d", len(real_1hop) - len(real_1hop_clean))
    logger.info("Real 2-hop: %d", len(real_2hop) - len(real_2hop_clean))
    logger.info("Permuted 1-hop: %d", len(permuted_1hop) - len(permuted_1hop_clean))
    logger.info("Permuted 2-hop: %d", len(permuted_2hop) - len(permuted_2hop_clean))
    logger.info("Master permuted: %d", len(master_permuted) - len(master_permuted_clean))

    logger.info("\n After removing self-loops:")
    logger.info("Real 1-hop rows: %d", len(real_1hop_clean))
    logger.info("Real 2-hop rows: %d", len(real_2hop_clean))
    logger.info("Permuted 1-hop rows: %d", len(permuted_1hop_clean))
    logger.info("Permuted 2-hop rows: %d", len(permuted_2hop_clean))
    logger.info("Master permuted pairs: %d", len(master_permuted_clean))

    logger.info("\n" + "=" * 80)
    logger.info(" UNIQUE SOURCE-TARGET PAIRS (No Self-Loops)")
    logger.info("=" * 80)

    real_1hop_pairs = set(zip(real_1hop_clean['source'], real_1hop_clean['target']))
    real_2hop_pairs = set(zip(real_2hop_clean['source'], real_2hop_clean['target']))
    real_combined_pairs = real_1hop_pairs.union(real_2hop_pairs)

    permuted_1hop_pairs = set(zip(permuted_1hop_clean['source'], permuted_1hop_clean['target']))
    permuted_2hop_pairs = set(zip(permuted_2hop_clean['source'], permuted_2hop_clean['target']))
    permuted_combined_pairs = permuted_1hop_pairs.union(permuted_2hop_pairs)

    master_pairs_clean = set(zip(master_permuted_clean['source'], master_permuted_clean['target']))

    logger.info("\nReal 1-hop unique pairs: %d", len(real_1hop_pairs))
    logger.info("Real 2-hop unique pairs: %d", len(real_2hop_pairs))
    logger.info("Real combined unique pairs: %d", len(real_combined_pairs))
    logger.info("\nPermuted 1-hop unique pairs: %d", len(permuted_1hop_pairs))
    logger.info("Permuted 2-hop unique pairs: %d", len(permuted_2hop_pairs))
    logger.info("Permuted combined unique pairs: %d", len(permuted_combined_pairs))
    logger.info("\nMaster permuted total pool: %d", len(master_pairs_clean))

    import os

    perturb_df = pd.read_csv("target_validation_expanded.csv")
    perturb_df = perturb_df[perturb_df['analysis_flag'] == "Use_for_analysis"]
    excluded_genes = ["TP53", "CDKN1A"]
    perturb_df = perturb_df[~perturb_df["Gene"].isin(excluded_genes)]

    real_total_pairs = []
    de_folder = "de_results_per_gene/"

    for _, row in perturb_df.iterrows():
        source_gene = row["Gene"]
        csv_path = os.path.join(de_folder, f"{source_gene}_vs_control.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df = df[df["pvals"] < 0.05]

        for _, target_row in df.iterrows():
            if source_gene != target_row["names"]:
                real_total_pairs.append((source_gene, target_row["names"]))

    real_total_pairs_set = set(real_total_pairs)
    logger.info("Real total pool (no self-loops): %d", len(real_total_pairs_set))

    logger.info("\n" + "=" * 80)
    logger.info(" COVERAGE RATES (No Self-Loops)")
    logger.info("=" * 80)

    real_1hop_pct = (len(real_1hop_pairs) / len(real_total_pairs_set)) * 100
    real_2hop_pct = (len(real_2hop_pairs) / len(real_total_pairs_set)) * 100
    real_combined_pct = (len(real_combined_pairs) / len(real_total_pairs_set)) * 100

    permuted_1hop_pct = (len(permuted_1hop_pairs) / len(master_pairs_clean)) * 100
    permuted_2hop_pct = (len(permuted_2hop_pairs) / len(master_pairs_clean)) * 100
    permuted_combined_pct = (len(permuted_combined_pairs) / len(master_pairs_clean)) * 100

    logger.info("\n%<25 %<15 %<15 %<15", 'Dataset', 'Explained', 'Total Pool', '% Explained')
    logger.info("-" * 70)
    # [corrupted format string removed]
    # [corrupted format string removed]
    logger.info(f"{'Real Combined':<25} {len(real_combined_pairs):<15,} {len(real_total_pairs_set):<15,} {real_combined_pct:<15.2f}%"))
    logger.info(f"{'Permuted 1-hop':<25} {len(permuted_1hop_pairs):<15,} {len(master_pairs_clean):<15,} {permuted_1hop_pct:<15.2f}%"))
    logger.info(f"{'Permuted 2-hop':<25} {len(permuted_2hop_pairs):<15,} {len(master_pairs_clean):<15,} {permuted_2hop_pct:<15.2f}%"))
    logger.info(f"{'Permuted Combined':<25} {len(permuted_combined_pairs):<15,} {len(master_pairs_clean):<15,} {permuted_combined_pct:<15.2f}%"))

    logger.info("\n" + "=" * 80)
    logger.info("  COVERAGE RATE COMPARISON (No Self-Loops)")
    logger.info("=" * 80)

    logger.info("\n%<40 %<15 %<15 %<15", 'Metric', 'Real %', 'Permuted %', 'Difference')
    logger.info("-" * 85)
    logger.info(f"{'1-hop coverage rate':<40} {real_1hop_pct:<15.2f}% {permuted_1hop_pct:<15.2f}% {real_1hop_pct - permuted_1hop_pct:+.2f}%"))
    logger.info(f"{'2-hop coverage rate':<40} {real_2hop_pct:<15.2f}% {permuted_2hop_pct:<15.2f}% {real_2hop_pct - permuted_2hop_pct:+.2f}%"))
    logger.info(f"{'Combined coverage rate':<40} {real_combined_pct:<15.2f}% {permuted_combined_pct:<15.2f}% {real_combined_pct - permuted_combined_pct:+.2f}%"))

    coverage_diff = real_combined_pct - permuted_combined_pct

    logger.info("\n" + "=" * 80)
    logger.info(" INTERPRETATION (No Self-Loops)")
    logger.info("=" * 80)

    if coverage_diff > 5:
        logger.info("\n EXCELLENT: Real pairs have %+.2f%% HIGHER coverage rate!", coverage_diff)
        logger.info("   Real: %.2f%% explained", real_combined_pct)
        logger.info("   Permuted: %.2f%% explained", permuted_combined_pct)
    else:
        logger.info("\n Coverage difference: %+.2f%%", coverage_diff)

    logger.info("\n Absolute numbers:")
    logger.info("   Real: %d of %d pairs", len(real_combined_pairs), len(real_total_pairs_set))
    logger.info("   Permuted: %d of %d pairs", len(permuted_combined_pairs), len(master_pairs_clean))
    logger.info("   Real has %d MORE pairs explained", len(real_combined_pairs) - len(permuted_combined_pairs))

    logger.info("\n" + "=" * 80)



if __name__ == "__main__":
    main()
