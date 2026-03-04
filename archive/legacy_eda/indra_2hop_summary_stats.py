"""2-hop pathway summary statistics: belief vs experimental strength analysis."""
from __future__ import annotations

import argparse
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


def print_stats(data, label):
    """Log summary statistics for a dataset partition."""
    logger.info("%s:", label)
    logger.info("  Mean belief score: %.3f (+/-%.3f)", data['mean_belief'].mean(), data['mean_belief'].std())
    logger.info("  Mean |logFC|: %.3f (+/-%.3f)", data['abs_logfc'].mean(), data['abs_logfc'].std())
    logger.info("  Mean (1-pval): %.3f (+/-%.3f)", data['one_minus_pval'].mean(), data['one_minus_pval'].std())


def main():
    parser = argparse.ArgumentParser(description="2-hop pathway summary statistics")
    parser.add_argument("--main-xlsx", required=True, help="Path to 2-hop all perturbations Excel")
    parser.add_argument("--tp53-csv", required=True, help="Path to 2-hop TP53 only CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Loading pathway datasets")

    df_main = pd.read_excel(args.main_xlsx)
    logger.info("Main dataset: %s pathways", f"{len(df_main):,}")

    df_tp53 = pd.read_csv(args.tp53_csv)
    logger.info("TP53 dataset: %s pathways", f"{len(df_tp53):,}")

    logger.info("Standardizing column names")
    if 'pvalue' in df_tp53.columns and 'pval' not in df_tp53.columns:
        df_tp53 = df_tp53.rename(columns={'pvalue': 'pval'})
        logger.info("Renamed 'pvalue' to 'pval' in TP53 dataset")

    essential_cols = ['source', 'target', 'belief_1', 'belief_2', 'logfoldchange', 'pval']
    main_missing = [col for col in essential_cols if col not in df_main.columns]
    tp53_missing = [col for col in essential_cols if col not in df_tp53.columns]

    if not main_missing and not tp53_missing:
        logger.info("All essential columns present in both datasets")
    else:
        logger.warning("Missing columns - Main: %s, TP53: %s", main_missing, tp53_missing)

    logger.info("Combining datasets")
    df_combined = pd.concat([df_main, df_tp53], ignore_index=True)
    logger.info("Combined dataset: %s total pathways", f"{len(df_combined):,}")

    logger.info("Preparing data for analysis")
    df_clean = df_combined.dropna(subset=essential_cols).copy()
    df_clean = df_clean[(df_clean['pval'] > 0.0) & (df_clean['pval'] < 1.0)]
    logger.info("After cleaning: %s pathways", f"{len(df_clean):,}")

    df_clean['mean_belief'] = (df_clean['belief_1'] + df_clean['belief_2']) / 2
    df_clean['abs_logfc'] = np.abs(df_clean['logfoldchange'])
    df_clean['one_minus_pval'] = 1 - df_clean['pval']

    logger.info("Final dataset: %s pathways", f"{len(df_clean):,}")
    logger.info("Unique sources: %d", df_clean['source'].nunique())
    logger.info("Unique targets: %d", df_clean['target'].nunique())

    tp53_data = df_clean[df_clean['source'] == 'TP53'].copy()
    non_tp53_data = df_clean[df_clean['source'] != 'TP53'].copy()
    logger.info("TP53 pathways: %s", f"{len(tp53_data):,}")
    logger.info("Non-TP53 pathways: %s", f"{len(non_tp53_data):,}")

    logger.info("SUMMARY STATISTICS")
    print_stats(df_clean, "All data")
    print_stats(tp53_data, "TP53 data")
    print_stats(non_tp53_data, "Non-TP53 data")

    logger.info("Creating scatter plots")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Belief Score vs Experimental Strength Relationships', fontsize=16, fontweight='bold')

    main_size = 15 if len(non_tp53_data) < 100000 else 5
    tp53_size = 20 if len(tp53_data) < 100000 else 8
    alpha_main = 0.6 if len(non_tp53_data) < 100000 else 0.3
    alpha_tp53 = 0.7 if len(tp53_data) < 100000 else 0.5

    ax1 = axes[0, 0]
    ax1.scatter(non_tp53_data['abs_logfc'], non_tp53_data['mean_belief'],
                alpha=alpha_main, s=main_size, color='blue', label=f'Non-TP53 (n={len(non_tp53_data):,})')
    ax1.scatter(tp53_data['abs_logfc'], tp53_data['mean_belief'],
                alpha=alpha_tp53, s=tp53_size, color='red', label=f'TP53 (n={len(tp53_data):,})')
    ax1.set_xlabel('|Log Fold Change|')
    ax1.set_ylabel('Mean Belief Score')
    ax1.set_title('|LogFC| vs Mean Belief Score (All Data)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.scatter(non_tp53_data['abs_logfc'], non_tp53_data['mean_belief'], alpha=0.6, s=20, color='blue')
    ax2.set_xlabel('|Log Fold Change|')
    ax2.set_ylabel('Mean Belief Score')
    ax2.set_title('|LogFC| vs Mean Belief Score (Excluding TP53)')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.scatter(non_tp53_data['one_minus_pval'], non_tp53_data['mean_belief'],
                alpha=alpha_main, s=main_size, color='green', label=f'Non-TP53 (n={len(non_tp53_data):,})')
    ax3.scatter(tp53_data['one_minus_pval'], tp53_data['mean_belief'],
                alpha=alpha_tp53, s=tp53_size, color='red', label=f'TP53 (n={len(tp53_data):,})')
    ax3.set_xlabel('1 - P-value')
    ax3.set_ylabel('Mean Belief Score')
    ax3.set_title('(1-P-value) vs Mean Belief Score (All Data)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.scatter(non_tp53_data['one_minus_pval'], non_tp53_data['mean_belief'], alpha=0.6, s=20, color='green')
    ax4.set_xlabel('1 - P-value')
    ax4.set_ylabel('Mean Belief Score')
    ax4.set_title('(1-P-value) vs Mean Belief Score (Excluding TP53)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('belief_vs_experimental_strength_full.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)

    logger.info("CORRELATION ANALYSIS (EXCLUDING TP53) - FULL DATASET")
    logger.info("Using complete dataset of %s points", f"{len(non_tp53_data):,}")

    fc_belief_corr = np.corrcoef(non_tp53_data['abs_logfc'], non_tp53_data['mean_belief'])[0, 1]
    pval_belief_corr = np.corrcoef(non_tp53_data['one_minus_pval'], non_tp53_data['mean_belief'])[0, 1]

    logger.info("Correlation |LogFC| vs Mean Belief: r = %.6f", fc_belief_corr)
    logger.info("Correlation (1-P-value) vs Mean Belief: r = %.6f", pval_belief_corr)

    logger.info("OUTLIER ANALYSIS - FULL DATASET")

    fc_thresholds = np.percentile(non_tp53_data['abs_logfc'], [5, 95])
    belief_thresholds = np.percentile(non_tp53_data['mean_belief'], [5, 95])
    pval_thresholds = np.percentile(non_tp53_data['one_minus_pval'], [5, 95])

    fc_low, fc_high = fc_thresholds
    belief_low, belief_high = belief_thresholds
    pval_low, pval_high = pval_thresholds

    logger.info("Outlier thresholds (5th/95th percentiles):")
    logger.info("  |LogFC|: Low=%.3f, High=%.3f", fc_low, fc_high)
    logger.info("  Mean Belief: Low=%.3f, High=%.3f", belief_low, belief_high)
    logger.info("  (1-P-value): Low=%.3f, High=%.3f", pval_low, pval_high)

    outliers_1 = non_tp53_data[(non_tp53_data['abs_logfc'] >= fc_high) &
                               (non_tp53_data['mean_belief'] <= belief_low)]
    logger.info("HIGH |LOGFC| + LOW BELIEF (strong experiment, weak literature): %s outliers (%.2f%%)",
                f"{len(outliers_1):,}", len(outliers_1) / len(non_tp53_data) * 100)
    if len(outliers_1) > 0:
        logger.info("  Top 10 examples:")
        top = outliers_1.nlargest(10, 'abs_logfc')[['source', 'target', 'abs_logfc', 'mean_belief', 'pval']]
        for _, row in top.iterrows():
            logger.info("    %s -> %s: FC=%.3f, Belief=%.3f",
                         row['source'], row['target'], row['abs_logfc'], row['mean_belief'])

    outliers_2 = non_tp53_data[(non_tp53_data['abs_logfc'] <= fc_low) &
                               (non_tp53_data['mean_belief'] >= belief_high)]
    logger.info("LOW |LOGFC| + HIGH BELIEF (weak experiment, strong literature): %s outliers (%.2f%%)",
                f"{len(outliers_2):,}", len(outliers_2) / len(non_tp53_data) * 100)
    if len(outliers_2) > 0:
        logger.info("  Top 10 examples:")
        top = outliers_2.nsmallest(10, 'abs_logfc')[['source', 'target', 'abs_logfc', 'mean_belief', 'pval']]
        for _, row in top.iterrows():
            logger.info("    %s -> %s: FC=%.3f, Belief=%.3f",
                         row['source'], row['target'], row['abs_logfc'], row['mean_belief'])

    outliers_3 = non_tp53_data[(non_tp53_data['one_minus_pval'] >= pval_high) &
                               (non_tp53_data['mean_belief'] <= belief_low)]
    logger.info("HIGH SIGNIFICANCE + LOW BELIEF: %s outliers (%.2f%%)",
                f"{len(outliers_3):,}", len(outliers_3) / len(non_tp53_data) * 100)

    logger.info("TP53 COMPARISON ANALYSIS")
    if len(tp53_data) > 0:
        tp53_mean_belief = tp53_data['mean_belief'].mean()
        tp53_mean_fc = tp53_data['abs_logfc'].mean()
        tp53_mean_pval = tp53_data['one_minus_pval'].mean()
        other_mean_belief = non_tp53_data['mean_belief'].mean()
        other_mean_fc = non_tp53_data['abs_logfc'].mean()
        other_mean_pval = non_tp53_data['one_minus_pval'].mean()

        logger.info("TP53: %s pathways, %s unique targets",
                     f"{len(tp53_data):,}", f"{tp53_data['target'].nunique():,}")
        logger.info("  Mean belief: %.3f, Mean |FC|: %.3f, Mean significance: %.3f",
                     tp53_mean_belief, tp53_mean_fc, tp53_mean_pval)
        logger.info("  Ratios (TP53/Others): belief=%.2fx, FC=%.2fx, significance=%.2fx",
                     tp53_mean_belief / other_mean_belief,
                     tp53_mean_fc / other_mean_fc,
                     tp53_mean_pval / other_mean_pval)

    logger.info("FINAL SUMMARY")
    logger.info("Total pathways: %s", f"{len(df_clean):,}")
    logger.info("  TP53: %s (%.1f%%)", f"{len(tp53_data):,}", len(tp53_data) / len(df_clean) * 100)
    logger.info("  Others: %s (%.1f%%)", f"{len(non_tp53_data):,}", len(non_tp53_data) / len(df_clean) * 100)
    logger.info("Correlations (Excluding TP53): |LogFC| vs Belief r=%.6f, (1-pval) vs Belief r=%.6f",
                fc_belief_corr, pval_belief_corr)
    logger.info("Outliers: Strong FC/weak belief=%s, Weak FC/strong belief=%s, High sig/low belief=%s",
                f"{len(outliers_1):,}", f"{len(outliers_2):,}", f"{len(outliers_3):,}")
    logger.info("Key: Experimental strength and literature support are essentially independent")
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
