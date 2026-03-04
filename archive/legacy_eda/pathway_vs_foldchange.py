"""Pathway count vs fold change analysis: target connectivity and response strength."""
from __future__ import annotations

import argparse
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def standardize_columns(df, dataset_name):
    """Standardize column names across datasets."""
    if 'belief_1' in df.columns and 'belief' not in df.columns:
        if 'belief_2' in df.columns:
            df['belief'] = (df['belief_1'] + df['belief_2']) / 2
            logger.info("Created mean belief from belief_1 and belief_2 in %s", dataset_name)
        else:
            df['belief'] = df['belief_1']
    if 'pvalue' in df.columns and 'pval' not in df.columns:
        df = df.rename(columns={'pvalue': 'pval'})
        logger.info("Renamed 'pvalue' to 'pval' in %s", dataset_name)
    return df


def count_pathway_types(pathway_list):
    """Count 1-hop and 2-hop pathways in a list."""
    onehop_count = pathway_list.count('1-hop')
    twohop_count = pathway_list.count('2-hop')
    return pd.Series({'onehop_count': onehop_count, 'twohop_count': twohop_count})


def main():
    parser = argparse.ArgumentParser(
        description="Pathway count vs fold change analysis (excluding TP53)")
    parser.add_argument("--hop1-csv", required=True, help="Path to 1-hop excluding TP53 CSV")
    parser.add_argument("--hop2-xlsx", required=True, help="Path to 2-hop excluding TP53 Excel")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Loading datasets (excluding TP53)")

    df_1hop = pd.read_csv(args.hop1_csv)
    logger.info("1-hop excluding TP53: %s pathways", f"{len(df_1hop):,}")

    df_2hop = pd.read_excel(args.hop2_xlsx)
    logger.info("2-hop excluding TP53: %s pathways", f"{len(df_2hop):,}")

    logger.info("Column structures:")
    logger.info("1-hop columns: %s", list(df_1hop.columns))
    logger.info("2-hop columns: %s", list(df_2hop.columns))

    df_1hop = standardize_columns(df_1hop, "1-hop")
    df_2hop = standardize_columns(df_2hop, "2-hop")

    df_1hop['pathway_type'] = '1-hop'
    df_2hop['pathway_type'] = '2-hop'

    logger.info("PROCESSING DATA FOR PATHWAY COUNT ANALYSIS")

    df_1hop_subset = df_1hop[['source', 'target', 'logfoldchange', 'pval', 'belief', 'pathway_type']].copy()
    df_2hop_subset = df_2hop[['source', 'target', 'logfoldchange', 'pval', 'belief', 'pathway_type']].copy()

    combined_pathways = pd.concat([df_1hop_subset, df_2hop_subset], ignore_index=True)
    logger.info("Combined pathways: %s", f"{len(combined_pathways):,}")

    before_filter = len(combined_pathways)
    combined_pathways = combined_pathways[combined_pathways['source'] != 'TP53']
    after_filter = len(combined_pathways)
    logger.info("Removed %d TP53 pathways, final: %s", before_filter - after_filter, f"{after_filter:,}")

    logger.info("COUNTING PATHWAYS PER TARGET")
    pathway_counts = combined_pathways.groupby('target').agg({
        'source': 'count',
        'logfoldchange': 'first',
        'pval': 'first',
        'pathway_type': lambda x: list(x)
    }).reset_index()
    pathway_counts.columns = ['target', 'pathway_count', 'logfoldchange', 'pval', 'pathway_types']

    logger.info("Unique targets: %s", f"{len(pathway_counts):,}")
    logger.info("Pathway count range: %d to %d",
                pathway_counts['pathway_count'].min(), pathway_counts['pathway_count'].max())

    pathway_type_counts = pathway_counts['pathway_types'].apply(count_pathway_types)
    pathway_counts = pd.concat([pathway_counts, pathway_type_counts], axis=1)
    pathway_counts['abs_logfoldchange'] = np.abs(pathway_counts['logfoldchange'])

    logger.info("Mean pathways per target: %.2f", pathway_counts['pathway_count'].mean())
    logger.info("Median pathways per target: %.0f", pathway_counts['pathway_count'].median())
    logger.info("Targets with 1-hop: %s", f"{len(pathway_counts[pathway_counts['onehop_count'] > 0]):,}")
    logger.info("Targets with 2-hop: %s", f"{len(pathway_counts[pathway_counts['twohop_count'] > 0]):,}")
    logger.info("Targets with both: %s",
                f"{len(pathway_counts[(pathway_counts['onehop_count'] > 0) & (pathway_counts['twohop_count'] > 0)]):,}")

    logger.info("Top 10 most connected targets:")
    top_targets = pathway_counts.nlargest(10, 'pathway_count')[
        ['target', 'pathway_count', 'abs_logfoldchange', 'onehop_count', 'twohop_count']]
    logger.info("\n%s", top_targets.to_string(index=False))

    logger.info("CORRELATION ANALYSIS")
    correlation_count_fc = np.corrcoef(pathway_counts['pathway_count'], pathway_counts['abs_logfoldchange'])[0, 1]
    correlation_count_pval = np.corrcoef(pathway_counts['pathway_count'], pathway_counts['pval'])[0, 1]
    logger.info("Pathway Count vs |LogFC|: r = %.6f", correlation_count_fc)
    logger.info("Pathway Count vs P-value: r = %.6f", correlation_count_pval)

    outlier_threshold = pathway_counts['pathway_count'].quantile(0.99)
    outliers = pathway_counts[pathway_counts['pathway_count'] > outlier_threshold]
    pathway_counts_filtered = pathway_counts[pathway_counts['pathway_count'] <= outlier_threshold]

    logger.info("Outlier threshold: %.0f pathways", outlier_threshold)
    logger.info("Outliers removed: %d targets, remaining: %s",
                len(outliers), f"{len(pathway_counts_filtered):,}")

    correlation_filtered = \
        np.corrcoef(pathway_counts_filtered['pathway_count'], pathway_counts_filtered['abs_logfoldchange'])[0, 1]
    logger.info("Correlation after outlier removal: r = %.6f", correlation_filtered)

    try:
        from scipy import stats
        r_stat, r_pval = stats.pearsonr(
            pathway_counts_filtered['pathway_count'], pathway_counts_filtered['abs_logfoldchange'])
        logger.info("Statistical significance (filtered): p = %.2e", r_pval)
    except ImportError:
        logger.info("scipy not available for significance testing")

    logger.info("CREATING VISUALIZATIONS")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Number of Pathways vs Fold Change Analysis (Excluding TP53)', fontsize=16, fontweight='bold')

    ax1 = axes[0, 0]
    ax1.scatter(pathway_counts_filtered['abs_logfoldchange'], pathway_counts_filtered['pathway_count'],
                alpha=0.6, s=30, color='darkblue')
    ax1.set_xlabel('|Log Fold Change|')
    ax1.set_ylabel('Number of Pathways (log scale)')
    ax1.set_yscale('log')
    ax1.set_title(f'Pathway Count vs |LogFC| (Filtered)\n(r = {correlation_filtered:.4f}, n = {len(pathway_counts_filtered):,})')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=10, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=100, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(0.02, 10, '10 pathways', transform=ax1.get_yaxis_transform(), fontsize=8, color='red')
    ax1.text(0.02, 100, '100 pathways', transform=ax1.get_yaxis_transform(), fontsize=8, color='orange')
    z = np.polyfit(pathway_counts_filtered['abs_logfoldchange'],
                   np.log10(pathway_counts_filtered['pathway_count']), 1)
    x_trend = np.linspace(pathway_counts_filtered['abs_logfoldchange'].min(),
                          pathway_counts_filtered['abs_logfoldchange'].max(), 100)
    y_trend = 10 ** (z[0] * x_trend + z[1])
    ax1.plot(x_trend, y_trend, "r-", alpha=0.8, linewidth=2, label='Trend line')

    ax2 = axes[0, 1]
    log_pathway_counts = np.log10(pathway_counts_filtered['pathway_count'])
    h = ax2.hist2d(pathway_counts_filtered['abs_logfoldchange'], log_pathway_counts, bins=50, cmap='Blues')
    plt.colorbar(h[3], ax=ax2)
    ax2.set_xlabel('|Log Fold Change|')
    ax2.set_ylabel('Log10(Number of Pathways)')
    ax2.set_title('Density Plot (Heat Map - Filtered)')
    y_ticks = ax2.get_yticks()
    ax2.set_yticklabels([f'{10 ** y:.0f}' if y >= 0 else '1' for y in y_ticks])

    ax3 = axes[0, 2]
    low_count = pathway_counts_filtered[pathway_counts_filtered['pathway_count'] <= 5]
    medium_count = pathway_counts_filtered[
        (pathway_counts_filtered['pathway_count'] > 5) & (pathway_counts_filtered['pathway_count'] <= 20)]
    high_count = pathway_counts_filtered[pathway_counts_filtered['pathway_count'] > 20]

    categories = ['Low\n(<=5)', 'Medium\n(6-20)', 'High\n(>20)']
    means = [
        low_count['abs_logfoldchange'].mean() if len(low_count) > 0 else 0,
        medium_count['abs_logfoldchange'].mean() if len(medium_count) > 0 else 0,
        high_count['abs_logfoldchange'].mean() if len(high_count) > 0 else 0
    ]
    counts = [len(low_count), len(medium_count), len(high_count)]

    bars = ax3.bar(categories, means, color=['lightcoral', 'gold', 'lightblue'], alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Connectivity Level')
    ax3.set_ylabel('Mean |Log Fold Change|')
    ax3.set_title('Mean |LogFC| by Connectivity')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'n={count:,}', ha='center', va='bottom', fontsize=9)

    ax4 = axes[1, 0]
    ax4.hist(pathway_counts_filtered['pathway_count'], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax4.set_xlabel('Number of Pathways per Target')
    ax4.set_ylabel('Count of Targets')
    ax4.set_xscale('log')
    ax4.set_title(
        f'Distribution of Pathway Counts (Filtered)\n'
        f'Range: {pathway_counts_filtered["pathway_count"].min()}-{pathway_counts_filtered["pathway_count"].max()}')
    ax4.grid(True, alpha=0.3)

    ax5 = axes[1, 1]
    if len(outliers) > 0:
        ax5.scatter(outliers['abs_logfoldchange'], outliers['pathway_count'],
                    alpha=0.8, s=60, color='red', edgecolor='black')
        ax5.set_xlabel('|Log Fold Change|')
        ax5.set_ylabel('Number of Pathways')
        ax5.set_title(f'Outliers Removed (Top 1%)\nn = {len(outliers)}')
        ax5.grid(True, alpha=0.3)
        for _, row in outliers.head(5).iterrows():
            ax5.annotate(row['target'], (row['abs_logfoldchange'], row['pathway_count']),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)
    else:
        ax5.text(0.5, 0.5, 'No outliers\nidentified', ha='center', va='center',
                 transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Outliers')

    ax6 = axes[1, 2]
    ax6.axis('off')
    rel_type = ("Strong NEGATIVE correlation" if correlation_filtered < -0.05
                else "Weak/no relationship" if abs(correlation_filtered) < 0.05
                else "Positive correlation")
    direction = "WEAKER" if correlation_filtered < 0 else "STRONGER"

    summary_text = f"PATHWAY COUNT ANALYSIS SUMMARY\n\n"
    summary_text += f"ORIGINAL DATA:\n  Total Targets: {len(pathway_counts):,}\n  Total Pathways: {after_filter:,}\n\n"
    summary_text += f"AFTER OUTLIER REMOVAL (Top 1%):\n  Targets: {len(pathway_counts_filtered):,}\n"
    summary_text += f"  Outliers: {len(outliers)}\n  Threshold: >{outlier_threshold:.0f} pathways\n\n"
    summary_text += f"CORRELATION RESULTS:\n  Original: r = {correlation_count_fc:.4f}\n"
    summary_text += f"  Filtered: r = {correlation_filtered:.4f}\n\n"
    summary_text += f"CONNECTIVITY BREAKDOWN:\n"
    summary_text += f"  Low (<=5): {len(low_count):,} targets\n    Mean |LogFC|: {means[0]:.3f}\n"
    summary_text += f"  Medium (6-20): {len(medium_count):,} targets\n    Mean |LogFC|: {means[1]:.3f}\n"
    summary_text += f"  High (>20): {len(high_count):,} targets\n    Mean |LogFC|: {means[2]:.3f}\n\n"
    summary_text += f"HYPOTHESIS TEST:\n{rel_type}\nWell-connected targets show {direction} responses"

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('pathway_count_vs_foldchange_improved.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)

    logger.info("Visualization saved as 'pathway_count_vs_foldchange_improved.png'")

    logger.info("DETAILED ANALYSIS RESULTS")
    logger.info("Targets analyzed: %s, outliers removed: %d",
                f"{len(pathway_counts_filtered):,}", len(outliers))
    logger.info("Average pathways per target: %.2f (std: %.2f)",
                pathway_counts_filtered['pathway_count'].mean(),
                pathway_counts_filtered['pathway_count'].std())

    logger.info("Targets by connectivity (filtered):")
    logger.info("  Low (<=5): %s targets", f"{len(low_count):,}")
    if len(low_count) > 0:
        logger.info("    Mean |LogFC|: %.4f", low_count['abs_logfoldchange'].mean())
    logger.info("  Medium (6-20): %s targets", f"{len(medium_count):,}")
    if len(medium_count) > 0:
        logger.info("    Mean |LogFC|: %.4f", medium_count['abs_logfoldchange'].mean())
    logger.info("  High (>20): %s targets", f"{len(high_count):,}")
    if len(high_count) > 0:
        logger.info("    Mean |LogFC|: %.4f", high_count['abs_logfoldchange'].mean())

    if len(low_count) > 0 and len(high_count) > 0:
        try:
            from scipy import stats
            t_stat, t_pval = stats.ttest_ind(high_count['abs_logfoldchange'], low_count['abs_logfoldchange'])
            logger.info("High vs Low connectivity: mean |LogFC| diff=%.4f, t-test p=%.2e",
                         high_count['abs_logfoldchange'].mean() - low_count['abs_logfoldchange'].mean(), t_pval)
        except ImportError:
            logger.info("High vs Low mean |LogFC| diff: %.4f",
                         high_count['abs_logfoldchange'].mean() - low_count['abs_logfoldchange'].mean())

    if len(outliers) > 0:
        logger.info("Outliers removed (Top 1%%, >%.0f pathways):", outlier_threshold)
        outlier_display = outliers[['target', 'pathway_count', 'abs_logfoldchange']].sort_values(
            'pathway_count', ascending=False)
        logger.info("\n%s", outlier_display.to_string(index=False, max_rows=10))

    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
