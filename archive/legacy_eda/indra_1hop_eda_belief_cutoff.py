"""1-hop pathway EDA with belief score cutoff (>=0.7) filtering."""
from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def standardize_columns(df, dataset_name):
    """Standardize column names across datasets."""
    if 'belief_1' in df.columns and 'belief' not in df.columns:
        df = df.rename(columns={'belief_1': 'belief'})
        logger.info("Renamed 'belief_1' to 'belief' in %s", dataset_name)
    if 'pvalue' in df.columns and 'pval' not in df.columns:
        df = df.rename(columns={'pvalue': 'pval'})
        logger.info("Renamed 'pvalue' to 'pval' in %s", dataset_name)
    return df


def process_deg_and_pathways(pathway_df, dataset_name, deg_files, include_tp53=True):
    """Process DEG files and combine with pathway data."""
    explained_data = []
    unexplained_data = []
    pathway_sources = pathway_df['source'].unique()
    logger.info("Processing %d unique sources in %s", len(pathway_sources), dataset_name)

    for deg_file in deg_files:
        gene_name = deg_file.stem.replace("_vs_control", "")
        if not include_tp53 and gene_name == 'TP53':
            continue
        if gene_name not in pathway_sources:
            continue
        try:
            deg_df = pd.read_csv(deg_file)
            significant_genes = deg_df[deg_df['pvals'] < 0.05].copy()
            if len(significant_genes) == 0:
                continue
            source_pathways = pathway_df[pathway_df['source'] == gene_name].copy()
            explained_targets = set(source_pathways['target'].unique())

            for _, pathway_row in source_pathways.iterrows():
                explained_data.append({
                    'source': gene_name, 'target': pathway_row['target'],
                    'belief': pathway_row['belief'], 'logfoldchange': pathway_row['logfoldchange'],
                    'pval': pathway_row['pval'], 'explained': True, 'pathway_type': '1-hop'
                })
            for _, deg_row in significant_genes.iterrows():
                target_gene = deg_row['names']
                if target_gene in explained_targets:
                    continue
                unexplained_data.append({
                    'source': gene_name, 'target': target_gene,
                    'belief': 0.5, 'logfoldchange': deg_row['logfoldchanges'],
                    'pval': deg_row['pvals'], 'explained': False, 'pathway_type': 'none'
                })
        except Exception as e:
            logger.warning("Error processing %s: %s", gene_name, e)
            continue

    combined_df = pd.DataFrame(explained_data + unexplained_data)
    logger.info("%s summary:", dataset_name)
    logger.info("  Explained pathways: %s", f"{len(explained_data):,}")
    logger.info("  Unexplained targets: %s", f"{len(unexplained_data):,}")
    logger.info("  Total data points: %s", f"{len(combined_df):,}")

    if len(combined_df) > 0:
        explained_count = combined_df['explained'].sum()
        unexplained_count = len(combined_df) - explained_count
        logger.info("  Explained percentage: %.1f%%", explained_count / len(combined_df) * 100)
        logger.info("  Unexplained percentage: %.1f%%", unexplained_count / len(combined_df) * 100)

    return combined_df


def print_dataset_stats(df, name):
    """Log summary statistics for a dataset."""
    if len(df) == 0:
        logger.info("%s: No data", name)
        return
    explained = df[df['explained']]
    unexplained = df[~df['explained']]

    logger.info("%s:", name)
    logger.info("  Total data points: %s", f"{len(df):,}")
    logger.info("  Explained: %s (%.1f%%)", f"{len(explained):,}", len(explained) / len(df) * 100)
    logger.info("  Unexplained: %s (%.1f%%)", f"{len(unexplained):,}", len(unexplained) / len(df) * 100)

    if len(explained) > 0:
        logger.info("  Explained - Mean |FC|: %.3f", np.abs(explained['logfoldchange']).mean())
        logger.info("  Explained - Mean belief: %.3f", explained['belief'].mean())
    if len(unexplained) > 0:
        logger.info("  Unexplained - Mean |FC|: %.3f", np.abs(unexplained['logfoldchange']).mean())
        logger.info("  Unexplained - Mean belief: %.3f", unexplained['belief'].mean())


def create_heatmap_plot(df, title, save_name):
    """Create comprehensive visualization with heat maps and separate histograms."""
    if len(df) == 0:
        logger.info("No data for %s", title)
        return

    df_plot = df.copy()
    df_plot['abs_logfc'] = np.abs(df_plot['logfoldchange'])
    explained = df_plot[df_plot['explained']]
    unexplained = df_plot[~df_plot['explained']]

    logger.info("Visualization for %s: Total=%s, Explained=%s, Unexplained=%s",
                title, f"{len(df_plot):,}", f"{len(explained):,}", f"{len(unexplained):,}")

    if len(explained) > 0:
        logger.info("  Belief score range (explained): %.3f to %.3f",
                     explained['belief'].min(), explained['belief'].max())
        logger.info("  Belief scores below 0.7: %d", len(explained[explained['belief'] < 0.7]))

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'{title} - 1-Hop Analysis (Belief >=0.7)', fontsize=16, fontweight='bold')
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    if len(explained) > 0:
        ax1.scatter(explained['abs_logfc'], explained['belief'], alpha=0.7, s=25,
                    color='blue', label=f'Explained (n={len(explained):,})')
    if len(unexplained) > 0:
        ax1.scatter(unexplained['abs_logfc'], unexplained['belief'], alpha=0.4, s=15,
                    color='red', label=f'Unexplained (n={len(unexplained):,})')
    ax1.set_xlabel('|Log Fold Change|')
    ax1.set_ylabel('Belief Score')
    ax1.set_title('All Data Types (Scatter)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    if len(explained) > 0:
        h2 = ax2.hist2d(explained['abs_logfc'], explained['belief'], bins=50, cmap='viridis')
        plt.colorbar(h2[3], ax=ax2)
        logger.info("Heat map: Using %s explained data points", f"{len(explained):,}")
    ax2.set_xlabel('|Log Fold Change|')
    ax2.set_ylabel('Belief Score')
    ax2.set_title('LogFC vs Belief Score (Heat Map)')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    if len(explained) > 0:
        ax3.hist(explained['belief'], bins=30, alpha=0.7, color='blue', density=True)
        ax3.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='Belief = 0.7')
    ax3.set_xlabel('Belief Score')
    ax3.set_ylabel('Density')
    ax3.set_title('Belief Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])
    if len(explained) > 0:
        ax4.hist(explained['abs_logfc'], bins=40, density=False, color='blue', alpha=0.7)
        ax4.set_title(f'1-hop Explained (n={len(explained):,})')
        logger.info("1-hop histogram: %s points, range %.2f-%.2f",
                     f"{len(explained):,}", explained['abs_logfc'].min(), explained['abs_logfc'].max())
    else:
        ax4.text(0.5, 0.5, 'No 1-hop data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('1-hop Explained (n=0)')
    ax4.set_xlabel('|Log Fold Change|')
    ax4.set_ylabel('Count')
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    if len(unexplained) > 0:
        ax5.hist(unexplained['abs_logfc'], bins=100, density=False, color='red', alpha=0.7)
        ax5.set_title(f'Unexplained (n={len(unexplained):,})')
        logger.info("Unexplained histogram: %s points, range %.2f-%.2f",
                     f"{len(unexplained):,}", unexplained['abs_logfc'].min(), unexplained['abs_logfc'].max())
    else:
        ax5.text(0.5, 0.5, 'No unexplained data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Unexplained (n=0)')
    ax5.set_xlabel('|Log Fold Change|')
    ax5.set_ylabel('Count')
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[0, 3])
    ax6.axis('off')
    summary_text = f"SUMMARY STATISTICS (Belief >=0.7)\n\nTotal Data Points: {len(df_plot):,}\n"
    summary_text += f"EXPLAINED: {len(explained):,} ({len(explained)/len(df_plot)*100:.1f}%)\n"
    summary_text += f"UNEXPLAINED: {len(unexplained):,} ({len(unexplained)/len(df_plot)*100:.1f}%)\n\nFOLD CHANGE STATS:\n"
    if len(explained) > 0:
        summary_text += f"  Explained mean: {explained['abs_logfc'].mean():.3f}\n"
        summary_text += f"  Belief range: {explained['belief'].min():.3f}-{explained['belief'].max():.3f}\n"
    if len(unexplained) > 0:
        summary_text += f"  Unexplained mean: {unexplained['abs_logfc'].mean():.3f}\n"
    if len(explained) > 0:
        high_fc_explained = len(explained[explained['abs_logfc'] > 1.0])
        summary_text += f"\nHIGH FC (>1.0):\n"
        summary_text += f"  Explained: {high_fc_explained} ({high_fc_explained/len(explained)*100:.1f}%)\n"
    if len(unexplained) > 0:
        high_fc_unexplained = len(unexplained[unexplained['abs_logfc'] > 1.0])
        summary_text += f"  Unexplained: {high_fc_unexplained} ({high_fc_unexplained/len(unexplained)*100:.1f}%)\n"
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    ax7 = fig.add_subplot(gs[1, 2:])
    ax7.axis('off')
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)

    logger.info("Visualization complete for %s", title)
    return df_plot


def comprehensive_analysis(df, name):
    """Run comprehensive statistical analysis on a dataset."""
    if len(df) == 0:
        logger.info("%s: No data for analysis", name)
        return
    explained = df[df['explained']]
    unexplained = df[~df['explained']]

    logger.info("%s:", name)
    logger.info("  Total data points: %s", f"{len(df):,}")
    logger.info("  Explained pathways: %s (%.2f%%)", f"{len(explained):,}", len(explained) / len(df) * 100)
    logger.info("  Unexplained targets: %s (%.2f%%)", f"{len(unexplained):,}", len(unexplained) / len(df) * 100)

    if len(df) > 0:
        logger.info("  Unique sources: %s", f"{df['source'].nunique():,}")
        logger.info("  Unique targets: %s", f"{df['target'].nunique():,}")

    if len(explained) > 0:
        abs_fc = np.abs(explained['logfoldchange'])
        belief = explained['belief']
        pval = explained['pval']

        logger.info("  Explained |Fold Change|: Mean=%.4f, Median=%.4f, Std=%.4f, Min=%.4f, Max=%.4f, 95th=%.4f",
                     abs_fc.mean(), abs_fc.median(), abs_fc.std(), abs_fc.min(), abs_fc.max(),
                     np.percentile(abs_fc, 95))
        logger.info("  Belief Score: Mean=%.4f, Median=%.4f, Std=%.4f, Min=%.4f, Max=%.4f",
                     belief.mean(), belief.median(), belief.std(), belief.min(), belief.max())
        logger.info("  P-value: Mean=%.6f, Median=%.6f, Min=%.2e, Max=%.4f",
                     pval.mean(), pval.median(), pval.min(), pval.max())

        correlation_fc_belief = np.corrcoef(abs_fc, belief)[0, 1]
        correlation_pval_belief = np.corrcoef(pval, belief)[0, 1]
        one_minus_pval = 1 - pval
        correlation_sig_belief = np.corrcoef(one_minus_pval, belief)[0, 1]
        logger.info("  Correlation |LogFC| vs Belief: r=%.6f", correlation_fc_belief)
        logger.info("  Correlation P-value vs Belief: r=%.6f", correlation_pval_belief)
        logger.info("  Correlation (1-P-value) vs Belief: r=%.6f", correlation_sig_belief)

        high_belief = explained[explained['belief'] >= 0.8]
        low_belief = explained[explained['belief'] <= 0.7]
        logger.info("  High belief (>=0.8): %s (%.1f%%)", f"{len(high_belief):,}",
                     len(high_belief) / len(explained) * 100)
        if len(high_belief) > 0:
            logger.info("    Mean |FC|: %.4f", np.abs(high_belief['logfoldchange']).mean())
        logger.info("  Low belief (<=0.7): %s (%.1f%%)", f"{len(low_belief):,}",
                     len(low_belief) / len(explained) * 100)
        if len(low_belief) > 0:
            logger.info("    Mean |FC|: %.4f", np.abs(low_belief['logfoldchange']).mean())

    if len(unexplained) > 0:
        abs_fc_unexp = np.abs(unexplained['logfoldchange'])
        pval_unexp = unexplained['pval']
        logger.info("  Unexplained |FC|: Mean=%.4f, Median=%.4f, Std=%.4f, 95th=%.4f",
                     abs_fc_unexp.mean(), abs_fc_unexp.median(), abs_fc_unexp.std(),
                     np.percentile(abs_fc_unexp, 95))
        logger.info("  Unexplained P-value: Mean=%.6f, Median=%.6f", pval_unexp.mean(), pval_unexp.median())

    if len(explained) > 0 and len(unexplained) > 0:
        abs_fc_exp = np.abs(explained['logfoldchange'])
        abs_fc_unexp = np.abs(unexplained['logfoldchange'])
        try:
            from scipy import stats
            t_stat, t_pval = stats.ttest_ind(abs_fc_exp, abs_fc_unexp)
            mw_stat, mw_pval = stats.mannwhitneyu(abs_fc_exp, abs_fc_unexp, alternative='two-sided')
            logger.info("  Explained vs Unexplained: Mean |FC| %.4f vs %.4f, fold=%.2fx",
                         abs_fc_exp.mean(), abs_fc_unexp.mean(),
                         abs_fc_exp.mean() / abs_fc_unexp.mean())
            logger.info("  T-test p=%.2e, Mann-Whitney U p=%.2e", t_pval, mw_pval)
            explained_high_fc = len(explained[np.abs(explained['logfoldchange']) > 1.0])
            unexplained_high_fc = len(unexplained[np.abs(unexplained['logfoldchange']) > 1.0])
            logger.info("  High |FC| (>1.0): Explained %s (%.1f%%), Unexplained %s (%.1f%%)",
                         f"{explained_high_fc:,}", explained_high_fc / len(explained) * 100,
                         f"{unexplained_high_fc:,}", unexplained_high_fc / len(unexplained) * 100)
        except ImportError:
            logger.info("  Explained vs Unexplained: Mean |FC| %.4f vs %.4f (scipy not available)",
                         abs_fc_exp.mean(), abs_fc_unexp.mean())


def create_summary_table(df_all, df_tp53, df_no_tp53):
    """Create and log a summary table."""
    datasets = [
        ("All Perturbations", df_all),
        ("TP53 Only", df_tp53),
        ("Excluding TP53", df_no_tp53)
    ]

    header = (f"{'Dataset':<20} {'Total':<8} {'Explained':<10} {'Unexplained':<12} "
              f"{'Expl %':<8} {'Mean |FC| (Exp)':<15} {'Mean |FC| (Unexp)':<16} {'Correlation':<12}")
    logger.info(header)
    logger.info("-" * 115)

    for name, df in datasets:
        if len(df) == 0:
            logger.info("%-20s %-8s %-10s %-12s %-8s %-15s %-16s %-12s",
                         name, "0", "0", "0", "0.0%", "N/A", "N/A", "N/A")
            continue

        explained = df[df['explained']]
        unexplained = df[~df['explained']]
        total = len(df)
        n_explained = len(explained)
        n_unexplained = len(unexplained)
        expl_pct = f"{n_explained / total * 100:.1f}%" if total > 0 else "0.0%"
        mean_fc_exp = f"{np.abs(explained['logfoldchange']).mean():.3f}" if len(explained) > 0 else "N/A"
        mean_fc_unexp = f"{np.abs(unexplained['logfoldchange']).mean():.3f}" if len(unexplained) > 0 else "N/A"
        if len(explained) >= 2:
            corr = np.corrcoef(np.abs(explained['logfoldchange']), explained['belief'])[0, 1]
            corr_str = f"{corr:.4f}"
        else:
            corr_str = "N/A"
        logger.info("%-20s %-8s %-10s %-12s %-8s %-15s %-16s %-12s",
                     name, str(total), str(n_explained), str(n_unexplained),
                     expl_pct, mean_fc_exp, mean_fc_unexp, corr_str)


def main():
    parser = argparse.ArgumentParser(description="1-hop pathway EDA with belief score cutoff")
    parser.add_argument("--hop1-all-xlsx", required=True, help="Path to 1-hop all perturbations Excel")
    parser.add_argument("--hop1-no-tp53-csv", required=True, help="Path to 1-hop no TP53 CSV")
    parser.add_argument("--deg-folder", required=True, help="Path to de_results_per_gene folder")
    parser.add_argument("--belief-cutoff", type=float, default=0.7, help="Belief score cutoff")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        from scipy import stats  # noqa: F401
    except ImportError:
        logger.warning("scipy not available for statistical tests")

    logger.info("Loading 1-hop pathway datasets with belief score cutoff >= %.1f", args.belief_cutoff)

    df_1hop_all = pd.read_excel(args.hop1_all_xlsx)
    logger.info("1-hop all perturbations (before filtering): %s pathways", f"{len(df_1hop_all):,}")

    df_1hop_no_tp53 = pd.read_csv(args.hop1_no_tp53_csv)
    logger.info("1-hop excluding TP53 (before filtering): %s pathways", f"{len(df_1hop_no_tp53):,}")

    logger.info("Column structures:")
    logger.info("All perturbations columns: %s", list(df_1hop_all.columns))
    logger.info("Excluding TP53 columns: %s", list(df_1hop_no_tp53.columns))

    df_1hop_all = standardize_columns(df_1hop_all, "all perturbations")
    df_1hop_no_tp53 = standardize_columns(df_1hop_no_tp53, "excluding TP53")

    logger.info("Applying belief score cutoff >= %.1f", args.belief_cutoff)
    df_1hop_all_filtered = df_1hop_all[df_1hop_all['belief'] >= args.belief_cutoff].copy()
    df_1hop_no_tp53_filtered = df_1hop_no_tp53[df_1hop_no_tp53['belief'] >= args.belief_cutoff].copy()

    logger.info("1-hop all (after filtering): %s pathways", f"{len(df_1hop_all_filtered):,}")
    logger.info("1-hop no TP53 (after filtering): %s pathways", f"{len(df_1hop_no_tp53_filtered):,}")
    logger.info("Filtered out: %s from all, %s from no-TP53",
                f"{len(df_1hop_all) - len(df_1hop_all_filtered):,}",
                f"{len(df_1hop_no_tp53) - len(df_1hop_no_tp53_filtered):,}")

    df_1hop_tp53_only_filtered = df_1hop_all_filtered[df_1hop_all_filtered['source'] == 'TP53'].copy()
    logger.info("1-hop TP53 only (after filtering): %s pathways", f"{len(df_1hop_tp53_only_filtered):,}")

    deg_folder = args.deg_folder
    if not os.path.exists(deg_folder):
        logger.error("DEG folder not found at %s", deg_folder)
        return

    deg_files = list(Path(deg_folder).glob("*_vs_control.csv"))
    logger.info("Found %d DEG files", len(deg_files))

    if len(deg_files) == 0:
        logger.error("No DEG files found matching pattern '*_vs_control.csv'")
        return

    logger.info("First few DEG files:")
    for f in deg_files[:5]:
        logger.info("  %s", f.name)
    if len(deg_files) > 5:
        logger.info("  ... and %d more files", len(deg_files) - 5)

    df_all_combined = process_deg_and_pathways(
        df_1hop_all_filtered, "All perturbations (filtered)", deg_files, include_tp53=True)
    df_tp53_combined = process_deg_and_pathways(
        df_1hop_tp53_only_filtered, "TP53 only (filtered)", deg_files, include_tp53=True)
    df_no_tp53_combined = process_deg_and_pathways(
        df_1hop_no_tp53_filtered, "Excluding TP53 (filtered)", deg_files, include_tp53=False)

    logger.info("SUMMARY STATISTICS (BELIEF CUTOFF >= %.1f)", args.belief_cutoff)
    print_dataset_stats(df_all_combined, "All perturbations (filtered)")
    print_dataset_stats(df_tp53_combined, "TP53 only (filtered)")
    print_dataset_stats(df_no_tp53_combined, "Excluding TP53 (filtered)")

    logger.info("Creating visualizations")
    create_heatmap_plot(df_all_combined, "All Perturbations", "1hop_all_perturbations_belief_cutoff")
    create_heatmap_plot(df_tp53_combined, "TP53 Only", "1hop_tp53_only_belief_cutoff")
    create_heatmap_plot(df_no_tp53_combined, "Excluding TP53", "1hop_excluding_tp53_belief_cutoff")

    logger.info("COMPREHENSIVE STATISTICAL ANALYSIS (BELIEF CUTOFF >= %.1f)", args.belief_cutoff)
    comprehensive_analysis(df_all_combined, "All Perturbations (Belief >=0.7)")
    comprehensive_analysis(df_tp53_combined, "TP53 Only (Belief >=0.7)")
    comprehensive_analysis(df_no_tp53_combined, "Excluding TP53 (Belief >=0.7)")

    logger.info("SUMMARY TABLE (BELIEF CUTOFF >= %.1f)", args.belief_cutoff)
    create_summary_table(df_all_combined, df_tp53_combined, df_no_tp53_combined)

    logger.info("Belief cutoff analysis complete")


if __name__ == "__main__":
    main()
