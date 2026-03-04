"""1-hop pathway EDA: explained vs unexplained target analysis with visualizations."""
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
    logger.info("%s: Explained=%s, Unexplained=%s, Total=%s",
                dataset_name, f"{len(explained_data):,}",
                f"{len(unexplained_data):,}", f"{len(combined_df):,}")
    if len(combined_df) > 0:
        explained_count = combined_df['explained'].sum()
        logger.info("  Explained: %.1f%%", explained_count / len(combined_df) * 100)
    return combined_df


def print_dataset_stats(df, name):
    """Log summary statistics for a dataset."""
    if len(df) == 0:
        logger.info("%s: No data", name)
        return
    explained = df[df['explained']]
    unexplained = df[~df['explained']]
    logger.info("%s:", name)
    logger.info("  Total: %s, Explained: %s (%.1f%%), Unexplained: %s (%.1f%%)",
                f"{len(df):,}", f"{len(explained):,}", len(explained) / len(df) * 100,
                f"{len(unexplained):,}", len(unexplained) / len(df) * 100)
    if len(explained) > 0:
        logger.info("  Explained - Mean |FC|: %.3f, Mean belief: %.3f",
                     np.abs(explained['logfoldchange']).mean(), explained['belief'].mean())
    if len(unexplained) > 0:
        logger.info("  Unexplained - Mean |FC|: %.3f", np.abs(unexplained['logfoldchange']).mean())


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

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'{title} - 1-Hop Analysis', fontsize=16, fontweight='bold')
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    if len(explained) > 0:
        h1 = ax1.hist2d(explained['abs_logfc'], explained['belief'], bins=50, cmap='Blues')
        plt.colorbar(h1[3], ax=ax1)
    ax1.set_xlabel('|Log Fold Change|')
    ax1.set_ylabel('Belief Score')
    ax1.set_title('Explained Data Only (Heat Map)')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    if len(explained) > 0:
        ax2.scatter(explained['abs_logfc'], explained['belief'], alpha=0.7, s=25,
                    color='blue', label=f'Explained (n={len(explained):,})')
    if len(unexplained) > 0:
        ax2.scatter(unexplained['abs_logfc'], unexplained['belief'], alpha=0.4, s=15,
                    color='red', label=f'Unexplained (n={len(unexplained):,})')
    ax2.set_xlabel('|Log Fold Change|')
    ax2.set_ylabel('Belief Score')
    ax2.set_title('All Data Types (Scatter)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    if len(explained) > 0:
        ax3.hist(explained['abs_logfc'], bins=40, density=False, color='blue', alpha=0.7)
        ax3.set_title(f'1-hop Explained (n={len(explained):,})')
    else:
        ax3.text(0.5, 0.5, 'No 1-hop data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('1-hop Explained (n=0)')
    ax3.set_xlabel('|Log Fold Change|')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    if len(unexplained) > 0:
        ax4.hist(unexplained['abs_logfc'], bins=100, density=False, color='red', alpha=0.7)
        ax4.set_title(f'Unexplained (n={len(unexplained):,})')
    else:
        ax4.text(0.5, 0.5, 'No unexplained data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Unexplained (n=0)')
    ax4.set_xlabel('|Log Fold Change|')
    ax4.set_ylabel('Count')
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[0, 2:])
    if len(explained) > 0:
        ax5.hist(explained['belief'], bins=30, alpha=0.7, color='blue', density=True)
        ax5.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='Belief = 0.7')
    ax5.set_xlabel('Belief Score')
    ax5.set_ylabel('Density')
    ax5.set_title('Belief Score Distribution (Explained Only)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.axis('off')
    summary_text = f"SUMMARY STATISTICS\n\nTotal: {len(df_plot):,}\n"
    summary_text += f"Explained: {len(explained):,} ({len(explained)/len(df_plot)*100:.1f}%)\n"
    summary_text += f"Unexplained: {len(unexplained):,} ({len(unexplained)/len(df_plot)*100:.1f}%)\n"
    if len(explained) > 0:
        summary_text += f"\nExplained mean |FC|: {explained['abs_logfc'].mean():.3f}\n"
    if len(unexplained) > 0:
        summary_text += f"Unexplained mean |FC|: {unexplained['abs_logfc'].mean():.3f}\n"
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)
    return df_plot


def comprehensive_analysis(df, name):
    """Run comprehensive statistical analysis on a dataset."""
    if len(df) == 0:
        logger.info("%s: No data for analysis", name)
        return
    explained = df[df['explained']]
    unexplained = df[~df['explained']]

    logger.info("%s:", name)
    logger.info("  Total: %s, Explained: %s (%.2f%%), Unexplained: %s (%.2f%%)",
                f"{len(df):,}", f"{len(explained):,}", len(explained) / len(df) * 100,
                f"{len(unexplained):,}", len(unexplained) / len(df) * 100)

    if len(explained) > 0:
        abs_fc = np.abs(explained['logfoldchange'])
        belief = explained['belief']
        correlation = np.corrcoef(abs_fc, belief)[0, 1]
        logger.info("  |LogFC| vs Belief: r = %.6f", correlation)

    if len(explained) > 0 and len(unexplained) > 0:
        abs_fc_exp = np.abs(explained['logfoldchange'])
        abs_fc_unexp = np.abs(unexplained['logfoldchange'])
        try:
            from scipy import stats
            t_stat, t_pval = stats.ttest_ind(abs_fc_exp, abs_fc_unexp)
            logger.info("  Explained vs Unexplained: Mean |FC| %.4f vs %.4f, t-test p=%.2e",
                         abs_fc_exp.mean(), abs_fc_unexp.mean(), t_pval)
        except ImportError:
            logger.info("  Explained vs Unexplained: Mean |FC| %.4f vs %.4f",
                         abs_fc_exp.mean(), abs_fc_unexp.mean())


def main():
    parser = argparse.ArgumentParser(description="1-hop pathway EDA")
    parser.add_argument("--hop1-all-xlsx", required=True, help="Path to 1-hop all perturbations Excel")
    parser.add_argument("--hop1-no-tp53-csv", required=True, help="Path to 1-hop no TP53 CSV")
    parser.add_argument("--deg-folder", required=True, help="Path to de_results_per_gene folder")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Loading 1-hop pathway datasets")

    df_1hop_all = pd.read_excel(args.hop1_all_xlsx)
    logger.info("1-hop all perturbations: %s pathways", f"{len(df_1hop_all):,}")

    df_1hop_no_tp53 = pd.read_csv(args.hop1_no_tp53_csv)
    logger.info("1-hop excluding TP53: %s pathways", f"{len(df_1hop_no_tp53):,}")

    df_1hop_tp53_only = df_1hop_all[df_1hop_all['source'] == 'TP53'].copy()

    df_1hop_all = standardize_columns(df_1hop_all, "all perturbations")
    df_1hop_no_tp53 = standardize_columns(df_1hop_no_tp53, "excluding TP53")
    df_1hop_tp53_only = standardize_columns(df_1hop_tp53_only, "TP53 only")

    deg_folder = args.deg_folder
    if not os.path.exists(deg_folder):
        logger.error("DEG folder not found at %s", deg_folder)
        return

    deg_files = list(Path(deg_folder).glob("*_vs_control.csv"))
    logger.info("Found %d DEG files", len(deg_files))

    df_all_combined = process_deg_and_pathways(df_1hop_all, "All perturbations", deg_files, include_tp53=True)
    df_tp53_combined = process_deg_and_pathways(df_1hop_tp53_only, "TP53 only", deg_files, include_tp53=True)
    df_no_tp53_combined = process_deg_and_pathways(df_1hop_no_tp53, "Excluding TP53", deg_files, include_tp53=False)

    print_dataset_stats(df_all_combined, "All perturbations")
    print_dataset_stats(df_tp53_combined, "TP53 only")
    print_dataset_stats(df_no_tp53_combined, "Excluding TP53")

    logger.info("Creating visualizations")
    create_heatmap_plot(df_all_combined, "All Perturbations", "1hop_all_perturbations")
    create_heatmap_plot(df_tp53_combined, "TP53 Only", "1hop_tp53_only")
    create_heatmap_plot(df_no_tp53_combined, "Excluding TP53", "1hop_excluding_tp53")

    comprehensive_analysis(df_all_combined, "All Perturbations")
    comprehensive_analysis(df_tp53_combined, "TP53 Only")
    comprehensive_analysis(df_no_tp53_combined, "Excluding TP53")

    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
