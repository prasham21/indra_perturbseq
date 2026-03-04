"""Combined 1-hop + 2-hop pathway EDA with priority logic and visualizations."""
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
        if 'belief_2' in df.columns:
            df['belief'] = (df['belief_1'] + df['belief_2']) / 2
            logger.info("Created mean belief from belief_1 and belief_2 in %s", dataset_name)
        else:
            df['belief'] = df['belief_1']
            logger.info("Renamed 'belief_1' to 'belief' in %s", dataset_name)
    if 'pvalue' in df.columns and 'pval' not in df.columns:
        df = df.rename(columns={'pvalue': 'pval'})
        logger.info("Renamed 'pvalue' to 'pval' in %s", dataset_name)
    return df


def create_combined_dataset(df_1hop, df_2hop, dataset_name, include_tp53=True):
    """Create combined dataset with priority logic: 1-hop first, then best 2-hop by belief."""
    logger.info("Processing %s...", dataset_name)
    all_pathways = []

    if len(df_1hop) > 0:
        for _, row in df_1hop.iterrows():
            if not include_tp53 and row['source'] == 'TP53':
                continue
            all_pathways.append({
                'source': row['source'], 'target': row['target'],
                'belief': row['belief'], 'logfoldchange': row['logfoldchange'],
                'pval': row['pval'], 'pathway_type': '1-hop', 'priority': 1
            })

    if len(df_2hop) > 0:
        for _, row in df_2hop.iterrows():
            if not include_tp53 and row['source'] == 'TP53':
                continue
            all_pathways.append({
                'source': row['source'], 'target': row['target'],
                'belief': row['belief'], 'logfoldchange': row['logfoldchange'],
                'pval': row['pval'], 'pathway_type': '2-hop', 'priority': 2
            })

    combined_df = pd.DataFrame(all_pathways)
    if len(combined_df) == 0:
        logger.info("  No data for %s", dataset_name)
        return pd.DataFrame()

    logger.info("  Combined raw data: %s pathways", f"{len(combined_df):,}")
    logger.info("    1-hop: %s", f"{len(combined_df[combined_df['pathway_type'] == '1-hop']):,}")
    logger.info("    2-hop: %s", f"{len(combined_df[combined_df['pathway_type'] == '2-hop']):,}")

    def select_best_pathway(group):
        sorted_group = group.sort_values(['priority', 'belief'], ascending=[True, False])
        return sorted_group.iloc[0]

    best_pathways = combined_df.groupby(['source', 'target']).apply(select_best_pathway).reset_index(drop=True)

    logger.info("  After priority logic: %s unique source-target pairs", f"{len(best_pathways):,}")
    logger.info("    1-hop selected: %s", f"{len(best_pathways[best_pathways['pathway_type'] == '1-hop']):,}")
    logger.info("    2-hop selected: %s", f"{len(best_pathways[best_pathways['pathway_type'] == '2-hop']):,}")

    reduction = len(combined_df) - len(best_pathways)
    logger.info("  Removed %s redundant pathways (%.1f%%)",
                f"{reduction:,}", reduction / len(combined_df) * 100)
    return best_pathways


def add_unexplained_targets(combined_df, dataset_name, deg_folder, include_tp53=True):
    """Add unexplained targets to combined dataset."""
    logger.info("Adding unexplained targets for %s...", dataset_name)
    deg_files = list(Path(deg_folder).glob("*_vs_control.csv"))

    unexplained_data = []
    explained_pairs = set()

    for _, row in combined_df.iterrows():
        explained_pairs.add((row['source'], row['target']))

    pathway_sources = combined_df['source'].unique() if len(combined_df) > 0 else []

    for deg_file in deg_files:
        gene_name = deg_file.stem.replace("_vs_control", "")
        if not include_tp53 and gene_name == 'TP53':
            continue
        if gene_name not in pathway_sources:
            continue
        try:
            deg_df = pd.read_csv(deg_file)
            significant_genes = deg_df[deg_df['pvals'] < 0.05].copy()
            for _, deg_row in significant_genes.iterrows():
                target_gene = deg_row['names']
                if (gene_name, target_gene) in explained_pairs:
                    continue
                unexplained_data.append({
                    'source': gene_name, 'target': target_gene,
                    'belief': 0.5, 'logfoldchange': deg_row['logfoldchanges'],
                    'pval': deg_row['pvals'], 'pathway_type': 'none',
                    'priority': 3, 'explained': False
                })
        except Exception:
            continue

    combined_df['explained'] = True
    if unexplained_data:
        unexplained_df = pd.DataFrame(unexplained_data)
        final_df = pd.concat([combined_df, unexplained_df], ignore_index=True)
    else:
        final_df = combined_df.copy()

    final_df['explained'] = final_df['explained'].fillna(False)
    explained_count = len(final_df[final_df['explained']])
    unexplained_count = len(final_df[~final_df['explained']])

    logger.info("  Final dataset: %s total pairs", f"{len(final_df):,}")
    logger.info("    Explained: %s (%.1f%%)", f"{explained_count:,}", explained_count / len(final_df) * 100)
    logger.info("    Unexplained: %s (%.1f%%)", f"{unexplained_count:,}", unexplained_count / len(final_df) * 100)
    return final_df


def print_combined_stats(df, name):
    """Log combined dataset statistics."""
    if len(df) == 0:
        logger.info("%s: No data", name)
        return
    explained = df[df['explained']]
    unexplained = df[~df['explained']]
    onehop = explained[explained['pathway_type'] == '1-hop']
    twohop = explained[explained['pathway_type'] == '2-hop']

    logger.info("%s:", name)
    logger.info("  Total pairs: %s", f"{len(df):,}")
    logger.info("  Explained: %s (%.1f%%)", f"{len(explained):,}", len(explained) / len(df) * 100)
    logger.info("    1-hop selected: %s (%.1f%% of explained)",
                f"{len(onehop):,}", len(onehop) / len(explained) * 100)
    logger.info("    2-hop selected: %s (%.1f%% of explained)",
                f"{len(twohop):,}", len(twohop) / len(explained) * 100)
    logger.info("  Unexplained: %s (%.1f%%)", f"{len(unexplained):,}", len(unexplained) / len(df) * 100)

    if len(explained) > 0:
        logger.info("  Explained - Mean |FC|: %.3f, Mean belief: %.3f",
                     np.abs(explained['logfoldchange']).mean(), explained['belief'].mean())
    if len(unexplained) > 0:
        logger.info("  Unexplained - Mean |FC|: %.3f", np.abs(unexplained['logfoldchange']).mean())


def create_combined_plot(df, title, save_name):
    """Create visualization for combined 1-hop + 2-hop analysis."""
    if len(df) == 0:
        logger.info("No data for %s", title)
        return

    df_plot = df.copy()
    df_plot['abs_logfc'] = np.abs(df_plot['logfoldchange'])
    explained = df_plot[df_plot['explained']]
    unexplained = df_plot[~df_plot['explained']]
    onehop = explained[explained['pathway_type'] == '1-hop']
    twohop = explained[explained['pathway_type'] == '2-hop']

    logger.info("Visualization for %s: Total=%s, 1-hop=%s, 2-hop=%s, Unexplained=%s",
                title, f"{len(df_plot):,}", f"{len(onehop):,}", f"{len(twohop):,}", f"{len(unexplained):,}")

    high_fc_threshold = 2.5
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'{title} - Combined 1-Hop + 2-Hop Analysis', fontsize=16, fontweight='bold')
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    if len(onehop) > 0:
        ax1.scatter(onehop['abs_logfc'], onehop['belief'], alpha=0.7, s=25,
                    color='blue', label=f'1-hop (n={len(onehop):,})')
    if len(twohop) > 0:
        ax1.scatter(twohop['abs_logfc'], twohop['belief'], alpha=0.7, s=25,
                    color='green', label=f'2-hop (n={len(twohop):,})')
    if len(unexplained) > 0:
        ax1.scatter(unexplained['abs_logfc'], unexplained['belief'], alpha=0.4, s=15,
                    color='red', label=f'Unexplained (n={len(unexplained):,})')
    ax1.set_xlabel('|Log Fold Change|')
    ax1.set_ylabel('Belief Score')
    ax1.set_title('All Pathway Types (Scatter)')
    ax1.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05))
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    if len(explained) > 0:
        h2 = ax2.hist2d(explained['abs_logfc'], explained['belief'], bins=50, cmap='viridis')
        plt.colorbar(h2[3], ax=ax2)
    ax2.set_xlabel('|Log Fold Change|')
    ax2.set_ylabel('Belief Score')
    ax2.set_title('LogFC vs Belief Score (Heat Map)')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    if len(onehop) > 0:
        ax3.hist(onehop['abs_logfc'], bins=40, density=False, color='blue', alpha=0.7)
        ax3.set_title(f'1-hop (n={len(onehop):,})')
    else:
        ax3.text(0.5, 0.5, 'No 1-hop data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('1-hop (n=0)')
    ax3.set_xlabel('|Log Fold Change|')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    if len(twohop) > 0:
        ax4.hist(twohop['abs_logfc'], bins=100, density=False, color='green', alpha=0.7)
        ax4.set_title(f'2-hop (n={len(twohop):,})')
    else:
        ax4.text(0.5, 0.5, 'No 2-hop data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('2-hop (n=0)')
    ax4.set_xlabel('|Log Fold Change|')
    ax4.set_ylabel('Count')
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    if len(unexplained) > 0:
        ax5.hist(unexplained['abs_logfc'], bins=100, density=False, color='red', alpha=0.7)
        ax5.set_title(f'Unexplained (n={len(unexplained):,})')
    else:
        ax5.text(0.5, 0.5, 'No unexplained data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Unexplained (n=0)')
    ax5.set_xlabel('|Log Fold Change|')
    ax5.set_ylabel('Count')
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[0, 2:])
    if len(onehop) > 0:
        ax6.hist(onehop['belief'], bins=100, alpha=0.7, color='blue',
                 label=f'1-hop (n={len(onehop):,})', density=True)
    if len(twohop) > 0:
        ax6.hist(twohop['belief'], bins=100, alpha=0.7, color='green',
                 label=f'2-hop (n={len(twohop):,})', density=True)
    ax6.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='Belief = 0.7')
    ax6.set_xlabel('Belief Score')
    ax6.set_ylabel('Density')
    ax6.set_title('Belief Score Distribution (Explained Only)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(gs[1, 3])
    ax7.axis('off')
    summary_text = f"SUMMARY STATISTICS\n\nTotal: {len(df_plot):,}\n\n"
    summary_text += f"EXPLAINED ({len(explained):,}):\n  1-hop: {len(onehop):,}\n  2-hop: {len(twohop):,}\n\n"
    summary_text += f"UNEXPLAINED: {len(unexplained):,}\n\nFOLD CHANGE STATS:\n"
    if len(explained) > 0:
        summary_text += f"  Explained mean: {explained['abs_logfc'].mean():.3f}\n"
    if len(unexplained) > 0:
        summary_text += f"  Unexplained mean: {unexplained['abs_logfc'].mean():.3f}\n"
    summary_text += f"\nHIGH FC (>{high_fc_threshold}):\n"
    if len(onehop) > 0:
        summary_text += f"  1-hop: {len(onehop[onehop['abs_logfc'] > high_fc_threshold])}\n"
    if len(twohop) > 0:
        summary_text += f"  2-hop: {len(twohop[twohop['abs_logfc'] > high_fc_threshold])}\n"
    if len(unexplained) > 0:
        summary_text += f"  Unexplained: {len(unexplained[unexplained['abs_logfc'] > high_fc_threshold])}\n"
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)
    logger.info("Visualization complete for %s", title)
    return df_plot


def comprehensive_combined_analysis(df, name):
    """Run comprehensive statistical analysis on combined dataset."""
    if len(df) == 0:
        logger.info("%s: No data for analysis", name)
        return
    explained = df[df['explained']]
    unexplained = df[~df['explained']]
    onehop = explained[explained['pathway_type'] == '1-hop']
    twohop = explained[explained['pathway_type'] == '2-hop']

    logger.info("%s:", name)
    logger.info("  Total: %s, Explained: %s (%.2f%%), Unexplained: %s (%.2f%%)",
                f"{len(df):,}", f"{len(explained):,}", len(explained) / len(df) * 100,
                f"{len(unexplained):,}", len(unexplained) / len(df) * 100)
    logger.info("    1-hop: %s (%.1f%%), 2-hop: %s (%.1f%%)",
                f"{len(onehop):,}", len(onehop) / len(explained) * 100,
                f"{len(twohop):,}", len(twohop) / len(explained) * 100)

    if len(df) > 0:
        logger.info("  Unique sources: %s, targets: %s",
                     f"{df['source'].nunique():,}", f"{df['target'].nunique():,}")

    if len(explained) > 0:
        abs_fc = np.abs(explained['logfoldchange'])
        belief = explained['belief']
        pval = explained['pval']
        logger.info("  Explained |FC|: Mean=%.4f, Median=%.4f, 95th=%.4f",
                     abs_fc.mean(), abs_fc.median(), np.percentile(abs_fc, 95))
        logger.info("  Belief: Mean=%.4f, Median=%.4f", belief.mean(), belief.median())

        corr_fc_belief = np.corrcoef(abs_fc, belief)[0, 1]
        corr_pval_belief = np.corrcoef(pval, belief)[0, 1]
        corr_sig_belief = np.corrcoef(1 - pval, belief)[0, 1]
        logger.info("  Correlation |LogFC| vs Belief: r=%.6f", corr_fc_belief)
        logger.info("  Correlation P-value vs Belief: r=%.6f", corr_pval_belief)
        logger.info("  Correlation (1-P-value) vs Belief: r=%.6f", corr_sig_belief)

        if len(onehop) > 0 and len(twohop) > 0:
            logger.info("  1-hop vs 2-hop: mean |FC| %.4f vs %.4f, mean belief %.4f vs %.4f",
                         np.abs(onehop['logfoldchange']).mean(), np.abs(twohop['logfoldchange']).mean(),
                         onehop['belief'].mean(), twohop['belief'].mean())

    if len(explained) > 0 and len(unexplained) > 0:
        abs_fc_exp = np.abs(explained['logfoldchange'])
        abs_fc_unexp = np.abs(unexplained['logfoldchange'])
        try:
            from scipy import stats
            t_stat, t_pval = stats.ttest_ind(abs_fc_exp, abs_fc_unexp)
            mw_stat, mw_pval = stats.mannwhitneyu(abs_fc_exp, abs_fc_unexp, alternative='two-sided')
            logger.info("  Explained vs Unexplained: Mean |FC| %.4f vs %.4f, fold=%.2fx",
                         abs_fc_exp.mean(), abs_fc_unexp.mean(), abs_fc_exp.mean() / abs_fc_unexp.mean())
            logger.info("  T-test p=%.2e, Mann-Whitney U p=%.2e", t_pval, mw_pval)
        except ImportError:
            logger.info("  Explained vs Unexplained: Mean |FC| %.4f vs %.4f (scipy not available)",
                         abs_fc_exp.mean(), abs_fc_unexp.mean())


def create_combined_summary_table(df_all, df_tp53, df_no_tp53):
    """Create and log summary table."""
    datasets = [("All Perturbations", df_all), ("TP53 Only", df_tp53), ("Excluding TP53", df_no_tp53)]
    header = (f"{'Dataset':<20} {'Total':<8} {'Explained':<10} {'1-hop':<8} {'2-hop':<8} "
              f"{'Unexpl':<8} {'Expl%':<8} {'Mean|FC|(Exp)':<13} {'Mean|FC|(Unexp)':<14} {'Correlation':<12}")
    logger.info(header)
    logger.info("-" * 125)

    for name, df in datasets:
        if len(df) == 0:
            logger.info("%-20s %-8s %-10s %-8s %-8s %-8s %-8s %-13s %-14s %-12s",
                         name, "0", "0", "0", "0", "0", "0.0%", "N/A", "N/A", "N/A")
            continue
        explained = df[df['explained']]
        unexplained = df[~df['explained']]
        onehop = explained[explained['pathway_type'] == '1-hop']
        twohop = explained[explained['pathway_type'] == '2-hop']
        total = len(df)
        expl_pct = f"{len(explained) / total * 100:.1f}%" if total > 0 else "0.0%"
        mean_fc_exp = f"{np.abs(explained['logfoldchange']).mean():.3f}" if len(explained) > 0 else "N/A"
        mean_fc_unexp = f"{np.abs(unexplained['logfoldchange']).mean():.3f}" if len(unexplained) > 0 else "N/A"
        if len(explained) >= 2:
            corr = np.corrcoef(np.abs(explained['logfoldchange']), explained['belief'])[0, 1]
            corr_str = f"{corr:.4f}"
        else:
            corr_str = "N/A"
        logger.info("%-20s %-8s %-10s %-8s %-8s %-8s %-8s %-13s %-14s %-12s",
                     name, str(total), str(len(explained)), str(len(onehop)), str(len(twohop)),
                     str(len(unexplained)), expl_pct, mean_fc_exp, mean_fc_unexp, corr_str)


def main():
    parser = argparse.ArgumentParser(description="Combined 1-hop + 2-hop pathway EDA")
    parser.add_argument("--hop1-all-xlsx", required=True, help="Path to 1-hop all perturbations Excel")
    parser.add_argument("--hop1-no-tp53-csv", required=True, help="Path to 1-hop no TP53 CSV")
    parser.add_argument("--hop2-no-tp53-xlsx", required=True, help="Path to 2-hop excluding TP53 Excel")
    parser.add_argument("--hop2-tp53-csv", required=True, help="Path to 2-hop TP53 only CSV")
    parser.add_argument("--deg-folder", required=True, help="Path to de_results_per_gene folder")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        from scipy import stats  # noqa: F401
    except ImportError:
        logger.warning("scipy not available for statistical tests")

    logger.info("Loading datasets for combined 1-hop + 2-hop analysis")

    df_1hop_all = pd.read_excel(args.hop1_all_xlsx)
    df_1hop_no_tp53 = pd.read_csv(args.hop1_no_tp53_csv)
    df_2hop_no_tp53 = pd.read_excel(args.hop2_no_tp53_xlsx)
    df_2hop_tp53_only = pd.read_csv(args.hop2_tp53_csv)

    logger.info("1-hop all: %s, 1-hop no TP53: %s, 2-hop no TP53: %s, 2-hop TP53: %s",
                f"{len(df_1hop_all):,}", f"{len(df_1hop_no_tp53):,}",
                f"{len(df_2hop_no_tp53):,}", f"{len(df_2hop_tp53_only):,}")

    logger.info("Standardizing column names")
    df_1hop_all = standardize_columns(df_1hop_all, "1-hop all")
    df_1hop_no_tp53 = standardize_columns(df_1hop_no_tp53, "1-hop no TP53")
    df_2hop_no_tp53 = standardize_columns(df_2hop_no_tp53, "2-hop no TP53")
    df_2hop_tp53_only = standardize_columns(df_2hop_tp53_only, "2-hop TP53 only")

    df_1hop_all['pathway_type'] = '1-hop'
    df_1hop_no_tp53['pathway_type'] = '1-hop'
    df_2hop_no_tp53['pathway_type'] = '2-hop'
    df_2hop_tp53_only['pathway_type'] = '2-hop'

    logger.info("Creating combined datasets with priority logic")
    df_combined_all = create_combined_dataset(
        df_1hop_all, pd.concat([df_2hop_no_tp53, df_2hop_tp53_only]),
        "All Perturbations", include_tp53=True)
    df_combined_no_tp53 = create_combined_dataset(
        df_1hop_no_tp53, df_2hop_no_tp53, "Excluding TP53", include_tp53=False)
    df_1hop_tp53_only = df_1hop_all[df_1hop_all['source'] == 'TP53'].copy()
    df_combined_tp53_only = create_combined_dataset(
        df_1hop_tp53_only, df_2hop_tp53_only, "TP53 Only", include_tp53=True)

    logger.info("Adding unexplained targets")
    df_final_all = add_unexplained_targets(df_combined_all, "All Perturbations",
                                           args.deg_folder, include_tp53=True)
    df_final_no_tp53 = add_unexplained_targets(df_combined_no_tp53, "Excluding TP53",
                                                args.deg_folder, include_tp53=False)
    df_final_tp53_only = add_unexplained_targets(df_combined_tp53_only, "TP53 Only",
                                                  args.deg_folder, include_tp53=True)

    logger.info("COMBINED DATASET SUMMARY STATISTICS")
    print_combined_stats(df_final_all, "All Perturbations (Combined)")
    print_combined_stats(df_final_tp53_only, "TP53 Only (Combined)")
    print_combined_stats(df_final_no_tp53, "Excluding TP53 (Combined)")

    logger.info("Creating combined visualizations")
    create_combined_plot(df_final_all, "All Perturbations", "combined_all_perturbations")
    create_combined_plot(df_final_tp53_only, "TP53 Only", "combined_tp53_only")
    create_combined_plot(df_final_no_tp53, "Excluding TP53", "combined_excluding_tp53")

    logger.info("COMPREHENSIVE STATISTICAL ANALYSIS")
    comprehensive_combined_analysis(df_final_all, "All Perturbations (Combined)")
    comprehensive_combined_analysis(df_final_tp53_only, "TP53 Only (Combined)")
    comprehensive_combined_analysis(df_final_no_tp53, "Excluding TP53 (Combined)")

    logger.info("COMPREHENSIVE SUMMARY TABLE")
    create_combined_summary_table(df_final_all, df_final_tp53_only, df_final_no_tp53)

    logger.info("Combined 1-hop + 2-hop analysis complete")


if __name__ == "__main__":
    main()
