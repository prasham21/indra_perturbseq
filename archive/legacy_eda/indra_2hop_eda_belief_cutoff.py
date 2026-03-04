"""Combined 1-hop + 2-hop pathway EDA with belief score cutoff (>=0.7) and advanced visualizations."""
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
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

try:
    from scipy.stats import gaussian_kde, ks_2samp
except ImportError:
    gaussian_kde = None
    ks_2samp = None

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
    """Create combined dataset with priority logic."""
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

    logger.info("  Combined raw: %s (1-hop: %s, 2-hop: %s)",
                f"{len(combined_df):,}",
                f"{len(combined_df[combined_df['pathway_type'] == '1-hop']):,}",
                f"{len(combined_df[combined_df['pathway_type'] == '2-hop']):,}")

    def select_best_pathway(group):
        return group.sort_values(['priority', 'belief'], ascending=[True, False]).iloc[0]

    best_pathways = combined_df.groupby(['source', 'target']).apply(select_best_pathway).reset_index(drop=True)
    logger.info("  After priority logic: %s unique pairs", f"{len(best_pathways):,}")
    reduction = len(combined_df) - len(best_pathways)
    logger.info("  Removed %s redundant (%.1f%%)", f"{reduction:,}", reduction / len(combined_df) * 100)
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
            for _, deg_row in deg_df[deg_df['pvals'] < 0.05].iterrows():
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
        final_df = pd.concat([combined_df, pd.DataFrame(unexplained_data)], ignore_index=True)
    else:
        final_df = combined_df.copy()
    final_df['explained'] = final_df['explained'].fillna(False)

    explained_count = len(final_df[final_df['explained']])
    unexplained_count = len(final_df[~final_df['explained']])
    logger.info("  Final: %s total, Explained: %s (%.1f%%), Unexplained: %s (%.1f%%)",
                f"{len(final_df):,}", f"{explained_count:,}", explained_count / len(final_df) * 100,
                f"{unexplained_count:,}", unexplained_count / len(final_df) * 100)
    return final_df


def print_combined_stats(df, name):
    """Log combined dataset stats."""
    if len(df) == 0:
        logger.info("%s: No data", name)
        return
    explained = df[df['explained']]
    unexplained = df[~df['explained']]
    onehop = explained[explained['pathway_type'] == '1-hop']
    twohop = explained[explained['pathway_type'] == '2-hop']

    logger.info("%s:", name)
    logger.info("  Total: %s, Explained: %s (%.1f%%)", f"{len(df):,}",
                f"{len(explained):,}", len(explained) / len(df) * 100)
    logger.info("    1-hop: %s (%.1f%%), 2-hop: %s (%.1f%%)",
                f"{len(onehop):,}", len(onehop) / len(explained) * 100,
                f"{len(twohop):,}", len(twohop) / len(explained) * 100)
    logger.info("  Unexplained: %s (%.1f%%)", f"{len(unexplained):,}", len(unexplained) / len(df) * 100)
    if len(explained) > 0:
        logger.info("  Explained - Mean |FC|: %.3f, belief: %.3f",
                     np.abs(explained['logfoldchange']).mean(), explained['belief'].mean())
    if len(unexplained) > 0:
        logger.info("  Unexplained - Mean |FC|: %.3f", np.abs(unexplained['logfoldchange']).mean())


def generate_high_fc_unexplained_csv(df, dataset_name, fc_threshold=2.5):
    """Generate CSV of unexplained targets with high fold change."""
    unexplained = df[~df['explained']].copy()
    if len(unexplained) == 0:
        logger.info("No unexplained targets in %s", dataset_name)
        return None
    unexplained['abs_logfc'] = np.abs(unexplained['logfoldchange'])
    high_fc = unexplained[unexplained['abs_logfc'] > fc_threshold].copy()
    if len(high_fc) == 0:
        logger.info("No unexplained targets with |LogFC| > %.1f in %s", fc_threshold, dataset_name)
        return None

    high_fc = high_fc.sort_values('abs_logfc', ascending=False)
    output_df = pd.DataFrame({
        'source_gene': high_fc['source'], 'target_gene': high_fc['target'],
        'log_fold_change': high_fc['logfoldchange'],
        'abs_log_fold_change': high_fc['abs_logfc'],
        'p_value': high_fc['pval'],
        'linear_fold_change': 2 ** high_fc['abs_logfc']
    })
    filename = f"high_fc_unexplained_{dataset_name.lower().replace(' ', '_')}.csv"
    output_df.to_csv(filename, index=False)
    logger.info("High FC unexplained CSV: %s (%s targets with |LogFC| > %.1f out of %s total)",
                filename, f"{len(output_df):,}", fc_threshold, f"{len(unexplained):,}")
    return output_df


def create_enhanced_combined_plot(df, title, save_name):
    """Create enhanced visualization with advanced plots."""
    if len(df) == 0:
        logger.info("No data for %s", title)
        return

    df_plot = df.copy()
    df_plot['abs_logfc'] = np.abs(df_plot['logfoldchange'])
    df_plot['neg_log10_pval'] = -np.log10(df_plot['pval'])

    explained = df_plot[df_plot['explained']]
    unexplained = df_plot[~df_plot['explained']]
    onehop = explained[explained['pathway_type'] == '1-hop']
    twohop = explained[explained['pathway_type'] == '2-hop']

    logger.info("Visualization for %s: Total=%s, 1-hop=%s, 2-hop=%s, Unexplained=%s",
                title, f"{len(df_plot):,}", f"{len(onehop):,}", f"{len(twohop):,}", f"{len(unexplained):,}")

    if len(explained) > 0:
        logger.info("  Belief range (explained): %.3f to %.3f",
                     explained['belief'].min(), explained['belief'].max())

    high_fc_threshold = 2.5

    # Main analysis figure
    fig1 = plt.figure(figsize=(20, 12))
    fig1.suptitle(f'{title} - Combined Analysis (Belief >=0.7)', fontsize=16, fontweight='bold')
    gs1 = fig1.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    ax1 = fig1.add_subplot(gs1[0, 0])
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

    ax2 = fig1.add_subplot(gs1[0, 1])
    if len(explained) > 0:
        try:
            h2 = ax2.hist2d(explained['abs_logfc'], explained['belief'],
                            bins=25, cmap='plasma', norm=LogNorm(vmin=1))
            plt.colorbar(h2[3], ax=ax2)
        except Exception:
            h2 = ax2.hist2d(explained['abs_logfc'], explained['belief'], bins=25, cmap='plasma')
            plt.colorbar(h2[3], ax=ax2)
    ax2.set_xlabel('|Log Fold Change|')
    ax2.set_ylabel('Belief Score')
    ax2.set_title('LogFC vs Belief (Enhanced Heat Map)')
    ax2.grid(True, alpha=0.3)

    ax3 = fig1.add_subplot(gs1[0, 2])
    if len(onehop) > 0:
        ax3.hist(onehop['belief'], bins=50, alpha=0.7, color='blue',
                 label=f'1-hop (n={len(onehop):,}, mean={onehop["belief"].mean():.3f})', density=True)
    if len(twohop) > 0:
        ax3.hist(twohop['belief'], bins=50, alpha=0.7, color='green',
                 label=f'2-hop (n={len(twohop):,}, mean={twohop["belief"].mean():.3f})', density=True)
    if len(onehop) > 0 and len(twohop) > 0 and ks_2samp is not None:
        ks_stat, ks_pval = ks_2samp(onehop['belief'], twohop['belief'])
        ax3.text(0.05, 0.95, f'KS test p-value: {ks_pval:.2e}',
                 transform=ax3.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax3.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='Belief = 0.7')
    ax3.set_xlabel('Belief Score')
    ax3.set_ylabel('Density')
    ax3.set_title('Belief Score Distribution (Enhanced)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = fig1.add_subplot(gs1[0, 3])
    ax4.axis('off')
    summary_text = f"SUMMARY (Belief >=0.7)\n\nTotal: {len(df_plot):,}\n\n"
    summary_text += f"EXPLAINED ({len(explained):,}):\n  1-hop: {len(onehop):,}\n  2-hop: {len(twohop):,}\n\n"
    summary_text += f"UNEXPLAINED: {len(unexplained):,}\n\n"
    if len(explained) > 0:
        summary_text += f"Explained mean |FC|: {explained['abs_logfc'].mean():.3f}\n"
        summary_text += f"Belief range: {explained['belief'].min():.3f}-{explained['belief'].max():.3f}\n"
    if len(unexplained) > 0:
        summary_text += f"Unexplained mean |FC|: {unexplained['abs_logfc'].mean():.3f}\n"
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    ax5 = fig1.add_subplot(gs1[1, 0])
    if len(onehop) > 0:
        ax5.hist(onehop['abs_logfc'], bins=40, density=False, color='blue', alpha=0.7)
        ax5.set_title(f'1-hop (n={len(onehop):,})')
    else:
        ax5.text(0.5, 0.5, 'No 1-hop data', ha='center', va='center', transform=ax5.transAxes)
    ax5.set_xlabel('|Log Fold Change|')
    ax5.set_ylabel('Count')
    ax5.grid(True, alpha=0.3)

    ax6 = fig1.add_subplot(gs1[1, 1])
    if len(twohop) > 0:
        ax6.hist(twohop['abs_logfc'], bins=100, density=False, color='green', alpha=0.7)
        ax6.set_title(f'2-hop (n={len(twohop):,})')
    else:
        ax6.text(0.5, 0.5, 'No 2-hop data', ha='center', va='center', transform=ax6.transAxes)
    ax6.set_xlabel('|Log Fold Change|')
    ax6.set_ylabel('Count')
    ax6.grid(True, alpha=0.3)

    ax7 = fig1.add_subplot(gs1[1, 2])
    if len(unexplained) > 0:
        ax7.hist(unexplained['abs_logfc'], bins=100, density=False, color='red', alpha=0.7)
        ax7.set_title(f'Unexplained (n={len(unexplained):,})')
    else:
        ax7.text(0.5, 0.5, 'No unexplained data', ha='center', va='center', transform=ax7.transAxes)
    ax7.set_xlabel('|Log Fold Change|')
    ax7.set_ylabel('Count')
    ax7.grid(True, alpha=0.3)

    ax8 = fig1.add_subplot(gs1[1, 3])
    if len(onehop) > 0:
        ax8.hist(onehop['abs_logfc'], bins=50, alpha=0.5, color='blue',
                 label=f'1-hop (n={len(onehop):,})', density=True)
        if gaussian_kde is not None and len(onehop) > 5:
            try:
                kde_1hop = gaussian_kde(onehop['abs_logfc'])
                x_range = np.linspace(onehop['abs_logfc'].min(), onehop['abs_logfc'].max(), 100)
                ax8.plot(x_range, kde_1hop(x_range), color='darkblue', linewidth=2)
            except Exception:
                pass
    if len(twohop) > 0:
        ax8.hist(twohop['abs_logfc'], bins=50, alpha=0.5, color='green',
                 label=f'2-hop (n={len(twohop):,})', density=True)
        if gaussian_kde is not None and len(twohop) > 5:
            try:
                kde_2hop = gaussian_kde(twohop['abs_logfc'])
                x_range = np.linspace(twohop['abs_logfc'].min(), twohop['abs_logfc'].max(), 100)
                ax8.plot(x_range, kde_2hop(x_range), color='darkgreen', linewidth=2)
            except Exception:
                pass
    if len(unexplained) > 0:
        ax8.hist(unexplained['abs_logfc'], bins=50, alpha=0.5, color='red',
                 label=f'Unexplained (n={len(unexplained):,})', density=True)
        if gaussian_kde is not None and len(unexplained) > 5:
            try:
                kde_unexp = gaussian_kde(unexplained['abs_logfc'])
                x_range = np.linspace(unexplained['abs_logfc'].min(), unexplained['abs_logfc'].max(), 100)
                ax8.plot(x_range, kde_unexp(x_range), color='darkred', linewidth=2)
            except Exception:
                pass
    ax8.set_xlabel('|Log Fold Change|')
    ax8.set_ylabel('Density')
    ax8.set_title('Overlaid Histograms (KDE Smoothed)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_name}_main.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)

    # Advanced visualizations figure
    fig2 = plt.figure(figsize=(16, 8))
    fig2.suptitle(f'{title} - Advanced Visualizations', fontsize=16, fontweight='bold')
    gs2 = fig2.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

    ax_3d = fig2.add_subplot(gs2[0, 0], projection='3d')
    if len(explained) > 10 and gaussian_kde is not None:
        try:
            x = explained['abs_logfc'].values
            y = explained['belief'].values
            kde = gaussian_kde([x, y])
            xi = np.linspace(x.min(), x.max(), 30)
            yi = np.linspace(y.min(), y.max(), 30)
            X, Y = np.meshgrid(xi, yi)
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = kde(positions).reshape(X.shape)
            ax_3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            ax_3d.set_xlabel('|Log Fold Change|')
            ax_3d.set_ylabel('Belief Score')
            ax_3d.set_zlabel('Density')
            ax_3d.set_title('3D KDE Surface')
        except Exception as e:
            ax_3d.text(0.5, 0.5, 0.5, f'3D KDE Error: {str(e)[:50]}',
                       ha='center', va='center', transform=ax_3d.transAxes)
    else:
        ax_3d.text(0.5, 0.5, 0.5, 'Insufficient data for 3D KDE',
                   ha='center', va='center', transform=ax_3d.transAxes)

    ax_pval = fig2.add_subplot(gs2[0, 1])
    if len(onehop) > 0:
        ax_pval.hist(onehop['neg_log10_pval'], bins=30, alpha=0.7, color='blue',
                     label=f'1-hop (n={len(onehop):,})', density=True)
    if len(twohop) > 0:
        ax_pval.hist(twohop['neg_log10_pval'], bins=30, alpha=0.7, color='green',
                     label=f'2-hop (n={len(twohop):,})', density=True)
    if len(unexplained) > 0:
        ax_pval.hist(unexplained['neg_log10_pval'], bins=30, alpha=0.7, color='red',
                     label=f'Unexplained (n={len(unexplained):,})', density=True)
    ax_pval.set_xlabel('-log10(P-value)')
    ax_pval.set_ylabel('Density')
    ax_pval.set_title('P-value Significance Distribution')
    ax_pval.axvline(x=-np.log10(0.05), color='orange', linestyle='--', alpha=0.7, label='p=0.05')
    ax_pval.axvline(x=-np.log10(0.01), color='red', linestyle='--', alpha=0.7, label='p=0.01')
    ax_pval.legend()
    ax_pval.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_name}_advanced.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)

    logger.info("Visualization complete for %s", title)
    return df_plot


def comprehensive_combined_analysis(df, name):
    """Run comprehensive statistical analysis."""
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
        logger.info("  Belief: Mean=%.4f, Median=%.4f, Range=%.4f-%.4f",
                     belief.mean(), belief.median(), belief.min(), belief.max())
        logger.info("  P-value: Mean=%.6f, Median=%.6f", pval.mean(), pval.median())

        corr_fc = np.corrcoef(abs_fc, belief)[0, 1]
        corr_pval = np.corrcoef(pval, belief)[0, 1]
        corr_sig = np.corrcoef(1 - pval, belief)[0, 1]
        logger.info("  Correlation |LogFC| vs Belief: r=%.6f", corr_fc)
        logger.info("  Correlation P-value vs Belief: r=%.6f", corr_pval)
        logger.info("  Correlation (1-P-value) vs Belief: r=%.6f", corr_sig)

        high_belief = explained[explained['belief'] >= 0.8]
        low_belief = explained[explained['belief'] <= 0.7]
        logger.info("  High belief (>=0.8): %s (%.1f%%)", f"{len(high_belief):,}",
                     len(high_belief) / len(explained) * 100)
        if len(high_belief) > 0:
            logger.info("    Mean |FC|: %.4f", np.abs(high_belief['logfoldchange']).mean())
        logger.info("  Low belief (<=0.7): %s (%.1f%%)", f"{len(low_belief):,}",
                     len(low_belief) / len(explained) * 100)

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
            logger.info("  Mean |FC|: Explained=%.4f, Unexplained=%.4f (scipy not available)",
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
            logger.info("%-20s (no data)", name)
            continue
        explained = df[df['explained']]
        unexplained = df[~df['explained']]
        onehop = explained[explained['pathway_type'] == '1-hop']
        twohop = explained[explained['pathway_type'] == '2-hop']
        total = len(df)
        expl_pct = f"{len(explained) / total * 100:.1f}%"
        mean_fc_exp = f"{np.abs(explained['logfoldchange']).mean():.3f}" if len(explained) > 0 else "N/A"
        mean_fc_unexp = f"{np.abs(unexplained['logfoldchange']).mean():.3f}" if len(unexplained) > 0 else "N/A"
        corr_str = "N/A"
        if len(explained) >= 2:
            corr = np.corrcoef(np.abs(explained['logfoldchange']), explained['belief'])[0, 1]
            corr_str = f"{corr:.4f}"
        logger.info("%-20s %-8s %-10s %-8s %-8s %-8s %-8s %-13s %-14s %-12s",
                     name, str(total), str(len(explained)), str(len(onehop)), str(len(twohop)),
                     str(len(unexplained)), expl_pct, mean_fc_exp, mean_fc_unexp, corr_str)


def main():
    parser = argparse.ArgumentParser(
        description="Combined 1-hop + 2-hop pathway EDA with belief cutoff")
    parser.add_argument("--hop1-all-xlsx", required=True, help="Path to 1-hop all perturbations Excel")
    parser.add_argument("--hop1-no-tp53-csv", required=True, help="Path to 1-hop no TP53 CSV")
    parser.add_argument("--hop2-no-tp53-xlsx", required=True, help="Path to 2-hop excluding TP53 Excel")
    parser.add_argument("--hop2-tp53-csv", required=True, help="Path to 2-hop TP53 only CSV")
    parser.add_argument("--deg-folder", required=True, help="Path to de_results_per_gene folder")
    parser.add_argument("--belief-cutoff", type=float, default=0.7, help="Belief score cutoff")
    parser.add_argument("--fc-threshold", type=float, default=2.5, help="High FC threshold for CSV export")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Loading datasets with belief cutoff >= %.1f", args.belief_cutoff)

    df_1hop_all = pd.read_excel(args.hop1_all_xlsx)
    df_1hop_no_tp53 = pd.read_csv(args.hop1_no_tp53_csv)
    df_2hop_no_tp53 = pd.read_excel(args.hop2_no_tp53_xlsx)
    df_2hop_tp53_only = pd.read_csv(args.hop2_tp53_csv)

    logger.info("Before filtering: 1-hop all=%s, 1-hop no-TP53=%s, 2-hop no-TP53=%s, 2-hop TP53=%s",
                f"{len(df_1hop_all):,}", f"{len(df_1hop_no_tp53):,}",
                f"{len(df_2hop_no_tp53):,}", f"{len(df_2hop_tp53_only):,}")

    df_1hop_all = standardize_columns(df_1hop_all, "1-hop all")
    df_1hop_no_tp53 = standardize_columns(df_1hop_no_tp53, "1-hop no TP53")
    df_2hop_no_tp53 = standardize_columns(df_2hop_no_tp53, "2-hop no TP53")
    df_2hop_tp53_only = standardize_columns(df_2hop_tp53_only, "2-hop TP53 only")

    logger.info("Applying belief score cutoff >= %.1f", args.belief_cutoff)
    cutoff = args.belief_cutoff
    df_1hop_all_f = df_1hop_all[df_1hop_all['belief'] >= cutoff].copy()
    df_1hop_no_tp53_f = df_1hop_no_tp53[df_1hop_no_tp53['belief'] >= cutoff].copy()
    df_2hop_no_tp53_f = df_2hop_no_tp53[df_2hop_no_tp53['belief'] >= cutoff].copy()
    df_2hop_tp53_only_f = df_2hop_tp53_only[df_2hop_tp53_only['belief'] >= cutoff].copy()

    logger.info("After filtering: 1-hop all=%s, 1-hop no-TP53=%s, 2-hop no-TP53=%s, 2-hop TP53=%s",
                f"{len(df_1hop_all_f):,}", f"{len(df_1hop_no_tp53_f):,}",
                f"{len(df_2hop_no_tp53_f):,}", f"{len(df_2hop_tp53_only_f):,}")

    df_1hop_all_f['pathway_type'] = '1-hop'
    df_1hop_no_tp53_f['pathway_type'] = '1-hop'
    df_2hop_no_tp53_f['pathway_type'] = '2-hop'
    df_2hop_tp53_only_f['pathway_type'] = '2-hop'

    logger.info("Creating combined datasets with priority logic")
    df_combined_all = create_combined_dataset(
        df_1hop_all_f, pd.concat([df_2hop_no_tp53_f, df_2hop_tp53_only_f]),
        "All Perturbations", include_tp53=True)
    df_combined_no_tp53 = create_combined_dataset(
        df_1hop_no_tp53_f, df_2hop_no_tp53_f, "Excluding TP53", include_tp53=False)
    df_1hop_tp53_only_f = df_1hop_all_f[df_1hop_all_f['source'] == 'TP53'].copy()
    df_combined_tp53_only = create_combined_dataset(
        df_1hop_tp53_only_f, df_2hop_tp53_only_f, "TP53 Only", include_tp53=True)

    logger.info("Adding unexplained targets")
    df_final_all = add_unexplained_targets(df_combined_all, "All Perturbations",
                                           args.deg_folder, include_tp53=True)
    df_final_no_tp53 = add_unexplained_targets(df_combined_no_tp53, "Excluding TP53",
                                                args.deg_folder, include_tp53=False)
    df_final_tp53_only = add_unexplained_targets(df_combined_tp53_only, "TP53 Only",
                                                  args.deg_folder, include_tp53=True)

    logger.info("COMBINED DATASET SUMMARY (BELIEF >= %.1f)", cutoff)
    print_combined_stats(df_final_all, "All Perturbations (Combined)")
    print_combined_stats(df_final_tp53_only, "TP53 Only (Combined)")
    print_combined_stats(df_final_no_tp53, "Excluding TP53 (Combined)")

    logger.info("Creating enhanced visualizations")
    create_enhanced_combined_plot(df_final_all, "All Perturbations", "enhanced_all_perturbations")
    create_enhanced_combined_plot(df_final_tp53_only, "TP53 Only", "enhanced_tp53_only")
    create_enhanced_combined_plot(df_final_no_tp53, "Excluding TP53", "enhanced_excluding_tp53")

    logger.info("Generating high FC unexplained targets CSV")
    generate_high_fc_unexplained_csv(df_final_no_tp53, "Excluding TP53", fc_threshold=args.fc_threshold)

    logger.info("COMPREHENSIVE STATISTICAL ANALYSIS (BELIEF >= %.1f)", cutoff)
    comprehensive_combined_analysis(df_final_all, "All Perturbations (Combined)")
    comprehensive_combined_analysis(df_final_tp53_only, "TP53 Only (Combined)")
    comprehensive_combined_analysis(df_final_no_tp53, "Excluding TP53 (Combined)")

    logger.info("SUMMARY TABLE")
    create_combined_summary_table(df_final_all, df_final_tp53_only, df_final_no_tp53)

    logger.info("Enhanced analysis complete")


if __name__ == "__main__":
    main()
