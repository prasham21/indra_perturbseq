"""Combined 1+2+3-hop pathway analysis with priority logic and visualizations."""
from __future__ import annotations

import argparse
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from scipy.stats import gaussian_kde, ks_2samp, ttest_ind

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def load_and_filter_pathway_datasets(hop1_file, hop2_file, hop3_file):
    """Load and filter all pathway datasets with belief cutoff >= 0.7."""
    logger.info("Loading and filtering pathway datasets")

    df_1hop = pd.read_csv(hop1_file)
    df_1hop_filtered = df_1hop[df_1hop['belief'] >= 0.7].copy()
    df_1hop_filtered['pathway_type'] = '1-hop'
    logger.info("1-hop: %s -> %s (after belief >= 0.7)", f"{len(df_1hop):,}", f"{len(df_1hop_filtered):,}")

    df_2hop = pd.read_excel(hop2_file)
    df_2hop['belief'] = (df_2hop['belief_1'] + df_2hop['belief_2']) / 2
    df_2hop_filtered = df_2hop[df_2hop['belief'] >= 0.7].copy()
    df_2hop_filtered['pathway_type'] = '2-hop'
    logger.info("2-hop: %s -> %s (after belief >= 0.7)", f"{len(df_2hop):,}", f"{len(df_2hop_filtered):,}")

    df_3hop = pd.read_csv(hop3_file)
    df_3hop['belief'] = (df_3hop['belief_1'] + df_3hop['belief_2'] + df_3hop['belief_3']) / 3
    df_3hop_filtered = df_3hop[df_3hop['belief'] >= 0.7].copy()
    df_3hop_filtered['pathway_type'] = '3-hop'
    logger.info("3-hop: %s -> %s (after belief >= 0.7)", f"{len(df_3hop):,}", f"{len(df_3hop_filtered):,}")

    return df_1hop_filtered, df_2hop_filtered, df_3hop_filtered


def create_combined_3pathway_dataset(df_1hop, df_2hop, df_3hop):
    """Create combined dataset with 3-level priority logic: 1-hop > 2-hop > 3-hop."""
    logger.info("Applying 3-level priority logic")

    all_pathways = []
    for priority, (df, ptype) in enumerate(
        [(df_1hop, '1-hop'), (df_2hop, '2-hop'), (df_3hop, '3-hop')], start=1
    ):
        for _, row in df.iterrows():
            all_pathways.append({
                'source': row['source'], 'target': row['target'],
                'belief': row['belief'], 'logfoldchange': row['logfoldchange'],
                'pval': row['pval'], 'pathway_type': ptype, 'priority': priority
            })

    combined_df = pd.DataFrame(all_pathways)
    logger.info("Combined raw data: %s pathways", f"{len(combined_df):,}")

    def select_best_pathway(group):
        return group.sort_values(['priority', 'belief'], ascending=[True, False]).iloc[0]

    best_pathways = combined_df.groupby(['source', 'target']).apply(select_best_pathway).reset_index(drop=True)

    logger.info("After priority logic: %s unique source-target pairs", f"{len(best_pathways):,}")
    for ptype in ['1-hop', '2-hop', '3-hop']:
        count = len(best_pathways[best_pathways['pathway_type'] == ptype])
        logger.info("  %s selected: %s", ptype, f"{count:,}")

    return best_pathways


def add_unexplained_targets_3hop(combined_df, target_validation_path, deg_folder):
    """Add unexplained targets following the same pattern as 1+2-hop analysis."""
    logger.info("Adding unexplained targets")

    allowed_sources = set(
        pd.read_csv(target_validation_path)
        .query("analysis_flag == 'Use_for_analysis'")["Gene"]
        .astype(str).str.strip()
        .tolist()
    )
    EXCLUDE_SOURCES = {"TP53", "CDKN1A"}
    allowed_sources = {g for g in allowed_sources if g not in EXCLUDE_SOURCES}

    explained_pairs = set(zip(combined_df['source'], combined_df['target']))
    pathway_sources = combined_df['source'].unique()

    unexplained_data = []
    processed_sources = 0

    for source in pathway_sources:
        if source not in allowed_sources:
            continue
        deg_file = Path(deg_folder) / f"{source}_vs_control.csv"
        if not deg_file.exists():
            continue
        try:
            deg_df = pd.read_csv(deg_file)
            if not {"names", "pvals", "logfoldchanges"}.issubset(deg_df.columns):
                continue
            significant_genes = deg_df[deg_df['pvals'] < 0.05].copy()
            for _, deg_row in significant_genes.iterrows():
                target_gene = deg_row['names']
                if (source, target_gene) in explained_pairs:
                    continue
                unexplained_data.append({
                    'source': source, 'target': target_gene,
                    'belief': 0.5, 'logfoldchange': deg_row['logfoldchanges'],
                    'pval': deg_row['pvals'], 'pathway_type': 'unexplained', 'priority': 4
                })
        except Exception:
            continue
        processed_sources += 1
        if processed_sources % 50 == 0:
            logger.info("Processed %d sources", processed_sources)

    combined_df['explained'] = True
    if unexplained_data:
        unexplained_df = pd.DataFrame(unexplained_data)
        unexplained_df['explained'] = False
        final_df = pd.concat([combined_df, unexplained_df], ignore_index=True)
    else:
        final_df = combined_df.copy()
        final_df['explained'] = True

    explained_count = len(final_df[final_df['explained']])
    unexplained_count = len(final_df[~final_df['explained']])
    logger.info("Final dataset: %s total pairs", f"{len(final_df):,}")
    logger.info("  Explained: %s (%.1f%%)", f"{explained_count:,}", explained_count / len(final_df) * 100)
    logger.info("  Unexplained: %s (%.1f%%)", f"{unexplained_count:,}", unexplained_count / len(final_df) * 100)

    return final_df


def calculate_pathway_statistics(df):
    """Calculate comprehensive statistics for each pathway type."""
    stats_summary = {}
    for ptype in ['1-hop', '2-hop', '3-hop', 'unexplained']:
        subset = df[df['pathway_type'] == ptype]
        if len(subset) == 0:
            stats_summary[ptype] = {
                'count': 0, 'mean_fc': 0, 'mean_abs_fc': 0, 'mean_neg_log10_pval': 0,
                'mean_belief': 0, 'std_fc': 0, 'median_fc': 0, 'fc_95th': 0
            }
            continue
        abs_fc = np.abs(subset['logfoldchange'])
        neg_log10_pval = -np.log10(subset['pval'].clip(lower=1e-50))
        stats_summary[ptype] = {
            'count': len(subset), 'mean_fc': subset['logfoldchange'].mean(),
            'mean_abs_fc': abs_fc.mean(), 'mean_neg_log10_pval': neg_log10_pval.mean(),
            'mean_belief': subset['belief'].mean(), 'std_fc': abs_fc.std(),
            'median_fc': abs_fc.median(), 'fc_95th': np.percentile(abs_fc, 95)
        }
    return stats_summary


def create_enhanced_4pathway_plot(df, title, save_name):
    """Create enhanced visualization with 4 pathway types."""
    logger.info("Creating enhanced 4-pathway analysis")

    df_plot = df.copy()
    df_plot['abs_logfc'] = np.abs(df_plot['logfoldchange'])
    df_plot['neg_log10_pval'] = -np.log10(df_plot['pval'].clip(lower=1e-50))

    onehop = df_plot[df_plot['pathway_type'] == '1-hop']
    twohop = df_plot[df_plot['pathway_type'] == '2-hop']
    threehop = df_plot[df_plot['pathway_type'] == '3-hop']
    unexplained = df_plot[df_plot['pathway_type'] == 'unexplained']

    for ptype, data in [('1-hop', onehop), ('2-hop', twohop), ('3-hop', threehop), ('Unexplained', unexplained)]:
        logger.info("  %s: %s", ptype, f"{len(data):,}")

    fig1 = plt.figure(figsize=(24, 12))
    fig1.suptitle(f'{title} - 3-Pathway Analysis (Belief >= 0.7)', fontsize=18, fontweight='bold')
    gs1 = fig1.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    ax1 = fig1.add_subplot(gs1[0, 0])
    for data, color, label, s, alpha, ec in [
        (onehop, 'blue', '1-hop', 40, 0.8, 'darkblue'),
        (twohop, 'green', '2-hop', 25, 0.7, 'darkgreen'),
        (threehop, 'orange', '3-hop', 20, 0.6, 'darkorange'),
        (unexplained, 'red', 'Unexplained', 15, 0.4, 'none'),
    ]:
        if len(data) > 0:
            ax1.scatter(data['abs_logfc'], data['belief'], alpha=alpha, s=s,
                        color=color, label=f'{label} (n={len(data):,})', edgecolors=ec)
    ax1.set_xlabel('|Log Fold Change|')
    ax1.set_ylabel('Belief Score')
    ax1.set_title('All Pathway Types')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig1.add_subplot(gs1[0, 1])
    explained = df_plot[df_plot['explained']]
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
    ax2.set_title('LogFC vs Belief Score')
    ax2.grid(True, alpha=0.3)

    ax3 = fig1.add_subplot(gs1[0, 2])
    belief_data = []
    labels = []
    for ptype, color in [('1-hop', 'blue'), ('2-hop', 'green'), ('3-hop', 'orange')]:
        subset = df_plot[df_plot['pathway_type'] == ptype]
        if len(subset) > 0:
            mean_belief = subset['belief'].mean()
            ax3.hist(subset['belief'], bins=30, alpha=0.7, color=color,
                     label=f'{ptype} (mu={mean_belief:.3f})', density=True)
            belief_data.append(subset['belief'].values)
            labels.append(ptype)

    if len(belief_data) >= 2:
        ks_results = []
        for i in range(len(belief_data) - 1):
            if len(belief_data[i]) > 1 and len(belief_data[i + 1]) > 1:
                ks_stat, ks_pval = ks_2samp(belief_data[i], belief_data[i + 1])
                ks_results.append(f'{labels[i]} vs {labels[i + 1]}: p={ks_pval:.2e}')
        if ks_results:
            ax3.text(0.05, 0.95, 'KS Tests:\n' + '\n'.join(ks_results),
                     transform=ax3.transAxes, fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax3.set_xlabel('Belief Score')
    ax3.set_ylabel('Density')
    ax3.set_title('Belief Distributions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = fig1.add_subplot(gs1[0, 3])
    ax4.axis('off')
    stats = calculate_pathway_statistics(df_plot)
    summary_text = f"3-PATHWAY ANALYSIS SUMMARY\n\nTotal: {len(df_plot):,} pairs\n\n"
    for ptype in ['1-hop', '2-hop', '3-hop', 'unexplained']:
        s = stats[ptype]
        summary_text += f"{ptype}: {s['count']:,} ({s['count'] / len(df_plot) * 100:.1f}%)\n"
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')

    for i, (ptype, color, pos) in enumerate([
        ('1-hop', 'blue', gs1[1, 0]), ('2-hop', 'green', gs1[1, 1]), ('3-hop', 'orange', gs1[1, 2])
    ]):
        ax = fig1.add_subplot(pos)
        data = df_plot[df_plot['pathway_type'] == ptype]
        if len(data) > 0:
            ax.hist(data['abs_logfc'], bins=40, color=color, alpha=0.7, edgecolor='black')
            ax.set_title(f'{ptype} (n={len(data):,})')
        else:
            ax.text(0.5, 0.5, f'No {ptype} data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{ptype} (n=0)')
        ax.set_xlabel('|Log Fold Change|')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)

    ax8 = fig1.add_subplot(gs1[1, 3])
    for ptype, color in [('1-hop', 'blue'), ('2-hop', 'green'), ('3-hop', 'orange'), ('unexplained', 'red')]:
        data = df_plot[df_plot['pathway_type'] == ptype]
        if len(data) > 0:
            ax8.hist(data['abs_logfc'], bins=40, alpha=0.5, color=color,
                     label=f'{ptype} (n={len(data):,})', density=True)
    ax8.set_xlabel('|Log Fold Change|')
    ax8.set_ylabel('Density')
    ax8.set_title('Overlaid Distributions')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_name}_main.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)

    create_advanced_plots(df_plot, title, save_name)
    return stats


def create_advanced_plots(df_plot, title, save_name):
    """Create advanced visualization plots."""
    fig2 = plt.figure(figsize=(20, 10))
    fig2.suptitle(f'{title} - Advanced Visualizations', fontsize=16, fontweight='bold')
    gs2 = fig2.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    explained = df_plot[df_plot['explained']]

    ax_3d = fig2.add_subplot(gs2[0, 0], projection='3d')
    if len(explained) > 20:
        try:
            x = explained['abs_logfc'].values
            y = explained['belief'].values
            kde = gaussian_kde([x, y])
            xi = np.linspace(x.min(), x.max(), 25)
            yi = np.linspace(y.min(), y.max(), 25)
            X, Y = np.meshgrid(xi, yi)
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            ax_3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            ax_3d.set_xlabel('|Log Fold Change|')
            ax_3d.set_ylabel('Belief Score')
            ax_3d.set_zlabel('Density')
            ax_3d.set_title('3D KDE: LogFC vs Belief')
        except Exception:
            ax_3d.text(0.5, 0.5, 0.5, '3D KDE Error', ha='center', va='center')

    ax_pval = fig2.add_subplot(gs2[0, 1])
    if len(explained) > 0:
        try:
            h_pval = ax_pval.hist2d(explained['abs_logfc'], explained['neg_log10_pval'],
                                    bins=30, cmap='hot', norm=LogNorm(vmin=1))
            plt.colorbar(h_pval[3], ax=ax_pval)
        except Exception:
            h_pval = ax_pval.hist2d(explained['abs_logfc'], explained['neg_log10_pval'], bins=30, cmap='hot')
            plt.colorbar(h_pval[3], ax=ax_pval)
    ax_pval.set_xlabel('|Log Fold Change|')
    ax_pval.set_ylabel('-log10(P-value)')
    ax_pval.set_title('Fold Change vs P-value distribution')
    ax_pval.grid(True, alpha=0.3)

    ax_pval_dist = fig2.add_subplot(gs2[1, 0])
    for ptype, color in [('1-hop', 'blue'), ('2-hop', 'green'), ('3-hop', 'orange'), ('unexplained', 'red')]:
        data = df_plot[df_plot['pathway_type'] == ptype]
        if len(data) > 0:
            ax_pval_dist.hist(data['neg_log10_pval'], bins=30, alpha=0.6, color=color,
                              label=f'{ptype} (n={len(data):,})', density=True)
    ax_pval_dist.axvline(x=-np.log10(0.05), color='orange', linestyle='--', alpha=0.7, label='p=0.05')
    ax_pval_dist.axvline(x=-np.log10(0.01), color='red', linestyle='--', alpha=0.7, label='p=0.01')
    ax_pval_dist.set_xlabel('-log10(P-value)')
    ax_pval_dist.set_ylabel('Density')
    ax_pval_dist.set_title('P-value Distributions')
    ax_pval_dist.legend(fontsize=8)
    ax_pval_dist.grid(True, alpha=0.3)

    ax4 = fig2.add_subplot(gs2[1, 1], projection='3d')
    explained_for_kde = df_plot[df_plot['explained']].copy()
    explained_for_kde['neg_log10_pval'] = -np.log10(explained_for_kde['pval'].clip(lower=1e-50))
    if len(explained_for_kde) > 20:
        try:
            x = explained_for_kde['abs_logfc'].values
            y = explained_for_kde['neg_log10_pval'].values
            kde = gaussian_kde([x, y])
            xi = np.linspace(x.min(), x.max(), 40)
            yi = np.linspace(y.min(), y.max(), 40)
            X, Y = np.meshgrid(xi, yi)
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            ax4.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8)
            ax4.set_xlabel('|Log Fold Change|')
            ax4.set_ylabel('-log10(P-value)')
            ax4.set_zlabel('Density')
            ax4.set_title('3D KDE: LogFC vs P-values')
        except Exception as e:
            ax4.text(0.5, 0.5, 0.5, f'3D KDE Error:\n{str(e)}', ha='center', va='center')
    else:
        ax4.text(0.5, 0.5, 0.5, 'Not enough data for KDE', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(f'{save_name}_advanced.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)


def generate_final_unexplained_csv(df, fc_threshold=2.0):
    """Generate final unexplained CSV with adjustable threshold."""
    logger.info("Generating final unexplained targets CSV")

    unexplained = df[(df['pathway_type'] == 'unexplained') & (df['source'] != df['target'])].copy()
    logger.info("Unexplained targets (excluding self-targeting): %s", f"{len(unexplained):,}")

    if len(unexplained) == 0:
        return None

    unexplained['abs_logfc'] = np.abs(unexplained['logfoldchange'])
    high_fc = unexplained[unexplained['abs_logfc'] >= fc_threshold].copy()

    if len(high_fc) == 0:
        for threshold in [1.5, 1.0, 0.8]:
            high_fc = unexplained[unexplained['abs_logfc'] >= threshold]
            logger.info("With threshold >= %s: %s targets", threshold, f"{len(high_fc):,}")
            if len(high_fc) >= 10:
                fc_threshold = threshold
                break

    if len(high_fc) == 0:
        high_fc = unexplained.nlargest(50, 'abs_logfc')
        logger.info("Using top 50 unexplained targets by fold change")

    high_fc = high_fc.sort_values('abs_logfc', ascending=False)

    output_df = pd.DataFrame({
        'source_gene': high_fc['source'], 'target_gene': high_fc['target'],
        'log_fold_change': high_fc['logfoldchange'], 'abs_log_fold_change': high_fc['abs_logfc'],
        'p_value': high_fc['pval'], 'linear_fold_change': 2 ** high_fc['abs_logfc'],
        'neg_log10_pval': -np.log10(high_fc['pval'].clip(lower=1e-50))
    })

    filename = "final_unexplained_targets_3hop_direct.csv"
    output_df.to_csv(filename, index=False)
    logger.info("Saved: %s (%s targets)", filename, f"{len(output_df):,}")

    return output_df


def main():
    parser = argparse.ArgumentParser(description="Combined 3-pathway analysis")
    parser.add_argument("--hop1-file", required=True, help="Path to 1-hop CSV (e.g. indra_1hop_no_v2.csv)")
    parser.add_argument("--hop2-file", required=True, help="Path to 2-hop Excel file")
    parser.add_argument("--hop3-file", required=True, help="Path to 3-hop CSV")
    parser.add_argument("--validation-csv", required=True, help="Path to target_validation_expanded.csv")
    parser.add_argument("--deg-folder", required=True, help="Path to de_results_per_gene folder")
    parser.add_argument("--fc-threshold", type=float, default=2.0, help="FC threshold for unexplained CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df_1hop, df_2hop, df_3hop = load_and_filter_pathway_datasets(
        args.hop1_file, args.hop2_file, args.hop3_file
    )
    combined_df = create_combined_3pathway_dataset(df_1hop, df_2hop, df_3hop)
    final_df = add_unexplained_targets_3hop(combined_df, args.validation_csv, args.deg_folder)

    stats = create_enhanced_4pathway_plot(final_df, "Direct 3-Pathway Analysis", "direct_3pathway_analysis")

    stats_df = pd.DataFrame(stats).T
    stats_df['pathway_type'] = stats_df.index
    cols = ['pathway_type', 'count', 'mean_abs_fc', 'mean_neg_log10_pval', 'mean_belief', 'std_fc', 'median_fc',
            'fc_95th']
    stats_df = stats_df[cols]
    stats_df.to_csv('direct_3pathway_statistics.csv', index=False)
    logger.info("Statistics saved")

    generate_final_unexplained_csv(final_df, fc_threshold=args.fc_threshold)

    pathway_types = ['1-hop', '2-hop', '3-hop']
    comparisons = [('1-hop', '2-hop'), ('1-hop', '3-hop'), ('2-hop', '3-hop')]
    for type1, type2 in comparisons:
        data1 = final_df[final_df['pathway_type'] == type1]
        data2 = final_df[final_df['pathway_type'] == type2]
        if len(data1) > 1 and len(data2) > 1:
            fc1 = np.abs(data1['logfoldchange'])
            fc2 = np.abs(data2['logfoldchange'])
            try:
                t_stat, t_pval = ttest_ind(fc1, fc2)
                logger.info("%s vs %s: Mean |FC| %.4f vs %.4f, t-test p=%.2e",
                            type1, type2, fc1.mean(), fc2.mean(), t_pval)
            except Exception:
                logger.warning("%s vs %s: Statistical test failed", type1, type2)

    logger.info("Analysis complete")
    logger.info("Total pairs analyzed: %s", f"{len(final_df):,}")
    logger.info("Coverage: %.1f%%", len(final_df[final_df['explained']]) / len(final_df) * 100)


if __name__ == "__main__":
    main()
