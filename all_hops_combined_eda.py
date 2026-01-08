import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, ks_2samp, ttest_ind
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("Direct 3-Hop Analysis Following Combined 1+2-Hop Pattern")
print("=" * 80)


def load_and_filter_pathway_datasets():
    """Load and filter all pathway datasets with belief cutoff ≥0.7"""

    print("Loading and filtering pathway datasets...")

    # Load 1-hop data (excluding TP53)
    df_1hop = pd.read_csv("/Users/prashammarfatia/Downloads/indra_1hop_no_v2.csv")
    df_1hop_filtered = df_1hop[df_1hop['belief'] >= 0.7].copy()
    df_1hop_filtered['pathway_type'] = '1-hop'
    print(f"1-hop: {len(df_1hop):,} -> {len(df_1hop_filtered):,} (after belief ≥0.7)")

    # Load 2-hop data (excluding TP53)
    df_2hop = pd.read_excel("/Users/prashammarfatia/Downloads/indra_2hop_all_perturbations.xlsx")
    df_2hop['belief'] = (df_2hop['belief_1'] + df_2hop['belief_2']) / 2
    df_2hop_filtered = df_2hop[df_2hop['belief'] >= 0.7].copy()
    df_2hop_filtered['pathway_type'] = '2-hop'
    print(f"2-hop: {len(df_2hop):,} -> {len(df_2hop_filtered):,} (after belief ≥0.7)")

    # Load 3-hop data
    df_3hop = pd.read_csv("/Users/prashammarfatia/Downloads/indra_3hop_optimized_all_perturbations_combined.csv")
    df_3hop['belief'] = (df_3hop['belief_1'] + df_3hop['belief_2'] + df_3hop['belief_3']) / 3
    df_3hop_filtered = df_3hop[df_3hop['belief'] >= 0.7].copy()
    df_3hop_filtered['pathway_type'] = '3-hop'
    print(f"3-hop: {len(df_3hop):,} -> {len(df_3hop_filtered):,} (after belief ≥0.7)")

    return df_1hop_filtered, df_2hop_filtered, df_3hop_filtered


def create_combined_3pathway_dataset(df_1hop, df_2hop, df_3hop):
    """Create combined dataset with 3-level priority logic: 1-hop > 2-hop > 3-hop"""

    print("\nApplying 3-level priority logic...")

    # Combine all pathways
    all_pathways = []

    # Add 1-hop data (highest priority)
    for _, row in df_1hop.iterrows():
        all_pathways.append({
            'source': row['source'],
            'target': row['target'],
            'belief': row['belief'],
            'logfoldchange': row['logfoldchange'],
            'pval': row['pval'],
            'pathway_type': '1-hop',
            'priority': 1
        })

    # Add 2-hop data (medium priority)
    for _, row in df_2hop.iterrows():
        all_pathways.append({
            'source': row['source'],
            'target': row['target'],
            'belief': row['belief'],
            'logfoldchange': row['logfoldchange'],
            'pval': row['pval'],
            'pathway_type': '2-hop',
            'priority': 2
        })

    # Add 3-hop data (lowest priority)
    for _, row in df_3hop.iterrows():
        all_pathways.append({
            'source': row['source'],
            'target': row['target'],
            'belief': row['belief'],
            'logfoldchange': row['logfoldchange'],
            'pval': row['pval'],
            'pathway_type': '3-hop',
            'priority': 3
        })

    combined_df = pd.DataFrame(all_pathways)
    print(f"Combined raw data: {len(combined_df):,} pathways")
    print(f"  1-hop: {len(combined_df[combined_df['pathway_type'] == '1-hop']):,}")
    print(f"  2-hop: {len(combined_df[combined_df['pathway_type'] == '2-hop']):,}")
    print(f"  3-hop: {len(combined_df[combined_df['pathway_type'] == '3-hop']):,}")

    # Apply priority logic: keep best pathway for each source-target pair
    def select_best_pathway(group):
        """Select highest priority pathway, then highest belief if tied"""
        sorted_group = group.sort_values(['priority', 'belief'], ascending=[True, False])
        return sorted_group.iloc[0]

    best_pathways = combined_df.groupby(['source', 'target']).apply(select_best_pathway).reset_index(drop=True)

    print(f"After priority logic: {len(best_pathways):,} unique source-target pairs")
    print(f"  1-hop selected: {len(best_pathways[best_pathways['pathway_type'] == '1-hop']):,}")
    print(f"  2-hop selected: {len(best_pathways[best_pathways['pathway_type'] == '2-hop']):,}")
    print(f"  3-hop selected: {len(best_pathways[best_pathways['pathway_type'] == '3-hop']):,}")

    reduction = len(combined_df) - len(best_pathways)
    print(f"Removed {reduction:,} redundant pathways ({reduction / len(combined_df) * 100:.1f}%)")

    return best_pathways


def add_unexplained_targets_3hop(combined_df):
    """Add unexplained targets following the same pattern as 1+2-hop analysis"""

    print("\nAdding unexplained targets...")

    # Load allowed sources
    tv_path = "/Users/prashammarfatia/Downloads/target_validation_expanded.csv"
    allowed_sources = set(
        pd.read_csv(tv_path)
        .query("Karen_Flag == 'Use_for_analysis'")["Gene"]
        .astype(str).str.strip()
        .tolist()
    )

    # Exclude TP53 and CDKN1A
    EXCLUDE_SOURCES = {"TP53", "CDKN1A"}
    allowed_sources = {g for g in allowed_sources if g not in EXCLUDE_SOURCES}

    # Get explained pairs
    explained_pairs = set()
    for _, row in combined_df.iterrows():
        explained_pairs.add((row['source'], row['target']))

    # Get pathway sources (those with at least one pathway)
    pathway_sources = combined_df['source'].unique()

    # Load DEG data and find unexplained targets
    deg_folder = "/Users/prashammarfatia/Downloads/de_results_per_gene"
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

                # Skip if already explained by a pathway
                if (source, target_gene) in explained_pairs:
                    continue

                # Add to unexplained data
                unexplained_data.append({
                    'source': source,
                    'target': target_gene,
                    'belief': 0.5,  # Assigned belief for unexplained
                    'logfoldchange': deg_row['logfoldchanges'],
                    'pval': deg_row['pvals'],
                    'pathway_type': 'unexplained',
                    'priority': 4
                })

        except Exception as e:
            continue

        processed_sources += 1
        if processed_sources % 50 == 0:
            print(f"Processed {processed_sources} sources...")

    # Add explained flag
    combined_df['explained'] = True

    # Combine explained and unexplained
    if unexplained_data:
        unexplained_df = pd.DataFrame(unexplained_data)
        unexplained_df['explained'] = False
        final_df = pd.concat([combined_df, unexplained_df], ignore_index=True)
    else:
        final_df = combined_df.copy()
        final_df['explained'] = True

    explained_count = len(final_df[final_df['explained'] == True])
    unexplained_count = len(final_df[final_df['explained'] == False])

    print(f"Final dataset: {len(final_df):,} total pairs")
    print(f"  Explained: {explained_count:,} ({explained_count / len(final_df) * 100:.1f}%)")
    print(f"  Unexplained: {unexplained_count:,} ({unexplained_count / len(final_df) * 100:.1f}%)")

    return final_df


def calculate_pathway_statistics(df):
    """Calculate comprehensive statistics for each pathway type"""

    stats_summary = {}
    pathway_types = ['1-hop', '2-hop', '3-hop', 'unexplained']

    for ptype in pathway_types:
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
            'count': len(subset),
            'mean_fc': subset['logfoldchange'].mean(),
            'mean_abs_fc': abs_fc.mean(),
            'mean_neg_log10_pval': neg_log10_pval.mean(),
            'mean_belief': subset['belief'].mean(),
            'std_fc': abs_fc.std(),
            'median_fc': abs_fc.median(),
            'fc_95th': np.percentile(abs_fc, 95)
        }

    return stats_summary


def create_enhanced_4pathway_plot(df, title, save_name):
    """Create enhanced visualization with 4 pathway types"""

    print(f"\n=== CREATING ENHANCED 4-PATHWAY ANALYSIS ===")

    df_plot = df.copy()
    df_plot['abs_logfc'] = np.abs(df_plot['logfoldchange'])
    df_plot['neg_log10_pval'] = -np.log10(df_plot['pval'].clip(lower=1e-50))

    # Separate data by pathway type
    onehop = df_plot[df_plot['pathway_type'] == '1-hop']
    twohop = df_plot[df_plot['pathway_type'] == '2-hop']
    threehop = df_plot[df_plot['pathway_type'] == '3-hop']
    unexplained = df_plot[df_plot['pathway_type'] == 'unexplained']

    print(f"Data breakdown:")
    print(f"  1-hop: {len(onehop):,}")
    print(f"  2-hop: {len(twohop):,}")
    print(f"  3-hop: {len(threehop):,}")
    print(f"  Unexplained: {len(unexplained):,}")

    # FIGURE 1: Main Analysis
    fig1 = plt.figure(figsize=(24, 12))
    fig1.suptitle(f'{title} - 3-Pathway Analysis (Belief ≥0.7)', fontsize=18, fontweight='bold')
    gs1 = fig1.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # Plot 1: Scatter plot
    ax1 = fig1.add_subplot(gs1[0, 0])
    if len(onehop) > 0:
        ax1.scatter(onehop['abs_logfc'], onehop['belief'], alpha=0.8, s=40,
                    color='blue', label=f'1-hop (n={len(onehop):,})', edgecolors='darkblue')
    if len(twohop) > 0:
        ax1.scatter(twohop['abs_logfc'], twohop['belief'], alpha=0.7, s=25,
                    color='green', label=f'2-hop (n={len(twohop):,})', edgecolors='darkgreen')
    if len(threehop) > 0:
        ax1.scatter(threehop['abs_logfc'], threehop['belief'], alpha=0.6, s=20,
                    color='orange', label=f'3-hop (n={len(threehop):,})', edgecolors='darkorange')
    if len(unexplained) > 0:
        ax1.scatter(unexplained['abs_logfc'], unexplained['belief'], alpha=0.4, s=15,
                    color='red', label=f'Unexplained (n={len(unexplained):,})')

    ax1.set_xlabel('|Log Fold Change|')
    ax1.set_ylabel('Belief Score')
    ax1.set_title('All Pathway Types')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: LogFC vs Belief heat map
    ax2 = fig1.add_subplot(gs1[0, 1])
    explained = df_plot[df_plot['explained'] == True]
    if len(explained) > 0:
        try:
            h2 = ax2.hist2d(explained['abs_logfc'], explained['belief'],
                            bins=25, cmap='plasma', norm=LogNorm(vmin=1))
            plt.colorbar(h2[3], ax=ax2)
        except:
            h2 = ax2.hist2d(explained['abs_logfc'], explained['belief'],
                            bins=25, cmap='plasma')
            plt.colorbar(h2[3], ax=ax2)
    ax2.set_xlabel('|Log Fold Change|')
    ax2.set_ylabel('Belief Score')
    ax2.set_title('LogFC vs Belief Score')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Belief distributions with KS tests
    ax3 = fig1.add_subplot(gs1[0, 2])
    belief_data = []
    labels = []

    for ptype, color in [('1-hop', 'blue'), ('2-hop', 'green'), ('3-hop', 'orange')]:
        subset = df_plot[df_plot['pathway_type'] == ptype]
        if len(subset) > 0:
            mean_belief = subset['belief'].mean()
            ax3.hist(subset['belief'], bins=30, alpha=0.7, color=color,
                     label=f'{ptype} (μ={mean_belief:.3f})', density=True)
            belief_data.append(subset['belief'].values)
            labels.append(ptype)

    # KS tests
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

    # Plot 4: Summary statistics
    ax4 = fig1.add_subplot(gs1[0, 3])
    ax4.axis('off')

    stats = calculate_pathway_statistics(df_plot)

    summary_text = f"""3-PATHWAY ANALYSIS SUMMARY

Total: {len(df_plot):,} pairs

PATHWAY BREAKDOWN:
• 1-hop: {stats['1-hop']['count']:,} ({stats['1-hop']['count'] / len(df_plot) * 100:.1f}%)
• 2-hop: {stats['2-hop']['count']:,} ({stats['2-hop']['count'] / len(df_plot) * 100:.1f}%)
• 3-hop: {stats['3-hop']['count']:,} ({stats['3-hop']['count'] / len(df_plot) * 100:.1f}%)
• Unexplained: {stats['unexplained']['count']:,} ({stats['unexplained']['count'] / len(df_plot) * 100:.1f}%)

MEAN |FOLD CHANGE|:
• 1-hop: {stats['1-hop']['mean_abs_fc']:.3f}
• 2-hop: {stats['2-hop']['mean_abs_fc']:.3f}
• 3-hop: {stats['3-hop']['mean_abs_fc']:.3f}
• Unexplained: {stats['unexplained']['mean_abs_fc']:.3f}

MEAN -log10(P-VALUE):
• 1-hop: {stats['1-hop']['mean_neg_log10_pval']:.2f}
• 2-hop: {stats['2-hop']['mean_neg_log10_pval']:.2f}
• 3-hop: {stats['3-hop']['mean_neg_log10_pval']:.2f}
• Unexplained: {stats['unexplained']['mean_neg_log10_pval']:.2f}
"""

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')

    # Bottom row: Individual + overlaid histograms
    for i, (ptype, color, pos) in enumerate([('1-hop', 'blue', gs1[1, 0]),
                                             ('2-hop', 'green', gs1[1, 1]),
                                             ('3-hop', 'orange', gs1[1, 2])]):
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

    # Overlaid histogram
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

    # FIGURE 2: Advanced plots
    create_advanced_plots(df_plot, title, save_name)

    return stats


def create_advanced_plots(df_plot, title, save_name):
    """Create advanced visualization plots"""

    fig2 = plt.figure(figsize=(20, 10))
    fig2.suptitle(f'{title} - Advanced Visualizations', fontsize=16, fontweight='bold')
    gs2 = fig2.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    explained = df_plot[df_plot['explained'] == True]

    # 3D KDE surface
    ax_3d = fig2.add_subplot(gs2[0, 0], projection='3d')
    if len(explained) > 20:
        try:
            x = explained['abs_logfc'].values
            y = explained['belief'].values
            kde = gaussian_kde([x, y])

            xi = np.linspace(x.min(), x.max(), 25)
            yi = np.linspace(y.min(), y.max(), 25)
            X, Y = np.meshgrid(xi, yi)

            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = kde(positions).reshape(X.shape)

            ax_3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            ax_3d.set_xlabel('|Log Fold Change|')
            ax_3d.set_ylabel('Belief Score')
            ax_3d.set_zlabel('Density')
            ax_3d.set_title('3D KDE: LogFC vs Belief')
        except:
            ax_3d.text(0.5, 0.5, 0.5, '3D KDE Error', ha='center', va='center')

    # Cool heatmap: LogFC vs P-values
    ax_pval = fig2.add_subplot(gs2[0, 1])
    if len(explained) > 0:
        try:
            h_pval = ax_pval.hist2d(explained['abs_logfc'], explained['neg_log10_pval'],
                                    bins=30, cmap='hot', norm=LogNorm(vmin=1))
            plt.colorbar(h_pval[3], ax=ax_pval)
        except:
            h_pval = ax_pval.hist2d(explained['abs_logfc'], explained['neg_log10_pval'],
                                    bins=30, cmap='hot')
            plt.colorbar(h_pval[3], ax=ax_pval)

    ax_pval.set_xlabel('|Log Fold Change|')
    ax_pval.set_ylabel('-log10(P-value)')
    ax_pval.set_title('Fold Change vs P-value distribution')
    ax_pval.grid(True, alpha=0.3)

    # P-value distributions
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

    # 3D KDE for LogFC vs P-values vs Density
    ax4 = fig2.add_subplot(gs2[1, 1], projection='3d')

    # Use explained targets only to avoid noise
    explained_for_kde = df_plot[df_plot['explained'] == True].copy()
    explained_for_kde['neg_log10_pval'] = -np.log10(explained_for_kde['pval'].clip(lower=1e-50))

    if len(explained_for_kde) > 20:
        try:
            # Prepare data for KDE
            x = explained_for_kde['abs_logfc'].values
            y = explained_for_kde['neg_log10_pval'].values
            kde = gaussian_kde([x, y])

            # Create grid
            xi = np.linspace(x.min(), x.max(), 40)
            yi = np.linspace(y.min(), y.max(), 40)
            X, Y = np.meshgrid(xi, yi)

            # Calculate KDE density
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = kde(positions).reshape(X.shape)

            # Plot surface
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
    """Generate final unexplained CSV with adjustable threshold"""

    print(f"\n=== GENERATING FINAL UNEXPLAINED TARGETS CSV ===")

    # Filter unexplained and remove self-targeting
    unexplained = df[(df['pathway_type'] == 'unexplained') & (df['source'] != df['target'])].copy()
    print(f"Unexplained targets (excluding self-targeting): {len(unexplained):,}")

    if len(unexplained) == 0:
        return None

    # Apply fold change threshold
    unexplained['abs_logfc'] = np.abs(unexplained['logfoldchange'])
    high_fc = unexplained[unexplained['abs_logfc'] >= fc_threshold].copy()

    # Adjust threshold if needed to get reasonable number
    if len(high_fc) == 0:
        for threshold in [1.5, 1.0, 0.8]:
            high_fc = unexplained[unexplained['abs_logfc'] >= threshold]
            print(f"With threshold >= {threshold}: {len(high_fc):,} targets")
            if len(high_fc) >= 10:
                fc_threshold = threshold
                break

    if len(high_fc) == 0:
        high_fc = unexplained.nlargest(50, 'abs_logfc')  # Top 50
        print(f"Using top 50 unexplained targets by fold change")

    # Sort by fold change
    high_fc = high_fc.sort_values('abs_logfc', ascending=False)

    # Create CSV
    output_df = pd.DataFrame({
        'source_gene': high_fc['source'],
        'target_gene': high_fc['target'],
        'log_fold_change': high_fc['logfoldchange'],
        'abs_log_fold_change': high_fc['abs_logfc'],
        'p_value': high_fc['pval'],
        'linear_fold_change': 2 ** high_fc['abs_logfc'],
        'neg_log10_pval': -np.log10(high_fc['pval'].clip(lower=1e-50))
    })

    filename = "final_unexplained_targets_3hop_direct.csv"
    output_df.to_csv(filename, index=False)

    print(f"Final CSV saved: {filename}")
    print(f"Targets included: {len(output_df):,}")
    print(
        f"Fold change range: {output_df['abs_log_fold_change'].min():.2f} to {output_df['abs_log_fold_change'].max():.2f}")

    print("\nTop 10 targets:")
    print(output_df.head(10)[['source_gene', 'target_gene', 'abs_log_fold_change', 'p_value']].to_string(index=False))

    return output_df


def main():
    """Main analysis execution"""

    # Load and filter pathway datasets
    df_1hop, df_2hop, df_3hop = load_and_filter_pathway_datasets()

    # Create combined dataset with priority logic
    combined_df = create_combined_3pathway_dataset(df_1hop, df_2hop, df_3hop)

    # Add unexplained targets
    final_df = add_unexplained_targets_3hop(combined_df)

    # Create visualizations
    print("\n" + "=" * 80)
    print("CREATING 3-PATHWAY VISUALIZATIONS")
    print("=" * 80)

    stats = create_enhanced_4pathway_plot(
        final_df,
        "Direct 3-Pathway Analysis",
        "direct_3pathway_analysis"
    )

    # Generate statistics table
    stats_df = pd.DataFrame(stats).T
    stats_df['pathway_type'] = stats_df.index
    cols = ['pathway_type', 'count', 'mean_abs_fc', 'mean_neg_log10_pval', 'mean_belief', 'std_fc', 'median_fc',
            'fc_95th']
    stats_df = stats_df[cols]

    stats_filename = 'direct_3pathway_statistics.csv'
    stats_df.to_csv(stats_filename, index=False)
    print(f"Statistics saved to: {stats_filename}")

    print("\nStatistical Summary:")
    print(stats_df.to_string(index=False, float_format='%.4f'))

    # Generate final unexplained CSV
    print("\n" + "=" * 80)
    print("GENERATING FINAL UNEXPLAINED TARGETS")
    print("=" * 80)

    final_csv = generate_final_unexplained_csv(final_df, fc_threshold=2.0)

    # Statistical tests
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISONS")
    print("=" * 80)

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
                print(f"\n{type1} vs {type2}:")
                print(f"  Mean |FC| - {type1}: {fc1.mean():.4f}")
                print(f"  Mean |FC| - {type2}: {fc2.mean():.4f}")
                print(f"  Difference: {fc1.mean() - fc2.mean():.4f}")
                print(f"  T-test p-value: {t_pval:.2e}")
            except:
                print(f"\n{type1} vs {type2}: Statistical test failed")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)

    print("Files generated:")
    print("1. direct_3pathway_analysis_main.png - Main 8-panel visualization")
    print("2. direct_3pathway_analysis_advanced.png - Advanced 4-panel visualization")
    print("3. direct_3pathway_statistics.csv - Statistical summary")
    print("4. final_unexplained_targets_3hop_direct.csv - Final unexplained targets")

    print(f"\nFinal Results:")
    print(f"- Total pairs analyzed: {len(final_df):,}")
    print(f"- Coverage by all pathway types: {len(final_df[final_df['explained'] == True]) / len(final_df) * 100:.1f}%")
    print(
        f"- Unexplained after all pathways: {len(final_df[final_df['explained'] == False]) / len(final_df) * 100:.1f}%")

    return final_df, stats, final_csv


# Execute the analysis
if __name__ == "__main__":
    final_dataset, statistics, unexplained_csv = main()
    print("\nDirect 3-pathway analysis complete!")