import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("Loading datasets for combined 1-hop + 2-hop analysis...")
print("=" * 70)

# Load 1-hop datasets
print("Loading 1-hop datasets...")
df_1hop_all = pd.read_excel("/Users/prashammarfatia/Downloads/indra_1hop_all_perturbations.xlsx")
df_1hop_no_tp53 = pd.read_csv("/Users/prashammarfatia/Downloads/indra_1hop_no_v2.csv")

# Load 2-hop datasets
print("Loading 2-hop datasets...")
df_2hop_no_tp53 = pd.read_excel("/Users/prashammarfatia/Downloads/indra_2hop_all_perturbations.xlsx")  # Excluding TP53
df_2hop_tp53_only = pd.read_csv("/Users/prashammarfatia/Downloads/indra_2-hop_gene_symbol_only_2.csv")  # TP53 only

print(f"1-hop all perturbations: {len(df_1hop_all):,} pathways")
print(f"1-hop excluding TP53: {len(df_1hop_no_tp53):,} pathways")
print(f"2-hop excluding TP53: {len(df_2hop_no_tp53):,} pathways")
print(f"2-hop TP53 only: {len(df_2hop_tp53_only):,} pathways")


# Standardize column names
def standardize_columns(df, dataset_name):
    """Standardize column names across datasets"""
    # Handle belief columns
    if 'belief_1' in df.columns and 'belief' not in df.columns:
        # For 2-hop: calculate mean belief
        if 'belief_2' in df.columns:
            df['belief'] = (df['belief_1'] + df['belief_2']) / 2
            print(f"Created mean belief from belief_1 and belief_2 in {dataset_name}")
        else:
            df['belief'] = df['belief_1']
            print(f"Renamed 'belief_1' to 'belief' in {dataset_name}")

    # Handle p-value columns
    if 'pvalue' in df.columns and 'pval' not in df.columns:
        df = df.rename(columns={'pvalue': 'pval'})
        print(f"Renamed 'pvalue' to 'pval' in {dataset_name}")

    return df


print("\nStandardizing column names...")
df_1hop_all = standardize_columns(df_1hop_all, "1-hop all")
df_1hop_no_tp53 = standardize_columns(df_1hop_no_tp53, "1-hop no TP53")
df_2hop_no_tp53 = standardize_columns(df_2hop_no_tp53, "2-hop no TP53")
df_2hop_tp53_only = standardize_columns(df_2hop_tp53_only, "2-hop TP53 only")

# Add pathway type indicators
df_1hop_all['pathway_type'] = '1-hop'
df_1hop_no_tp53['pathway_type'] = '1-hop'
df_2hop_no_tp53['pathway_type'] = '2-hop'
df_2hop_tp53_only['pathway_type'] = '2-hop'

print("\nColumn structures after standardization:")
print(f"1-hop columns: {list(df_1hop_all.columns)}")
print(f"2-hop columns: {list(df_2hop_no_tp53.columns)}")


def create_combined_dataset(df_1hop, df_2hop, dataset_name, include_tp53=True):
    """
    Create combined dataset with priority logic:
    1. Use 1-hop pathway if available
    2. If no 1-hop, use 2-hop with highest belief score
    """
    print(f"\nProcessing {dataset_name}...")

    # Combine datasets
    all_pathways = []

    # Add 1-hop data
    if len(df_1hop) > 0:
        for _, row in df_1hop.iterrows():
            if not include_tp53 and row['source'] == 'TP53':
                continue
            all_pathways.append({
                'source': row['source'],
                'target': row['target'],
                'belief': row['belief'],
                'logfoldchange': row['logfoldchange'],
                'pval': row['pval'],
                'pathway_type': '1-hop',
                'priority': 1  # Highest priority
            })

    # Add 2-hop data
    if len(df_2hop) > 0:
        for _, row in df_2hop.iterrows():
            if not include_tp53 and row['source'] == 'TP53':
                continue
            all_pathways.append({
                'source': row['source'],
                'target': row['target'],
                'belief': row['belief'],
                'logfoldchange': row['logfoldchange'],
                'pval': row['pval'],
                'pathway_type': '2-hop',
                'priority': 2  # Lower priority
            })

    # Convert to DataFrame
    combined_df = pd.DataFrame(all_pathways)

    if len(combined_df) == 0:
        print(f"  No data for {dataset_name}")
        return pd.DataFrame()

    print(f"  Combined raw data: {len(combined_df):,} pathways")
    print(f"    1-hop: {len(combined_df[combined_df['pathway_type'] == '1-hop']):,}")
    print(f"    2-hop: {len(combined_df[combined_df['pathway_type'] == '2-hop']):,}")

    # Apply priority logic: for each (source, target) pair, keep the best pathway
    def select_best_pathway(group):
        """Select best pathway for a source-target pair"""
        # Sort by priority (1-hop first), then by belief score (highest first)
        sorted_group = group.sort_values(['priority', 'belief'], ascending=[True, False])
        return sorted_group.iloc[0]

    # Group by source-target pairs and select best pathway
    best_pathways = combined_df.groupby(['source', 'target']).apply(select_best_pathway).reset_index(drop=True)

    print(f"  After applying priority logic: {len(best_pathways):,} unique source-target pairs")
    print(f"    1-hop selected: {len(best_pathways[best_pathways['pathway_type'] == '1-hop']):,}")
    print(f"    2-hop selected: {len(best_pathways[best_pathways['pathway_type'] == '2-hop']):,}")

    # Calculate coverage statistics
    total_before = len(combined_df)
    total_after = len(best_pathways)
    reduction = total_before - total_after

    print(f"  Removed {reduction:,} redundant pathways ({reduction / total_before * 100:.1f}%)")

    return best_pathways


# Create combined datasets
print("\n" + "=" * 70)
print("CREATING COMBINED DATASETS WITH PRIORITY LOGIC")
print("=" * 70)

# All perturbations (including TP53)
df_combined_all = create_combined_dataset(
    df_1hop_all,
    pd.concat([df_2hop_no_tp53, df_2hop_tp53_only]),
    "All Perturbations",
    include_tp53=True
)

# Excluding TP53
df_combined_no_tp53 = create_combined_dataset(
    df_1hop_no_tp53,
    df_2hop_no_tp53,
    "Excluding TP53",
    include_tp53=False
)

# TP53 only
df_1hop_tp53_only = df_1hop_all[df_1hop_all['source'] == 'TP53'].copy()
df_combined_tp53_only = create_combined_dataset(
    df_1hop_tp53_only,
    df_2hop_tp53_only,
    "TP53 Only",
    include_tp53=True
)


# Add unexplained targets for each dataset
def add_unexplained_targets(combined_df, dataset_name, include_tp53=True):
    """Add unexplained targets to combined dataset"""
    print(f"\nAdding unexplained targets for {dataset_name}...")

    # Load DEG data
    deg_folder = "/Users/prashammarfatia/Downloads/de_results_per_gene"
    deg_files = list(Path(deg_folder).glob("*_vs_control.csv"))

    unexplained_data = []
    explained_pairs = set()

    # Get all explained source-target pairs
    for _, row in combined_df.iterrows():
        explained_pairs.add((row['source'], row['target']))

    # Get unique sources from combined data
    if len(combined_df) > 0:
        pathway_sources = combined_df['source'].unique()
    else:
        pathway_sources = []

    for deg_file in deg_files:
        gene_name = deg_file.stem.replace("_vs_control", "")

        # Skip TP53 if not including it
        if not include_tp53 and gene_name == 'TP53':
            continue

        # Skip if gene not in pathway sources or if no pathways for this gene
        if gene_name not in pathway_sources:
            continue

        try:
            # Load DEG data
            deg_df = pd.read_csv(deg_file)
            significant_genes = deg_df[deg_df['pvals'] < 0.05].copy()

            for _, deg_row in significant_genes.iterrows():
                target_gene = deg_row['names']

                # Skip if this source-target pair is already explained
                if (gene_name, target_gene) in explained_pairs:
                    continue

                # Add to unexplained data
                unexplained_data.append({
                    'source': gene_name,
                    'target': target_gene,
                    'belief': 0.5,  # Assigned belief for unexplained
                    'logfoldchange': deg_row['logfoldchanges'],
                    'pval': deg_row['pvals'],
                    'pathway_type': 'none',
                    'priority': 3,
                    'explained': False
                })
        except Exception as e:
            continue

    # Add explained flag to combined data
    combined_df['explained'] = True

    # Combine explained and unexplained
    if unexplained_data:
        unexplained_df = pd.DataFrame(unexplained_data)
        final_df = pd.concat([combined_df, unexplained_df], ignore_index=True)
    else:
        final_df = combined_df.copy()

    # Fill missing explained flag
    final_df['explained'] = final_df['explained'].fillna(False)

    explained_count = len(final_df[final_df['explained'] == True])
    unexplained_count = len(final_df[final_df['explained'] == False])

    print(f"  Final dataset: {len(final_df):,} total pairs")
    print(f"    Explained: {explained_count:,} ({explained_count / len(final_df) * 100:.1f}%)")
    print(f"    Unexplained: {unexplained_count:,} ({unexplained_count / len(final_df) * 100:.1f}%)")

    return final_df


# Add unexplained targets to each dataset
print("\n" + "=" * 70)
print("ADDING UNEXPLAINED TARGETS")
print("=" * 70)

df_final_all = add_unexplained_targets(df_combined_all, "All Perturbations", include_tp53=True)
df_final_no_tp53 = add_unexplained_targets(df_combined_no_tp53, "Excluding TP53", include_tp53=False)
df_final_tp53_only = add_unexplained_targets(df_combined_tp53_only, "TP53 Only", include_tp53=True)

# Summary statistics
print("\n" + "=" * 70)
print("COMBINED DATASET SUMMARY STATISTICS")
print("=" * 70)


def print_combined_stats(df, name):
    if len(df) == 0:
        print(f"{name}: No data")
        return

    explained = df[df['explained'] == True]
    unexplained = df[df['explained'] == False]

    # Pathway type breakdown for explained
    onehop = explained[explained['pathway_type'] == '1-hop']
    twohop = explained[explained['pathway_type'] == '2-hop']

    print(f"\n{name}:")
    print(f"  Total pairs: {len(df):,}")
    print(f"  Explained: {len(explained):,} ({len(explained) / len(df) * 100:.1f}%)")
    print(f"    1-hop selected: {len(onehop):,} ({len(onehop) / len(explained) * 100:.1f}% of explained)")
    print(f"    2-hop selected: {len(twohop):,} ({len(twohop) / len(explained) * 100:.1f}% of explained)")
    print(f"  Unexplained: {len(unexplained):,} ({len(unexplained) / len(df) * 100:.1f}%)")

    if len(explained) > 0:
        print(f"  Explained - Mean |FC|: {np.abs(explained['logfoldchange']).mean():.3f}")
        print(f"  Explained - Mean belief: {explained['belief'].mean():.3f}")
    if len(unexplained) > 0:
        print(f"  Unexplained - Mean |FC|: {np.abs(unexplained['logfoldchange']).mean():.3f}")


print_combined_stats(df_final_all, "All Perturbations (Combined)")
print_combined_stats(df_final_tp53_only, "TP53 Only (Combined)")
print_combined_stats(df_final_no_tp53, "Excluding TP53 (Combined)")


# Create visualization function with data integrity checks
def create_combined_plot(df, title, save_name):
    """Create visualization for combined 1-hop + 2-hop analysis with data integrity checks"""

    if len(df) == 0:
        print(f"No data for {title}")
        return

    print(f"\n=== DATA INTEGRITY CHECKS FOR {title.upper()} ===")

    df_plot = df.copy()
    df_plot['abs_logfc'] = np.abs(df_plot['logfoldchange'])

    # Separate data by explanation type
    explained = df_plot[df_plot['explained'] == True]
    unexplained = df_plot[df_plot['explained'] == False]
    onehop = explained[explained['pathway_type'] == '1-hop']
    twohop = explained[explained['pathway_type'] == '2-hop']

    # Data integrity checks
    print(f"Total data points: {len(df_plot):,}")
    print(f"  Explained: {len(explained):,}")
    print(f"    1-hop: {len(onehop):,}")
    print(f"    2-hop: {len(twohop):,}")
    print(f"  Unexplained: {len(unexplained):,}")
    print(f"  Sum check: {len(explained) + len(unexplained)} == {len(df_plot)} ✓" if len(explained) + len(
        unexplained) == len(df_plot) else f"  ❌ Sum mismatch!")

    # Fold change range checks
    print(f"\nFold Change Ranges:")
    if len(onehop) > 0:
        print(f"  1-hop: {onehop['abs_logfc'].min():.3f} to {onehop['abs_logfc'].max():.3f}")
    if len(twohop) > 0:
        print(f"  2-hop: {twohop['abs_logfc'].min():.3f} to {twohop['abs_logfc'].max():.3f}")
    if len(unexplained) > 0:
        print(f"  Unexplained: {unexplained['abs_logfc'].min():.3f} to {unexplained['abs_logfc'].max():.3f}")

    # Overall range
    print(f"  Overall range: {df_plot['abs_logfc'].min():.3f} to {df_plot['abs_logfc'].max():.3f}")

    # High fold change analysis
    high_fc_threshold = 2.5
    high_fc_total = len(df_plot[df_plot['abs_logfc'] > high_fc_threshold])
    print(f"\nHigh fold change (>{high_fc_threshold}) distribution:")
    print(f"  Total: {high_fc_total}")
    if len(onehop) > 0:
        high_fc_1hop = len(onehop[onehop['abs_logfc'] > high_fc_threshold])
        print(f"  1-hop: {high_fc_1hop} ({high_fc_1hop / len(onehop) * 100:.1f}%)")
    if len(twohop) > 0:
        high_fc_2hop = len(twohop[twohop['abs_logfc'] > high_fc_threshold])
        print(f"  2-hop: {high_fc_2hop} ({high_fc_2hop / len(twohop) * 100:.1f}%)")
    if len(unexplained) > 0:
        high_fc_unexp = len(unexplained[unexplained['abs_logfc'] > high_fc_threshold])
        print(f"  Unexplained: {high_fc_unexp} ({high_fc_unexp / len(unexplained) * 100:.1f}%)")

    # Missing data checks
    print(f"\nMissing Data Checks:")
    print(f"  Missing abs_logfc: {df_plot['abs_logfc'].isna().sum()}")
    print(f"  Missing belief: {df_plot['belief'].isna().sum()}")
    print(f"  Missing explained flag: {df_plot['explained'].isna().sum()}")

    # Create improved visualization with separate panels
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'{title} - Combined 1-Hop + 2-Hop Analysis', fontsize=16, fontweight='bold')

    # Create grid: 2 rows, 4 columns
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # Plot 1: Pathway type comparison scatter plot (MOVED from Plot 2)
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

    # Plot 2: Heat map for LogFC vs Belief Score (NEW - explained data only)
    ax2 = fig.add_subplot(gs[0, 1])
    if len(explained) > 0:
        h2 = ax2.hist2d(explained['abs_logfc'], explained['belief'], bins=50, cmap='viridis')
        plt.colorbar(h2[3], ax=ax2)
        print(f"Heat map: Using {len(explained):,} explained data points")
    ax2.set_xlabel('|Log Fold Change|')
    ax2.set_ylabel('Belief Score')
    ax2.set_title('LogFC vs Belief Score (Heat Map)')
    ax2.grid(True, alpha=0.3)

    # Plots 3-5: Separate fold change histograms
    # 1-hop histogram
    ax3 = fig.add_subplot(gs[1, 0])
    if len(onehop) > 0:
        ax3.hist(onehop['abs_logfc'], bins=40, density=False, color='blue', alpha=0.7)
        ax3.set_title(f'1-hop (n={len(onehop):,})')
        print(
            f"1-hop histogram: {len(onehop):,} points, range {onehop['abs_logfc'].min():.2f}-{onehop['abs_logfc'].max():.2f}")
    else:
        ax3.text(0.5, 0.5, 'No 1-hop data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('1-hop (n=0)')
    ax3.set_xlabel('|Log Fold Change|')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3)

    # 2-hop histogram
    ax4 = fig.add_subplot(gs[1, 1])
    if len(twohop) > 0:
        ax4.hist(twohop['abs_logfc'], bins=100, density=False, color='green', alpha=0.7)
        ax4.set_title(f'2-hop (n={len(twohop):,})')
        print(
            f"2-hop histogram: {len(twohop):,} points, range {twohop['abs_logfc'].min():.2f}-{twohop['abs_logfc'].max():.2f}")
    else:
        ax4.text(0.5, 0.5, 'No 2-hop data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('2-hop (n=0)')
    ax4.set_xlabel('|Log Fold Change|')
    ax4.set_ylabel('Count')
    ax4.grid(True, alpha=0.3)

    # Unexplained histogram
    ax5 = fig.add_subplot(gs[1, 2])
    if len(unexplained) > 0:
        ax5.hist(unexplained['abs_logfc'], bins=100, density=False, color='red', alpha=0.7)
        ax5.set_title(f'Unexplained (n={len(unexplained):,})')
        print(
            f"Unexplained histogram: {len(unexplained):,} points, range {unexplained['abs_logfc'].min():.2f}-{unexplained['abs_logfc'].max():.2f}")
    else:
        ax5.text(0.5, 0.5, 'No unexplained data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Unexplained (n=0)')
    ax5.set_xlabel('|Log Fold Change|')
    ax5.set_ylabel('Count')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Belief score distribution (explained only)
    ax6 = fig.add_subplot(gs[0, 2:])  # Span 2 columns
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

    # Summary statistics panel
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.axis('off')

    # Create summary text
    summary_text = f"""SUMMARY STATISTICS

Total Data Points: {len(df_plot):,}

EXPLAINED ({len(explained):,}):
• 1-hop: {len(onehop):,}
• 2-hop: {len(twohop):,}

UNEXPLAINED: {len(unexplained):,}

FOLD CHANGE STATS:
"""

    if len(explained) > 0:
        summary_text += f"• Explained mean: {explained['abs_logfc'].mean():.3f}\n"
    if len(unexplained) > 0:
        summary_text += f"• Unexplained mean: {unexplained['abs_logfc'].mean():.3f}\n"

    summary_text += f"\nHIGH FC (>{high_fc_threshold}):\n"
    if len(onehop) > 0:
        summary_text += f"• 1-hop: {len(onehop[onehop['abs_logfc'] > high_fc_threshold])}\n"
    if len(twohop) > 0:
        summary_text += f"• 2-hop: {len(twohop[twohop['abs_logfc'] > high_fc_threshold])}\n"
    if len(unexplained) > 0:
        summary_text += f"• Unexplained: {len(unexplained[unexplained['abs_logfc'] > high_fc_threshold])}\n"

    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)

    print(f"=== VISUALIZATION COMPLETE FOR {title.upper()} ===\n")

    return df_plot


# Create combined plots
print("\n" + "=" * 70)
print("CREATING COMBINED VISUALIZATIONS")
print("=" * 70)

create_combined_plot(df_final_all, "All Perturbations", "combined_all_perturbations")
create_combined_plot(df_final_tp53_only, "TP53 Only", "combined_tp53_only")
create_combined_plot(df_final_no_tp53, "Excluding TP53", "combined_excluding_tp53")

# Add comprehensive statistical analysis
print("\n" + "=" * 70)
print("COMPREHENSIVE STATISTICAL ANALYSIS")
print("=" * 70)


def comprehensive_combined_analysis(df, name):
    if len(df) == 0:
        print(f"\n{name}: No data for analysis")
        return

    explained = df[df['explained'] == True]
    unexplained = df[df['explained'] == False]
    onehop = explained[explained['pathway_type'] == '1-hop']
    twohop = explained[explained['pathway_type'] == '2-hop']

    print(f"\n{name}:")
    print("-" * (len(name) + 1))

    # Basic counts
    print(f"Dataset Summary:")
    print(f"  Total data points: {len(df):,}")
    print(f"  Explained pathways: {len(explained):,} ({len(explained) / len(df) * 100:.2f}%)")
    print(f"    1-hop: {len(onehop):,} ({len(onehop) / len(explained) * 100:.1f}% of explained)")
    print(f"    2-hop: {len(twohop):,} ({len(twohop) / len(explained) * 100:.1f}% of explained)")
    print(f"  Unexplained targets: {len(unexplained):,} ({len(unexplained) / len(df) * 100:.2f}%)")

    if len(df) > 0:
        print(f"  Unique sources: {df['source'].nunique():,}")
        print(f"  Unique targets: {df['target'].nunique():,}")

    # Explained pathways statistics
    if len(explained) > 0:
        abs_fc_explained = np.abs(explained['logfoldchange'])
        belief_explained = explained['belief']
        pval_explained = explained['pval']

        print(f"\nExplained Pathways Statistics:")
        print(f"  |Fold Change|:")
        print(f"    Mean: {abs_fc_explained.mean():.4f}")
        print(f"    Median: {abs_fc_explained.median():.4f}")
        print(f"    Std: {abs_fc_explained.std():.4f}")
        print(f"    Min: {abs_fc_explained.min():.4f}")
        print(f"    Max: {abs_fc_explained.max():.4f}")
        print(f"    95th percentile: {np.percentile(abs_fc_explained, 95):.4f}")

        print(f"  Belief Score:")
        print(f"    Mean: {belief_explained.mean():.4f}")
        print(f"    Median: {belief_explained.median():.4f}")
        print(f"    Std: {belief_explained.std():.4f}")
        print(f"    Min: {belief_explained.min():.4f}")
        print(f"    Max: {belief_explained.max():.4f}")

        print(f"  P-value:")
        print(f"    Mean: {pval_explained.mean():.6f}")
        print(f"    Median: {pval_explained.median():.6f}")
        print(f"    Min: {pval_explained.min():.2e}")
        print(f"    Max: {pval_explained.max():.4f}")

        # Correlation analysis
        correlation_fc_belief = np.corrcoef(abs_fc_explained, belief_explained)[0, 1]
        correlation_pval_belief = np.corrcoef(pval_explained, belief_explained)[0, 1]
        one_minus_pval = 1 - pval_explained
        correlation_sig_belief = np.corrcoef(one_minus_pval, belief_explained)[0, 1]

        print(f"\nCorrelation Analysis (Explained only):")
        print(f"  |LogFC| vs Belief Score: r = {correlation_fc_belief:.6f}")
        print(f"  P-value vs Belief Score: r = {correlation_pval_belief:.6f}")
        print(f"  (1-P-value) vs Belief Score: r = {correlation_sig_belief:.6f}")

        # High/low belief score analysis
        high_belief = explained[explained['belief'] >= 0.8]
        low_belief = explained[explained['belief'] <= 0.7]

        print(f"\nBelief Score Categories:")
        print(f"  High belief (≥0.8): {len(high_belief):,} ({len(high_belief) / len(explained) * 100:.1f}%)")
        if len(high_belief) > 0:
            print(f"    Mean |FC|: {np.abs(high_belief['logfoldchange']).mean():.4f}")
        print(f"  Low belief (≤0.7): {len(low_belief):,} ({len(low_belief) / len(explained) * 100:.1f}%)")
        if len(low_belief) > 0:
            print(f"    Mean |FC|: {np.abs(low_belief['logfoldchange']).mean():.4f}")

        # 1-hop vs 2-hop comparison
        if len(onehop) > 0 and len(twohop) > 0:
            abs_fc_1hop = np.abs(onehop['logfoldchange'])
            abs_fc_2hop = np.abs(twohop['logfoldchange'])
            belief_1hop = onehop['belief']
            belief_2hop = twohop['belief']

            print(f"\n1-hop vs 2-hop Comparison:")
            print(f"  1-hop mean |FC|: {abs_fc_1hop.mean():.4f}")
            print(f"  2-hop mean |FC|: {abs_fc_2hop.mean():.4f}")
            print(f"  1-hop mean belief: {belief_1hop.mean():.4f}")
            print(f"  2-hop mean belief: {belief_2hop.mean():.4f}")

    # Unexplained targets statistics
    if len(unexplained) > 0:
        abs_fc_unexplained = np.abs(unexplained['logfoldchange'])
        pval_unexplained = unexplained['pval']

        print(f"\nUnexplained Targets Statistics:")
        print(f"  |Fold Change|:")
        print(f"    Mean: {abs_fc_unexplained.mean():.4f}")
        print(f"    Median: {abs_fc_unexplained.median():.4f}")
        print(f"    Std: {abs_fc_unexplained.std():.4f}")
        print(f"    Min: {abs_fc_unexplained.min():.4f}")
        print(f"    Max: {abs_fc_unexplained.max():.4f}")
        print(f"    95th percentile: {np.percentile(abs_fc_unexplained, 95):.4f}")

        print(f"  P-value:")
        print(f"    Mean: {pval_unexplained.mean():.6f}")
        print(f"    Median: {pval_unexplained.median():.6f}")
        print(f"    Min: {pval_unexplained.min():.2e}")
        print(f"    Max: {pval_unexplained.max():.4f}")

    # Comparison between explained vs unexplained
    if len(explained) > 0 and len(unexplained) > 0:
        abs_fc_explained = np.abs(explained['logfoldchange'])
        abs_fc_unexplained = np.abs(unexplained['logfoldchange'])

        # Statistical comparison
        try:
            from scipy import stats
            t_stat, t_pval = stats.ttest_ind(abs_fc_explained, abs_fc_unexplained)
            mannwhitney_stat, mannwhitney_pval = stats.mannwhitneyu(abs_fc_explained, abs_fc_unexplained,
                                                                    alternative='two-sided')

            print(f"\nExplained vs Unexplained Comparison:")
            print(f"  Mean |FC| - Explained: {abs_fc_explained.mean():.4f}")
            print(f"  Mean |FC| - Unexplained: {abs_fc_unexplained.mean():.4f}")
            print(f"  Fold difference: {abs_fc_explained.mean() / abs_fc_unexplained.mean():.2f}x")
            print(f"  T-test p-value: {t_pval:.2e}")
            print(f"  Mann-Whitney U p-value: {mannwhitney_pval:.2e}")

            # Effect size categories
            explained_high_fc = len(explained[np.abs(explained['logfoldchange']) > 1.0])
            unexplained_high_fc = len(unexplained[np.abs(unexplained['logfoldchange']) > 1.0])

            print(
                f"  High |FC| (>1.0) - Explained: {explained_high_fc:,} ({explained_high_fc / len(explained) * 100:.1f}%)")
            print(
                f"  High |FC| (>1.0) - Unexplained: {unexplained_high_fc:,} ({unexplained_high_fc / len(unexplained) * 100:.1f}%)")
        except ImportError:
            print(f"\nExplained vs Unexplained Comparison (scipy not available):")
            print(f"  Mean |FC| - Explained: {abs_fc_explained.mean():.4f}")
            print(f"  Mean |FC| - Unexplained: {abs_fc_unexplained.mean():.4f}")


# Add scipy import at the top
try:
    from scipy import stats
except ImportError:
    print("Warning: scipy not available for statistical tests")

# Run comprehensive analysis on all datasets
comprehensive_combined_analysis(df_final_all, "All Perturbations (Combined)")
comprehensive_combined_analysis(df_final_tp53_only, "TP53 Only (Combined)")
comprehensive_combined_analysis(df_final_no_tp53, "Excluding TP53 (Combined)")

# Create comprehensive summary table
print("\n" + "=" * 70)
print("COMPREHENSIVE SUMMARY TABLE")
print("=" * 70)


def create_combined_summary_table():
    datasets = [
        ("All Perturbations", df_final_all),
        ("TP53 Only", df_final_tp53_only),
        ("Excluding TP53", df_final_no_tp53)
    ]

    print(
        f"{'Dataset':<20} {'Total':<8} {'Explained':<10} {'1-hop':<8} {'2-hop':<8} {'Unexpl':<8} {'Expl%':<8} {'Mean|FC|(Exp)':<13} {'Mean|FC|(Unexp)':<14} {'Correlation':<12}")
    print("-" * 125)

    for name, df in datasets:
        if len(df) == 0:
            print(
                f"{name:<20} {'0':<8} {'0':<10} {'0':<8} {'0':<8} {'0':<8} {'0.0%':<8} {'N/A':<13} {'N/A':<14} {'N/A':<12}")
            continue

        explained = df[df['explained'] == True]
        unexplained = df[df['explained'] == False]
        onehop = explained[explained['pathway_type'] == '1-hop']
        twohop = explained[explained['pathway_type'] == '2-hop']

        total = len(df)
        n_explained = len(explained)
        n_1hop = len(onehop)
        n_2hop = len(twohop)
        n_unexplained = len(unexplained)
        expl_pct = f"{n_explained / total * 100:.1f}%" if total > 0 else "0.0%"

        mean_fc_exp = f"{np.abs(explained['logfoldchange']).mean():.3f}" if len(explained) > 0 else "N/A"
        mean_fc_unexp = f"{np.abs(unexplained['logfoldchange']).mean():.3f}" if len(unexplained) > 0 else "N/A"

        if len(explained) >= 2:
            corr = np.corrcoef(np.abs(explained['logfoldchange']), explained['belief'])[0, 1]
            corr_str = f"{corr:.4f}"
        else:
            corr_str = "N/A"

        print(
            f"{name:<20} {total:<8} {n_explained:<10} {n_1hop:<8} {n_2hop:<8} {n_unexplained:<8} {expl_pct:<8} {mean_fc_exp:<13} {mean_fc_unexp:<14} {corr_str:<12}")


create_combined_summary_table()

print("\nCombined 1-hop + 2-hop analysis complete!")
print("Priority logic applied: 1-hop preferred, then best 2-hop by belief score.")
print("Improved plots with separate histogram panels and data integrity checks.")
print("Comprehensive statistical analysis including correlations and comparisons completed.")