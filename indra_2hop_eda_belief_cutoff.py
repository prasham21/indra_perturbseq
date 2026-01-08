import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, ks_2samp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("Loading datasets for combined 1-hop + 2-hop analysis with BELIEF SCORE CUTOFF ≥0.7...")
print("=" * 80)

# Load 1-hop datasets
print("Loading 1-hop datasets...")
df_1hop_all = pd.read_excel("/Users/prashammarfatia/Downloads/indra_1hop_all_perturbations.xlsx")
df_1hop_no_tp53 = pd.read_csv("/Users/prashammarfatia/Downloads/indra_1hop_no_v2.csv")

# Load 2-hop datasets
print("Loading 2-hop datasets...")
df_2hop_no_tp53 = pd.read_excel("/Users/prashammarfatia/Downloads/indra_2hop_all_perturbations.xlsx")
df_2hop_tp53_only = pd.read_csv("/Users/prashammarfatia/Downloads/indra_2-hop_gene_symbol_only_2.csv")

print(f"1-hop all perturbations (before filtering): {len(df_1hop_all):,} pathways")
print(f"1-hop excluding TP53 (before filtering): {len(df_1hop_no_tp53):,} pathways")
print(f"2-hop excluding TP53 (before filtering): {len(df_2hop_no_tp53):,} pathways")
print(f"2-hop TP53 only (before filtering): {len(df_2hop_tp53_only):,} pathways")


def standardize_columns(df, dataset_name):
    """Standardize column names across datasets"""
    if 'belief_1' in df.columns and 'belief' not in df.columns:
        if 'belief_2' in df.columns:
            df['belief'] = (df['belief_1'] + df['belief_2']) / 2
            print(f"Created mean belief from belief_1 and belief_2 in {dataset_name}")
        else:
            df['belief'] = df['belief_1']
            print(f"Renamed 'belief_1' to 'belief' in {dataset_name}")

    if 'pvalue' in df.columns and 'pval' not in df.columns:
        df = df.rename(columns={'pvalue': 'pval'})
        print(f"Renamed 'pvalue' to 'pval' in {dataset_name}")

    return df


print("\nStandardizing column names...")
df_1hop_all = standardize_columns(df_1hop_all, "1-hop all")
df_1hop_no_tp53 = standardize_columns(df_1hop_no_tp53, "1-hop no TP53")
df_2hop_no_tp53 = standardize_columns(df_2hop_no_tp53, "2-hop no TP53")
df_2hop_tp53_only = standardize_columns(df_2hop_tp53_only, "2-hop TP53 only")

print("\n" + "=" * 80)
print("APPLYING BELIEF SCORE CUTOFF ≥0.7")
print("=" * 80)

# Apply belief score cutoff ≥0.7 to all datasets
print("Filtering pathways with belief score ≥0.7...")
df_1hop_all_filtered = df_1hop_all[df_1hop_all['belief'] >= 0.7].copy()
df_1hop_no_tp53_filtered = df_1hop_no_tp53[df_1hop_no_tp53['belief'] >= 0.7].copy()
df_2hop_no_tp53_filtered = df_2hop_no_tp53[df_2hop_no_tp53['belief'] >= 0.7].copy()
df_2hop_tp53_only_filtered = df_2hop_tp53_only[df_2hop_tp53_only['belief'] >= 0.7].copy()

print(f"1-hop all perturbations (after filtering): {len(df_1hop_all_filtered):,} pathways")
print(f"1-hop excluding TP53 (after filtering): {len(df_1hop_no_tp53_filtered):,} pathways")
print(f"2-hop excluding TP53 (after filtering): {len(df_2hop_no_tp53_filtered):,} pathways")
print(f"2-hop TP53 only (after filtering): {len(df_2hop_tp53_only_filtered):,} pathways")

print(f"\nFiltered out: {len(df_1hop_all) - len(df_1hop_all_filtered):,} 1-hop pathways from all perturbations")
print(f"Filtered out: {len(df_1hop_no_tp53) - len(df_1hop_no_tp53_filtered):,} 1-hop pathways from excluding TP53")
print(f"Filtered out: {len(df_2hop_no_tp53) - len(df_2hop_no_tp53_filtered):,} 2-hop pathways from excluding TP53")
print(f"Filtered out: {len(df_2hop_tp53_only) - len(df_2hop_tp53_only_filtered):,} 2-hop pathways from TP53 only")

# Add pathway type indicators
df_1hop_all_filtered['pathway_type'] = '1-hop'
df_1hop_no_tp53_filtered['pathway_type'] = '1-hop'
df_2hop_no_tp53_filtered['pathway_type'] = '2-hop'
df_2hop_tp53_only_filtered['pathway_type'] = '2-hop'

print("\nColumn structures after standardization and filtering:")
print(f"1-hop columns: {list(df_1hop_all_filtered.columns)}")
print(f"2-hop columns: {list(df_2hop_no_tp53_filtered.columns)}")


def create_combined_dataset(df_1hop, df_2hop, dataset_name, include_tp53=True):
    """Create combined dataset with priority logic"""
    print(f"\nProcessing {dataset_name}...")

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
                'priority': 1
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
                'priority': 2
            })

    combined_df = pd.DataFrame(all_pathways)

    if len(combined_df) == 0:
        print(f"  No data for {dataset_name}")
        return pd.DataFrame()

    print(f"  Combined raw data: {len(combined_df):,} pathways")
    print(f"    1-hop: {len(combined_df[combined_df['pathway_type'] == '1-hop']):,}")
    print(f"    2-hop: {len(combined_df[combined_df['pathway_type'] == '2-hop']):,}")

    def select_best_pathway(group):
        sorted_group = group.sort_values(['priority', 'belief'], ascending=[True, False])
        return sorted_group.iloc[0]

    best_pathways = combined_df.groupby(['source', 'target']).apply(select_best_pathway).reset_index(drop=True)

    print(f"  After applying priority logic: {len(best_pathways):,} unique source-target pairs")
    print(f"    1-hop selected: {len(best_pathways[best_pathways['pathway_type'] == '1-hop']):,}")
    print(f"    2-hop selected: {len(best_pathways[best_pathways['pathway_type'] == '2-hop']):,}")

    total_before = len(combined_df)
    total_after = len(best_pathways)
    reduction = total_before - total_after

    print(f"  Removed {reduction:,} redundant pathways ({reduction / total_before * 100:.1f}%)")

    return best_pathways


print("\n" + "=" * 80)
print("CREATING COMBINED DATASETS WITH PRIORITY LOGIC (FILTERED)")
print("=" * 80)

# All perturbations (including TP53)
df_combined_all = create_combined_dataset(
    df_1hop_all_filtered,
    pd.concat([df_2hop_no_tp53_filtered, df_2hop_tp53_only_filtered]),
    "All Perturbations",
    include_tp53=True
)

# Excluding TP53
df_combined_no_tp53 = create_combined_dataset(
    df_1hop_no_tp53_filtered,
    df_2hop_no_tp53_filtered,
    "Excluding TP53",
    include_tp53=False
)

# TP53 only
df_1hop_tp53_only_filtered = df_1hop_all_filtered[df_1hop_all_filtered['source'] == 'TP53'].copy()
df_combined_tp53_only = create_combined_dataset(
    df_1hop_tp53_only_filtered,
    df_2hop_tp53_only_filtered,
    "TP53 Only",
    include_tp53=True
)


def add_unexplained_targets(combined_df, dataset_name, include_tp53=True):
    """Add unexplained targets to combined dataset"""
    print(f"\nAdding unexplained targets for {dataset_name}...")

    deg_folder = "/Users/prashammarfatia/Downloads/de_results_per_gene"
    deg_files = list(Path(deg_folder).glob("*_vs_control.csv"))

    unexplained_data = []
    explained_pairs = set()

    for _, row in combined_df.iterrows():
        explained_pairs.add((row['source'], row['target']))

    if len(combined_df) > 0:
        pathway_sources = combined_df['source'].unique()
    else:
        pathway_sources = []

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
                    'source': gene_name,
                    'target': target_gene,
                    'belief': 0.5,
                    'logfoldchange': deg_row['logfoldchanges'],
                    'pval': deg_row['pvals'],
                    'pathway_type': 'none',
                    'priority': 3,
                    'explained': False
                })
        except Exception as e:
            continue

    combined_df['explained'] = True

    if unexplained_data:
        unexplained_df = pd.DataFrame(unexplained_data)
        final_df = pd.concat([combined_df, unexplained_df], ignore_index=True)
    else:
        final_df = combined_df.copy()

    final_df['explained'] = final_df['explained'].fillna(False)

    explained_count = len(final_df[final_df['explained'] == True])
    unexplained_count = len(final_df[final_df['explained'] == False])

    print(f"  Final dataset: {len(final_df):,} total pairs")
    print(f"    Explained: {explained_count:,} ({explained_count / len(final_df) * 100:.1f}%)")
    print(f"    Unexplained: {unexplained_count:,} ({unexplained_count / len(final_df) * 100:.1f}%)")

    return final_df


print("\n" + "=" * 80)
print("ADDING UNEXPLAINED TARGETS")
print("=" * 80)

df_final_all = add_unexplained_targets(df_combined_all, "All Perturbations", include_tp53=True)
df_final_no_tp53 = add_unexplained_targets(df_combined_no_tp53, "Excluding TP53", include_tp53=False)
df_final_tp53_only = add_unexplained_targets(df_combined_tp53_only, "TP53 Only", include_tp53=True)


def print_combined_stats(df, name):
    if len(df) == 0:
        print(f"{name}: No data")
        return

    explained = df[df['explained'] == True]
    unexplained = df[df['explained'] == False]

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


print("\n" + "=" * 80)
print("COMBINED DATASET SUMMARY STATISTICS (BELIEF CUTOFF ≥0.7)")
print("=" * 80)

print_combined_stats(df_final_all, "All Perturbations (Combined)")
print_combined_stats(df_final_tp53_only, "TP53 Only (Combined)")
print_combined_stats(df_final_no_tp53, "Excluding TP53 (Combined)")


def generate_high_fc_unexplained_csv(df, dataset_name, fc_threshold=2.5):
    """Generate CSV of unexplained targets with high fold change"""
    unexplained = df[df['explained'] == False].copy()

    if len(unexplained) == 0:
        print(f"No unexplained targets in {dataset_name}")
        return

    unexplained['abs_logfc'] = np.abs(unexplained['logfoldchange'])
    high_fc_unexplained = unexplained[unexplained['abs_logfc'] > fc_threshold].copy()

    if len(high_fc_unexplained) == 0:
        print(f"No unexplained targets with |LogFC| > {fc_threshold} in {dataset_name}")
        return

    # Sort by absolute fold change (highest first)
    high_fc_unexplained = high_fc_unexplained.sort_values('abs_logfc', ascending=False)

    # Create clean output DataFrame
    output_df = pd.DataFrame({
        'source_gene': high_fc_unexplained['source'],
        'target_gene': high_fc_unexplained['target'],
        'log_fold_change': high_fc_unexplained['logfoldchange'],
        'abs_log_fold_change': high_fc_unexplained['abs_logfc'],
        'p_value': high_fc_unexplained['pval'],
        'linear_fold_change': 2 ** high_fc_unexplained['abs_logfc']
    })

    # Save to CSV
    filename = f"high_fc_unexplained_{dataset_name.lower().replace(' ', '_')}.csv"
    output_df.to_csv(filename, index=False)

    print(f"\nHigh fold change unexplained targets CSV generated: {filename}")
    print(f"Total unexplained targets: {len(unexplained):,}")
    print(f"High FC targets (|LogFC| > {fc_threshold}): {len(output_df):,}")
    print("\nTargets saved to CSV:")
    print(output_df.to_string(index=False))

    return output_df


def create_3d_kde_plot(explained_data, title, save_name):
    """Create 3D KDE surface plot"""
    if len(explained_data) < 10:
        print(f"Insufficient data for 3D KDE plot: {title}")
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare data
    x = explained_data['abs_logfc'].values
    y = explained_data['belief'].values

    # Create 2D KDE
    try:
        kde = gaussian_kde([x, y])

        # Create grid
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        xi = np.linspace(x_min, x_max, 50)
        yi = np.linspace(y_min, y_max, 50)
        X, Y = np.meshgrid(xi, yi)

        # Evaluate KDE on grid
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(positions).reshape(X.shape)

        # Create 3D surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

        ax.set_xlabel('|Log Fold Change|')
        ax.set_ylabel('Belief Score')
        ax.set_zlabel('Density')
        ax.set_title(f'3D KDE Surface: {title}')

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        plt.savefig(f'{save_name}_3d_kde.png', dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(0.1)

    except Exception as e:
        print(f"Error creating 3D KDE plot: {e}")


def create_enhanced_combined_plot(df, title, save_name):
    """Create enhanced visualization with all requested improvements"""
    if len(df) == 0:
        print(f"No data for {title}")
        return

    print(f"\n=== DATA INTEGRITY CHECKS FOR {title.upper()} ===")

    df_plot = df.copy()
    df_plot['abs_logfc'] = np.abs(df_plot['logfoldchange'])
    df_plot['neg_log10_pval'] = -np.log10(df_plot['pval'])

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

    if len(explained) > 0:
        print(f"  Belief score range (explained): {explained['belief'].min():.3f} to {explained['belief'].max():.3f}")

    print(f"\nFold Change Ranges:")
    if len(onehop) > 0:
        print(f"  1-hop: {onehop['abs_logfc'].min():.3f} to {onehop['abs_logfc'].max():.3f}")
    if len(twohop) > 0:
        print(f"  2-hop: {twohop['abs_logfc'].min():.3f} to {twohop['abs_logfc'].max():.3f}")
    if len(unexplained) > 0:
        print(f"  Unexplained: {unexplained['abs_logfc'].min():.3f} to {unexplained['abs_logfc'].max():.3f}")

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

    # FIGURE 1: Main Analysis
    fig1 = plt.figure(figsize=(20, 12))
    fig1.suptitle(f'{title} - Combined 1-Hop + 2-Hop Analysis (Belief ≥0.7)', fontsize=16, fontweight='bold')

    gs1 = fig1.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # Plot 1: Pathway type comparison scatter plot
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

    # Plot 2: Enhanced heat map with better scaling
    ax2 = fig1.add_subplot(gs1[0, 1])
    if len(explained) > 0:
        try:
            # Use fewer bins and log normalization for better visibility
            h2 = ax2.hist2d(explained['abs_logfc'], explained['belief'],
                            bins=25, cmap='plasma', norm=LogNorm(vmin=1))
            plt.colorbar(h2[3], ax=ax2)
        except:
            # Fallback without log normalization
            h2 = ax2.hist2d(explained['abs_logfc'], explained['belief'],
                            bins=25, cmap='plasma')
            plt.colorbar(h2[3], ax=ax2)
        print(f"Enhanced heat map: Using {len(explained):,} explained data points")
    ax2.set_xlabel('|Log Fold Change|')
    ax2.set_ylabel('Belief Score')
    ax2.set_title('LogFC vs Belief Score (Enhanced Heat Map)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Belief score distribution with mean labels and KS test
    ax3 = fig1.add_subplot(gs1[0, 2])
    if len(onehop) > 0:
        onehop_mean = onehop['belief'].mean()
        ax3.hist(onehop['belief'], bins=50, alpha=0.7, color='blue',
                 label=f'1-hop (n={len(onehop):,}, mean={onehop_mean:.3f})', density=True)
    if len(twohop) > 0:
        twohop_mean = twohop['belief'].mean()
        ax3.hist(twohop['belief'], bins=50, alpha=0.7, color='green',
                 label=f'2-hop (n={len(twohop):,}, mean={twohop_mean:.3f})', density=True)

    # Kolmogorov-Smirnov test
    if len(onehop) > 0 and len(twohop) > 0:
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

    # Plot 4: Summary statistics panel
    ax4 = fig1.add_subplot(gs1[0, 3])
    ax4.axis('off')

    summary_text = f"""SUMMARY STATISTICS (Belief ≥0.7)

Total Data Points: {len(df_plot):,}

EXPLAINED ({len(explained):,}):
• 1-hop: {len(onehop):,}
• 2-hop: {len(twohop):,}

UNEXPLAINED: {len(unexplained):,}

FOLD CHANGE STATS:
"""

    if len(explained) > 0:
        summary_text += f"• Explained mean: {explained['abs_logfc'].mean():.3f}\n"
        summary_text += f"• Belief range: {explained['belief'].min():.3f}-{explained['belief'].max():.3f}\n"
    if len(unexplained) > 0:
        summary_text += f"• Unexplained mean: {unexplained['abs_logfc'].mean():.3f}\n"

    summary_text += f"\nHIGH FC (>{high_fc_threshold}):\n"
    if len(onehop) > 0:
        summary_text += f"• 1-hop: {len(onehop[onehop['abs_logfc'] > high_fc_threshold])}\n"
    if len(twohop) > 0:
        summary_text += f"• 2-hop: {len(twohop[twohop['abs_logfc'] > high_fc_threshold])}\n"
    if len(unexplained) > 0:
        summary_text += f"• Unexplained: {len(unexplained[unexplained['abs_logfc'] > high_fc_threshold])}\n"

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    # Plot 5: 1-hop histogram
    ax5 = fig1.add_subplot(gs1[1, 0])
    if len(onehop) > 0:
        ax5.hist(onehop['abs_logfc'], bins=40, density=False, color='blue', alpha=0.7)
        ax5.set_title(f'1-hop (n={len(onehop):,})')
    else:
        ax5.text(0.5, 0.5, 'No 1-hop data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('1-hop (n=0)')
    ax5.set_xlabel('|Log Fold Change|')
    ax5.set_ylabel('Count')
    ax5.grid(True, alpha=0.3)

    # Plot 6: 2-hop histogram
    ax6 = fig1.add_subplot(gs1[1, 1])
    if len(twohop) > 0:
        ax6.hist(twohop['abs_logfc'], bins=100, density=False, color='green', alpha=0.7)
        ax6.set_title(f'2-hop (n={len(twohop):,})')
    else:
        ax6.text(0.5, 0.5, 'No 2-hop data', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('2-hop (n=0)')
    ax6.set_xlabel('|Log Fold Change|')
    ax6.set_ylabel('Count')
    ax6.grid(True, alpha=0.3)

    # Plot 7: Unexplained histogram
    ax7 = fig1.add_subplot(gs1[1, 2])
    if len(unexplained) > 0:
        ax7.hist(unexplained['abs_logfc'], bins=100, density=False, color='red', alpha=0.7)
        ax7.set_title(f'Unexplained (n={len(unexplained):,})')
    else:
        ax7.text(0.5, 0.5, 'No unexplained data', ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Unexplained (n=0)')
    ax7.set_xlabel('|Log Fold Change|')
    ax7.set_ylabel('Count')
    ax7.grid(True, alpha=0.3)

    # Plot 8: Overlaid histograms with KDE smoothing
    ax8 = fig1.add_subplot(gs1[1, 3])

    # Create overlaid histograms with KDE smoothing
    if len(onehop) > 0:
        ax8.hist(onehop['abs_logfc'], bins=50, alpha=0.5, color='blue',
                 label=f'1-hop (n={len(onehop):,})', density=True)

        # Add KDE curve for 1-hop
        if len(onehop) > 5:
            try:
                kde_1hop = gaussian_kde(onehop['abs_logfc'])
                x_range = np.linspace(onehop['abs_logfc'].min(), onehop['abs_logfc'].max(), 100)
                ax8.plot(x_range, kde_1hop(x_range), color='darkblue', linewidth=2)
            except:
                pass

    if len(twohop) > 0:
        ax8.hist(twohop['abs_logfc'], bins=50, alpha=0.5, color='green',
                 label=f'2-hop (n={len(twohop):,})', density=True)

        # Add KDE curve for 2-hop
        if len(twohop) > 5:
            try:
                kde_2hop = gaussian_kde(twohop['abs_logfc'])
                x_range = np.linspace(twohop['abs_logfc'].min(), twohop['abs_logfc'].max(), 100)
                ax8.plot(x_range, kde_2hop(x_range), color='darkgreen', linewidth=2)
            except:
                pass

    if len(unexplained) > 0:
        ax8.hist(unexplained['abs_logfc'], bins=50, alpha=0.5, color='red',
                 label=f'Unexplained (n={len(unexplained):,})', density=True)

        # Add KDE curve for unexplained
        if len(unexplained) > 5:
            try:
                kde_unexp = gaussian_kde(unexplained['abs_logfc'])
                x_range = np.linspace(unexplained['abs_logfc'].min(), unexplained['abs_logfc'].max(), 100)
                ax8.plot(x_range, kde_unexp(x_range), color='darkred', linewidth=2)
            except:
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

    # FIGURE 2: Advanced Visualizations
    fig2 = plt.figure(figsize=(16, 8))
    fig2.suptitle(f'{title} - Advanced Visualizations', fontsize=16, fontweight='bold')

    gs2 = fig2.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

    # Plot A: 3D KDE surface plot (inline version)
    ax_3d = fig2.add_subplot(gs2[0, 0], projection='3d')
    if len(explained) > 10:
        try:
            x = explained['abs_logfc'].values
            y = explained['belief'].values

            kde = gaussian_kde([x, y])

            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()

            xi = np.linspace(x_min, x_max, 30)
            yi = np.linspace(y_min, y_max, 30)
            X, Y = np.meshgrid(xi, yi)

            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = kde(positions).reshape(X.shape)

            surf = ax_3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

            ax_3d.set_xlabel('|Log Fold Change|')
            ax_3d.set_ylabel('Belief Score')
            ax_3d.set_zlabel('Density')
            ax_3d.set_title('3D KDE Surface')

        except Exception as e:
            ax_3d.text(0.5, 0.5, 0.5, f'3D KDE Error: {str(e)[:50]}...',
                       ha='center', va='center', transform=ax_3d.transAxes)
    else:
        ax_3d.text(0.5, 0.5, 0.5, 'Insufficient data for 3D KDE',
                   ha='center', va='center', transform=ax_3d.transAxes)

    # Plot B: P-value significance analysis
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
    ax_pval.legend()
    ax_pval.grid(True, alpha=0.3)

    # Add significance threshold lines
    ax_pval.axvline(x=-np.log10(0.05), color='orange', linestyle='--', alpha=0.7, label='p=0.05')
    ax_pval.axvline(x=-np.log10(0.01), color='red', linestyle='--', alpha=0.7, label='p=0.01')

    plt.tight_layout()
    plt.savefig(f'{save_name}_advanced.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)

    print(f"=== VISUALIZATION COMPLETE FOR {title.upper()} ===\n")

    return df_plot


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

    print(f"Dataset Summary:")
    print(f"  Total data points: {len(df):,}")
    print(f"  Explained pathways: {len(explained):,} ({len(explained) / len(df) * 100:.2f}%)")
    print(f"    1-hop: {len(onehop):,} ({len(onehop) / len(explained) * 100:.1f}% of explained)")
    print(f"    2-hop: {len(twohop):,} ({len(twohop) / len(explained) * 100:.1f}% of explained)")
    print(f"  Unexplained targets: {len(unexplained):,} ({len(unexplained) / len(df) * 100:.2f}%)")

    if len(df) > 0:
        print(f"  Unique sources: {df['source'].nunique():,}")
        print(f"  Unique targets: {df['target'].nunique():,}")

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

        correlation_fc_belief = np.corrcoef(abs_fc_explained, belief_explained)[0, 1]
        correlation_pval_belief = np.corrcoef(pval_explained, belief_explained)[0, 1]
        one_minus_pval = 1 - pval_explained
        correlation_sig_belief = np.corrcoef(one_minus_pval, belief_explained)[0, 1]

        print(f"\nCorrelation Analysis (Explained only):")
        print(f"  |LogFC| vs Belief Score: r = {correlation_fc_belief:.6f}")
        print(f"  P-value vs Belief Score: r = {correlation_pval_belief:.6f}")
        print(f"  (1-P-value) vs Belief Score: r = {correlation_sig_belief:.6f}")

        high_belief = explained[explained['belief'] >= 0.8]
        low_belief = explained[explained['belief'] <= 0.7]

        print(f"\nBelief Score Categories:")
        print(f"  High belief (≥0.8): {len(high_belief):,} ({len(high_belief) / len(explained) * 100:.1f}%)")
        if len(high_belief) > 0:
            print(f"    Mean |FC|: {np.abs(high_belief['logfoldchange']).mean():.4f}")
        print(f"  Low belief (≤0.7): {len(low_belief):,} ({len(low_belief) / len(explained) * 100:.1f}%)")
        if len(low_belief) > 0:
            print(f"    Mean |FC|: {np.abs(low_belief['logfoldchange']).mean():.4f}")

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

    if len(explained) > 0 and len(unexplained) > 0:
        abs_fc_explained = np.abs(explained['logfoldchange'])
        abs_fc_unexplained = np.abs(unexplained['logfoldchange'])

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


# Create enhanced plots
print("\n" + "=" * 80)
print("CREATING ENHANCED VISUALIZATIONS (BELIEF CUTOFF ≥0.7)")
print("=" * 80)

create_enhanced_combined_plot(df_final_all, "All Perturbations", "enhanced_all_perturbations")
create_enhanced_combined_plot(df_final_tp53_only, "TP53 Only", "enhanced_tp53_only")
create_enhanced_combined_plot(df_final_no_tp53, "Excluding TP53", "enhanced_excluding_tp53")

# Generate CSV for high fold change unexplained targets
print("\n" + "=" * 80)
print("GENERATING HIGH FOLD CHANGE UNEXPLAINED TARGETS CSV")
print("=" * 80)

generate_high_fc_unexplained_csv(df_final_no_tp53, "Excluding TP53", fc_threshold=2.5)

# Comprehensive statistical analysis
print("\n" + "=" * 80)
print("COMPREHENSIVE STATISTICAL ANALYSIS (BELIEF CUTOFF ≥0.7)")
print("=" * 80)

comprehensive_combined_analysis(df_final_all, "All Perturbations (Combined - Belief ≥0.7)")
comprehensive_combined_analysis(df_final_tp53_only, "TP53 Only (Combined - Belief ≥0.7)")
comprehensive_combined_analysis(df_final_no_tp53, "Excluding TP53 (Combined - Belief ≥0.7)")

# Create comprehensive summary table
print("\n" + "=" * 80)
print("COMPREHENSIVE SUMMARY TABLE (BELIEF CUTOFF ≥0.7)")
print("=" * 80)

create_combined_summary_table()

print("\n" + "=" * 80)
print("ENHANCED ANALYSIS COMPLETE!")
print("=" * 80)
print("Enhanced combined 1-hop + 2-hop analysis complete with all requested improvements:")
print("- Fixed heat map scaling with log normalization")
print("- Added 3D KDE surface plots")
print("- Enhanced belief distributions with mean labels and KS tests")
print("- Added overlaid histograms with KDE smoothing")
print("- Added -log10(p-value) significance plots")
print("- Generated CSV for high fold change unexplained targets")
print("All plots saved with '_main.png' and '_advanced.png' suffixes.")
print("CSV saved as 'high_fc_unexplained_excluding_tp53.csv'.")