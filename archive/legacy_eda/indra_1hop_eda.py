import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("Loading 1-hop pathway datasets...")
print("=" * 60)

# Load 1-hop data
print("Loading 1-hop all perturbations data...")
df_1hop_all = pd.read_excel("/Users/prashammarfatia/Downloads/indra_1hop_all_perturbations.xlsx")
print(f"1-hop all perturbations: {len(df_1hop_all):,} pathways")

print("Loading 1-hop excluding TP53 data...")
df_1hop_no_tp53 = pd.read_csv("/Users/prashammarfatia/Downloads/indra_1hop_no_v2.csv")
print(f"1-hop excluding TP53: {len(df_1hop_no_tp53):,} pathways")

# Extract TP53-only data
df_1hop_tp53_only = df_1hop_all[df_1hop_all['source'] == 'TP53'].copy()
print(f"1-hop TP53 only: {len(df_1hop_tp53_only):,} pathways")

# Check column structures
print("\nColumn structures:")
print(f"All perturbations columns: {list(df_1hop_all.columns)}")
print(f"Excluding TP53 columns: {list(df_1hop_no_tp53.columns)}")

# Standardize column names if needed
essential_cols = ['source', 'target', 'belief', 'logfoldchange', 'pval']


# Check if we need to rename any columns
def standardize_columns(df, dataset_name):
    """Standardize column names across datasets"""
    if 'belief_1' in df.columns and 'belief' not in df.columns:
        df = df.rename(columns={'belief_1': 'belief'})
        print(f"Renamed 'belief_1' to 'belief' in {dataset_name}")
    if 'pvalue' in df.columns and 'pval' not in df.columns:
        df = df.rename(columns={'pvalue': 'pval'})
        print(f"Renamed 'pvalue' to 'pval' in {dataset_name}")
    return df


df_1hop_all = standardize_columns(df_1hop_all, "all perturbations")
df_1hop_no_tp53 = standardize_columns(df_1hop_no_tp53, "excluding TP53")
df_1hop_tp53_only = standardize_columns(df_1hop_tp53_only, "TP53 only")

print("\nProcessing DEG files to identify explained vs unexplained targets...")
print("=" * 60)

# Load DEG data for all perturbations
deg_folder = "/Users/prashammarfatia/Downloads/de_results_per_gene"

# Verify the folder exists
if not os.path.exists(deg_folder):
    print(f"ERROR: DEG folder not found at {deg_folder}")
    exit()

deg_files = list(Path(deg_folder).glob("*_vs_control.csv"))
print(f"Found {len(deg_files)} DEG files")

# Show first few files for verification
if len(deg_files) > 0:
    print("First few DEG files found:")
    for i, file in enumerate(deg_files[:5]):
        print(f"  {file.name}")
    if len(deg_files) > 5:
        print(f"  ... and {len(deg_files) - 5} more files")
else:
    print("ERROR: No DEG files found matching pattern '*_vs_control.csv'")
    exit()


def process_deg_and_pathways(pathway_df, dataset_name, include_tp53=True):
    """Process DEG files and combine with pathway data"""

    explained_data = []
    unexplained_data = []

    # Get unique sources from pathway data
    pathway_sources = pathway_df['source'].unique()

    for deg_file in deg_files:
        # Extract gene name from filename
        gene_name = deg_file.stem.replace("_vs_control", "")

        # Skip TP53 if not including it
        if not include_tp53 and gene_name == 'TP53':
            continue

        # Skip if gene not in pathway sources
        if gene_name not in pathway_sources:
            continue

        try:
            # Load DEG data
            deg_df = pd.read_csv(deg_file)

            # Filter for significant genes (p < 0.05)
            significant_genes = deg_df[deg_df['pvals'] < 0.05].copy()

            if len(significant_genes) == 0:
                continue

            # Get pathway data for this source
            source_pathways = pathway_df[pathway_df['source'] == gene_name].copy()

            # Identify explained targets (targets that appear in pathways)
            explained_targets = set(source_pathways['target'].unique())

            # Add explained targets to explained_data
            for _, pathway_row in source_pathways.iterrows():
                explained_data.append({
                    'source': gene_name,
                    'target': pathway_row['target'],
                    'belief': pathway_row['belief'],
                    'logfoldchange': pathway_row['logfoldchange'],
                    'pval': pathway_row['pval'],
                    'explained': True,
                    'pathway_type': '1-hop'
                })

            # Identify unexplained targets
            for _, deg_row in significant_genes.iterrows():
                target_gene = deg_row['names']

                # Skip if this target is already explained by a pathway
                if target_gene in explained_targets:
                    continue

                # Add to unexplained data
                unexplained_data.append({
                    'source': gene_name,
                    'target': target_gene,
                    'belief': 0.5,  # Assigned belief for unexplained
                    'logfoldchange': deg_row['logfoldchanges'],
                    'pval': deg_row['pvals'],
                    'explained': False,
                    'pathway_type': 'none'
                })

        except Exception as e:
            print(f"Error processing {gene_name}: {e}")
            continue

    # Combine explained and unexplained data
    combined_data = explained_data + unexplained_data
    combined_df = pd.DataFrame(combined_data)

    print(f"\n{dataset_name} summary:")
    print(f"  Explained pathways: {len(explained_data):,}")
    print(f"  Unexplained targets: {len(unexplained_data):,}")
    print(f"  Total data points: {len(combined_df):,}")

    if len(combined_df) > 0:
        explained_count = combined_df['explained'].sum()
        unexplained_count = len(combined_df) - explained_count
        print(f"  Explained percentage: {explained_count / len(combined_df) * 100:.1f}%")
        print(f"  Unexplained percentage: {unexplained_count / len(combined_df) * 100:.1f}%")

    return combined_df


# Process datasets
print("\nProcessing all perturbations (including TP53)...")
df_all_combined = process_deg_and_pathways(df_1hop_all, "All perturbations", include_tp53=True)

print("\nProcessing TP53 only...")
df_tp53_combined = process_deg_and_pathways(df_1hop_tp53_only, "TP53 only", include_tp53=True)

print("\nProcessing excluding TP53...")
df_no_tp53_combined = process_deg_and_pathways(df_1hop_no_tp53, "Excluding TP53", include_tp53=False)

# Summary statistics
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)


def print_dataset_stats(df, name):
    if len(df) == 0:
        print(f"{name}: No data")
        return

    explained = df[df['explained'] == True]
    unexplained = df[df['explained'] == False]

    print(f"\n{name}:")
    print(f"  Total data points: {len(df):,}")
    print(f"  Explained: {len(explained):,} ({len(explained) / len(df) * 100:.1f}%)")
    print(f"  Unexplained: {len(unexplained):,} ({len(unexplained) / len(df) * 100:.1f}%)")

    if len(explained) > 0:
        print(f"  Explained - Mean |FC|: {np.abs(explained['logfoldchange']).mean():.3f}")
        print(f"  Explained - Mean belief: {explained['belief'].mean():.3f}")

    if len(unexplained) > 0:
        print(f"  Unexplained - Mean |FC|: {np.abs(unexplained['logfoldchange']).mean():.3f}")
        print(f"  Unexplained - Mean belief: {unexplained['belief'].mean():.3f}")


print_dataset_stats(df_all_combined, "All perturbations")
print_dataset_stats(df_tp53_combined, "TP53 only")
print_dataset_stats(df_no_tp53_combined, "Excluding TP53")


# Create updated visualization function with heat maps and separate histograms
def create_heatmap_plot(df, title, save_name):
    """Create comprehensive visualization with heat maps and separate histograms"""

    if len(df) == 0:
        print(f"No data for {title}")
        return

    # Prepare data
    df_plot = df.copy()
    df_plot['abs_logfc'] = np.abs(df_plot['logfoldchange'])

    # Separate explained and unexplained
    explained = df_plot[df_plot['explained'] == True]
    unexplained = df_plot[df_plot['explained'] == False]

    print(f"\n=== DATA INTEGRITY CHECKS FOR {title.upper()} ===")
    print(f"Total data points: {len(df_plot):,}")
    print(f"  Explained: {len(explained):,}")
    print(f"  Unexplained: {len(unexplained):,}")
    print(f"  Sum check: {len(explained) + len(unexplained)} == {len(df_plot)} ✓" if len(explained) + len(unexplained) == len(df_plot) else f"  ❌ Sum mismatch!")

    # Create improved visualization with separate panels - 2 rows, 4 columns like combined implementation
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'{title} - 1-Hop Analysis', fontsize=16, fontweight='bold')

    # Create grid: 2 rows, 4 columns
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # Plot 1: Heat map of explained data only
    ax1 = fig.add_subplot(gs[0, 0])
    if len(explained) > 0:
        h1 = ax1.hist2d(explained['abs_logfc'], explained['belief'], bins=50, cmap='Blues')
        plt.colorbar(h1[3], ax=ax1)
        print(f"Heat map: Using {len(explained):,} explained data points")
    ax1.set_xlabel('|Log Fold Change|')
    ax1.set_ylabel('Belief Score')
    ax1.set_title('Explained Data Only (Heat Map)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Explained vs Unexplained scatter plot (KEEP as requested)
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

    # Plot 3: SEPARATE 1-hop histogram (using COUNTS)
    ax3 = fig.add_subplot(gs[1, 0])
    if len(explained) > 0:
        ax3.hist(explained['abs_logfc'], bins=40, density=False, color='blue', alpha=0.7)
        ax3.set_title(f'1-hop Explained (n={len(explained):,})')
        print(f"1-hop histogram: {len(explained):,} points, range {explained['abs_logfc'].min():.2f}-{explained['abs_logfc'].max():.2f}")
    else:
        ax3.text(0.5, 0.5, 'No 1-hop data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('1-hop Explained (n=0)')
    ax3.set_xlabel('|Log Fold Change|')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3)

    # Plot 4: SEPARATE Unexplained histogram (using COUNTS)
    ax4 = fig.add_subplot(gs[1, 1])
    if len(unexplained) > 0:
        ax4.hist(unexplained['abs_logfc'], bins=100, density=False, color='red', alpha=0.7)
        ax4.set_title(f'Unexplained (n={len(unexplained):,})')
        print(f"Unexplained histogram: {len(unexplained):,} points, range {unexplained['abs_logfc'].min():.2f}-{unexplained['abs_logfc'].max():.2f}")
    else:
        ax4.text(0.5, 0.5, 'No unexplained data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Unexplained (n=0)')
    ax4.set_xlabel('|Log Fold Change|')
    ax4.set_ylabel('Count')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Belief score distribution (explained only)
    ax5 = fig.add_subplot(gs[0, 2:])  # Span 2 columns
    if len(explained) > 0:
        ax5.hist(explained['belief'], bins=30, alpha=0.7, color='blue', density=True)
        ax5.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='Belief = 0.7')
    ax5.set_xlabel('Belief Score')
    ax5.set_ylabel('Density')
    ax5.set_title('Belief Score Distribution (Explained Only)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Summary statistics panel
    ax6 = fig.add_subplot(gs[1, 2:])  # Span 2 columns
    ax6.axis('off')

    # Create summary text
    summary_text = f"""SUMMARY STATISTICS

Total Data Points: {len(df_plot):,}

EXPLAINED: {len(explained):,} ({len(explained)/len(df_plot)*100:.1f}%)
UNEXPLAINED: {len(unexplained):,} ({len(unexplained)/len(df_plot)*100:.1f}%)

FOLD CHANGE STATS:
"""

    if len(explained) > 0:
        summary_text += f"• Explained mean: {explained['abs_logfc'].mean():.3f}\n"
    if len(unexplained) > 0:
        summary_text += f"• Unexplained mean: {unexplained['abs_logfc'].mean():.3f}\n"

    if len(explained) > 0:
        high_fc_explained = len(explained[explained['abs_logfc'] > 1.0])
        summary_text += f"\nHIGH FC (>1.0):\n"
        summary_text += f"• Explained: {high_fc_explained}\n"
    if len(unexplained) > 0:
        high_fc_unexplained = len(unexplained[unexplained['abs_logfc'] > 1.0])
        summary_text += f"• Unexplained: {high_fc_unexplained}\n"

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)

    print(f"=== VISUALIZATION COMPLETE FOR {title.upper()} ===\n")

    return df_plot


# Create plots
print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

print("Creating plots for all perturbations...")
df_all_plot = create_heatmap_plot(df_all_combined, "All Perturbations", "1hop_all_perturbations")

print("Creating plots for TP53 only...")
df_tp53_plot = create_heatmap_plot(df_tp53_combined, "TP53 Only", "1hop_tp53_only")

print("Creating plots for excluding TP53...")
df_no_tp53_plot = create_heatmap_plot(df_no_tp53_combined, "Excluding TP53", "1hop_excluding_tp53")

# Calculate correlations and comprehensive statistics
print("\n" + "=" * 60)
print("COMPREHENSIVE STATISTICAL ANALYSIS")
print("=" * 60)


def comprehensive_analysis(df, name):
    if len(df) == 0:
        print(f"\n{name}: No data for analysis")
        return

    explained = df[df['explained'] == True]
    unexplained = df[df['explained'] == False]

    print(f"\n{name}:")
    print("-" * (len(name) + 1))

    # Basic counts
    print(f"Dataset Summary:")
    print(f"  Total data points: {len(df):,}")
    print(f"  Explained pathways: {len(explained):,} ({len(explained) / len(df) * 100:.2f}%)")
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


# Add scipy import at the top
try:
    from scipy import stats
except ImportError:
    print("Warning: scipy not available for statistical tests")

# Run comprehensive analysis
comprehensive_analysis(df_all_combined, "All Perturbations")
comprehensive_analysis(df_tp53_combined, "TP53 Only")
comprehensive_analysis(df_no_tp53_combined, "Excluding TP53")

# Overall summary table
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)


def create_summary_table():
    datasets = [
        ("All Perturbations", df_all_combined),
        ("TP53 Only", df_tp53_combined),
        ("Excluding TP53", df_no_tp53_combined)
    ]

    print(
        f"{'Dataset':<20} {'Total':<8} {'Explained':<10} {'Unexplained':<12} {'Expl %':<8} {'Mean |FC| (Exp)':<15} {'Mean |FC| (Unexp)':<16} {'Correlation':<12}")
    print("-" * 115)

    for name, df in datasets:
        if len(df) == 0:
            print(f"{name:<20} {'0':<8} {'0':<10} {'0':<12} {'0.0%':<8} {'N/A':<15} {'N/A':<16} {'N/A':<12}")
            continue

        explained = df[df['explained'] == True]
        unexplained = df[df['explained'] == False]

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

        print(
            f"{name:<20} {total:<8} {n_explained:<10} {n_unexplained:<12} {expl_pct:<8} {mean_fc_exp:<15} {mean_fc_unexp:<16} {corr_str:<12}")


create_summary_table()

print("\nDataset preparation and initial plotting complete!")
print("Plots saved as PNG files with heat map visualizations.")