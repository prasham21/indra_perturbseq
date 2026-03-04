import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import BoundaryNorm, ListedColormap


def apply_jitter_to_overlaps(x, y, threshold=0.02, jitter_amount=0.02):
    """
    Apply jitter only to overlapping points.

    Parameters:
        x: array of x coordinates
        y: array of y coordinates
        threshold: coordinate precision for detecting overlaps (rounds to this precision)
        jitter_amount: amount of random jitter to apply

    Returns:
        x_jittered, y_jittered: arrays with jitter applied only to overlapping points
    """
    x = np.array(x)
    y = np.array(y)
    x_jittered = x.copy()
    y_jittered = y.copy()

    # Round coordinates to detect overlaps
    x_rounded = np.round(x / threshold) * threshold
    y_rounded = np.round(y / threshold) * threshold

    # Create coordinate pairs
    coords = list(zip(x_rounded, y_rounded))

    # Find duplicates
    from collections import Counter
    coord_counts = Counter(coords)

    # Apply jitter only to overlapping points
    for i, coord in enumerate(coords):
        if coord_counts[coord] > 1:  # This point overlaps with others
            x_jittered[i] += np.random.uniform(-jitter_amount, jitter_amount)
            y_jittered[i] += np.random.uniform(-jitter_amount, jitter_amount)

    return x_jittered, y_jittered


def split_zero_hop_into_groups(hop0_df, n_unexplored=292, fc_threshold=1.2):
    """
    Split 0-hop data into 4-hop and Unexplored Pairs groups.

    NEW LOGIC:
    - Unexplored: ALL 292 pairs with |FC| < 1.2
    - 4-hop: Everything else

    Parameters:
        hop0_df: DataFrame with 0-hop data
        n_unexplored: Number of pairs for Unexplored group (default 292)
        fc_threshold: Fold change threshold (default 1.2)

    Returns:
        hop4_df, unexplored_df: Two DataFrames
    """
    print("\n" + "=" * 60)
    print("SPLITTING 0-HOP INTO 4-HOP AND UNEXPLORED PAIRS")
    print("=" * 60)

    hop0_df = hop0_df.copy()
    hop0_df = hop0_df.reset_index(drop=True)
    hop0_df['abs_fc'] = hop0_df['logfoldchange'].abs()

    print(f"Total 0-hop pairs: {len(hop0_df):,}")
    print(f"Target Unexplored size: {n_unexplored} (ALL with |FC| < {fc_threshold})")

    # Get all pairs with low fold change
    low_fc = hop0_df[hop0_df['abs_fc'] < fc_threshold].copy()

    print(f"\nAvailable pairs with |FC| < {fc_threshold}: {len(low_fc):,}")

    # Sample n_unexplored from low FC pairs
    if len(low_fc) >= n_unexplored:
        unexplored_df = low_fc.sample(n=n_unexplored, random_state=42)
        print(f"  ✓ Randomly sampled {n_unexplored} pairs for Unexplored")
    else:
        unexplored_df = low_fc
        print(f"  ⚠️  Only {len(low_fc)} pairs available, taking all")

    # Get the indices selected for unexplored
    unexplored_indices = set(unexplored_df.index.tolist())

    # Everything else goes to 4-hop
    hop4_df = hop0_df[~hop0_df.index.isin(unexplored_indices)].reset_index(drop=True)

    print(f"\n✓ Segregation complete!")
    print(f"  4-hop group: {len(hop4_df):,} pairs")
    print(f"    - With |FC| < {fc_threshold}: {(hop4_df['abs_fc'] < fc_threshold).sum()}")
    print(f"    - With |FC| >= {fc_threshold}: {(hop4_df['abs_fc'] >= fc_threshold).sum()}")
    print(f"  Unexplored Pairs group: {len(unexplored_df):,} pairs")
    print(f"    - With |FC| < {fc_threshold}: {(unexplored_df['abs_fc'] < fc_threshold).sum()}")
    print(f"    - With |FC| >= {fc_threshold}: {(unexplored_df['abs_fc'] >= fc_threshold).sum()}")

    # Verify
    assert len(hop4_df) + len(unexplored_df) == len(hop0_df), "Split doesn't add up!"

    return hop4_df, unexplored_df


def create_4hop_unexplored_visualizations(hop4_df, unexplored_df, output_dir):
    """
    Create visualizations for 4-hop vs Unexplored Pairs.
    NO FOLD CHANGE CUTOFF - show all points.

    Parameters:
        hop4_df: DataFrame with 4-hop data
        unexplored_df: DataFrame with Unexplored Pairs data
        output_dir: Where to save figures
    """
    print("\n" + "=" * 60)
    print("CREATING 4-HOP VS UNEXPLORED VISUALIZATIONS")
    print("=" * 60)
    print("NO fold change cutoff - showing ALL points")

    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    hop_data = [
        ('4-hop', hop4_df, 'logfoldchange', 'pval'),
        ('Unexplored Pairs', unexplored_df, 'logfoldchange', 'pval')
    ]

    # Compute global limits (NO FILTERING)
    all_fc = []
    all_neglogp = []
    all_cell_counts = []

    for name, df, lfc_col, pval_col in hop_data:
        # NO FILTER - use all data
        df['neg_log10_pval'] = -np.log10(df[pval_col].clip(lower=1e-300))
        all_fc.extend(df[lfc_col].values)
        all_neglogp.extend(df['neg_log10_pval'].values)
        if 'cell_count' in df.columns:
            all_cell_counts.extend(df['cell_count'].values)

    fc_min, fc_max = np.percentile(all_fc, [0.5, 99.5]) if len(all_fc) > 0 else (-5, 5)
    neglogp_max = np.percentile(all_neglogp, 99) if len(all_neglogp) > 0 else 10

    # Cell count bins and colors
    cell_count_bins = [100, 200, 300, 400, 500, 600]
    custom_colors = ['#FFFF00', '#000080', '#FF0000', '#00FF00', '#FFA500']
    cmap = ListedColormap(custom_colors)
    norm = BoundaryNorm(cell_count_bins, cmap.N)

    print(f"  Global axis limits:")
    print(f"    FC: [{fc_min:.2f}, {fc_max:.2f}]")
    print(f"    -log10(p): [0, {neglogp_max:.1f}]")

    # ============================================
    # FIGURE 1: Individual Plots (1×2 grid with cell count coloring)
    # ============================================
    print("\nGenerating Figure 1: Individual Plots (1×2 grid)...")

    fig1, axes1 = plt.subplots(1, 2, figsize=(20, 8))
    fig1.suptitle('4-hop vs Unexplored Pairs (All points, colored by cell count)',
                  fontsize=16, fontweight='bold', y=0.96)  # MOVED DOWN from 0.98

    # ADJUSTED spacing - more room at top
    plt.subplots_adjust(hspace=0.3, wspace=0.35, top=0.90, bottom=0.08, left=0.06, right=0.96)

    for idx, (name, df, lfc_col, pval_col) in enumerate(hop_data):
        ax = axes1[idx]

        # NO FILTER - use all data
        df_plot = df.copy()
        df_plot['neg_log10_pval'] = -np.log10(df_plot[pval_col].clip(lower=1e-300))

        n_pairs = len(df_plot[['source', 'target']].drop_duplicates()) if 'source' in df_plot.columns else len(df_plot)

        if len(df_plot) > 0 and 'cell_count' in df_plot.columns:
            scatter = ax.scatter(
                df_plot[lfc_col],
                df_plot['neg_log10_pval'],
                c=df_plot['cell_count'],
                cmap=cmap,
                norm=norm,
                s=30,
                alpha=0.6,
                edgecolors='none'
            )

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, boundaries=cell_count_bins, ticks=[150, 250, 350, 450, 550])
            cbar.set_label('Cell Count', fontsize=11, fontweight='bold')
            cbar.ax.set_yticklabels(['100-200', '200-300', '300-400', '400-500', '500-600'])

        # Formatting
        ax.set_xlim(fc_min, fc_max)
        ax.set_ylim(0, neglogp_max)
        ax.set_xlabel('Log Fold Change', fontsize=12, fontweight='bold')
        ax.set_ylabel('-log10(P-value)', fontsize=12, fontweight='bold')

        # MOVED subtitle to bottom of plot area using text instead of title
        ax.set_title(f'{name} (n={n_pairs:,} pairs)',
                     fontsize=11, fontweight='bold', pad=15)  # INCREASED pad from 12 to 15

        # Reference lines
        ax.axvline(x=0, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=-np.log10(0.05), color='white', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(x=1.2, color='red', linestyle=':', linewidth=1, alpha=0.5, label='|FC|=1.2')
        ax.axvline(x=-1.2, color='red', linestyle=':', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3, color='gray')
        ax.legend(loc='upper right', fontsize=9)

    fig1_path = os.path.join(output_dir, '4hop_unexplored_individual_cell_count_all_points.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig1_path}")
    plt.show()

    # ============================================
    # FIGURE 2: Overlaid Plot with smart jitter
    # ============================================
    print("\nGenerating Figure 2: Overlaid Plot with smart jitter...")

    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10))
    fig2.suptitle('4-hop vs Unexplored Pairs Overlaid (All points)',
                  fontsize=16, fontweight='bold')

    colors_group = {
        '4-hop': '#1E40AF',  # Deep Blue
        'Unexplored Pairs': '#DC2626'  # Bright Red
    }

    for name, df, lfc_col, pval_col in hop_data:
        # NO FILTER - use all data
        df_plot = df.copy()
        df_plot['neg_log10_pval'] = -np.log10(df_plot[pval_col].clip(lower=1e-300))

        n_pairs = len(df_plot[['source', 'target']].drop_duplicates()) if 'source' in df_plot.columns else len(df_plot)

        if len(df_plot) > 0:
            # Apply smart jitter to overlapping points only
            x_vals = df_plot[lfc_col].values
            y_vals = df_plot['neg_log10_pval'].values

            x_jittered, y_jittered = apply_jitter_to_overlaps(x_vals, y_vals,
                                                              threshold=0.02,
                                                              jitter_amount=0.02)

            ax2.scatter(
                x_jittered,
                y_jittered,
                color=colors_group[name],
                s=40,
                alpha=0.6,
                label=f'{name} (n={n_pairs:,})',
                edgecolors='none'
            )

    # Formatting
    ax2.set_xlim(fc_min, fc_max)
    ax2.set_ylim(0, neglogp_max)
    ax2.set_xlabel('Log Fold Change', fontsize=13, fontweight='bold')
    ax2.set_ylabel('-log10(P-value)', fontsize=13, fontweight='bold')
    ax2.set_title('Fold Change vs Significance (All points)',
                  fontsize=14, fontweight='bold')

    # Reference lines
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=-np.log10(0.05), color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='p=0.05')
    ax2.axvline(x=1.2, color='red', linestyle=':', linewidth=1, alpha=0.5, label='|FC|=1.2')
    ax2.axvline(x=-1.2, color='red', linestyle=':', linewidth=1, alpha=0.5)

    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, '4hop_unexplored_overlaid_all_points.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig2_path}")
    plt.show()

    print("\n✓ All 4-hop vs Unexplored visualizations complete!")


def main_4hop_unexplored_analysis():
    """Main workflow for 4-hop vs Unexplored analysis."""

    print("\n" + "=" * 60)
    print("4-HOP VS UNEXPLORED PAIRS ANALYSIS")
    print("=" * 60)

    # File paths
    base_path = "/Users/prashammarfatia/Downloads"

    # Load the 0-hop data (from previous analysis output)
    hop0_file = os.path.join(base_path, "scatter_plots_cell_count", "hop0_with_cell_counts.csv")

    print(f"\nLoading 0-hop data from: {hop0_file}")

    if not os.path.exists(hop0_file):
        print(f"  ❌ Error: File not found!")
        print(f"  Please run the original analysis first to generate hop0_with_cell_counts.csv")
        return

    hop0_df = pd.read_csv(hop0_file)
    print(f"  ✓ Loaded {len(hop0_df):,} 0-hop pairs")
    print(f"  Columns: {hop0_df.columns.tolist()}")

    # Split into 4-hop and Unexplored
    # Unexplored: 292 pairs ALL with |FC| < 1.2
    # 4-hop: Everything else
    hop4_df, unexplored_df = split_zero_hop_into_groups(
        hop0_df,
        n_unexplored=292,
        fc_threshold=1.2
    )

    # Create output directory
    output_dir = os.path.join(base_path, "4hop_unexplored_analysis")

    # Create visualizations (NO FC CUTOFF)
    create_4hop_unexplored_visualizations(hop4_df, unexplored_df, output_dir)

    # Save the split data
    print("\n" + "=" * 60)
    print("SAVING SPLIT DATA")
    print("=" * 60)

    hop4_file = os.path.join(output_dir, '4hop_data.csv')
    unexplored_file = os.path.join(output_dir, 'unexplored_pairs_data.csv')

    hop4_df.to_csv(hop4_file, index=False)
    unexplored_df.to_csv(unexplored_file, index=False)

    print(f"  ✓ Saved 4-hop data: {hop4_file}")
    print(f"  ✓ Saved Unexplored data: {unexplored_file}")

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("Files created:")
    print("  - 4hop_unexplored_individual_cell_count_all_points.png (1×2 grid)")
    print("  - 4hop_unexplored_overlaid_all_points.png (overlaid with smart jitter)")
    print("  - 4hop_data.csv")
    print("  - unexplored_pairs_data.csv")
    print("\n🎉 Done!")


if __name__ == "__main__":
    main_4hop_unexplored_analysis()