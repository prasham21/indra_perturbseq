"""Scatter plots for 0-hop data split into 4-hop and unexplored pair groups."""
from __future__ import annotations

import argparse
import logging
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

logger = logging.getLogger(__name__)


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

    x_rounded = np.round(x / threshold) * threshold
    y_rounded = np.round(y / threshold) * threshold

    coords = list(zip(x_rounded, y_rounded))
    coord_counts = Counter(coords)

    for i, coord in enumerate(coords):
        if coord_counts[coord] > 1:
            x_jittered[i] += np.random.uniform(-jitter_amount, jitter_amount)
            y_jittered[i] += np.random.uniform(-jitter_amount, jitter_amount)

    return x_jittered, y_jittered


def split_zero_hop_into_groups(hop0_df, n_unexplored=292, fc_threshold=1.2):
    """
    Split 0-hop data into 4-hop and Unexplored Pairs groups.

    Unexplored: ALL n_unexplored pairs with |FC| < fc_threshold.
    4-hop: Everything else.

    Parameters:
        hop0_df: DataFrame with 0-hop data
        n_unexplored: Number of pairs for Unexplored group (default 292)
        fc_threshold: Fold change threshold (default 1.2)

    Returns:
        hop4_df, unexplored_df: Two DataFrames
    """
    logger.info("Splitting 0-hop into 4-hop and unexplored pairs")

    hop0_df = hop0_df.copy()
    hop0_df = hop0_df.reset_index(drop=True)
    hop0_df['abs_fc'] = hop0_df['logfoldchange'].abs()

    logger.info("Total 0-hop pairs: %s", f"{len(hop0_df):,}")
    logger.info("Target Unexplored size: %d (all with |FC| < %s)", n_unexplored, fc_threshold)

    low_fc = hop0_df[hop0_df['abs_fc'] < fc_threshold].copy()
    logger.info("Available pairs with |FC| < %s: %s", fc_threshold, f"{len(low_fc):,}")

    if len(low_fc) >= n_unexplored:
        unexplored_df = low_fc.sample(n=n_unexplored, random_state=42)
        logger.info("Randomly sampled %d pairs for Unexplored", n_unexplored)
    else:
        unexplored_df = low_fc
        logger.warning("Only %d pairs available, taking all", len(low_fc))

    unexplored_indices = set(unexplored_df.index.tolist())
    hop4_df = hop0_df[~hop0_df.index.isin(unexplored_indices)].reset_index(drop=True)

    logger.info("Segregation complete")
    logger.info("  4-hop group: %s pairs", f"{len(hop4_df):,}")
    logger.info("    With |FC| < %s: %d", fc_threshold, (hop4_df['abs_fc'] < fc_threshold).sum())
    logger.info("    With |FC| >= %s: %d", fc_threshold, (hop4_df['abs_fc'] >= fc_threshold).sum())
    logger.info("  Unexplored Pairs group: %s pairs", f"{len(unexplored_df):,}")

    assert len(hop4_df) + len(unexplored_df) == len(hop0_df), "Split doesn't add up!"

    return hop4_df, unexplored_df


def create_4hop_unexplored_visualizations(hop4_df, unexplored_df, output_dir):
    """
    Create visualizations for 4-hop vs Unexplored Pairs.
    No fold change cutoff -- show all points.

    Parameters:
        hop4_df: DataFrame with 4-hop data
        unexplored_df: DataFrame with Unexplored Pairs data
        output_dir: Where to save figures
    """
    logger.info("Creating 4-hop vs unexplored visualizations (all points)")

    os.makedirs(output_dir, exist_ok=True)

    hop_data = [
        ('4-hop', hop4_df, 'logfoldchange', 'pval'),
        ('Unexplored Pairs', unexplored_df, 'logfoldchange', 'pval')
    ]

    all_fc = []
    all_neglogp = []

    for name, df, lfc_col, pval_col in hop_data:
        df['neg_log10_pval'] = -np.log10(df[pval_col].clip(lower=1e-300))
        all_fc.extend(df[lfc_col].values)
        all_neglogp.extend(df['neg_log10_pval'].values)

    fc_min, fc_max = np.percentile(all_fc, [0.5, 99.5]) if len(all_fc) > 0 else (-5, 5)
    neglogp_max = np.percentile(all_neglogp, 99) if len(all_neglogp) > 0 else 10

    cell_count_bins = [100, 200, 300, 400, 500, 600]
    custom_colors = ['#FFFF00', '#000080', '#FF0000', '#00FF00', '#FFA500']
    cmap = ListedColormap(custom_colors)
    norm = BoundaryNorm(cell_count_bins, cmap.N)

    logger.info("Global axis limits: FC=[%.2f, %.2f], -log10(p)=[0, %.1f]", fc_min, fc_max, neglogp_max)

    # Individual plots (1x2 grid with cell count coloring)
    logger.info("Generating Figure 1: Individual Plots (1x2 grid)")

    fig1, axes1 = plt.subplots(1, 2, figsize=(20, 8))
    fig1.suptitle('4-hop vs Unexplored Pairs (All points, colored by cell count)',
                  fontsize=16, fontweight='bold', y=0.96)
    plt.subplots_adjust(hspace=0.3, wspace=0.35, top=0.90, bottom=0.08, left=0.06, right=0.96)

    for idx, (name, df, lfc_col, pval_col) in enumerate(hop_data):
        ax = axes1[idx]

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
            cbar = plt.colorbar(scatter, ax=ax, boundaries=cell_count_bins, ticks=[150, 250, 350, 450, 550])
            cbar.set_label('Cell Count', fontsize=11, fontweight='bold')
            cbar.ax.set_yticklabels(['100-200', '200-300', '300-400', '400-500', '500-600'])

        ax.set_xlim(fc_min, fc_max)
        ax.set_ylim(0, neglogp_max)
        ax.set_xlabel('Log Fold Change', fontsize=12, fontweight='bold')
        ax.set_ylabel('-log10(P-value)', fontsize=12, fontweight='bold')
        ax.set_title(f'{name} (n={n_pairs:,} pairs)', fontsize=11, fontweight='bold', pad=15)

        ax.axvline(x=0, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=-np.log10(0.05), color='white', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(x=1.2, color='red', linestyle=':', linewidth=1, alpha=0.5, label='|FC|=1.2')
        ax.axvline(x=-1.2, color='red', linestyle=':', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3, color='gray')
        ax.legend(loc='upper right', fontsize=9)

    fig1_path = os.path.join(output_dir, '4hop_unexplored_individual_cell_count_all_points.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    logger.info("Saved: %s", fig1_path)
    plt.show()

    # Overlaid plot with smart jitter
    logger.info("Generating Figure 2: Overlaid Plot with smart jitter")

    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10))
    fig2.suptitle('4-hop vs Unexplored Pairs Overlaid (All points)',
                  fontsize=16, fontweight='bold')

    colors_group = {
        '4-hop': '#1E40AF',
        'Unexplored Pairs': '#DC2626'
    }

    for name, df, lfc_col, pval_col in hop_data:
        df_plot = df.copy()
        df_plot['neg_log10_pval'] = -np.log10(df_plot[pval_col].clip(lower=1e-300))

        n_pairs = len(df_plot[['source', 'target']].drop_duplicates()) if 'source' in df_plot.columns else len(df_plot)

        if len(df_plot) > 0:
            x_vals = df_plot[lfc_col].values
            y_vals = df_plot['neg_log10_pval'].values
            x_jittered, y_jittered = apply_jitter_to_overlaps(
                x_vals, y_vals, threshold=0.02, jitter_amount=0.02
            )
            ax2.scatter(
                x_jittered, y_jittered,
                color=colors_group[name], s=40, alpha=0.6,
                label=f'{name} (n={n_pairs:,})', edgecolors='none'
            )

    ax2.set_xlim(fc_min, fc_max)
    ax2.set_ylim(0, neglogp_max)
    ax2.set_xlabel('Log Fold Change', fontsize=13, fontweight='bold')
    ax2.set_ylabel('-log10(P-value)', fontsize=13, fontweight='bold')
    ax2.set_title('Fold Change vs Significance (All points)', fontsize=14, fontweight='bold')

    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=-np.log10(0.05), color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='p=0.05')
    ax2.axvline(x=1.2, color='red', linestyle=':', linewidth=1, alpha=0.5, label='|FC|=1.2')
    ax2.axvline(x=-1.2, color='red', linestyle=':', linewidth=1, alpha=0.5)

    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, '4hop_unexplored_overlaid_all_points.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    logger.info("Saved: %s", fig2_path)
    plt.show()

    logger.info("All 4-hop vs Unexplored visualizations complete")


def main():
    parser = argparse.ArgumentParser(description="4-hop vs Unexplored Pairs analysis")
    parser.add_argument("--hop0-file", required=True, help="Path to hop0_with_cell_counts.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory for figures and CSVs")
    parser.add_argument("--n-unexplored", type=int, default=292, help="Number of unexplored pairs")
    parser.add_argument("--fc-threshold", type=float, default=1.2, help="Fold change threshold")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("4-HOP VS UNEXPLORED PAIRS ANALYSIS")

    if not os.path.exists(args.hop0_file):
        logger.error("File not found: %s", args.hop0_file)
        return

    hop0_df = pd.read_csv(args.hop0_file)
    logger.info("Loaded %s 0-hop pairs", f"{len(hop0_df):,}")
    logger.info("Columns: %s", hop0_df.columns.tolist())

    hop4_df, unexplored_df = split_zero_hop_into_groups(
        hop0_df, n_unexplored=args.n_unexplored, fc_threshold=args.fc_threshold
    )

    create_4hop_unexplored_visualizations(hop4_df, unexplored_df, args.output_dir)

    hop4_file = os.path.join(args.output_dir, '4hop_data.csv')
    unexplored_file = os.path.join(args.output_dir, 'unexplored_pairs_data.csv')
    hop4_df.to_csv(hop4_file, index=False)
    unexplored_df.to_csv(unexplored_file, index=False)

    logger.info("Saved 4-hop data: %s", hop4_file)
    logger.info("Saved Unexplored data: %s", unexplored_file)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Done")


if __name__ == "__main__":
    main()
