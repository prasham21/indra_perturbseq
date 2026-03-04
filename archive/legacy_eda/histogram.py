"""Mutually exclusive hop-level histogram and distribution analysis."""
from __future__ import annotations

import argparse
import logging
import os
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def load_hop_data(file_path, hop_name):
    """Load hop data with progress indication."""
    logger.info("Loading %s data from: %s", hop_name, file_path)
    df = pd.read_csv(file_path)
    logger.info("  Loaded %s rows", f"{len(df):,}")
    return df


def deduplicate_hop_data(df, hop_name):
    """De-duplicate hop data to one row per unique source-target pair (lowest p-value)."""
    logger.info("De-duplicating %s data", hop_name)
    original_count = len(df)
    unique_pairs_before = len(df[['source', 'target']].drop_duplicates())
    df_unique = df.loc[df.groupby(['source', 'target'])['pval'].idxmin()].reset_index(drop=True)
    logger.info("  Original: %s, Unique pairs: %s, After: %s",
                f"{original_count:,}", f"{unique_pairs_before:,}", f"{len(df_unique):,}")
    return df_unique


def get_validated_sources(validation_file):
    """Get set of validated SOURCE genes (filtered by analysis_flag)."""
    logger.info("Loading target validation data")
    df = pd.read_csv(validation_file)
    df_analysis = df[df['analysis_flag'] == 'Use_for_analysis']
    df_analysis = df_analysis[df_analysis['Gene'] != 'TP53']
    validated_sources = set(df_analysis['Gene'].unique())
    logger.info("  Source genes 'Use_for_analysis' (excluding TP53): %s", f"{len(validated_sources):,}")
    return validated_sources


def identify_zero_hop_pairs(deg_folder, validated_sources, hop1_pairs, hop2_pairs, hop3_pairs):
    """Identify source-target pairs that are not explained by any hop (0-hop)."""
    logger.info("Identifying 0-hop pairs (unexplained by pathways)")

    all_explained_pairs = hop1_pairs | hop2_pairs | hop3_pairs
    logger.info("  Total unique explained pairs: %s", f"{len(all_explained_pairs):,}")

    all_deg_pairs = {}
    processed_sources = 0
    missing_deg_files = []

    for i, source_gene in enumerate(sorted(validated_sources), 1):
        deg_file = os.path.join(deg_folder, f"{source_gene}_vs_control.csv")
        if not os.path.exists(deg_file):
            missing_deg_files.append(source_gene)
            continue
        processed_sources += 1
        if i % 50 == 0 or i == len(validated_sources):
            logger.info("  Processing %d/%d: %s", i, len(validated_sources), source_gene)
        try:
            deg_df = pd.read_csv(deg_file)
            deg_df_sig = deg_df[deg_df['pvals'] < 0.05].copy()
            for _, row in deg_df_sig.iterrows():
                target_gene = row['names']
                pair = (source_gene, target_gene)
                if pair not in all_deg_pairs or row['pvals'] < all_deg_pairs[pair][1]:
                    all_deg_pairs[pair] = (row['logfoldchanges'], row['pvals'])
        except Exception as e:
            logger.warning("  Error processing %s: %s", source_gene, str(e))
            continue

    logger.info("DEG collection complete")
    logger.info("  Sources with DEG files: %d/%d", processed_sources, len(validated_sources))
    if missing_deg_files:
        logger.info("  Missing DEG files: %d", len(missing_deg_files))
    logger.info("  Total actual DEG pairs collected: %s", f"{len(all_deg_pairs):,}")

    zero_hop_data = []
    explained_count = len(all_explained_pairs & set(all_deg_pairs.keys()))

    for pair, (logfc, pval) in all_deg_pairs.items():
        if pair not in all_explained_pairs:
            zero_hop_data.append({
                'source': pair[0], 'target': pair[1],
                'logfoldchange': logfc, 'pval': pval
            })

    logger.info("  Total actual DEG pairs: %s", f"{len(all_deg_pairs):,}")
    logger.info("  Explained by pathways: %s", f"{explained_count:,}")
    logger.info("  Unexplained pairs (0-hop): %s", f"{len(zero_hop_data):,}")
    logger.info("  Coverage: %.1f%%", explained_count / len(all_deg_pairs) * 100)

    zero_hop_df = pd.DataFrame(zero_hop_data)
    all_pairs_list = [{'source': s, 'target': t, 'logfoldchange': lfc, 'pval': pv}
                      for (s, t), (lfc, pv) in all_deg_pairs.items()]
    all_deg_pairs_df = pd.DataFrame(all_pairs_list)

    return zero_hop_df, all_deg_pairs_df


def create_mutually_exclusive_datasets(hop1_df, hop2_df, hop3_df):
    """Create mutually exclusive hop datasets."""
    logger.info("Creating mutually exclusive hop datasets")

    hop1_pairs = set(zip(hop1_df['source'], hop1_df['target']))
    hop2_pairs = set(zip(hop2_df['source'], hop2_df['target']))
    hop3_pairs = set(zip(hop3_df['source'], hop3_df['target']))

    logger.info("  Original unique pairs: 1-hop=%s, 2-hop=%s, 3-hop=%s",
                f"{len(hop1_pairs):,}", f"{len(hop2_pairs):,}", f"{len(hop3_pairs):,}")

    hop1_unique = hop1_pairs
    hop2_unique_only = hop2_pairs - hop1_pairs
    hop3_unique_only = hop3_pairs - hop1_pairs - hop2_pairs

    logger.info("  Mutually exclusive: 1-hop=%s, 2-hop=%s, 3-hop=%s",
                f"{len(hop1_unique):,}", f"{len(hop2_unique_only):,}", f"{len(hop3_unique_only):,}")

    hop1_filtered = hop1_df.copy()
    hop2_filtered = hop2_df[
        hop2_df.apply(lambda row: (row['source'], row['target']) in hop2_unique_only, axis=1)].copy()
    hop3_filtered = hop3_df[
        hop3_df.apply(lambda row: (row['source'], row['target']) in hop3_unique_only, axis=1)].copy()

    return hop1_filtered, hop2_filtered, hop3_filtered, hop1_unique, hop2_unique_only, hop3_unique_only


def create_all_visualizations(hop0_df, hop1_df, hop2_df, hop3_df, all_deg_pairs_df, output_dir=None):
    """Create all visualizations."""
    logger.info("Creating visualizations")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info("Identifying top 100 pairs by absolute fold change")
    all_deg_pairs_df['abs_fc'] = all_deg_pairs_df['logfoldchange'].abs()
    top_100 = all_deg_pairs_df.nlargest(100, 'abs_fc')
    top_100_pairs = set(zip(top_100['source'], top_100['target']))

    hop_data = [
        ('0-hop', hop0_df, 'logfoldchange', 'pval'),
        ('1-hop', hop1_df, 'logfoldchange', 'pval'),
        ('2-hop', hop2_df, 'logfoldchange', 'pval'),
        ('3-hop', hop3_df, 'logfoldchange', 'pval')
    ]

    for name, df, lfc_col, pval_col in hop_data:
        if 'source' in df.columns and 'target' in df.columns:
            df['is_top100'] = df.apply(lambda row: (row['source'], row['target']) in top_100_pairs, axis=1)

    colors = {'0-hop': '#FF6B6B', '1-hop': '#95E1D3', '2-hop': '#4A90E2', '3-hop': '#FFA07A'}

    # P-value histograms (2x2)
    logger.info("Generating Figure 1: P-value Histograms")
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('P-value Distributions Across Hop Levels', fontsize=16, fontweight='bold')
    axes1 = axes1.flatten()

    for idx, (name, df, lfc_col, pval_col) in enumerate(hop_data):
        ax = axes1[idx]
        n_pairs = len(df[['source', 'target']].drop_duplicates()) if 'source' in df.columns else len(df)
        pvals = df[pval_col].dropna()
        ax.hist(pvals, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        if 'is_top100' in df.columns and df['is_top100'].any():
            top_df = df[df['is_top100']]
            counts, _ = np.histogram(pvals, bins=50)
            n_top = len(top_df[['source', 'target']].drop_duplicates())
            y_pos = np.full(len(top_df), counts.max() * 1.05)
            ax.scatter(top_df[pval_col], y_pos, color='red', s=30, alpha=0.8,
                       label=f'Top 100 (n={n_top})', edgecolors='darkred', zorder=5)
            ax.legend(loc='upper right', fontsize=8)
        ax.set_xlabel('P-value', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{name} (n={n_pairs:,})', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pvalue_distributions_2x2.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Log FC histograms (2x2)
    logger.info("Generating Figure 2: Log Fold Change Histograms")
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Log Fold Change Distributions Across Hop Levels', fontsize=16, fontweight='bold')
    axes2 = axes2.flatten()

    for idx, (name, df, lfc_col, pval_col) in enumerate(hop_data):
        ax = axes2[idx]
        n_pairs = len(df[['source', 'target']].drop_duplicates()) if 'source' in df.columns else len(df)
        lfcs = df[lfc_col].dropna()
        ax.hist(lfcs, bins=50, color='coral', alpha=0.7, edgecolor='black')
        if 'is_top100' in df.columns and df['is_top100'].any():
            top_df = df[df['is_top100']]
            counts, _ = np.histogram(lfcs, bins=50)
            n_top = len(top_df[['source', 'target']].drop_duplicates())
            y_pos = np.full(len(top_df), counts.max() * 1.05)
            ax.scatter(top_df[lfc_col], y_pos, color='darkgreen', s=30, alpha=0.8,
                       label=f'Top 100 (n={n_top})', edgecolors='black', zorder=5)
            ax.legend(loc='upper right', fontsize=8)
        ax.set_xlabel('Log Fold Change', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{name} (n={n_pairs:,})', fontsize=10, fontweight='bold')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'logfc_distributions_2x2.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Overlaid histograms (100 bins)
    logger.info("Generating Figure 3: Overlaid Histograms (100 bins, filtered |FC| >= 0.5)")
    fig3, (ax_pval, ax_lfc) = plt.subplots(1, 2, figsize=(16, 6))
    fig3.suptitle('Overlaid Distributions Across Hop Levels (Filtered |FC| >= 0.5)', fontsize=16, fontweight='bold')

    for name, df, lfc_col, pval_col in hop_data:
        df_filtered = df[df[lfc_col].abs() >= 0.5].copy()
        n_pairs = len(df[['source', 'target']].drop_duplicates()) if 'source' in df.columns else len(df)
        n_pairs_filtered = (len(df_filtered[['source', 'target']].drop_duplicates())
                            if 'source' in df_filtered.columns else len(df_filtered))
        pvals = df[pval_col].dropna()
        lfcs_filtered = df_filtered[lfc_col].dropna()
        if len(pvals) > 0:
            ax_pval.hist(pvals, bins=100, alpha=0.5, label=f'{name} (n={n_pairs:,})',
                         color=colors[name], edgecolor='none')
        if len(lfcs_filtered) > 0:
            ax_lfc.hist(lfcs_filtered, bins=100, alpha=0.5, label=f'{name} (n={n_pairs_filtered:,})',
                        color=colors[name], edgecolor='none')

    ax_pval.set_xlabel('P-value', fontsize=12, fontweight='bold')
    ax_pval.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax_pval.set_title('P-value Distributions', fontsize=13, fontweight='bold')
    ax_pval.legend(loc='best', fontsize=10, framealpha=0.9)
    ax_pval.grid(True, alpha=0.3)

    ax_lfc.set_xlabel('Log Fold Change', fontsize=12, fontweight='bold')
    ax_lfc.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax_lfc.set_title('Log Fold Change Distributions', fontsize=13, fontweight='bold')
    ax_lfc.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_lfc.legend(loc='best', fontsize=10, framealpha=0.9)
    ax_lfc.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'overlaid_histograms_100bins.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 2D landscape (FC vs -log10(pval))
    logger.info("Generating Figure 4: 2D Landscape Plots")
    fig4, axes4 = plt.subplots(2, 2, figsize=(20, 18))
    fig4.suptitle('Fold Change vs Significance Landscape by Hop Level', fontsize=16, fontweight='bold', y=0.998)
    axes4 = axes4.flatten()
    plt.subplots_adjust(hspace=0.25, wspace=0.3, top=0.93, bottom=0.05, left=0.08, right=0.95)

    for idx, (name, df, lfc_col, pval_col) in enumerate(hop_data):
        ax = axes4[idx]
        n_pairs = len(df[['source', 'target']].drop_duplicates()) if 'source' in df.columns else len(df)
        plot_df = df[[lfc_col, pval_col]].dropna().copy()
        plot_df['neg_log10_pval'] = -np.log10(plot_df[pval_col].clip(lower=1e-300))

        if len(plot_df) > 1:
            hexbin = ax.hexbin(plot_df[lfc_col], plot_df['neg_log10_pval'],
                               gridsize=50, cmap='YlOrRd', mincnt=1, alpha=0.8)
            cb = plt.colorbar(hexbin, ax=ax)
            cb.set_label('Count', fontsize=10, fontweight='bold')
            if 'is_top100' in df.columns and df['is_top100'].any():
                top_df = df[df['is_top100']][[lfc_col, pval_col]].dropna().copy()
                top_df['neg_log10_pval'] = -np.log10(top_df[pval_col].clip(lower=1e-300))
                n_top = len(df[df['is_top100']][['source', 'target']].drop_duplicates())
                ax.scatter(top_df[lfc_col], top_df['neg_log10_pval'],
                           color='blue', s=50, alpha=0.7, edgecolors='darkblue',
                           linewidths=1.5, label=f'Top 100 (n={n_top})', zorder=5)

        ax.set_xlabel('Log Fold Change', fontsize=11, fontweight='bold')
        ax.set_ylabel('-log10(P-value)', fontsize=11, fontweight='bold')
        ax.set_title(f'{name} (n={n_pairs:,})', fontsize=9, fontweight='bold', pad=12)
        ax.axvline(x=0, color='white', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y=-np.log10(0.05), color='white', linestyle='--', linewidth=1, alpha=0.7, label='p=0.05')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, color='white')

    if output_dir:
        plt.savefig(os.path.join(output_dir, '2d_landscape_fc_vs_pval.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Pairwise p-value comparisons
    logger.info("Generating Figure 5: Pairwise P-value Comparisons")
    hop_combinations = list(combinations(range(4), 2))

    fig5, axes5 = plt.subplots(2, 3, figsize=(18, 10))
    fig5.suptitle('Pairwise P-value Distribution Comparisons', fontsize=16, fontweight='bold')
    axes5 = axes5.flatten()

    for idx, (i, j) in enumerate(hop_combinations):
        ax = axes5[idx]
        name_i, df_i, lfc_i, pval_i = hop_data[i]
        name_j, df_j, lfc_j, pval_j = hop_data[j]
        n_pairs_i = len(df_i[['source', 'target']].drop_duplicates()) if 'source' in df_i.columns else len(df_i)
        n_pairs_j = len(df_j[['source', 'target']].drop_duplicates()) if 'source' in df_j.columns else len(df_j)
        pvals_i = df_i[pval_i].dropna()
        pvals_j = df_j[pval_j].dropna()
        if len(pvals_i) > 0:
            ax.hist(pvals_i, bins=100, alpha=0.6, label=f'{name_i} (n={n_pairs_i:,})',
                    color=colors[name_i], edgecolor='none')
        if len(pvals_j) > 0:
            ax.hist(pvals_j, bins=100, alpha=0.6, label=f'{name_j} (n={n_pairs_j:,})',
                    color=colors[name_j], edgecolor='none')
        ax.set_xlabel('P-value', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{name_i} vs {name_j}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pairwise_pvalue_comparisons.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Pairwise -log10(p-value) comparisons
    logger.info("Generating Figure 6: Pairwise -log10(P-value) Comparisons")
    fig6, axes6 = plt.subplots(2, 3, figsize=(18, 10))
    fig6.suptitle('Pairwise -log10(P-value) Distribution Comparisons', fontsize=16, fontweight='bold')
    axes6 = axes6.flatten()

    all_neglog_pvals = []
    for _, df, _, pval_col in hop_data:
        vals = -np.log10(df[pval_col].dropna().clip(lower=1e-300))
        vals = np.clip(vals, 0, 40)
        all_neglog_pvals.extend(vals.values)
    xmax = np.percentile(all_neglog_pvals, 99.5)
    xlim = (0, max(10, xmax))

    for idx, (i, j) in enumerate(hop_combinations):
        ax = axes6[idx]
        name_i, df_i, lfc_i, pval_i = hop_data[i]
        name_j, df_j, lfc_j, pval_j = hop_data[j]
        n_i = len(df_i[['source', 'target']].drop_duplicates()) if 'source' in df_i.columns else len(df_i)
        n_j = len(df_j[['source', 'target']].drop_duplicates()) if 'source' in df_j.columns else len(df_j)
        neglog_i = np.clip(-np.log10(df_i[pval_i].dropna().clip(lower=1e-300)), 0, 40)
        neglog_j = np.clip(-np.log10(df_j[pval_j].dropna().clip(lower=1e-300)), 0, 40)
        bins = np.linspace(0, xlim[1], 100)
        ax.hist(neglog_i, bins=bins, alpha=0.6, label=f'{name_i} (n={n_i:,})',
                color=colors[name_i], edgecolor='none')
        ax.hist(neglog_j, bins=bins, alpha=0.6, label=f'{name_j} (n={n_j:,})',
                color=colors[name_j], edgecolor='none')
        ax.set_xlim(xlim)
        ax.set_xlabel('-log10(P-value)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{name_i} vs {name_j}', fontsize=12, fontweight='bold')
        ax.axvline(x=-np.log10(0.05), color='black', linestyle='--', linewidth=1, alpha=0.5, label='p=0.05')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pairwise_neglog_pvalue_comparisons_scaled.png'),
                    dpi=300, bbox_inches='tight')
    plt.show()

    # Pairwise log FC comparisons (tail-focused)
    logger.info("Generating Figure 7: Pairwise Log FC Comparisons (tail-focused)")
    fig7, axes7 = plt.subplots(2, 3, figsize=(18, 10))
    fig7.suptitle('Pairwise Log Fold Change Comparisons (Filtered |FC| >= 1.0)', fontsize=16, fontweight='bold')
    axes7 = axes7.flatten()

    xlim_fc = (-3, 3)
    bins_fc = np.linspace(xlim_fc[0], xlim_fc[1], 100)
    global_ymax = 0
    for _, df, lfc_col, _ in hop_data:
        vals = df[lfc_col].dropna()
        vals = vals[np.abs(vals) >= 1.0]
        vals = np.clip(vals, -4, 4)
        counts, _ = np.histogram(vals, bins=bins_fc)
        global_ymax = max(global_ymax, counts.max())

    for idx, (i, j) in enumerate(hop_combinations):
        ax = axes7[idx]
        name_i, df_i, lfc_i, _ = hop_data[i]
        name_j, df_j, lfc_j, _ = hop_data[j]
        df_i_filt = df_i[np.abs(df_i[lfc_i]) >= 1.0]
        df_j_filt = df_j[np.abs(df_j[lfc_j]) >= 1.0]
        n_i = len(df_i_filt[['source', 'target']].drop_duplicates()) if 'source' in df_i_filt.columns else len(df_i_filt)
        n_j = len(df_j_filt[['source', 'target']].drop_duplicates()) if 'source' in df_j_filt.columns else len(df_j_filt)
        lfcs_i = np.clip(df_i_filt[lfc_i].dropna(), -4, 4)
        lfcs_j = np.clip(df_j_filt[lfc_j].dropna(), -4, 4)
        ax.hist(lfcs_i, bins=bins_fc, alpha=0.6, label=f'{name_i} (n={n_i:,})',
                color=colors[name_i], edgecolor='none')
        ax.hist(lfcs_j, bins=bins_fc, alpha=0.6, label=f'{name_j} (n={n_j:,})',
                color=colors[name_j], edgecolor='none')
        ax.set_xlim(xlim_fc)
        ax.set_ylim(0, global_ymax * 1.1)
        ax.set_xlabel('Log Fold Change', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{name_i} vs {name_j}', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.6)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pairwise_logfc_comparisons_tailfocused.png'),
                    dpi=300, bbox_inches='tight')
    plt.show()

    logger.info("All visualizations complete")


def main():
    parser = argparse.ArgumentParser(description="Mutually exclusive hop histogram analysis")
    parser.add_argument("--hop1-file", required=True, help="Path to 1-hop CSV")
    parser.add_argument("--hop2-file", required=True, help="Path to 2-hop CSV")
    parser.add_argument("--hop3-file", required=True, help="Path to 3-hop CSV")
    parser.add_argument("--validation-file", required=True, help="Path to target_validation_expanded.csv")
    parser.add_argument("--deg-folder", required=True, help="Path to de_results_per_gene folder")
    parser.add_argument("--output-dir", required=True, help="Output directory for figures")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("HOP ANALYSIS - MUTUALLY EXCLUSIVE")

    hop1_df = load_hop_data(args.hop1_file, "1-hop")
    hop2_df = load_hop_data(args.hop2_file, "2-hop")
    hop3_df = load_hop_data(args.hop3_file, "3-hop")

    hop1_df = deduplicate_hop_data(hop1_df, "1-hop")
    hop2_df = deduplicate_hop_data(hop2_df, "2-hop")

    validated_sources = get_validated_sources(args.validation_file)

    hop1_filt, hop2_filt, hop3_filt, hop1_p, hop2_p, hop3_p = \
        create_mutually_exclusive_datasets(hop1_df, hop2_df, hop3_df)

    hop0_df, all_deg_df = identify_zero_hop_pairs(
        args.deg_folder, validated_sources, hop1_p, hop2_p, hop3_p)

    create_all_visualizations(hop0_df, hop1_filt, hop2_filt, hop3_filt, all_deg_df, args.output_dir)

    hop0_n = len(hop0_df)
    hop1_n = len(hop1_filt[['source', 'target']].drop_duplicates())
    hop2_n = len(hop2_filt[['source', 'target']].drop_duplicates())
    hop3_n = len(hop3_filt[['source', 'target']].drop_duplicates())
    total = hop0_n + hop1_n + hop2_n + hop3_n

    logger.info("Summary: 0-hop=%s (%.1f%%), 1-hop=%s (%.1f%%), 2-hop=%s (%.1f%%), 3-hop=%s (%.1f%%), Total=%s",
                f"{hop0_n:,}", hop0_n / total * 100,
                f"{hop1_n:,}", hop1_n / total * 100,
                f"{hop2_n:,}", hop2_n / total * 100,
                f"{hop3_n:,}", hop3_n / total * 100,
                f"{total:,}")
    logger.info("Figures saved to: %s", args.output_dir)
    logger.info("Done")


if __name__ == "__main__":
    main()
