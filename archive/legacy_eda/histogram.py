import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import glob


def load_hop_data(file_path, hop_name):
    """Load hop data with progress indication."""
    print(f"Loading {hop_name} data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"  ✓ Loaded {len(df):,} rows")
    return df


def deduplicate_hop_data(df, hop_name):
    """
    De-duplicate hop data to one row per unique source-target pair.
    Keeps the row with the lowest p-value for each pair.
    """
    print(f"\nDe-duplicating {hop_name} data...")
    original_count = len(df)
    unique_pairs_before = len(df[['source', 'target']].drop_duplicates())

    # Keep the row with lowest p-value for each unique source-target pair
    df_unique = df.loc[df.groupby(['source', 'target'])['pval'].idxmin()].reset_index(drop=True)

    print(f"  Original rows: {original_count:,}")
    print(f"  Unique pairs: {unique_pairs_before:,}")
    print(f"  After de-duplication: {len(df_unique):,} rows")

    return df_unique


def get_validated_sources(validation_file):
    """Get set of validated SOURCE genes (filtered by Karen_Flag)."""
    print(f"Loading target validation data...")
    df = pd.read_csv(validation_file)

    # Filter to only genes flagged for analysis
    df_analysis = df[df['Karen_Flag'] == 'Use_for_analysis']

    # Exclude TP53
    df_analysis = df_analysis[df_analysis['Gene'] != 'TP53']

    validated_sources = set(df_analysis['Gene'].unique())

    print(f"  ✓ Total genes in file: {len(df):,}")
    print(f"  ✓ Source genes marked 'Use_for_analysis' (excluding TP53): {len(validated_sources):,}")

    return validated_sources


def identify_zero_hop_pairs(deg_folder, validated_sources, hop1_pairs, hop2_pairs, hop3_pairs):
    """
    Identify source-target pairs that are not explained by any hop (0-hop).
    """
    print("\n" + "=" * 60)
    print("IDENTIFYING 0-HOP PAIRS (Unexplained by pathways)")
    print("=" * 60)

    # Combine all explained pairs
    all_explained_pairs = hop1_pairs | hop2_pairs | hop3_pairs
    print(f"  ✓ Total unique explained pairs: {len(all_explained_pairs):,}")

    print(f"\nCollecting DEG pairs from {len(validated_sources)} validated sources...")
    print(f"  Filtering: pval < 0.05")

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
            print(f"  Processing {i}/{len(validated_sources)}: {source_gene}")

        try:
            deg_df = pd.read_csv(deg_file)
            deg_df_sig = deg_df[deg_df['pvals'] < 0.05].copy()

            for _, row in deg_df_sig.iterrows():
                target_gene = row['names']
                pair = (source_gene, target_gene)
                if pair not in all_deg_pairs or row['pvals'] < all_deg_pairs[pair][1]:
                    all_deg_pairs[pair] = (row['logfoldchanges'], row['pvals'])

        except Exception as e:
            print(f"  ⚠️  Error processing {source_gene}: {str(e)}")
            continue

    print(f"\n✓ DEG collection complete!")
    print(f"  Sources with DEG files: {processed_sources}/{len(validated_sources)}")
    if missing_deg_files:
        print(f"  Missing DEG files: {len(missing_deg_files)}")
        if len(missing_deg_files) <= 10:
            print(f"    {', '.join(missing_deg_files)}")
    print(f"  Total actual DEG pairs collected: {len(all_deg_pairs):,}")

    zero_hop_data = []
    explained_count = len(all_explained_pairs & set(all_deg_pairs.keys()))

    for pair, (logfc, pval) in all_deg_pairs.items():
        if pair not in all_explained_pairs:
            zero_hop_data.append({
                'source': pair[0],
                'target': pair[1],
                'logfoldchange': logfc,
                'pval': pval
            })

    print(f"\n✓ Analysis complete!")
    print(f"  Total actual DEG pairs: {len(all_deg_pairs):,}")
    print(f"  Explained by pathways: {explained_count:,}")
    print(f"  Unexplained pairs (0-hop): {len(zero_hop_data):,}")
    print(f"  Coverage: {explained_count / len(all_deg_pairs) * 100:.1f}%")

    zero_hop_df = pd.DataFrame(zero_hop_data)
    all_pairs_list = [{'source': s, 'target': t, 'logfoldchange': lfc, 'pval': pv}
                      for (s, t), (lfc, pv) in all_deg_pairs.items()]
    all_deg_pairs_df = pd.DataFrame(all_pairs_list)

    return zero_hop_df, all_deg_pairs_df


def create_mutually_exclusive_datasets(hop1_df, hop2_df, hop3_df):
    """Create mutually exclusive hop datasets."""
    print("\n" + "=" * 60)
    print("CREATING MUTUALLY EXCLUSIVE HOP DATASETS")
    print("=" * 60)

    hop1_pairs = set(zip(hop1_df['source'], hop1_df['target']))
    hop2_pairs = set(zip(hop2_df['source'], hop2_df['target']))
    hop3_pairs = set(zip(hop3_df['source'], hop3_df['target']))

    print(f"\nOriginal unique pairs:")
    print(f"  1-hop: {len(hop1_pairs):,}")
    print(f"  2-hop: {len(hop2_pairs):,}")
    print(f"  3-hop: {len(hop3_pairs):,}")

    overlap_1_2 = len(hop1_pairs & hop2_pairs)
    overlap_1_3 = len(hop1_pairs & hop3_pairs)
    overlap_2_3 = len(hop2_pairs & hop3_pairs)

    print(f"\nOverlaps:")
    print(f"  1&2: {overlap_1_2:,}, 1&3: {overlap_1_3:,}, 2&3: {overlap_2_3:,}")

    hop1_unique = hop1_pairs
    hop2_unique_only = hop2_pairs - hop1_pairs
    hop3_unique_only = hop3_pairs - hop1_pairs - hop2_pairs

    print(f"\nMutually exclusive:")
    print(f"  1-hop: {len(hop1_unique):,}")
    print(f"  2-hop only: {len(hop2_unique_only):,}")
    print(f"  3-hop only: {len(hop3_unique_only):,}")

    hop1_filtered = hop1_df.copy()
    hop2_filtered = hop2_df[
        hop2_df.apply(lambda row: (row['source'], row['target']) in hop2_unique_only, axis=1)].copy()
    hop3_filtered = hop3_df[
        hop3_df.apply(lambda row: (row['source'], row['target']) in hop3_unique_only, axis=1)].copy()

    return hop1_filtered, hop2_filtered, hop3_filtered, hop1_unique, hop2_unique_only, hop3_unique_only


def create_all_visualizations(hop0_df, hop1_df, hop2_df, hop3_df, all_deg_pairs_df, output_dir=None):
    """Create all visualizations."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Identify top 100 pairs
    print("\nIdentifying top 100 pairs by absolute fold change...")
    all_deg_pairs_df['abs_fc'] = all_deg_pairs_df['logfoldchange'].abs()
    top_100 = all_deg_pairs_df.nlargest(100, 'abs_fc')
    top_100_pairs = set(zip(top_100['source'], top_100['target']))
    print(f"  ✓ Top 100 identified")

    hop_data = [
        ('0-hop', hop0_df, 'logfoldchange', 'pval'),
        ('1-hop', hop1_df, 'logfoldchange', 'pval'),
        ('2-hop', hop2_df, 'logfoldchange', 'pval'),
        ('3-hop', hop3_df, 'logfoldchange', 'pval')
    ]

    # Mark top 100 in dataframes
    for name, df, lfc_col, pval_col in hop_data:
        if 'source' in df.columns and 'target' in df.columns:
            df['is_top100'] = df.apply(lambda row: (row['source'], row['target']) in top_100_pairs, axis=1)

    colors = {'0-hop': '#FF6B6B', '1-hop': '#95E1D3', '2-hop': '#4A90E2', '3-hop': '#FFA07A'}

    # ============================================
    # FIGURE 1: P-VALUE HISTOGRAMS (2x2)
    # ============================================
    print("\nGenerating Figure 1: P-value Histograms (2x2)...")
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('P-value Distributions Across Hop Levels', fontsize=16, fontweight='bold')
    axes1 = axes1.flatten()

    for idx, (name, df, lfc_col, pval_col) in enumerate(hop_data):
        ax = axes1[idx]

        if 'source' in df.columns:
            n_sources = df['source'].nunique()
            n_targets = df['target'].nunique()
            n_pairs = len(df[['source', 'target']].drop_duplicates())
            title = f'{name} (n={n_pairs:,}) | {n_sources} sources, {n_targets} targets'
        else:
            n_pairs = len(df)
            title = f'{name} (n={n_pairs:,})'

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
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pvalue_distributions_2x2.png'), dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved")
    plt.show()

    # ============================================
    # FIGURE 2: LOG FC HISTOGRAMS (2x2)
    # ============================================
    print("Generating Figure 2: Log Fold Change Histograms (2x2)...")
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Log Fold Change Distributions Across Hop Levels', fontsize=16, fontweight='bold')
    axes2 = axes2.flatten()

    for idx, (name, df, lfc_col, pval_col) in enumerate(hop_data):
        ax = axes2[idx]

        if 'source' in df.columns:
            n_sources = df['source'].nunique()
            n_targets = df['target'].nunique()
            n_pairs = len(df[['source', 'target']].drop_duplicates())
            title = f'{name} (n={n_pairs:,}) | {n_sources} sources, {n_targets} targets'
        else:
            n_pairs = len(df)
            title = f'{name} (n={n_pairs:,})'

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
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'logfc_distributions_2x2.png'), dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved")
    plt.show()

    # ============================================
    # FIGURE 3: OVERLAID HISTOGRAMS (100 bins) - FILTERED
    # ============================================
    print("Generating Figure 3: Overlaid Histograms (100 bins, filtered |FC| >= 0.5)...")
    fig3, (ax_pval, ax_lfc) = plt.subplots(1, 2, figsize=(16, 6))
    fig3.suptitle('Overlaid Distributions Across Hop Levels (Filtered |FC| ≥ 0.5)', fontsize=16, fontweight='bold')

    for name, df, lfc_col, pval_col in hop_data:
        # Apply fold change filter for log FC plot
        df_filtered = df[df[lfc_col].abs() >= 0.5].copy()

        n_pairs = len(df[['source', 'target']].drop_duplicates()) if 'source' in df.columns else len(df)
        n_pairs_filtered = len(
            df_filtered[['source', 'target']].drop_duplicates()) if 'source' in df_filtered.columns else len(
            df_filtered)

        pvals = df[pval_col].dropna()  # No filter for pval
        lfcs_filtered = df_filtered[lfc_col].dropna()  # Filtered for FC

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
        print(f"  ✓ Saved")
    plt.show()

    # ============================================
    # FIGURE 4: 2D LANDSCAPE (FC vs -log10(pval))
    # ============================================
    print("Generating Figure 4: 2D Landscape Plots...")
    fig4, axes4 = plt.subplots(2, 2, figsize=(20, 18))
    fig4.suptitle('Fold Change vs Significance Landscape by Hop Level',
                  fontsize=16, fontweight='bold', y=0.998)
    axes4 = axes4.flatten()

    # Adjust subplot spacing - reduce top margin to bring top row closer to bottom
    plt.subplots_adjust(hspace=0.25, wspace=0.3, top=0.93, bottom=0.05, left=0.08, right=0.95)

    for idx, (name, df, lfc_col, pval_col) in enumerate(hop_data):
        ax = axes4[idx]

        if 'source' in df.columns:
            n_sources = df['source'].nunique()
            n_targets = df['target'].nunique()
            n_pairs = len(df[['source', 'target']].drop_duplicates())
            title = f'{name} (n={n_pairs:,}) | {n_sources} sources, {n_targets} targets'
        else:
            n_pairs = len(df)
            title = f'{name} (n={n_pairs:,})'

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
        ax.set_title(title, fontsize=9, fontweight='bold', pad=12)
        ax.axvline(x=0, color='white', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y=-np.log10(0.05), color='white', linestyle='--', linewidth=1, alpha=0.7,
                   label='p=0.05')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, color='white')

    # Don't use tight_layout, we already set spacing with subplots_adjust
    # plt.tight_layout() - removed
    if output_dir:
        plt.savefig(os.path.join(output_dir, '2d_landscape_fc_vs_pval.png'), dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved")
    plt.show()

    # ============================================
    # FIGURE 5: PAIRWISE P-VALUE COMPARISONS (6 plots)
    # ============================================
    print("Generating Figure 5: Pairwise P-value Comparisons...")

    # Create all pairwise combinations
    from itertools import combinations
    hop_combinations = list(combinations(range(4), 2))  # [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

    fig5, axes5 = plt.subplots(2, 3, figsize=(18, 10))
    fig5.suptitle('Pairwise P-value Distribution Comparisons',
                  fontsize=16, fontweight='bold')
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
        print(f"  ✓ Saved")
    plt.show()

    # ============================================
    # FIGURE 6: PAIRWISE -LOG10(P-VALUE) COMPARISONS (Improved Scaling)
    # ============================================
    print("Generating Figure 6: Pairwise -log10(P-value) Comparisons (improved scaling)...")

    from itertools import combinations
    hop_combinations = list(combinations(range(4), 2))

    fig6, axes6 = plt.subplots(2, 3, figsize=(18, 10))
    fig6.suptitle('Pairwise -log10(P-value) Distribution Comparisons', fontsize=16, fontweight='bold')
    axes6 = axes6.flatten()

    # --- Compute global xlim across all datasets ---
    all_neglog_pvals = []
    for _, df, _, pval_col in hop_data:
        vals = -np.log10(df[pval_col].dropna().clip(lower=1e-300))
        vals = np.clip(vals, 0, 40)  # cap extreme values
        all_neglog_pvals.extend(vals.values)

    xmax = np.percentile(all_neglog_pvals, 99.5)  # robust upper limit
    xlim = (0, max(10, xmax))  # ensure reasonable width, minimum 10

    for idx, (i, j) in enumerate(hop_combinations):
        ax = axes6[idx]
        name_i, df_i, lfc_i, pval_i = hop_data[i]
        name_j, df_j, lfc_j, pval_j = hop_data[j]

        n_i = len(df_i[['source', 'target']].drop_duplicates()) if 'source' in df_i.columns else len(df_i)
        n_j = len(df_j[['source', 'target']].drop_duplicates()) if 'source' in df_j.columns else len(df_j)

        neglog_i = -np.log10(df_i[pval_i].dropna().clip(lower=1e-300))
        neglog_j = -np.log10(df_j[pval_j].dropna().clip(lower=1e-300))
        neglog_i = np.clip(neglog_i, 0, 40)
        neglog_j = np.clip(neglog_j, 0, 40)

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
        print("  ✓ Saved (scaled version)")
    plt.show()

    # ============================================
    # FIGURE 7: PAIRWISE LOG FC COMPARISONS (Filtered |FC| >= 1.0, Tail-Focused)
    # ============================================
    print("Generating Figure 7: Pairwise Log FC Comparisons (tail-focused)...")

    from itertools import combinations
    hop_combinations = list(combinations(range(4), 2))

    fig7, axes7 = plt.subplots(2, 3, figsize=(18, 10))
    fig7.suptitle('Pairwise Log Fold Change Comparisons (Filtered |FC| ≥ 1.0)',
                  fontsize=16, fontweight='bold')
    axes7 = axes7.flatten()

    # Compute global axis ranges for consistent scaling
    all_lfcs = []
    for _, df, lfc_col, _ in hop_data:
        vals = df[lfc_col].dropna()
        vals = vals[np.abs(vals) >= 1.0]  # stronger filter
        vals = np.clip(vals, -4, 4)
        all_lfcs.extend(vals.values)
    xlim = (-3, 3)
    bins = np.linspace(xlim[0], xlim[1], 100)

    # Precompute max y for uniform scaling
    global_ymax = 0
    for _, df, lfc_col, _ in hop_data:
        vals = df[lfc_col].dropna()
        vals = vals[np.abs(vals) >= 1.0]
        vals = np.clip(vals, -4, 4)
        counts, _ = np.histogram(vals, bins=bins)
        global_ymax = max(global_ymax, counts.max())

    # Plot all pairwise combinations
    for idx, (i, j) in enumerate(hop_combinations):
        ax = axes7[idx]
        name_i, df_i, lfc_i, pval_i = hop_data[i]
        name_j, df_j, lfc_j, pval_j = hop_data[j]

        # Filter to |FC| ≥ 1.0 for better tail view
        df_i_filt = df_i[np.abs(df_i[lfc_i]) >= 1.0]
        df_j_filt = df_j[np.abs(df_j[lfc_j]) >= 1.0]

        n_i = len(df_i_filt[['source', 'target']].drop_duplicates()) if 'source' in df_i_filt.columns else len(
            df_i_filt)
        n_j = len(df_j_filt[['source', 'target']].drop_duplicates()) if 'source' in df_j_filt.columns else len(
            df_j_filt)

        lfcs_i = np.clip(df_i_filt[lfc_i].dropna(), -4, 4)
        lfcs_j = np.clip(df_j_filt[lfc_j].dropna(), -4, 4)

        ax.hist(lfcs_i, bins=bins, alpha=0.6, label=f'{name_i} (n={n_i:,})',
                color=colors[name_i], edgecolor='none')
        ax.hist(lfcs_j, bins=bins, alpha=0.6, label=f'{name_j} (n={n_j:,})',
                color=colors[name_j], edgecolor='none')

        ax.set_xlim(xlim)
        ax.set_ylim(0, global_ymax * 1.1)
        ax.set_xlabel('Log Fold Change', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{name_i} vs {name_j}', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.6)
        ax.axvline(x=1, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.axvline(x=-1, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Frame styling
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("#555")

    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pairwise_logfc_comparisons_tailfocused.png'),
                    dpi=300, bbox_inches='tight')
        print("  ✓ Saved (tail-focused version)")
    plt.show()

    print("\n✓ All visualizations complete!")


def main():
    print("\n" + "=" * 60)
    print("HOP ANALYSIS - MUTUALLY EXCLUSIVE")
    print("=" * 60)

    base_path = "/Users/prashammarfatia/Downloads"
    hop1_file = os.path.join(base_path, "indra_1hop_with_statements_ (1).csv")
    hop2_file = os.path.join(base_path, "indra_2hop_all_perturbations.csv")
    hop3_file = os.path.join(base_path, "indra_3hop_cleaned_results.csv")
    validation_file = os.path.join(base_path, "target_validation_expanded.csv")
    deg_folder = os.path.join(base_path, "de_results_per_gene")
    output_dir = os.path.join(base_path, "hop_analysis_final")

    print(f"\nOutput: {output_dir}")

    # Load and process
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    hop1_df = load_hop_data(hop1_file, "1-hop")
    hop2_df = load_hop_data(hop2_file, "2-hop")
    hop3_df = load_hop_data(hop3_file, "3-hop")

    hop1_df = deduplicate_hop_data(hop1_df, "1-hop")
    hop2_df = deduplicate_hop_data(hop2_df, "2-hop")
    print("\n3-hop already unique")

    validated_sources = get_validated_sources(validation_file)

    hop1_filt, hop2_filt, hop3_filt, hop1_p, hop2_p, hop3_p = \
        create_mutually_exclusive_datasets(hop1_df, hop2_df, hop3_df)

    hop0_df, all_deg_df = identify_zero_hop_pairs(
        deg_folder, validated_sources, hop1_p, hop2_p, hop3_p)

    create_all_visualizations(hop0_df, hop1_filt, hop2_filt, hop3_filt, all_deg_df, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    hop0_n = len(hop0_df)
    hop1_n = len(hop1_filt[['source', 'target']].drop_duplicates())
    hop2_n = len(hop2_filt[['source', 'target']].drop_duplicates())
    hop3_n = len(hop3_filt[['source', 'target']].drop_duplicates())
    total = hop0_n + hop1_n + hop2_n + hop3_n

    print(f"0-hop: {hop0_n:,} ({hop0_n / total * 100:.1f}%)")
    print(f"1-hop: {hop1_n:,} ({hop1_n / total * 100:.1f}%)")
    print(f"2-hop: {hop2_n:,} ({hop2_n / total * 100:.1f}%)")
    print(f"3-hop: {hop3_n:,} ({hop3_n / total * 100:.1f}%)")
    print(f"Total: {total:,}")
    print(f"\nFigures saved to: {output_dir}")
    print(f"  1. pvalue_distributions_2x2.png")
    print(f"  2. logfc_distributions_2x2.png")
    print(f"  3. overlaid_histograms_100bins.png (filtered |FC| >= 0.5)")
    print(f"  4. 2d_landscape_fc_vs_pval.png")
    print(f"  5. pairwise_pvalue_comparisons.png")
    print(f"  6. pairwise_neglog_pvalue_comparisons.png")
    print(f"  7. pairwise_logfc_comparisons_filtered.png (filtered |FC| >= 0.5)")
    print(f"\nTotal: 7 figure files created")
    print("\n🎉 Done!")


if __name__ == "__main__":
    main()