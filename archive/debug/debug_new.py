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


def identify_zero_hop_pairs(deg_folder, validated_sources, hop3_df):
    """
    Identify source-target pairs that are not explained by any hop (0-hop).
    Sources are limited to the 357 validated genes.
    Targets are ANY genes from the DEG files (not limited to the 357 set).

    Parameters:
    deg_folder: Path to folder containing DEG files
    validated_sources: Set of source genes marked for analysis
    hop3_df: DataFrame of 3-hop data (contains all explained pairs)

    Returns:
    DataFrame with 0-hop pairs and their pvals/logfoldchanges
    """
    print("\n" + "=" * 60)
    print("IDENTIFYING 0-HOP PAIRS (Unexplained by pathways)")
    print("=" * 60)

    # Get all source-target pairs from 3-hop data
    print("Extracting all explained pairs from 3-hop data...")
    explained_pairs = set(zip(hop3_df['source'], hop3_df['target']))
    print(f"  ✓ Found {len(explained_pairs):,} explained source-target pairs")

    # First pass: collect all actual DEG pairs from the experiment
    print(f"\nCollecting actual DEG pairs from {len(validated_sources)} validated sources...")

    all_deg_pairs = {}  # (source, target) -> (logfc, pval)
    processed_sources = 0
    missing_deg_files = []

    for i, source_gene in enumerate(sorted(validated_sources), 1):
        # Construct DEG file path
        deg_file = os.path.join(deg_folder, f"{source_gene}_vs_control.csv")

        if not os.path.exists(deg_file):
            missing_deg_files.append(source_gene)
            continue

        processed_sources += 1

        if i % 50 == 0 or i == len(validated_sources):
            print(f"  Processing {i}/{len(validated_sources)}: {source_gene}")

        try:
            # Read DEG file
            deg_df = pd.read_csv(deg_file)

            # Filter to significant targets only (pval < 0.05)
            deg_df_sig = deg_df[deg_df['pvals'] < 0.05].copy()

            # Store each source-target pair with its values
            for _, row in deg_df_sig.iterrows():
                target_gene = row['names']
                pair = (source_gene, target_gene)
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

    # Second pass: identify which DEG pairs are NOT explained by 3-hop
    print(f"\nIdentifying unexplained pairs...")

    zero_hop_data = []
    explained_count = 0

    for pair, (logfc, pval) in all_deg_pairs.items():
        source_gene, target_gene = pair

        if pair in explained_pairs:
            explained_count += 1
        else:
            # This pair is NOT explained by pathways (0-hop)
            zero_hop_data.append({
                'source': source_gene,
                'target': target_gene,
                'logfoldchange': logfc,
                'pval': pval
            })

    zero_hop_count = len(zero_hop_data)

    print(f"\n✓ Analysis complete!")
    print(f"  Total actual DEG pairs: {len(all_deg_pairs):,}")
    print(f"  Explained by pathways (in 3-hop): {explained_count:,}")
    print(f"  Unexplained pairs (0-hop): {zero_hop_count:,}")
    print(f"  Coverage: {explained_count / len(all_deg_pairs) * 100:.1f}% of DEG pairs explained")

    zero_hop_df = pd.DataFrame(zero_hop_data)
    return zero_hop_df


def create_histograms(hop0_df, hop1_df, hop2_df, hop3_df, output_dir=None):
    """
    Create two figures with 4 subplots each for pval and logfoldchange distributions.

    Parameters:
    hop0_df, hop1_df, hop2_df, hop3_df: DataFrames for each hop level
    output_dir: Directory to save figures (optional)
    """
    print("\n" + "=" * 60)
    print("CREATING HISTOGRAMS")
    print("=" * 60)

    # Prepare data
    hop_data = [
        ('0-hop', hop0_df, 'logfoldchange', 'pval'),
        ('1-hop', hop1_df, 'logfoldchange', 'pval'),
        ('2-hop', hop2_df, 'logfoldchange', 'pval'),
        ('3-hop', hop3_df, 'logfoldchange', 'pval')
    ]

    # Print statistics
    print("\nData Statistics:")
    for name, df, lfc_col, pval_col in hop_data:
        print(f"\n{name}:")
        print(f"  Total entries: {len(df):,}")
        print(f"  LogFC range: [{df[lfc_col].min():.3f}, {df[lfc_col].max():.3f}]")
        print(f"  P-value range: [{df[pval_col].min():.2e}, {df[pval_col].max():.2e}]")

    # ============================================
    # FIGURE 1: P-VALUE DISTRIBUTIONS
    # ============================================
    print("\nGenerating Figure 1: P-value Distributions...")
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('P-value Distributions Across Hop Levels', fontsize=16, fontweight='bold')
    axes1 = axes1.flatten()

    for idx, (name, df, lfc_col, pval_col) in enumerate(hop_data):
        ax = axes1[idx]

        # Create histogram
        pvals = df[pval_col].dropna()
        ax.hist(pvals, bins=50, color='steelblue', alpha=0.7, edgecolor='black')

        # Formatting
        ax.set_xlabel('P-value', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{name} (n={len(pvals):,})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add statistics text
        median_pval = pvals.median()
        mean_pval = pvals.mean()
        stats_text = f'Mean: {mean_pval:.2e}\nMedian: {median_pval:.2e}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

    plt.tight_layout()

    # Save Figure 1
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig1_path = os.path.join(output_dir, 'pvalue_distributions_by_hop.png')
        plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {fig1_path}")

    # ============================================
    # FIGURE 2: LOG FOLD CHANGE DISTRIBUTIONS
    # ============================================
    print("Generating Figure 2: Log Fold Change Distributions...")
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Log Fold Change Distributions Across Hop Levels', fontsize=16, fontweight='bold')
    axes2 = axes2.flatten()

    for idx, (name, df, lfc_col, pval_col) in enumerate(hop_data):
        ax = axes2[idx]

        # Create histogram
        lfcs = df[lfc_col].dropna()
        ax.hist(lfcs, bins=50, color='coral', alpha=0.7, edgecolor='black')

        # Formatting
        ax.set_xlabel('Log Fold Change', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{name} (n={len(lfcs):,})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add statistics text
        median_lfc = lfcs.median()
        mean_lfc = lfcs.mean()
        stats_text = f'Mean: {mean_lfc:.3f}\nMedian: {median_lfc:.3f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                fontsize=9)

        # Add vertical line at 0
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()

    # Save Figure 2
    if output_dir:
        fig2_path = os.path.join(output_dir, 'logfoldchange_distributions_by_hop.png')
        plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {fig2_path}")

    plt.show()
    print("\n✓ Histograms generated successfully!")


def main():
    """Main function to orchestrate the analysis."""

    print("\n" + "=" * 60)
    print("HOP ANALYSIS - HISTOGRAM GENERATION")
    print("=" * 60)

    # File paths
    base_path = "/Users/prashammarfatia/Downloads"

    hop1_file = os.path.join(base_path, "indra_1hop_with_statements_ (1).csv")
    hop2_file = os.path.join(base_path, "indra_2hop_all_perturbations.csv")
    hop3_file = os.path.join(base_path, "indra_3hop_cleaned_results.csv")
    validation_file = os.path.join(base_path, "target_validation_expanded.csv")
    deg_folder = os.path.join(base_path, "de_results_per_gene")

    # Output directory for saving figures
    output_dir = os.path.join(base_path, "hop_analysis_figures")

    print(f"\nConfiguration:")
    print(f"  1-hop file: {hop1_file}")
    print(f"  2-hop file: {hop2_file}")
    print(f"  3-hop file: {hop3_file}")
    print(f"  Validation file: {validation_file}")
    print(f"  DEG folder: {deg_folder}")
    print(f"  Output directory: {output_dir}")

    # Check if files exist
    print("\nVerifying files exist...")
    for label, path in [("1-hop", hop1_file), ("2-hop", hop2_file),
                        ("3-hop", hop3_file), ("Validation", validation_file)]:
        if os.path.exists(path):
            print(f"  ✓ {label} file found")
        else:
            print(f"  ✗ {label} file NOT found: {path}")
            return

    if os.path.exists(deg_folder):
        deg_count = len(glob.glob(os.path.join(deg_folder, "*_vs_control.csv")))
        print(f"  ✓ DEG folder found with {deg_count} files")
    else:
        print(f"  ✗ DEG folder NOT found: {deg_folder}")
        return

    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    hop1_df = load_hop_data(hop1_file, "1-hop")
    hop2_df = load_hop_data(hop2_file, "2-hop")
    hop3_df = load_hop_data(hop3_file, "3-hop")

    # Get validated SOURCE genes (filtered by Karen_Flag = 'Use_for_analysis')
    validated_sources = get_validated_sources(validation_file)

    # Identify 0-hop pairs
    hop0_df = identify_zero_hop_pairs(deg_folder, validated_sources, hop3_df)

    if len(hop0_df) == 0:
        print("\n⚠️  WARNING: No 0-hop pairs found!")
        print("This means all source-target pairs are explained by pathways.")

    # Create histograms
    create_histograms(hop0_df, hop1_df, hop2_df, hop3_df, output_dir)

    # Final summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  0-hop pairs (unexplained): {len(hop0_df):,}")
    print(f"  1-hop pairs: {len(hop1_df):,}")
    print(f"  2-hop pairs: {len(hop2_df):,}")
    print(f"  3-hop pairs: {len(hop3_df):,}")
    print(f"\nFigures saved to: {output_dir}")
    print(f"  - pvalue_distributions_by_hop.png")
    print(f"  - logfoldchange_distributions_by_hop.png")
    print("\n🎉 Done!")


if __name__ == "__main__":
    main()