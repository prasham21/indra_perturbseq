import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')

# Output directory
OUTPUT_DIR = '/Users/prashammarfatia/Downloads/Evidence_Analysis'


def load_data():
    """Load all datasets"""
    print("Loading datasets for debug analysis...")

    # Load 1-hop data
    hop1 = pd.read_csv('/Users/prashammarfatia/Downloads/indra_1hop_gene_names_cleaned.csv')
    hop1['evidence_count_mean'] = hop1['evidence_count']

    # Load 2-hop data
    hop2 = pd.read_csv('/Users/prashammarfatia/Downloads/indra_2hop_with_evidence_statements_.csv')
    hop2['evidence_count_mean'] = (hop2['evidence_1'] + hop2['evidence_2']) / 2

    # Load 3-hop data
    hop3 = pd.read_csv('/Users/prashammarfatia/Downloads/indra_3hop_cleaned_results.csv')
    hop3['evidence_count_mean'] = (hop3['evidence_1'] + hop3['evidence_2'] + hop3['evidence_3']) / 3

    print(f"1-hop: {len(hop1)} rows")
    print(f"2-hop: {len(hop2)} rows")
    print(f"3-hop: {len(hop3)} rows")

    return hop1, hop2, hop3


def debug_data_quality(hop1, hop2, hop3):
    """Check data quality and distributions"""
    print("\n" + "=" * 50)
    print("DATA QUALITY DEBUG")
    print("=" * 50)

    datasets = [hop1, hop2, hop3]
    names = ['1-hop', '2-hop', '3-hop']

    for data, name in zip(datasets, names):
        evidence = data['evidence_count_mean']
        logfc = data['logfoldchange']

        print(f"\n{name}:")
        print(f"  Evidence count - Min: {evidence.min():.2f}, Max: {evidence.max():.2f}")
        print(f"  Evidence count - Unique values: {evidence.nunique()}")
        print(f"  Log fold change - Min: {logfc.min():.2f}, Max: {logfc.max():.2f}")
        print(f"  Log fold change - Unique values: {logfc.nunique()}")
        print(f"  NaN in evidence: {evidence.isna().sum()}")
        print(f"  NaN in logfc: {logfc.isna().sum()}")
        print(f"  Infinite in evidence: {np.isinf(evidence).sum()}")
        print(f"  Infinite in logfc: {np.isinf(logfc).sum()}")

        # Check for suspicious clustering
        logfc_rounded = logfc.round(2)
        most_common_logfc = logfc_rounded.value_counts().head(5)
        print(f"  Most common log fold change values:")
        for val, count in most_common_logfc.items():
            print(f"    {val}: {count} times ({count / len(logfc) * 100:.1f}%)")


def create_raw_scatter_plots(hop1, hop2, hop3):
    """Create raw scatter plots to see actual data distribution"""
    print("\nCreating raw scatter plots...")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    datasets = [hop1, hop2, hop3]
    names = ['1-hop Raw Scatter', '2-hop Raw Scatter', '3-hop Raw Scatter']
    colors = ['blue', 'red', 'green']

    for i, (data, name, color) in enumerate(zip(datasets, names, colors)):
        evidence = data['evidence_count_mean']
        logfc = data['logfoldchange']

        # Clean data
        mask = np.isfinite(evidence) & np.isfinite(logfc)
        evidence_clean = evidence[mask]
        logfc_clean = logfc[mask]

        print(f"  {names[i].split()[0]}: {len(evidence_clean):,} valid points")

        # Sample for large datasets
        if len(evidence_clean) > 5000:
            sample_size = 5000
            sample_idx = np.random.choice(len(evidence_clean), sample_size, replace=False)
            evidence_plot = evidence_clean.iloc[sample_idx] if hasattr(evidence_clean, 'iloc') else evidence_clean[
                sample_idx]
            logfc_plot = logfc_clean.iloc[sample_idx] if hasattr(logfc_clean, 'iloc') else logfc_clean[sample_idx]
            title_suffix = f"(showing {sample_size:,} of {len(evidence_clean):,} points)"
        else:
            evidence_plot = evidence_clean
            logfc_plot = logfc_clean
            title_suffix = f"(all {len(evidence_clean):,} points)"

        # Create scatter plot
        axes[i].scatter(evidence_plot, logfc_plot, alpha=0.4, s=12, color=color, edgecolors='none')

        # Calculate correlation
        correlation = np.corrcoef(evidence_clean, logfc_clean)[0, 1]

        axes[i].set_xlabel('Mean Evidence Count')
        axes[i].set_ylabel('Log Fold Change')
        axes[i].set_title(f'{name}\n{title_suffix}\nr = {correlation:.4f}')
        axes[i].grid(True, alpha=0.3)

        # Add some basic stats
        axes[i].text(0.05, 0.95,
                     f'Evidence range: {evidence_clean.min():.1f} - {evidence_clean.max():.1f}\nLogFC range: {logfc_clean.min():.2f} - {logfc_clean.max():.2f}',
                     transform=axes[i].transAxes,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     verticalalignment='top', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'debug_raw_scatter_plots.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def create_zoomed_scatter_plots(hop1, hop2, hop3):
    """Create zoomed scatter plots focusing on bulk of data"""
    print("\nCreating zoomed scatter plots (excluding outliers)...")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    datasets = [hop1, hop2, hop3]
    names = ['1-hop Zoomed', '2-hop Zoomed', '3-hop Zoomed']
    colors = ['blue', 'red', 'green']

    for i, (data, name, color) in enumerate(zip(datasets, names, colors)):
        evidence = data['evidence_count_mean']
        logfc = data['logfoldchange']

        # Clean data
        mask = np.isfinite(evidence) & np.isfinite(logfc)
        evidence_clean = evidence[mask]
        logfc_clean = logfc[mask]

        # Zoom to reasonable ranges (exclude extreme outliers)
        evidence_95 = np.percentile(evidence_clean, 95)
        logfc_5 = np.percentile(logfc_clean, 5)
        logfc_95 = np.percentile(logfc_clean, 95)

        zoom_mask = (evidence_clean <= evidence_95) & (logfc_clean >= logfc_5) & (logfc_clean <= logfc_95)
        evidence_zoom = evidence_clean[zoom_mask]
        logfc_zoom = logfc_clean[zoom_mask]

        print(f"  {names[i].split()[0]}: {len(evidence_zoom):,} points in zoom range")

        # Sample if still too many
        if len(evidence_zoom) > 5000:
            sample_idx = np.random.choice(len(evidence_zoom), 5000, replace=False)
            evidence_plot = evidence_zoom.iloc[sample_idx] if hasattr(evidence_zoom, 'iloc') else evidence_zoom[
                sample_idx]
            logfc_plot = logfc_zoom.iloc[sample_idx] if hasattr(logfc_zoom, 'iloc') else logfc_zoom[sample_idx]
        else:
            evidence_plot = evidence_zoom
            logfc_plot = logfc_zoom

        # Create scatter plot
        axes[i].scatter(evidence_plot, logfc_plot, alpha=0.5, s=15, color=color, edgecolors='none')

        # Calculate correlation
        correlation = np.corrcoef(evidence_zoom, logfc_zoom)[0, 1]

        axes[i].set_xlabel('Mean Evidence Count')
        axes[i].set_ylabel('Log Fold Change')
        axes[i].set_title(f'{name} (5th-95th percentiles)\nr = {correlation:.4f}')
        axes[i].grid(True, alpha=0.3)

        # Show the zoom ranges
        axes[i].text(0.05, 0.95,
                     f'Evidence: ≤{evidence_95:.1f}\nLogFC: {logfc_5:.2f} to {logfc_95:.2f}\nPoints: {len(evidence_plot):,}',
                     transform=axes[i].transAxes,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     verticalalignment='top', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'debug_zoomed_scatter_plots.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def create_distribution_check_plots(hop1, hop2, hop3):
    """Check if fold change data has suspicious patterns"""
    print("\nCreating distribution check plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    datasets = [hop1, hop2, hop3]
    names = ['1-hop', '2-hop', '3-hop']

    for i, (data, name) in enumerate(zip(datasets, names)):
        evidence = data['evidence_count_mean']
        logfc = data['logfoldchange']

        # Clean data
        mask = np.isfinite(evidence) & np.isfinite(logfc)
        evidence_clean = evidence[mask]
        logfc_clean = logfc[mask]

        # Top row: Evidence count histograms
        axes[0, i].hist(evidence_clean, bins=50, alpha=0.7, edgecolor='black')
        axes[0, i].set_xlabel('Evidence Count')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].set_title(f'{name}: Evidence Distribution')
        axes[0, i].set_xlim(0, np.percentile(evidence_clean, 95))

        # Bottom row: Log fold change histograms
        axes[1, i].hist(logfc_clean, bins=50, alpha=0.7, edgecolor='black')
        axes[1, i].set_xlabel('Log Fold Change')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].set_title(f'{name}: LogFC Distribution')

        # Add stats
        axes[0, i].axvline(evidence_clean.median(), color='red', linestyle='--', alpha=0.8)
        axes[1, i].axvline(logfc_clean.median(), color='red', linestyle='--', alpha=0.8)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'debug_distribution_check.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def main():
    """Debug analysis to understand raw data patterns"""
    print("DEBUG ANALYSIS: Raw Data Investigation")
    print("=" * 60)

    # Load data
    hop1, hop2, hop3 = load_data()

    # Check data quality
    debug_data_quality(hop1, hop2, hop3)

    # Create raw scatter plots
    create_raw_scatter_plots(hop1, hop2, hop3)

    # Create zoomed scatter plots
    create_zoomed_scatter_plots(hop1, hop2, hop3)

    # Check distributions
    create_distribution_check_plots(hop1, hop2, hop3)

    print("\n" + "=" * 60)
    print("DEBUG ANALYSIS COMPLETE!")
    print("Files generated:")
    print("- debug_raw_scatter_plots.png")
    print("- debug_zoomed_scatter_plots.png")
    print("- debug_distribution_check.png")
    print("=" * 60)
    print("\nKEY QUESTIONS TO ANSWER:")
    print("1. Do the scatter plots look random or show patterns?")
    print("2. Are fold change values clustered at specific numbers?")
    print("3. Do the correlations match what you see visually?")
    print("4. Are there obvious data quality issues?")


if __name__ == "__main__":
    main()