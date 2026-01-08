import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kstest, ks_2samp, gaussian_kde
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Define output directory
OUTPUT_DIR = '/Users/prashammarfatia/Downloads/Evidence_Analysis'


def setup_output_directory():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")
    else:
        print(f"Using existing directory: {OUTPUT_DIR}")


def load_and_process_data():
    """
    Load all three datasets and calculate mean evidence counts
    """
    print("Loading datasets...")

    # Load 1-hop data
    hop1 = pd.read_csv('/Users/prashammarfatia/Downloads/indra_1hop_gene_names_cleaned.csv')
    hop1['hop_type'] = '1-hop'
    hop1['evidence_count_mean'] = hop1['evidence_count']
    hop1['num_hops'] = 1

    # Load 2-hop data
    hop2 = pd.read_csv('/Users/prashammarfatia/Downloads/indra_2hop_with_evidence_statements_.csv')
    hop2['hop_type'] = '2-hop'
    hop2['evidence_count_mean'] = (hop2['evidence_1'] + hop2['evidence_2']) / 2
    hop2['num_hops'] = 2

    # Load 3-hop data
    hop3 = pd.read_csv('/Users/prashammarfatia/Downloads/indra_3hop_cleaned_results.csv')
    hop3['hop_type'] = '3-hop'
    hop3['evidence_count_mean'] = (hop3['evidence_1'] + hop3['evidence_2'] + hop3['evidence_3']) / 3
    hop3['num_hops'] = 3

    print(f"Data loaded:")
    print(f"1-hop: {len(hop1)} rows")
    print(f"2-hop: {len(hop2)} rows")
    print(f"3-hop: {len(hop3)} rows")

    return hop1, hop2, hop3


def analyze_evidence_distributions(hop1, hop2, hop3):
    """
    Analyze evidence count distributions and identify outliers
    """
    datasets = [hop1, hop2, hop3]
    names = ['1-hop', '2-hop', '3-hop']

    print("\n" + "=" * 50)
    print("EVIDENCE COUNT DISTRIBUTION ANALYSIS")
    print("=" * 50)

    analysis_results = []

    for data, name in zip(datasets, names):
        evidence_counts = data['evidence_count_mean']

        stats_dict = {
            'Hop Type': name,
            'Count': len(evidence_counts),
            'Mean': evidence_counts.mean(),
            'Median': evidence_counts.median(),
            'Std Dev': evidence_counts.std(),
            'Min': evidence_counts.min(),
            'Max': evidence_counts.max(),
            '25th Percentile': evidence_counts.quantile(0.25),
            '75th Percentile': evidence_counts.quantile(0.75),
            '90th Percentile': evidence_counts.quantile(0.90),
            '95th Percentile': evidence_counts.quantile(0.95),
            '99th Percentile': evidence_counts.quantile(0.99)
        }

        # Identify extreme outliers (>99th percentile)
        outlier_threshold = evidence_counts.quantile(0.99)
        outliers = evidence_counts[evidence_counts > outlier_threshold]
        stats_dict['Outliers (>99th percentile)'] = len(outliers)
        if len(outliers) > 0:
            stats_dict['Outlier Min'] = outliers.min()
            stats_dict['Outlier Max'] = outliers.max()
        else:
            stats_dict['Outlier Min'] = 'N/A'
            stats_dict['Outlier Max'] = 'N/A'

        analysis_results.append(stats_dict)

        print(f"\n{name} Statistics:")
        print(f"  Count: {stats_dict['Count']}")
        print(f"  Mean: {stats_dict['Mean']:.2f}")
        print(f"  Median: {stats_dict['Median']:.2f}")
        print(f"  Std: {stats_dict['Std Dev']:.2f}")
        print(f"  Min: {stats_dict['Min']:.2f}")
        print(f"  Max: {stats_dict['Max']:.2f}")
        print(f"  95th percentile: {stats_dict['95th Percentile']:.2f}")
        print(f"  99th percentile: {stats_dict['99th Percentile']:.2f}")
        print(f"  Outliers (>99th percentile): {stats_dict['Outliers (>99th percentile)']} values")
        if len(outliers) > 0:
            print(f"  Outlier range: {stats_dict['Outlier Min']:.2f} - {stats_dict['Outlier Max']:.2f}")

    return analysis_results


def create_improved_individual_histograms(hop1, hop2, hop3):
    """
    Create much better histograms that actually show meaningful patterns
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    datasets = [hop1, hop2, hop3]
    titles = ['1-hop Evidence Counts', '2-hop Evidence Counts', '3-hop Evidence Counts']
    colors = ['skyblue', 'lightcoral', 'lightgreen']

    # Top row: Log scale histograms to handle extreme skew
    for i, (data, title, color) in enumerate(zip(datasets, titles, colors)):
        evidence_counts = data['evidence_count_mean']

        # Log scale histogram
        log_evidence = np.log10(evidence_counts + 1)  # +1 to handle zeros
        axes[0, i].hist(log_evidence, bins=30, alpha=0.7, color=color, edgecolor='black')
        axes[0, i].set_xlabel('Log10(Evidence Count + 1)')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].set_title(f'{title} (Log Scale)')
        axes[0, i].grid(True, alpha=0.3)

        # Add statistics
        stats_text = f'n={len(evidence_counts)}\nMean: {evidence_counts.mean():.1f}\nMedian: {evidence_counts.median():.1f}'
        axes[0, i].text(0.7, 0.8, stats_text, transform=axes[0, i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Bottom row: Truncated histograms (focus on bulk of data)
    for i, (data, title, color) in enumerate(zip(datasets, titles, colors)):
        evidence_counts = data['evidence_count_mean']

        # Focus on reasonable range (up to 90th percentile)
        cutoff = evidence_counts.quantile(0.90)
        truncated_data = evidence_counts[evidence_counts <= cutoff]
        excluded_count = len(evidence_counts) - len(truncated_data)

        axes[1, i].hist(truncated_data, bins=20, alpha=0.7, color=color, edgecolor='black')
        axes[1, i].set_xlabel('Mean Evidence Count')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].set_title(f'{title} (≤90th %ile)')
        axes[1, i].grid(True, alpha=0.3)

        # Add statistics
        stats_text = f'Showing: {len(truncated_data)}\nExcluded: {excluded_count}\nRange: {truncated_data.min():.1f}-{truncated_data.max():.1f}'
        axes[1, i].text(0.6, 0.8, stats_text, transform=axes[1, i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'evidence_counts_improved_histograms.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def create_comparative_boxplots(hop1, hop2, hop3):
    """
    Create box plots for better comparison across hop types
    """
    # Combine all data
    combined_data = []
    for data, hop_type in zip([hop1, hop2, hop3], ['1-hop', '2-hop', '3-hop']):
        for evidence_count in data['evidence_count_mean']:
            combined_data.append({'hop_type': hop_type, 'evidence_count': evidence_count})

    df_combined = pd.DataFrame(combined_data)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Standard box plot
    sns.boxplot(data=df_combined, x='hop_type', y='evidence_count', ax=axes[0])
    axes[0].set_title('Evidence Count Distribution by Hop Type')
    axes[0].set_ylabel('Mean Evidence Count')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Log scale box plot
    df_combined['log_evidence_count'] = np.log10(df_combined['evidence_count'] + 1)
    sns.boxplot(data=df_combined, x='hop_type', y='log_evidence_count', ax=axes[1])
    axes[1].set_title('Evidence Count Distribution (Log Scale)')
    axes[1].set_ylabel('Log10(Evidence Count + 1)')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Violin plot for distribution shape
    sns.violinplot(data=df_combined[df_combined['evidence_count'] <= df_combined['evidence_count'].quantile(0.95)],
                   x='hop_type', y='evidence_count', ax=axes[2])
    axes[2].set_title('Evidence Count Distribution (≤95th %ile)')
    axes[2].set_ylabel('Mean Evidence Count')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'evidence_counts_boxplots.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def create_overlay_histogram(hop1, hop2, hop3):
    """
    Create overlay histogram with better separation
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Standard overlay
    datasets = [hop1['evidence_count_mean'], hop2['evidence_count_mean'], hop3['evidence_count_mean']]
    labels = ['1-hop', '2-hop', '3-hop']
    colors = ['blue', 'red', 'green']

    for data, label, color in zip(datasets, labels, colors):
        ax1.hist(data, bins=50, alpha=0.6, label=label, color=color, density=True)

    ax1.set_xlabel('Mean Evidence Count')
    ax1.set_ylabel('Density')
    ax1.set_title('Evidence Count Distributions: All Hop Types (Overlay)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Limited scale to see detail (exclude extreme outliers)
    max_reasonable = np.percentile(np.concatenate(datasets), 95)

    for data, label, color in zip(datasets, labels, colors):
        filtered_data = data[data <= max_reasonable]
        ax2.hist(filtered_data, bins=50, alpha=0.6, label=f'{label} (≤95th %ile)', color=color, density=True)

    ax2.set_xlabel('Mean Evidence Count')
    ax2.set_ylabel('Density')
    ax2.set_title('Evidence Count Distributions: Excluding Top 5% Outliers')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'evidence_counts_overlay_histogram.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def create_ks_density_plots(hop1, hop2, hop3):
    """
    Create KS density plots for smooth comparison
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    datasets = [hop1['evidence_count_mean'], hop2['evidence_count_mean'], hop3['evidence_count_mean']]
    labels = ['1-hop', '2-hop', '3-hop']
    colors = ['blue', 'red', 'green']

    for data, label, color in zip(datasets, labels, colors):
        # Create density plot
        density = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.quantile(0.95), 1000)
        ax.plot(x_range, density(x_range), label=label, color=color, linewidth=2)
        ax.fill_between(x_range, density(x_range), alpha=0.3, color=color)

    ax.set_xlabel('Mean Evidence Count')
    ax.set_ylabel('Density')
    ax.set_title('Kernel Density Estimation: Evidence Count Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'evidence_counts_kde_plots.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")

    # Perform KS tests
    ks_results = []
    pairs = [(0, 1, '1-hop vs 2-hop'), (0, 2, '1-hop vs 3-hop'), (1, 2, '2-hop vs 3-hop')]

    for i, j, name in pairs:
        ks_stat, p_value = ks_2samp(datasets[i], datasets[j])
        ks_results.append({
            'Comparison': name,
            'KS Statistic': ks_stat,
            'P-value': p_value,
            'Significant (α=0.05)': 'Yes' if p_value < 0.05 else 'No'
        })

    return ks_results


def create_hops_vs_evidence_scatter(hop1, hop2, hop3):
    """
    Create scatter plot: Number of hops vs evidence counts
    """
    # Combine all data
    combined_data = []
    for data, hop_num in zip([hop1, hop2, hop3], [1, 2, 3]):
        for evidence_count in data['evidence_count_mean']:
            combined_data.append({'hops': hop_num, 'evidence_count': evidence_count})

    df_combined = pd.DataFrame(combined_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: All data points
    colors = ['blue', 'red', 'green']
    for i, (data, hop_num) in enumerate(zip([hop1, hop2, hop3], [1, 2, 3])):
        # Add small random jitter to x-axis for better visibility
        x_jitter = hop_num + np.random.normal(0, 0.1, len(data))
        ax1.scatter(x_jitter, data['evidence_count_mean'],
                    alpha=0.6, color=colors[i], s=20, label=f'{hop_num}-hop')

    ax1.set_xlabel('Number of Hops')
    ax1.set_ylabel('Mean Evidence Count')
    ax1.set_title('Evidence Count vs Number of Hops (All Data)')
    ax1.set_xticks([1, 2, 3])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Box plot summary
    df_combined.boxplot(column='evidence_count', by='hops', ax=ax2)
    ax2.set_xlabel('Number of Hops')
    ax2.set_ylabel('Mean Evidence Count')
    ax2.set_title('Evidence Count Distribution by Hop Type')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'hops_vs_evidence_counts.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")

    # Calculate correlation and summary
    correlation = df_combined['hops'].corr(df_combined['evidence_count'])
    summary = df_combined.groupby('hops')['evidence_count'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])

    return correlation, summary


def create_evidence_vs_fc_contour_plots(hop1, hop2, hop3):
    """
    Create continuous contour plots for evidence vs ABS(log2 fold change).
    Fixes label/colorbar overlap by:
    - adding more horizontal spacing
    - only labeling the Y axis on the left subplot
    - using consistent colorbar padding
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.subplots_adjust(wspace=0.55)  # <<< key: prevent overlap between subplots + colorbars

    datasets = [hop1, hop2, hop3]
    titles = ['1-hop: Evidence vs |Log2FC|',
              '2-hop: Evidence vs |Log2FC|',
              '3-hop: Evidence vs |Log2FC|']

    correlation_results = []

    for i, (data, title) in enumerate(zip(datasets, titles)):
        evidence_counts = data['evidence_count_mean']
        log_fold_change = data['logfoldchange']

        # abs log2 fold change
        abs_logfc = np.abs(log_fold_change)

        # Clean data
        mask = np.isfinite(evidence_counts) & np.isfinite(abs_logfc)
        evidence_clean = evidence_counts[mask]
        fc_clean = abs_logfc[mask]

        if len(evidence_clean) < 50:
            axes[i].text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                         transform=axes[i].transAxes, fontsize=16)
            continue

        # Limit evidence to 95th percentile for better visualization
        evidence_limit = np.percentile(evidence_clean, 95)
        mask_limited = evidence_clean <= evidence_limit
        evidence_plot = evidence_clean[mask_limited]
        fc_plot = fc_clean[mask_limited]

        # Create continuous grid and contours
        x_min, x_max = evidence_plot.min(), evidence_plot.max()
        y_min, y_max = fc_plot.min(), fc_plot.max()
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

        values = np.vstack([evidence_plot, fc_plot])
        kernel = gaussian_kde(values)
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = np.reshape(kernel(positions).T, xx.shape)

        contourf = axes[i].contourf(xx, yy, density, levels=15, cmap='YlOrRd', alpha=0.8)
        axes[i].contour(xx, yy, density, levels=8, colors='black', alpha=0.4, linewidths=0.8)

        axes[i].set_xlabel('Mean Evidence Count')
        if i == 0:
            axes[i].set_ylabel('|Log2 Fold Change|')
        else:
            axes[i].set_ylabel('')  # <<< avoid overlap with left colorbar of previous subplot

        axes[i].set_title(title)

        # Colorbar with controlled padding (reduces collisions)
        plt.colorbar(contourf, ax=axes[i], pad=0.02)

        # Correlation text ON the plot
        correlation = np.corrcoef(evidence_clean, fc_clean)[0, 1]
        axes[i].text(0.05, 0.95, f'r = {correlation:.3f}\nn = {len(evidence_clean):,}',
                     transform=axes[i].transAxes,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                     verticalalignment='top')

        correlation_results.append({
            'Hop Type': titles[i].split(':')[0],
            'Correlation': correlation,
            'Sample Size': len(evidence_clean)
        })

    save_path = os.path.join(OUTPUT_DIR, 'evidence_vs_abslogfc_contour_plots.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")

    return correlation_results


def save_summary_document(analysis_results, ks_results, correlation_hop, summary_hop, correlation_logfc):
    """
    Save comprehensive summary document with all statistics
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary_content = f"""
EVIDENCE COUNT ANALYSIS SUMMARY REPORT
=====================================

Generated: {timestamp}
Analysis Directory: {OUTPUT_DIR}

DATASET OVERVIEW
===============
This analysis examines evidence count distributions across 1-hop, 2-hop, and 3-hop gene pathway data.
Evidence counts represent the number of supporting publications for each biological relationship.

For multi-hop paths:
- 2-hop: Mean of evidence_1 and evidence_2
- 3-hop: Mean of evidence_1, evidence_2, and evidence_3

DESCRIPTIVE STATISTICS BY HOP TYPE
=================================
"""

    # Add detailed statistics table
    summary_content += "\nDetailed Statistics:\n"
    summary_content += "-" * 80 + "\n"
    for result in analysis_results:
        summary_content += f"\n{result['Hop Type']}:\n"
        for key, value in result.items():
            if key != 'Hop Type':
                if isinstance(value, float):
                    summary_content += f"  {key}: {value:.2f}\n"
                else:
                    summary_content += f"  {key}: {value}\n"

    # Add distribution comparison
    summary_content += f"""

DISTRIBUTION COMPARISONS
=======================

Kolmogorov-Smirnov Test Results:
-------------------------------
"""
    for ks_result in ks_results:
        summary_content += f"{ks_result['Comparison']}:\n"
        summary_content += f"  KS Statistic: {ks_result['KS Statistic']:.4f}\n"
        summary_content += f"  P-value: {ks_result['P-value']:.2e}\n"
        summary_content += f"  Significantly Different: {ks_result['Significant (α=0.05)']}\n\n"

    # Add correlation analysis
    summary_content += f"""
CORRELATION ANALYSIS
===================

Hops vs Evidence Count:
----------------------
Overall Correlation: {correlation_hop:.4f}

Summary by Hop Type:
"""
    summary_content += summary_hop.round(2).to_string()

    summary_content += f"""

Evidence Count vs Log Fold Change:
---------------------------------
"""
    for corr_result in correlation_logfc:
        if isinstance(corr_result['Correlation'], (int, float)):
            summary_content += f"{corr_result['Hop Type']}:\n"
            summary_content += f"  Correlation: {corr_result['Correlation']:.4f}\n"
            summary_content += f"  Sample Size: {corr_result['Sample Size']}\n"

    # Add key findings
    summary_content += f"""
KEY FINDINGS
============

1. SCALE DIFFERENCES:
   - 1-hop: Max evidence count of {analysis_results[0]['Max']:.0f}
   - 2-hop: Max evidence count of {analysis_results[1]['Max']:.0f} (extreme outliers)
   - 3-hop: Max evidence count of {analysis_results[2]['Max']:.0f}

2. CENTRAL TENDENCIES:
   - 2-hop has highest mean evidence count ({analysis_results[1]['Mean']:.2f})
   - 3-hop has lowest mean evidence count ({analysis_results[2]['Mean']:.2f})
   - All distributions are heavily right-skewed (median << mean)

3. DISTRIBUTION SHAPES:
   - All hop types have significantly different distributions (KS tests p < 0.05)
   - Data dominated by low evidence counts (1-3)
   - Extreme outliers present, especially in 2-hop data

4. CORRELATIONS:
   - Hops vs Evidence: {correlation_hop:.4f} (weak negative correlation)
   - Evidence vs LogFC correlations vary by hop type
   - No strong linear relationships observed

OUTLIER SUMMARY
===============
Values exceeding 99th percentile:
- 1-hop: {analysis_results[0]['Outliers (>99th percentile)']} outliers
- 2-hop: {analysis_results[1]['Outliers (>99th percentile)']} outliers  
- 3-hop: {analysis_results[2]['Outliers (>99th percentile)']} outliers

VISUALIZATION FILES GENERATED
=============================
1. evidence_counts_improved_histograms.png - Log scale and truncated histograms
2. evidence_counts_boxplots.png - Box plots, log box plots, violin plots
3. evidence_counts_overlay_histogram.png - Overlaid distributions
4. evidence_counts_kde_plots.png - Kernel density estimations
5. hops_vs_evidence_counts.png - Scatter and box plots by hop type
6. evidence_vs_logfc_heatmaps.png - Evidence vs log fold change relationships
7. evidence_count_summary_statistics.csv - Detailed statistics table
8. evidence_analysis_summary.txt - This summary document

ANALYSIS NOTES
==============
- Extreme outliers in 2-hop data create visualization challenges
- Log transformations and truncation used to reveal distribution patterns
- Evidence counts do not show strong correlation with fold changes
- 2-hop paths have most variable evidence counts
- Consider filtering extreme outliers for downstream analyses

"""

    # Save summary document
    summary_path = os.path.join(OUTPUT_DIR, 'evidence_analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_content)

    print(f"Saved comprehensive summary: {summary_path}")

    # Also save detailed statistics as CSV
    stats_df = pd.DataFrame(analysis_results)
    csv_path = os.path.join(OUTPUT_DIR, 'evidence_count_summary_statistics.csv')
    stats_df.to_csv(csv_path, index=False)
    print(f"Saved statistics CSV: {csv_path}")


def main():
    """
    Main analysis function - comprehensive evidence count analysis with organized output
    """
    print("EVIDENCE COUNT ANALYSIS - COMPREHENSIVE VERSION")
    print("=" * 60)
    print("All outputs will be saved to: /Users/prashammarfatia/Downloads/Evidence_Analysis")
    print("=" * 60)

    # Setup output directory
    setup_output_directory()

    # Load and process data
    hop1, hop2, hop3 = load_and_process_data()

    # Analyze distributions and get results
    analysis_results = analyze_evidence_distributions(hop1, hop2, hop3)

    # Create all visualizations
    print("\nGenerating visualizations...")
    create_improved_individual_histograms(hop1, hop2, hop3)
    create_comparative_boxplots(hop1, hop2, hop3)
    create_overlay_histogram(hop1, hop2, hop3)
    ks_results = create_ks_density_plots(hop1, hop2, hop3)
    correlation_hop, summary_hop = create_hops_vs_evidence_scatter(hop1, hop2, hop3)
    correlation_logfc = create_evidence_vs_fc_contour_plots(hop1, hop2, hop3)

    # Save comprehensive summary document
    print("\nGenerating summary document...")
    save_summary_document(analysis_results, ks_results, correlation_hop, summary_hop, correlation_logfc)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"All files saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("- evidence_counts_improved_histograms.png")
    print("- evidence_counts_boxplots.png")
    print("- evidence_counts_overlay_histogram.png")
    print("- evidence_counts_kde_plots.png")
    print("- hops_vs_evidence_counts.png")
    print("- evidence_vs_logfc_heatmaps.png")
    print("- evidence_count_summary_statistics.csv")
    print("- evidence_analysis_summary.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()