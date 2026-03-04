"""Evidence count distribution analysis across 1-hop, 2-hop, and 3-hop pathways."""
from __future__ import annotations

import argparse
import logging
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp, gaussian_kde

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_and_process_data(hop1_file, hop2_file, hop3_file):
    """Load all three datasets and calculate mean evidence counts."""
    logger.info("Loading datasets")

    hop1 = pd.read_csv(hop1_file)
    hop1['hop_type'] = '1-hop'
    hop1['evidence_count_mean'] = hop1['evidence_count']
    hop1['num_hops'] = 1

    hop2 = pd.read_csv(hop2_file)
    hop2['hop_type'] = '2-hop'
    hop2['evidence_count_mean'] = (hop2['evidence_1'] + hop2['evidence_2']) / 2
    hop2['num_hops'] = 2

    hop3 = pd.read_csv(hop3_file)
    hop3['hop_type'] = '3-hop'
    hop3['evidence_count_mean'] = (hop3['evidence_1'] + hop3['evidence_2'] + hop3['evidence_3']) / 3
    hop3['num_hops'] = 3

    logger.info("1-hop: %d rows, 2-hop: %d rows, 3-hop: %d rows", len(hop1), len(hop2), len(hop3))

    return hop1, hop2, hop3


def analyze_evidence_distributions(hop1, hop2, hop3):
    """Analyze evidence count distributions and identify outliers."""
    datasets = [hop1, hop2, hop3]
    names = ['1-hop', '2-hop', '3-hop']

    logger.info("Evidence count distribution analysis")

    analysis_results = []
    for data, name in zip(datasets, names):
        evidence_counts = data['evidence_count_mean']
        outlier_threshold = evidence_counts.quantile(0.99)
        outliers = evidence_counts[evidence_counts > outlier_threshold]

        stats_dict = {
            'Hop Type': name, 'Count': len(evidence_counts),
            'Mean': evidence_counts.mean(), 'Median': evidence_counts.median(),
            'Std Dev': evidence_counts.std(), 'Min': evidence_counts.min(),
            'Max': evidence_counts.max(),
            '25th Percentile': evidence_counts.quantile(0.25),
            '75th Percentile': evidence_counts.quantile(0.75),
            '90th Percentile': evidence_counts.quantile(0.90),
            '95th Percentile': evidence_counts.quantile(0.95),
            '99th Percentile': evidence_counts.quantile(0.99),
            'Outliers (>99th percentile)': len(outliers),
            'Outlier Min': outliers.min() if len(outliers) > 0 else 'N/A',
            'Outlier Max': outliers.max() if len(outliers) > 0 else 'N/A',
        }
        analysis_results.append(stats_dict)
        logger.info("%s: Count=%d, Mean=%.2f, Median=%.2f, Max=%.2f",
                     name, stats_dict['Count'], stats_dict['Mean'],
                     stats_dict['Median'], stats_dict['Max'])

    return analysis_results


def create_improved_individual_histograms(hop1, hop2, hop3, output_dir):
    """Create histograms that show meaningful patterns."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    datasets = [hop1, hop2, hop3]
    titles = ['1-hop Evidence Counts', '2-hop Evidence Counts', '3-hop Evidence Counts']
    colors = ['skyblue', 'lightcoral', 'lightgreen']

    for i, (data, title, color) in enumerate(zip(datasets, titles, colors)):
        evidence_counts = data['evidence_count_mean']
        log_evidence = np.log10(evidence_counts + 1)
        axes[0, i].hist(log_evidence, bins=30, alpha=0.7, color=color, edgecolor='black')
        axes[0, i].set_xlabel('Log10(Evidence Count + 1)')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].set_title(f'{title} (Log Scale)')
        axes[0, i].grid(True, alpha=0.3)
        stats_text = f'n={len(evidence_counts)}\nMean: {evidence_counts.mean():.1f}\nMedian: {evidence_counts.median():.1f}'
        axes[0, i].text(0.7, 0.8, stats_text, transform=axes[0, i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    for i, (data, title, color) in enumerate(zip(datasets, titles, colors)):
        evidence_counts = data['evidence_count_mean']
        cutoff = evidence_counts.quantile(0.90)
        truncated_data = evidence_counts[evidence_counts <= cutoff]
        excluded_count = len(evidence_counts) - len(truncated_data)
        axes[1, i].hist(truncated_data, bins=20, alpha=0.7, color=color, edgecolor='black')
        axes[1, i].set_xlabel('Mean Evidence Count')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].set_title(f'{title} (<=90th %ile)')
        axes[1, i].grid(True, alpha=0.3)
        stats_text = f'Showing: {len(truncated_data)}\nExcluded: {excluded_count}\nRange: {truncated_data.min():.1f}-{truncated_data.max():.1f}'
        axes[1, i].text(0.6, 0.8, stats_text, transform=axes[1, i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'evidence_counts_improved_histograms.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info("Saved: %s", save_path)


def create_comparative_boxplots(hop1, hop2, hop3, output_dir):
    """Create box plots for comparison across hop types."""
    combined_data = []
    for data, hop_type in zip([hop1, hop2, hop3], ['1-hop', '2-hop', '3-hop']):
        for evidence_count in data['evidence_count_mean']:
            combined_data.append({'hop_type': hop_type, 'evidence_count': evidence_count})
    df_combined = pd.DataFrame(combined_data)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.boxplot(data=df_combined, x='hop_type', y='evidence_count', ax=axes[0])
    axes[0].set_title('Evidence Count Distribution by Hop Type')
    axes[0].set_ylabel('Mean Evidence Count')
    axes[0].grid(True, alpha=0.3)

    df_combined['log_evidence_count'] = np.log10(df_combined['evidence_count'] + 1)
    sns.boxplot(data=df_combined, x='hop_type', y='log_evidence_count', ax=axes[1])
    axes[1].set_title('Evidence Count Distribution (Log Scale)')
    axes[1].set_ylabel('Log10(Evidence Count + 1)')
    axes[1].grid(True, alpha=0.3)

    sns.violinplot(data=df_combined[df_combined['evidence_count'] <= df_combined['evidence_count'].quantile(0.95)],
                   x='hop_type', y='evidence_count', ax=axes[2])
    axes[2].set_title('Evidence Count Distribution (<=95th %ile)')
    axes[2].set_ylabel('Mean Evidence Count')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'evidence_counts_boxplots.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info("Saved: %s", save_path)


def create_overlay_histogram(hop1, hop2, hop3, output_dir):
    """Create overlay histogram with better separation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
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

    max_reasonable = np.percentile(np.concatenate(datasets), 95)
    for data, label, color in zip(datasets, labels, colors):
        filtered_data = data[data <= max_reasonable]
        ax2.hist(filtered_data, bins=50, alpha=0.6, label=f'{label} (<=95th %ile)', color=color, density=True)
    ax2.set_xlabel('Mean Evidence Count')
    ax2.set_ylabel('Density')
    ax2.set_title('Evidence Count Distributions: Excluding Top 5% Outliers')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'evidence_counts_overlay_histogram.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info("Saved: %s", save_path)


def create_ks_density_plots(hop1, hop2, hop3, output_dir):
    """Create KS density plots for smooth comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    datasets = [hop1['evidence_count_mean'], hop2['evidence_count_mean'], hop3['evidence_count_mean']]
    labels = ['1-hop', '2-hop', '3-hop']
    colors = ['blue', 'red', 'green']

    for data, label, color in zip(datasets, labels, colors):
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
    save_path = os.path.join(output_dir, 'evidence_counts_kde_plots.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info("Saved: %s", save_path)

    ks_results = []
    pairs = [(0, 1, '1-hop vs 2-hop'), (0, 2, '1-hop vs 3-hop'), (1, 2, '2-hop vs 3-hop')]
    for i, j, name in pairs:
        ks_stat, p_value = ks_2samp(datasets[i], datasets[j])
        ks_results.append({
            'Comparison': name, 'KS Statistic': ks_stat,
            'P-value': p_value, 'Significant (alpha=0.05)': 'Yes' if p_value < 0.05 else 'No'
        })
    return ks_results


def create_hops_vs_evidence_scatter(hop1, hop2, hop3, output_dir):
    """Create scatter plot: Number of hops vs evidence counts."""
    combined_data = []
    for data, hop_num in zip([hop1, hop2, hop3], [1, 2, 3]):
        for evidence_count in data['evidence_count_mean']:
            combined_data.append({'hops': hop_num, 'evidence_count': evidence_count})
    df_combined = pd.DataFrame(combined_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['blue', 'red', 'green']
    for i, (data, hop_num) in enumerate(zip([hop1, hop2, hop3], [1, 2, 3])):
        x_jitter = hop_num + np.random.normal(0, 0.1, len(data))
        ax1.scatter(x_jitter, data['evidence_count_mean'],
                    alpha=0.6, color=colors[i], s=20, label=f'{hop_num}-hop')
    ax1.set_xlabel('Number of Hops')
    ax1.set_ylabel('Mean Evidence Count')
    ax1.set_title('Evidence Count vs Number of Hops (All Data)')
    ax1.set_xticks([1, 2, 3])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    df_combined.boxplot(column='evidence_count', by='hops', ax=ax2)
    ax2.set_xlabel('Number of Hops')
    ax2.set_ylabel('Mean Evidence Count')
    ax2.set_title('Evidence Count Distribution by Hop Type')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'hops_vs_evidence_counts.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info("Saved: %s", save_path)

    correlation = df_combined['hops'].corr(df_combined['evidence_count'])
    summary = df_combined.groupby('hops')['evidence_count'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    return correlation, summary


def create_evidence_vs_fc_contour_plots(hop1, hop2, hop3, output_dir):
    """Create continuous contour plots for evidence vs ABS(log2 fold change)."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.subplots_adjust(wspace=0.55)

    datasets = [hop1, hop2, hop3]
    titles = ['1-hop: Evidence vs |Log2FC|', '2-hop: Evidence vs |Log2FC|', '3-hop: Evidence vs |Log2FC|']
    correlation_results = []

    for i, (data, title) in enumerate(zip(datasets, titles)):
        evidence_counts = data['evidence_count_mean']
        abs_logfc = np.abs(data['logfoldchange'])

        mask = np.isfinite(evidence_counts) & np.isfinite(abs_logfc)
        evidence_clean = evidence_counts[mask]
        fc_clean = abs_logfc[mask]

        if len(evidence_clean) < 50:
            axes[i].text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                         transform=axes[i].transAxes, fontsize=16)
            continue

        evidence_limit = np.percentile(evidence_clean, 95)
        mask_limited = evidence_clean <= evidence_limit
        evidence_plot = evidence_clean[mask_limited]
        fc_plot = fc_clean[mask_limited]

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
            axes[i].set_ylabel('')
        axes[i].set_title(title)
        plt.colorbar(contourf, ax=axes[i], pad=0.02)

        correlation = np.corrcoef(evidence_clean, fc_clean)[0, 1]
        axes[i].text(0.05, 0.95, f'r = {correlation:.3f}\nn = {len(evidence_clean):,}',
                     transform=axes[i].transAxes,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                     verticalalignment='top')
        correlation_results.append({
            'Hop Type': titles[i].split(':')[0], 'Correlation': correlation,
            'Sample Size': len(evidence_clean)
        })

    save_path = os.path.join(output_dir, 'evidence_vs_abslogfc_contour_plots.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info("Saved: %s", save_path)

    return correlation_results


def save_summary_document(analysis_results, ks_results, correlation_hop, summary_hop,
                          correlation_logfc, output_dir):
    """Save comprehensive summary document with all statistics."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_content = f"EVIDENCE COUNT ANALYSIS SUMMARY REPORT\nGenerated: {timestamp}\n\n"

    for result in analysis_results:
        summary_content += f"\n{result['Hop Type']}:\n"
        for key, value in result.items():
            if key != 'Hop Type':
                summary_content += f"  {key}: {value:.2f}\n" if isinstance(value, float) else f"  {key}: {value}\n"

    summary_content += f"\nHops vs Evidence Correlation: {correlation_hop:.4f}\n"

    summary_path = os.path.join(output_dir, 'evidence_analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_content)
    logger.info("Saved comprehensive summary: %s", summary_path)

    stats_df = pd.DataFrame(analysis_results)
    csv_path = os.path.join(output_dir, 'evidence_count_summary_statistics.csv')
    stats_df.to_csv(csv_path, index=False)
    logger.info("Saved statistics CSV: %s", csv_path)


def main():
    parser = argparse.ArgumentParser(description="Evidence count analysis")
    parser.add_argument("--hop1-file", required=True, help="Path to 1-hop CSV with evidence_count")
    parser.add_argument("--hop2-file", required=True, help="Path to 2-hop CSV with evidence_1/evidence_2")
    parser.add_argument("--hop3-file", required=True, help="Path to 3-hop CSV with evidence columns")
    parser.add_argument("--output-dir", required=True, help="Output directory for all results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("All outputs will be saved to: %s", args.output_dir)

    hop1, hop2, hop3 = load_and_process_data(args.hop1_file, args.hop2_file, args.hop3_file)
    analysis_results = analyze_evidence_distributions(hop1, hop2, hop3)

    logger.info("Generating visualizations")
    create_improved_individual_histograms(hop1, hop2, hop3, args.output_dir)
    create_comparative_boxplots(hop1, hop2, hop3, args.output_dir)
    create_overlay_histogram(hop1, hop2, hop3, args.output_dir)
    ks_results = create_ks_density_plots(hop1, hop2, hop3, args.output_dir)
    correlation_hop, summary_hop = create_hops_vs_evidence_scatter(hop1, hop2, hop3, args.output_dir)
    correlation_logfc = create_evidence_vs_fc_contour_plots(hop1, hop2, hop3, args.output_dir)

    logger.info("Generating summary document")
    save_summary_document(analysis_results, ks_results, correlation_hop, summary_hop,
                          correlation_logfc, args.output_dir)

    logger.info("Analysis complete. All files saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
