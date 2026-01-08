import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def load_data():
    hop1 = pd.read_csv('/Users/prashammarfatia/Downloads/indra_1hop_gene_names_cleaned.csv')
    hop2 = pd.read_csv('/Users/prashammarfatia/Downloads/indra_2hop_with_evidence_statements_.csv')
    hop3 = pd.read_csv('/Users/prashammarfatia/Downloads/indra_3hop_cleaned_results.csv')
    return hop1, hop2, hop3


def bin_fc_analysis(hop1, hop2, hop3):
    # Combine all fold changes
    all_fc = []
    all_fc.extend(hop1['logfoldchange'].dropna())
    all_fc.extend(hop2['logfoldchange'].dropna())
    all_fc.extend(hop3['logfoldchange'].dropna())

    # Create 10 bins
    bins = pd.cut(all_fc, bins=10, retbins=True)[1]

    # For each bin, count how many are explained by each hop type
    results = []
    for i in range(len(bins) - 1):
        bin_start, bin_end = bins[i], bins[i + 1]

        hop1_in_bin = len(hop1[(hop1['logfoldchange'] >= bin_start) &
                               (hop1['logfoldchange'] < bin_end)])
        hop2_in_bin = len(hop2[(hop2['logfoldchange'] >= bin_start) &
                               (hop2['logfoldchange'] < bin_end)])
        hop3_in_bin = len(hop3[(hop3['logfoldchange'] >= bin_start) &
                               (hop3['logfoldchange'] < bin_end)])

        total = hop1_in_bin + hop2_in_bin + hop3_in_bin

        if total > 0:
            pct_1hop = (hop1_in_bin / total) * 100
            pct_2hop = (hop2_in_bin / total) * 100
            pct_3hop = (hop3_in_bin / total) * 100
        else:
            pct_1hop = pct_2hop = pct_3hop = 0

        results.append({
            'Bin': i + 1,
            'FC_Range': f'[{bin_start:.2f}, {bin_end:.2f}]',
            'Total': total,
            '1-hop_count': hop1_in_bin,
            '1-hop_%': pct_1hop,
            '2-hop_count': hop2_in_bin,
            '2-hop_%': pct_2hop,
            '3-hop_count': hop3_in_bin,
            '3-hop_%': pct_3hop
        })

    return pd.DataFrame(results), bins


def plot_combined_results(results, bins):
    """Plot combined stacked bar chart with bin ranges"""
    fig, ax = plt.subplots(figsize=(14, 8))

    x = range(len(results))
    ax.bar(x, results['1-hop_%'], label='1-hop', alpha=0.7)
    ax.bar(x, results['2-hop_%'], bottom=results['1-hop_%'], label='2-hop', alpha=0.7)
    ax.bar(x, results['3-hop_%'], bottom=results['1-hop_%'] + results['2-hop_%'],
           label='3-hop', alpha=0.7)

    ax.set_xlabel('Fold Change Bins')
    ax.set_ylabel('Percentage')
    ax.set_title('What % of each FC bin is explained by 1-hop vs 2-hop vs 3-hop')
    ax.legend()

    # Create x-axis labels with bin ranges
    bin_labels = []
    for i in range(len(bins) - 1):
        bin_labels.append(f'Bin {i + 1}\n[{bins[i]:.2f}, {bins[i + 1]:.2f}]')

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('/Users/prashammarfatia/Downloads/Evidence_Analysis/fc_bin_combined.png', dpi=300)
    plt.show()


def plot_individual_results(results, bins):
    """Plot separate bar charts for each hop type with proper y-axis spacing"""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))  # Increased figure width

    hop_types = ['1-hop', '2-hop', '3-hop']
    colors = ['skyblue', 'lightcoral', 'lightgreen']

    x = range(len(results))

    for i, (hop_type, color) in enumerate(zip(hop_types, colors)):
        bars = axes[i].bar(x, results[f'{hop_type}_%'], color=color, alpha=0.7)
        axes[i].set_xlabel('Fold Change Bins')
        axes[i].set_ylabel('Percentage of Bin')
        axes[i].set_title(f'{hop_type}: % of each FC bin')
        axes[i].grid(True, alpha=0.3, axis='y')

        # Move y-axis label to avoid overlap
        axes[i].yaxis.set_label_coords(-0.12, 0.5)  # Move y-label further left

        # Create x-axis labels with bin ranges
        bin_labels = []
        for j in range(len(bins) - 1):
            bin_labels.append(f'Bin {j + 1}\n[{bins[j]:.2f}, {bins[j + 1]:.2f}]')

        axes[i].set_xticks(x)
        axes[i].set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)

        # Add count annotations with proper positioning
        max_height = max(results[f'{hop_type}_%']) if max(results[f'{hop_type}_%']) > 0 else 1
        y_offset = max_height * 0.02

        for j, (pct, count) in enumerate(zip(results[f'{hop_type}_%'], results[f'{hop_type}_count'])):
            if pct > 0:
                text_y = pct + y_offset
                axes[i].text(j, text_y, f'{count}', ha='center', va='bottom', fontsize=8)

        # Set y-axis limit to accommodate text
        axes[i].set_ylim(0, max_height * 1.15)

    # Adjust margins and spacing
    plt.subplots_adjust(left=0.1, right=0.98, bottom=0.15, top=0.9, wspace=0.35)

    plt.savefig('/Users/prashammarfatia/Downloads/Evidence_Analysis/fc_bin_individual.png', dpi=300,
                bbox_inches='tight')
    plt.show()


def main():
    hop1, hop2, hop3 = load_data()
    results, bins = bin_fc_analysis(hop1, hop2, hop3)

    print("Fold Change Bin Analysis:")
    print(results.round(1))

    results.to_csv('/Users/prashammarfatia/Downloads/Evidence_Analysis/fc_bin_results.csv', index=False)

    # Plot combined results
    plot_combined_results(results, bins)

    # Plot individual results
    plot_individual_results(results, bins)

    print("Files saved:")
    print("- fc_bin_combined.png (stacked bar chart with bin ranges)")
    print("- fc_bin_individual.png (separate plots with bin ranges)")
    print("- fc_bin_results.csv")


if __name__ == "__main__":
    main()
