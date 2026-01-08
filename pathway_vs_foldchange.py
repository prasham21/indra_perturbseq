import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("Loading datasets for Number of Pathways vs Fold Change Analysis...")
print("Excluding TP53 perturbations")
print("=" * 70)

# Load 1-hop and 2-hop datasets (excluding TP53)
print("Loading 1-hop excluding TP53 data...")
df_1hop = pd.read_csv("/Users/prashammarfatia/Downloads/indra_1hop_no_v2.csv")
print(f"1-hop excluding TP53: {len(df_1hop):,} pathways")

print("Loading 2-hop excluding TP53 data...")
df_2hop = pd.read_excel("/Users/prashammarfatia/Downloads/indra_2hop_all_perturbations.xlsx")
print(f"2-hop excluding TP53: {len(df_2hop):,} pathways")

print("\nColumn structures:")
print(f"1-hop columns: {list(df_1hop.columns)}")
print(f"2-hop columns: {list(df_2hop.columns)}")


# Standardize column names if needed
def standardize_columns(df, dataset_name):
    """Standardize column names across datasets"""
    if 'belief_1' in df.columns and 'belief' not in df.columns:
        if 'belief_2' in df.columns:
            df['belief'] = (df['belief_1'] + df['belief_2']) / 2
            print(f"Created mean belief from belief_1 and belief_2 in {dataset_name}")
        else:
            df['belief'] = df['belief_1']

    if 'pvalue' in df.columns and 'pval' not in df.columns:
        df = df.rename(columns={'pvalue': 'pval'})
        print(f"Renamed 'pvalue' to 'pval' in {dataset_name}")

    return df


# Standardize columns
df_1hop = standardize_columns(df_1hop, "1-hop")
df_2hop = standardize_columns(df_2hop, "2-hop")

# Add pathway type indicators
df_1hop['pathway_type'] = '1-hop'
df_2hop['pathway_type'] = '2-hop'

print("\n" + "=" * 70)
print("PROCESSING DATA FOR PATHWAY COUNT ANALYSIS")
print("=" * 70)

# Extract essential columns for pathway counting
print("Extracting essential columns...")
df_1hop_subset = df_1hop[['source', 'target', 'logfoldchange', 'pval', 'belief', 'pathway_type']].copy()
df_2hop_subset = df_2hop[['source', 'target', 'logfoldchange', 'pval', 'belief', 'pathway_type']].copy()

print(f"1-hop subset: {len(df_1hop_subset):,} pathways")
print(f"2-hop subset: {len(df_2hop_subset):,} pathways")

# Combine both datasets
print("\nCombining 1-hop and 2-hop datasets...")
combined_pathways = pd.concat([df_1hop_subset, df_2hop_subset], ignore_index=True)
print(f"Combined pathways: {len(combined_pathways):,} total pathways")

# Remove any potential TP53 entries (double-check)
before_tp53_filter = len(combined_pathways)
combined_pathways = combined_pathways[combined_pathways['source'] != 'TP53']
after_tp53_filter = len(combined_pathways)
print(f"Removed {before_tp53_filter - after_tp53_filter} TP53 pathways (if any)")
print(f"Final combined pathways: {after_tp53_filter:,}")

print("\n" + "=" * 70)
print("COUNTING PATHWAYS PER TARGET")
print("=" * 70)

# Count pathways per target gene
print("Counting total pathways per target gene...")
pathway_counts = combined_pathways.groupby('target').agg({
    'source': 'count',  # Count of pathways
    'logfoldchange': 'first',  # Target's fold change (should be same for all pathways to same target)
    'pval': 'first',  # Target's p-value
    'pathway_type': lambda x: list(x)  # List of pathway types for this target
}).reset_index()

# Rename columns for clarity
pathway_counts.columns = ['target', 'pathway_count', 'logfoldchange', 'pval', 'pathway_types']

print(f"Unique targets: {len(pathway_counts):,}")
print(f"Pathway count range: {pathway_counts['pathway_count'].min()} to {pathway_counts['pathway_count'].max()}")

# Add breakdown by pathway type
print("\nCalculating pathway type breakdown per target...")


def count_pathway_types(pathway_list):
    """Count 1-hop and 2-hop pathways"""
    onehop_count = pathway_list.count('1-hop')
    twohop_count = pathway_list.count('2-hop')
    return pd.Series({'onehop_count': onehop_count, 'twohop_count': twohop_count})


pathway_type_counts = pathway_counts['pathway_types'].apply(count_pathway_types)
pathway_counts = pd.concat([pathway_counts, pathway_type_counts], axis=1)

# Add absolute fold change
pathway_counts['abs_logfoldchange'] = np.abs(pathway_counts['logfoldchange'])

print("\nPathway count statistics:")
print(f"  Mean pathways per target: {pathway_counts['pathway_count'].mean():.2f}")
print(f"  Median pathways per target: {pathway_counts['pathway_count'].median():.0f}")
print(f"  Targets with 1-hop pathways: {len(pathway_counts[pathway_counts['onehop_count'] > 0]):,}")
print(f"  Targets with 2-hop pathways: {len(pathway_counts[pathway_counts['twohop_count'] > 0]):,}")
print(
    f"  Targets with both types: {len(pathway_counts[(pathway_counts['onehop_count'] > 0) & (pathway_counts['twohop_count'] > 0)]):,}")

# Show examples of highly connected targets
print("\nTop 10 most connected targets:")
top_targets = pathway_counts.nlargest(10, 'pathway_count')[
    ['target', 'pathway_count', 'abs_logfoldchange', 'onehop_count', 'twohop_count']]
print(top_targets.to_string(index=False))

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

# Calculate correlation between pathway count and fold change
correlation_count_fc = np.corrcoef(pathway_counts['pathway_count'], pathway_counts['abs_logfoldchange'])[0, 1]
correlation_count_pval = np.corrcoef(pathway_counts['pathway_count'], pathway_counts['pval'])[0, 1]

print(f"Correlation Analysis (All data):")
print(f"  Pathway Count vs |LogFC|: r = {correlation_count_fc:.6f}")
print(f"  Pathway Count vs P-value: r = {correlation_count_pval:.6f}")

# Remove outliers - top 1% most connected targets (Option B)
print(f"\nApplying outlier removal (Option B - top 1%)...")
outlier_threshold = pathway_counts['pathway_count'].quantile(0.99)  # Top 1%
outliers = pathway_counts[pathway_counts['pathway_count'] > outlier_threshold]
pathway_counts_filtered = pathway_counts[pathway_counts['pathway_count'] <= outlier_threshold]

print(f"  Outlier threshold: {outlier_threshold:.0f} pathways")
print(f"  Outliers removed: {len(outliers)} targets")
print(f"  Targets remaining: {len(pathway_counts_filtered):,}")
print(f"  Outlier targets: {', '.join(map(str, outliers['target'].head(10).tolist()))}")

# Recalculate correlation after outlier removal
correlation_filtered = \
np.corrcoef(pathway_counts_filtered['pathway_count'], pathway_counts_filtered['abs_logfoldchange'])[0, 1]
print(f"  Correlation after outlier removal: r = {correlation_filtered:.6f}")

# Statistical significance test
try:
    from scipy import stats

    r_stat, r_pval = stats.pearsonr(pathway_counts_filtered['pathway_count'],
                                    pathway_counts_filtered['abs_logfoldchange'])
    print(f"  Statistical significance (filtered): p = {r_pval:.2e}")
except ImportError:
    print("  (scipy not available for significance testing)")

print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

# Create comprehensive visualization with log scale and outlier removal
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Number of Pathways vs Fold Change Analysis (Excluding TP53)', fontsize=16, fontweight='bold')

# Plot 1: Main scatter plot with log scale - Pathway Count vs |Fold Change| (filtered data)
ax1 = axes[0, 0]
ax1.scatter(pathway_counts_filtered['abs_logfoldchange'], pathway_counts_filtered['pathway_count'],
            alpha=0.6, s=30, color='darkblue')
ax1.set_xlabel('|Log Fold Change|')
ax1.set_ylabel('Number of Pathways (log scale)')
ax1.set_yscale('log')
ax1.set_title(
    f'Pathway Count vs |LogFC| (Filtered)\n(r = {correlation_filtered:.4f}, n = {len(pathway_counts_filtered):,})')
ax1.grid(True, alpha=0.3)

# Add reference lines
ax1.axhline(y=10, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax1.axhline(y=100, color='orange', linestyle='--', alpha=0.5, linewidth=1)
ax1.text(0.02, 10, '10 pathways', transform=ax1.get_yaxis_transform(), fontsize=8, color='red')
ax1.text(0.02, 100, '100 pathways', transform=ax1.get_yaxis_transform(), fontsize=8, color='orange')

# Add trend line
z = np.polyfit(pathway_counts_filtered['abs_logfoldchange'], np.log10(pathway_counts_filtered['pathway_count']), 1)
x_trend = np.linspace(pathway_counts_filtered['abs_logfoldchange'].min(),
                      pathway_counts_filtered['abs_logfoldchange'].max(), 100)
y_trend = 10 ** (z[0] * x_trend + z[1])
ax1.plot(x_trend, y_trend, "r-", alpha=0.8, linewidth=2, label='Trend line')

# Plot 2: Heat map of pathway count vs fold change (filtered data)
ax2 = axes[0, 1]
# Use log scale for binning
log_pathway_counts = np.log10(pathway_counts_filtered['pathway_count'])
h = ax2.hist2d(pathway_counts_filtered['abs_logfoldchange'], log_pathway_counts,
               bins=50, cmap='Blues')
plt.colorbar(h[3], ax=ax2)
ax2.set_xlabel('|Log Fold Change|')
ax2.set_ylabel('Log10(Number of Pathways)')
ax2.set_title('Density Plot (Heat Map - Filtered)')

# Convert y-axis labels back to original scale
y_ticks = ax2.get_yticks()
ax2.set_yticklabels([f'{10 ** y:.0f}' if y >= 0 else '1' for y in y_ticks])

# Plot 3: Connectivity categories comparison
ax3 = axes[0, 2]
# Use filtered data for categories
low_count_filt = pathway_counts_filtered[pathway_counts_filtered['pathway_count'] <= 5]
medium_count_filt = pathway_counts_filtered[
    (pathway_counts_filtered['pathway_count'] > 5) & (pathway_counts_filtered['pathway_count'] <= 20)]
high_count_filt = pathway_counts_filtered[pathway_counts_filtered['pathway_count'] > 20]

categories = ['Low\n(≤5)', 'Medium\n(6-20)', 'High\n(>20)']
means = [
    low_count_filt['abs_logfoldchange'].mean() if len(low_count_filt) > 0 else 0,
    medium_count_filt['abs_logfoldchange'].mean() if len(medium_count_filt) > 0 else 0,
    high_count_filt['abs_logfoldchange'].mean() if len(high_count_filt) > 0 else 0
]
counts = [len(low_count_filt), len(medium_count_filt), len(high_count_filt)]

bars = ax3.bar(categories, means, color=['lightcoral', 'gold', 'lightblue'], alpha=0.7, edgecolor='black')
ax3.set_xlabel('Connectivity Level')
ax3.set_ylabel('Mean |Log Fold Change|')
ax3.set_title('Mean |LogFC| by Connectivity')
ax3.grid(True, alpha=0.3, axis='y')

# Add count labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'n={count:,}', ha='center', va='bottom', fontsize=9)

# Plot 4: Distribution of pathway counts (filtered, log scale)
ax4 = axes[1, 0]
ax4.hist(pathway_counts_filtered['pathway_count'], bins=50, alpha=0.7, color='green', edgecolor='black')
ax4.set_xlabel('Number of Pathways per Target')
ax4.set_ylabel('Count of Targets')
ax4.set_xscale('log')
ax4.set_title(
    f'Distribution of Pathway Counts (Filtered)\nRange: {pathway_counts_filtered["pathway_count"].min()}-{pathway_counts_filtered["pathway_count"].max()}')
ax4.grid(True, alpha=0.3)

# Plot 5: Outliers information
ax5 = axes[1, 1]
if len(outliers) > 0:
    ax5.scatter(outliers['abs_logfoldchange'], outliers['pathway_count'],
                alpha=0.8, s=60, color='red', edgecolor='black')
    ax5.set_xlabel('|Log Fold Change|')
    ax5.set_ylabel('Number of Pathways')
    ax5.set_title(f'Outliers Removed (Top 1%)\nn = {len(outliers)}')
    ax5.grid(True, alpha=0.3)

    # Add target labels for outliers
    for _, row in outliers.head(5).iterrows():  # Label top 5 outliers
        ax5.annotate(row['target'], (row['abs_logfoldchange'], row['pathway_count']),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)
else:
    ax5.text(0.5, 0.5, 'No outliers\nidentified', ha='center', va='center',
             transform=ax5.transAxes, fontsize=12)
    ax5.set_title('Outliers')

# Plot 6: Summary statistics
ax6 = axes[1, 2]
ax6.axis('off')

summary_text = f"""PATHWAY COUNT ANALYSIS SUMMARY

ORIGINAL DATA:
• Total Targets: {len(pathway_counts):,}
• Total Pathways: {after_tp53_filter:,}

AFTER OUTLIER REMOVAL (Top 1%):
• Targets analyzed: {len(pathway_counts_filtered):,}
• Outliers removed: {len(outliers)}
• Outlier threshold: >{outlier_threshold:.0f} pathways

CORRELATION RESULTS:
• Original: r = {correlation_count_fc:.4f}
• Filtered: r = {correlation_filtered:.4f}

CONNECTIVITY BREAKDOWN (Filtered):
• Low (≤5): {len(low_count_filt):,} targets
  Mean |LogFC|: {means[0]:.3f}
• Medium (6-20): {len(medium_count_filt):,} targets  
  Mean |LogFC|: {means[1]:.3f}
• High (>20): {len(high_count_filt):,} targets
  Mean |LogFC|: {means[2]:.3f}

HYPOTHESIS TEST:
{"Strong NEGATIVE correlation" if correlation_filtered < -0.05 else "Weak/no relationship" if abs(correlation_filtered) < 0.05 else "Positive correlation"}
Well-connected targets show {"WEAKER" if correlation_filtered < 0 else "STRONGER"} responses
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('pathway_count_vs_foldchange_improved.png', dpi=300, bbox_inches='tight')
plt.show(block=False)
plt.pause(0.1)

print("Improved visualization complete! Saved as 'pathway_count_vs_foldchange_improved.png'")

print("\n" + "=" * 70)
print("DETAILED ANALYSIS RESULTS")
print("=" * 70)

# Detailed breakdown using filtered data
print(f"\nDetailed Results (After Outlier Removal):")
print(f"  Targets analyzed: {len(pathway_counts_filtered):,}")
print(f"  Outliers removed: {len(outliers)} targets")
print(f"  Average pathways per target: {pathway_counts_filtered['pathway_count'].mean():.2f}")
print(f"  Standard deviation: {pathway_counts_filtered['pathway_count'].std():.2f}")

# Recalculate categories with filtered data
low_count_filt = pathway_counts_filtered[pathway_counts_filtered['pathway_count'] <= 5]
medium_count_filt = pathway_counts_filtered[
    (pathway_counts_filtered['pathway_count'] > 5) & (pathway_counts_filtered['pathway_count'] <= 20)]
high_count_filt = pathway_counts_filtered[pathway_counts_filtered['pathway_count'] > 20]

print(f"\nTargets by connectivity (Filtered data):")
print(f"  Low connectivity (≤5 pathways): {len(low_count_filt):,} targets")
if len(low_count_filt) > 0:
    print(f"    Mean |LogFC|: {low_count_filt['abs_logfoldchange'].mean():.4f}")
print(f"  Medium connectivity (6-20 pathways): {len(medium_count_filt):,} targets")
if len(medium_count_filt) > 0:
    print(f"    Mean |LogFC|: {medium_count_filt['abs_logfoldchange'].mean():.4f}")
print(f"  High connectivity (>20 pathways): {len(high_count_filt):,} targets")
if len(high_count_filt) > 0:
    print(f"    Mean |LogFC|: {high_count_filt['abs_logfoldchange'].mean():.4f}")

# Statistical comparison using filtered data
if len(low_count_filt) > 0 and len(high_count_filt) > 0:
    try:
        from scipy import stats

        t_stat, t_pval = stats.ttest_ind(high_count_filt['abs_logfoldchange'], low_count_filt['abs_logfoldchange'])
        print(f"\nStatistical comparison (High vs Low connectivity - Filtered):")
        print(
            f"  Mean |LogFC| difference: {high_count_filt['abs_logfoldchange'].mean() - low_count_filt['abs_logfoldchange'].mean():.4f}")
        print(f"  T-test p-value: {t_pval:.2e}")
    except ImportError:
        print(
            f"\nMean |LogFC| difference (High vs Low - Filtered): {high_count_filt['abs_logfoldchange'].mean() - low_count_filt['abs_logfoldchange'].mean():.4f}")

# Show outlier information
if len(outliers) > 0:
    print(f"\nOutliers removed (Top 1% - >{outlier_threshold:.0f} pathways):")
    outlier_display = outliers[['target', 'pathway_count', 'abs_logfoldchange']].copy()
    outlier_display = outlier_display.sort_values('pathway_count', ascending=False)
    print(outlier_display.to_string(index=False, max_rows=10))

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
print("Number of Pathways vs Fold Change analysis finished.")
print("Results show the relationship between target connectivity and experimental response strength.")