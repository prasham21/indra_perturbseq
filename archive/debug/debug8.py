import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ======== Load total descendants ========
desc_counts = pd.read_excel("/Users/prashammarfatia/Downloads/descendant_counts.xlsx")
desc_counts = desc_counts.rename(columns={
    "Gene": "source",
    "Num_Significant_DEGs": "total_descendants"
})

# ======== Load 2-hop INDRA file ========
two_hop_path = "/Users/prashammarfatia/Downloads/indra_2hop_all_perturbations.csv"
df2 = pd.read_csv(two_hop_path, skiprows=1)
df2 = df2[['source', 'target']].dropna().drop_duplicates()

# ======== Count explained targets per source ========
two_hop_counts = df2.groupby('source')['target'].nunique().reset_index()
two_hop_counts = two_hop_counts.rename(columns={'target': 'explained_2hop'})

# ======== Merge with total descendant counts ========
merged_2hop = pd.merge(desc_counts, two_hop_counts, on='source', how='left').fillna(0)

# ======== Compute percentage explained ========
merged_2hop['percent_explained'] = (
    merged_2hop['explained_2hop'] / merged_2hop['total_descendants'] * 100
)

# ======== Plot histogram ========
plt.figure(figsize=(12, 8))
counts, bins, patches = plt.hist(
    merged_2hop['percent_explained'],
    bins=100, alpha=0.7, edgecolor='black', linewidth=0.5
)

# Label each bar with count
for count, bin_left, bin_right in zip(counts, bins[:-1], bins[1:]):
    if count > 0:
        plt.text(
            (bin_left + bin_right) / 2,
            count,
            str(int(count)),
            ha='center', va='bottom', fontsize=7
        )

plt.xlabel('Percentage of Descendants Explained by 2-Hop')
plt.ylabel('Number of Perturbations')
plt.title('Distribution of 2-Hop Coverage Across Perturbations')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('hist_2hop_percent_explained.png', dpi=300, bbox_inches='tight')
plt.show()
