import pandas as pd
import matplotlib.pyplot as plt

path = "/Users/prashammarfatia/Downloads/indra_2hop_all_perturbations.csv"

# Read with second row as header
df = pd.read_csv(path, skiprows=1, low_memory=False)

# Drop NaN and duplicates
pairs = df[["source", "target"]].dropna().drop_duplicates()

# Count unique targets per source
counts_per_source = pairs.groupby("source")["target"].nunique()

# Make 100-bin histogram
plt.figure(figsize=(12,6))
counts, bins, _ = plt.hist(counts_per_source, bins=100, alpha=0.7, edgecolor='black')

# Add count labels on top of bars
bin_w = bins[1] - bins[0]
for c, left in zip(counts, bins[:-1]):
    if c > 0:
        plt.text(left + bin_w/2, c, str(int(c)),
                 ha='center', va='bottom', fontsize=7, rotation=90)

plt.xlabel('# Descendants explained (2-hop)')
plt.ylabel('# Perturbations')
plt.title('2-hop Explained Descendants per Perturbation (100 bins) — with Counts')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('explained_descendants_2hop_hist_100bins_labels.png', dpi=300)
plt.show()
