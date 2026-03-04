import os
import pandas as pd
import numpy as np

# === LOAD SOURCES ===
perturb_df = pd.read_csv("/Users/prashammarfatia/Downloads/target_validation_expanded.csv")
perturb_df = perturb_df[perturb_df['Karen_Flag'] == "Use_for_analysis"]

# EXCLUDE TP53 and CDKN1A (outliers, no DE files)
excluded_genes = ["TP53", "CDKN1A"]
perturb_df = perturb_df[~perturb_df["Gene"].isin(excluded_genes)]

source_genes = perturb_df["Gene"].tolist()
print(f"✅ {len(source_genes)} source genes loaded (excluded: {excluded_genes})")
print(f"   Sources: {source_genes}")

# === LOAD ALL TARGETS WITH THEIR DATA ===
all_targets = []
de_folder = "/Users/prashammarfatia/Downloads/de_results_per_gene/"

for source_gene in source_genes:
    csv_path = os.path.join(de_folder, f"{source_gene}_vs_control.csv")
    if not os.path.exists(csv_path):
        print(f"⚠️ Missing DE file for {source_gene} (skipping)")
        continue

    df = pd.read_csv(csv_path)
    # Filter by pval < 0.05
    df = df[df["pvals"] < 0.05]

    print(f"   {source_gene}: {len(df)} targets with pval < 0.05")

    for _, row in df.iterrows():
        all_targets.append({
            "original_source": source_gene,
            "target": row["names"],
            "logfoldchange": row["logfoldchanges"],
            "pval": row["pvals"]
        })

targets_df = pd.DataFrame(all_targets)
print(f"\n✅ Total: {len(targets_df)} source-target pairs loaded")
print(f"   Unique sources with targets: {targets_df['original_source'].nunique()}")
print(f"   Unique targets: {targets_df['target'].nunique()}")

# === SHUFFLE SOURCES (ONLY FROM THE ALLOWED SOURCE GENES) ===
np.random.seed(42)  # Fixed seed for reproducibility

# Create a list of sources to shuffle from (only the valid source genes, repeated for each target)
available_sources = source_genes.copy()
shuffled_sources = []

# For each target, randomly assign a source from the available pool
for _ in range(len(targets_df)):
    shuffled_sources.append(np.random.choice(available_sources))

targets_df["permuted_source"] = shuffled_sources

print(f"\n🔀 Sources shuffled (seed=42)")
print(f"   TP53 and CDKN1A will NOT appear as permuted sources")

# === REMOVE ACCIDENTAL TRUE MATCHES ===
true_pairs = set(zip(targets_df["original_source"], targets_df["target"]))
targets_df["is_true_pair"] = targets_df.apply(
    lambda row: (row["permuted_source"], row["target"]) in true_pairs, axis=1
)

before_removal = len(targets_df)
targets_df_filtered = targets_df[~targets_df["is_true_pair"]].copy()
after_removal = len(targets_df_filtered)

print(f"✓ Removed {before_removal - after_removal} accidental true matches")

# === VERIFY NO TP53 OR CDKN1A AS PERMUTED SOURCES ===
tp53_count = (targets_df_filtered["permuted_source"] == "TP53").sum()
cdkn1a_count = (targets_df_filtered["permuted_source"] == "CDKN1A").sum()
print(f"\n✓ Verification:")
print(f"   TP53 as permuted source: {tp53_count} (should be 0)")
print(f"   CDKN1A as permuted source: {cdkn1a_count} (should be 0)")

# === SAVE MASTER PERMUTED DATASET ===
permuted_pairs = targets_df_filtered[["permuted_source", "target", "logfoldchange", "pval", "original_source"]].copy()
permuted_pairs.columns = ["source", "target", "logfoldchange", "pval", "original_source"]
permuted_pairs.to_csv("/Users/prashammarfatia/Downloads/MASTER_permuted_source_target_pairs.csv", index=False)

print(f"\n✅ Saved master permuted dataset: MASTER_permuted_source_target_pairs.csv")
print(f"   Rows: {len(permuted_pairs)}")
print(f"   Unique permuted sources: {permuted_pairs['source'].nunique()}")
print(f"   Unique targets: {permuted_pairs['target'].nunique()}")

# === SAVE SUMMARY STATISTICS ===
summary = {
    "total_pairs": len(permuted_pairs),
    "unique_sources": permuted_pairs['source'].nunique(),
    "unique_targets": permuted_pairs['target'].nunique(),
    "accidental_matches_removed": before_removal - after_removal,
    "excluded_genes": excluded_genes,
    "sources_used": source_genes
}

with open("/Users/prashammarfatia/Downloads/permutation_summary.txt", "w") as f:
    f.write("PERMUTATION SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total permuted pairs: {summary['total_pairs']}\n")
    f.write(f"Unique permuted sources: {summary['unique_sources']}\n")
    f.write(f"Unique targets: {summary['unique_targets']}\n")
    f.write(f"Accidental true matches removed: {summary['accidental_matches_removed']}\n")
    f.write(f"\nExcluded genes (outliers): {', '.join(excluded_genes)}\n")
    f.write(f"\nSource genes used for permutation:\n")
    for gene in summary['sources_used']:
        f.write(f"  - {gene}\n")

print(f"✅ Saved summary: permutation_summary.txt")

# === SHOW SAMPLE COMPARISON ===
print("\n" + "=" * 80)
print("📋 SAMPLE: Real vs Permuted Pairings")
print("=" * 80)
sample = targets_df_filtered[["original_source", "permuted_source", "target", "logfoldchange", "pval"]].head(15)
sample.columns = ["Real Source", "Permuted Source", "Target", "LogFC", "P-value"]
print(sample.to_string(index=False))
print("=" * 80)

# === PERMUTED SOURCE DISTRIBUTION ===
grouped = permuted_pairs.groupby("source")

print(f"\n📊 Permuted sources distribution:")
for source in sorted(grouped.groups.keys()):
    count = len(grouped.get_group(source))
    print(f"   {source}: {count} targets")

print(f"\n🎯 This master file can now be used for ALL hop analyses (1-hop, 2-hop, 3-hop, 4-hop)")
print(f"   TP53 and CDKN1A are completely excluded from permuted sources")