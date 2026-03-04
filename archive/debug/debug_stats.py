import pandas as pd
from pathlib import Path
import numpy as np

print("Debugging the 391,769 vs ~57,000 discrepancy")
print("=" * 60)

# Load DEG data to understand the discrepancy
deg_folder = "/Users/prashammarfatia/Downloads/de_results_per_gene"
deg_files = list(Path(deg_folder).glob("*_vs_control.csv"))

print(f"Found {len(deg_files)} DEG files")

# Method 1: My original approach (creating pairs)
experimental_pairs = set()
perturbation_target_counts = {}
all_targets_with_source = []

print("Processing all DEG files...")
processed_count = 0

for deg_file in deg_files:
    gene_name = deg_file.stem.replace("_vs_control", "")

    # Skip TP53 and CDKN1A
    if gene_name in ['TP53', 'CDKN1A']:
        continue

    try:
        deg_df = pd.read_csv(deg_file)
        significant_genes = deg_df[deg_df['pvals'] < 0.05].copy()

        # Store target count for this perturbation
        perturbation_target_counts[gene_name] = len(significant_genes)

        # Add to experimental pairs (my original method)
        for _, row in significant_genes.iterrows():
            experimental_pairs.add((gene_name, row['names']))
            all_targets_with_source.append((gene_name, row['names']))

        processed_count += 1
        if processed_count % 500 == 0:
            print(f"Processed {processed_count} files...")

    except Exception as e:
        print(f"Error processing {gene_name}: {e}")

print(f"\nProcessing complete. Processed {processed_count} perturbations.")

# Calculate totals
sum_of_targets = sum(perturbation_target_counts.values())
unique_pairs = len(experimental_pairs)
total_pairs_with_duplicates = len(all_targets_with_source)

print(f"\nMethod Comparison:")
print(f"Sum of targets per perturbation (your method): {sum_of_targets:,}")
print(f"Unique source-target pairs (my method): {unique_pairs:,}")
print(f"Total pairs with duplicates: {total_pairs_with_duplicates:,}")

# Check for duplicates
print(f"\nDuplicate Analysis:")
print(f"Difference between methods: {unique_pairs - sum_of_targets:,}")
print(f"Duplication rate: {(total_pairs_with_duplicates - unique_pairs) / total_pairs_with_duplicates * 100:.2f}%")

# Analyze target overlap between perturbations
print("\nAnalyzing target overlap between perturbations...")
all_targets = []
perturbation_targets = {}

for gene_name in list(perturbation_target_counts.keys())[:20]:  # Test with first 20
    try:
        deg_file = Path(deg_folder) / f"{gene_name}_vs_control.csv"
        deg_df = pd.read_csv(deg_file)
        significant_genes = deg_df[deg_df['pvals'] < 0.05]['names'].tolist()
        perturbation_targets[gene_name] = set(significant_genes)
        all_targets.extend(significant_genes)
    except:
        pass

print(f"\nSample Analysis (first 20 perturbations):")
print(f"Total targets (with duplicates): {len(all_targets)}")
print(f"Unique targets: {len(set(all_targets))}")

# Find most common targets
from collections import Counter

target_counts = Counter(all_targets)
most_common = target_counts.most_common(10)

print(f"\nMost frequently appearing targets:")
for target, count in most_common:
    print(f"  {target}: appears in {count} perturbations")

# Check if same target appears with same source multiple times
print(f"\nChecking for same (source, target) appearing multiple times...")
pair_counter = Counter(all_targets_with_source)
duplicate_pairs = [(pair, count) for pair, count in pair_counter.items() if count > 1]

if duplicate_pairs:
    print(f"Found {len(duplicate_pairs)} duplicate source-target pairs:")
    for pair, count in duplicate_pairs[:10]:
        print(f"  {pair}: {count} times")
else:
    print("No duplicate source-target pairs found - each (source, target) is unique")

# Final verification
print(f"\nFinal Verification:")
print(f"Expected (sum method): {sum_of_targets:,}")
print(f"Actual (set method): {unique_pairs:,}")

if sum_of_targets == unique_pairs:
    print("✓ Methods match! No discrepancy.")
else:
    print(f"✗ Discrepancy: {abs(sum_of_targets - unique_pairs):,} pairs")
    if unique_pairs > sum_of_targets:
        print("Possible cause: Same target appears in multiple perturbations")
    else:
        print("Possible cause: Error in calculation logic")

# Show statistics per perturbation
print(f"\nPer-perturbation statistics:")
target_counts_list = list(perturbation_target_counts.values())
print(f"Mean targets per perturbation: {np.mean(target_counts_list):.1f}")
print(f"Median targets per perturbation: {np.median(target_counts_list):.1f}")
print(f"Min targets per perturbation: {min(target_counts_list)}")
print(f"Max targets per perturbation: {max(target_counts_list)}")
print(f"Total perturbations: {len(perturbation_target_counts)}")

# Show top and bottom perturbations by target count
print(f"\nTop 5 perturbations by target count:")
sorted_perturbations = sorted(perturbation_target_counts.items(), key=lambda x: x[1], reverse=True)
for gene, count in sorted_perturbations[:5]:
    print(f"  {gene}: {count} targets")

print(f"\nBottom 5 perturbations by target count:")
for gene, count in sorted_perturbations[-5:]:
    print(f"  {gene}: {count} targets")