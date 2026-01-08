import pandas as pd

print("=" * 80)
print(" VERIFYING 2-HOP PAIRS (Removing Self-Loops)")
print("=" * 80)

# === LOAD FILES ===
real_1hop = pd.read_csv("/Users/prashammarfatia/Downloads/indra_1hop_with_statements_.csv")
real_2hop = pd.read_csv("/Users/prashammarfatia/Downloads/indra_2hop_all_perturbations.csv")
permuted_1hop = pd.read_csv("/Users/prashammarfatia/Downloads/indra_1hop_PERMUTED.csv")
permuted_2hop = pd.read_csv("/Users/prashammarfatia/Downloads/indra_2hop_PERMUTED.csv")
master_permuted = pd.read_csv("/Users/prashammarfatia/Downloads/MASTER_permuted_source_target_pairs.csv")

print("\n Before removing self-loops:")
print(f"Real 1-hop rows: {len(real_1hop):,}")
print(f"Real 2-hop rows: {len(real_2hop):,}")
print(f"Permuted 1-hop rows: {len(permuted_1hop):,}")
print(f"Permuted 2-hop rows: {len(permuted_2hop):,}")
print(f"Master permuted pairs: {len(master_permuted):,}")

# === REMOVE SELF-LOOPS ===
real_1hop_clean = real_1hop[real_1hop['source'] != real_1hop['target']].copy()
real_2hop_clean = real_2hop[real_2hop['source'] != real_2hop['target']].copy()
permuted_1hop_clean = permuted_1hop[permuted_1hop['source'] != permuted_1hop['target']].copy()
permuted_2hop_clean = permuted_2hop[permuted_2hop['source'] != permuted_2hop['target']].copy()
master_permuted_clean = master_permuted[master_permuted['source'] != master_permuted['target']].copy()

# Check what was removed
print("\n🗑️  Self-loops removed:")
print(f"Real 1-hop: {len(real_1hop) - len(real_1hop_clean):,}")
print(f"Real 2-hop: {len(real_2hop) - len(real_2hop_clean):,}")
print(f"Permuted 1-hop: {len(permuted_1hop) - len(permuted_1hop_clean):,}")
print(f"Permuted 2-hop: {len(permuted_2hop) - len(permuted_2hop_clean):,}")
print(f"Master permuted: {len(master_permuted) - len(master_permuted_clean):,}")

print("\n After removing self-loops:")
print(f"Real 1-hop rows: {len(real_1hop_clean):,}")
print(f"Real 2-hop rows: {len(real_2hop_clean):,}")
print(f"Permuted 1-hop rows: {len(permuted_1hop_clean):,}")
print(f"Permuted 2-hop rows: {len(permuted_2hop_clean):,}")
print(f"Master permuted pairs: {len(master_permuted_clean):,}")

# === COUNT UNIQUE PAIRS ===
print("\n" + "=" * 80)
print(" UNIQUE SOURCE-TARGET PAIRS (No Self-Loops)")
print("=" * 80)

real_1hop_pairs = set(zip(real_1hop_clean['source'], real_1hop_clean['target']))
real_2hop_pairs = set(zip(real_2hop_clean['source'], real_2hop_clean['target']))
real_combined_pairs = real_1hop_pairs.union(real_2hop_pairs)

permuted_1hop_pairs = set(zip(permuted_1hop_clean['source'], permuted_1hop_clean['target']))
permuted_2hop_pairs = set(zip(permuted_2hop_clean['source'], permuted_2hop_clean['target']))
permuted_combined_pairs = permuted_1hop_pairs.union(permuted_2hop_pairs)

master_pairs_clean = set(zip(master_permuted_clean['source'], master_permuted_clean['target']))

print(f"\nReal 1-hop unique pairs: {len(real_1hop_pairs):,}")
print(f"Real 2-hop unique pairs: {len(real_2hop_pairs):,}")
print(f"Real combined unique pairs: {len(real_combined_pairs):,}")
print(f"\nPermuted 1-hop unique pairs: {len(permuted_1hop_pairs):,}")
print(f"Permuted 2-hop unique pairs: {len(permuted_2hop_pairs):,}")
print(f"Permuted combined unique pairs: {len(permuted_combined_pairs):,}")
print(f"\nMaster permuted total pool: {len(master_pairs_clean):,}")

# === CALCULATE REAL TOTAL POOL (without self-loops) ===
import os

perturb_df = pd.read_csv("/Users/prashammarfatia/Downloads/target_validation_expanded.csv")
perturb_df = perturb_df[perturb_df['Karen_Flag'] == "Use_for_analysis"]
excluded_genes = ["TP53", "CDKN1A"]
perturb_df = perturb_df[~perturb_df["Gene"].isin(excluded_genes)]

real_total_pairs = []
de_folder = "/Users/prashammarfatia/Downloads/de_results_per_gene/"

for _, row in perturb_df.iterrows():
    source_gene = row["Gene"]
    csv_path = os.path.join(de_folder, f"{source_gene}_vs_control.csv")
    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path)
    df = df[df["pvals"] < 0.05]

    for _, target_row in df.iterrows():
        # EXCLUDE SELF-LOOPS
        if source_gene != target_row["names"]:
            real_total_pairs.append((source_gene, target_row["names"]))

real_total_pairs_set = set(real_total_pairs)
print(f"Real total pool (no self-loops): {len(real_total_pairs_set):,}")

# === CALCULATE COVERAGE RATES ===
print("\n" + "=" * 80)
print(" COVERAGE RATES (No Self-Loops)")
print("=" * 80)

real_1hop_pct = (len(real_1hop_pairs) / len(real_total_pairs_set)) * 100
real_2hop_pct = (len(real_2hop_pairs) / len(real_total_pairs_set)) * 100
real_combined_pct = (len(real_combined_pairs) / len(real_total_pairs_set)) * 100

permuted_1hop_pct = (len(permuted_1hop_pairs) / len(master_pairs_clean)) * 100
permuted_2hop_pct = (len(permuted_2hop_pairs) / len(master_pairs_clean)) * 100
permuted_combined_pct = (len(permuted_combined_pairs) / len(master_pairs_clean)) * 100

print(f"\n{'Dataset':<25} {'Explained':<15} {'Total Pool':<15} {'% Explained':<15}")
print("-" * 70)
print(f"{'Real 1-hop':<25} {len(real_1hop_pairs):<15,} {len(real_total_pairs_set):<15,} {real_1hop_pct:<15.2f}%")
print(f"{'Real 2-hop':<25} {len(real_2hop_pairs):<15,} {len(real_total_pairs_set):<15,} {real_2hop_pct:<15.2f}%")
print(
    f"{'Real Combined':<25} {len(real_combined_pairs):<15,} {len(real_total_pairs_set):<15,} {real_combined_pct:<15.2f}%")
print()
print(
    f"{'Permuted 1-hop':<25} {len(permuted_1hop_pairs):<15,} {len(master_pairs_clean):<15,} {permuted_1hop_pct:<15.2f}%")
print(
    f"{'Permuted 2-hop':<25} {len(permuted_2hop_pairs):<15,} {len(master_pairs_clean):<15,} {permuted_2hop_pct:<15.2f}%")
print(
    f"{'Permuted Combined':<25} {len(permuted_combined_pairs):<15,} {len(master_pairs_clean):<15,} {permuted_combined_pct:<15.2f}%")

# === COMPARISON ===
print("\n" + "=" * 80)
print("⚖️  COVERAGE RATE COMPARISON (No Self-Loops)")
print("=" * 80)

print(f"\n{'Metric':<40} {'Real %':<15} {'Permuted %':<15} {'Difference':<15}")
print("-" * 85)
print(
    f"{'1-hop coverage rate':<40} {real_1hop_pct:<15.2f}% {permuted_1hop_pct:<15.2f}% {real_1hop_pct - permuted_1hop_pct:+.2f}%")
print(
    f"{'2-hop coverage rate':<40} {real_2hop_pct:<15.2f}% {permuted_2hop_pct:<15.2f}% {real_2hop_pct - permuted_2hop_pct:+.2f}%")
print(
    f"{'Combined coverage rate':<40} {real_combined_pct:<15.2f}% {permuted_combined_pct:<15.2f}% {real_combined_pct - permuted_combined_pct:+.2f}%")

coverage_diff = real_combined_pct - permuted_combined_pct

print("\n" + "=" * 80)
print(" INTERPRETATION (No Self-Loops)")
print("=" * 80)

if coverage_diff > 5:
    print(f"\n EXCELLENT: Real pairs have {coverage_diff:+.2f}% HIGHER coverage rate!")
    print(f"   Real: {real_combined_pct:.2f}% explained")
    print(f"   Permuted: {permuted_combined_pct:.2f}% explained")
else:
    print(f"\n Coverage difference: {coverage_diff:+.2f}%")

print(f"\n Absolute numbers:")
print(f"   Real: {len(real_combined_pairs):,} of {len(real_total_pairs_set):,} pairs")
print(f"   Permuted: {len(permuted_combined_pairs):,} of {len(master_pairs_clean):,} pairs")
print(f"   Real has {len(real_combined_pairs) - len(permuted_combined_pairs):,} MORE pairs explained")

print("\n" + "=" * 80)