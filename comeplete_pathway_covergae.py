import pandas as pd
import numpy as np
from pathlib import Path
import os

print("Deep Statistical Coverage Analysis for 1-hop, 2-hop, 3-hop Pathways")
print("=" * 80)

# ----------------------------------------------------------------------
# Load pathway datasets (NO belief filtering for coverage analysis)
# ----------------------------------------------------------------------
print("Loading pathway datasets...")

# 1-hop data (excluding TP53)
df_1hop_combined = pd.read_csv("/Users/prashammarfatia/Downloads/indra_1hop_no_v2.csv")

# 2-hop data (excluding TP53)
df_2hop_combined = pd.read_excel("/Users/prashammarfatia/Downloads/indra_2hop_all_perturbations.xlsx")

# 3-hop data
df_3hop = pd.read_csv('/Users/prashammarfatia/Downloads/indra_3hop_optimized_all_perturbations_combined.csv')

print(f"1-hop total pathways: {len(df_1hop_combined):,}")
print(f"2-hop total pathways: {len(df_2hop_combined):,}")
print(f"3-hop total pathways: {len(df_3hop):,}")

# Extract unique source-target pairs for each pathway type
pairs_1hop = set(zip(df_1hop_combined['source'], df_1hop_combined['target']))
pairs_2hop = set(zip(df_2hop_combined['source'], df_2hop_combined['target']))
pairs_3hop = set(zip(df_3hop['source'], df_3hop['target']))

print(f"\nUnique source-target pairs by pathway type (unfiltered):")
print(f"1-hop: {len(pairs_1hop):,}")
print(f"2-hop: {len(pairs_2hop):,}")
print(f"3-hop: {len(pairs_3hop):,}")

# ----------------------------------------------------------------------
# Build denominator: ONLY the 358 "Use_for_analysis" perturbations
# ----------------------------------------------------------------------
print("\nLoading DEG data for total experimental pairs...")

tv_path = "/Users/prashammarfatia/Downloads/target_validation_expanded.csv"
deg_folder = "/Users/prashammarfatia/Downloads/de_results_per_gene"

# Allowed sources (358 perturbations with Karen_Flag == Use_for_analysis)
allowed_sources = set(
    pd.read_csv(tv_path)
      .query("Karen_Flag == 'Use_for_analysis'")["Gene"]
      .astype(str).str.strip()
      .tolist()
)

# Optional: exclude TP53 and CDKN1A
EXCLUDE_SOURCES = {"TP53", "CDKN1A"}
allowed_sources = {g for g in allowed_sources if g not in EXCLUDE_SOURCES}

experimental_pairs = set()
perturbation_stats = {}

for src in sorted(allowed_sources):
    deg_path = Path(deg_folder) / f"{src}_vs_control.csv"
    if not deg_path.exists():
        continue

    df = pd.read_csv(deg_path)
    if not {"names", "pvals"}.issubset(df.columns):
        continue

    sig = df[df["pvals"] < 0.05].copy()
    targets = set(sig["names"].dropna().astype(str))

    for tgt in targets:
        experimental_pairs.add((src, tgt))

    perturbation_stats[src] = {
        "total_targets": len(targets),
        "targets": targets,
    }

total_experimental_pairs = len(experimental_pairs)
print(f"Total experimental (source,target) pairs (allowed perturbations only): {total_experimental_pairs:,}")
print(f"Perturbations with valid DEG: {len(perturbation_stats):,}")

# ----------------------------------------------------------------------
# Align hop pairs to same source set as denominator
# ----------------------------------------------------------------------
pairs_1hop = {(s, t) for (s, t) in pairs_1hop if s in allowed_sources}
pairs_2hop = {(s, t) for (s, t) in pairs_2hop if s in allowed_sources}
pairs_3hop = {(s, t) for (s, t) in pairs_3hop if s in allowed_sources}

print(f"\nUnique source-target pairs by pathway type (filtered to 358 sources):")
print(f"1-hop: {len(pairs_1hop):,}")
print(f"2-hop: {len(pairs_2hop):,}")
print(f"3-hop: {len(pairs_3hop):,}")

# ----------------------------------------------------------------------
# INDIVIDUAL COVERAGE ANALYSIS
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("INDIVIDUAL COVERAGE ANALYSIS")
print("=" * 80)

coverage_1hop = (len(pairs_1hop.intersection(experimental_pairs)) / total_experimental_pairs) * 100
coverage_2hop = (len(pairs_2hop.intersection(experimental_pairs)) / total_experimental_pairs) * 100
coverage_3hop = (len(pairs_3hop.intersection(experimental_pairs)) / total_experimental_pairs) * 100

print("Individual Coverage (without priority logic):")
print(f"1-hop coverage: {len(pairs_1hop.intersection(experimental_pairs)):,} / {total_experimental_pairs:,} = {coverage_1hop:.2f}%")
print(f"2-hop coverage: {len(pairs_2hop.intersection(experimental_pairs)):,} / {total_experimental_pairs:,} = {coverage_2hop:.2f}%")
print(f"3-hop coverage: {len(pairs_3hop.intersection(experimental_pairs)):,} / {total_experimental_pairs:,} = {coverage_3hop:.2f}%")

# ----------------------------------------------------------------------
# EXCLUSIVE COVERAGE ANALYSIS
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("EXCLUSIVE COVERAGE ANALYSIS (WITH PRIORITY LOGIC)")
print("=" * 80)

explained_1hop_only = pairs_1hop.intersection(experimental_pairs)
explained_2hop_only = (pairs_2hop - pairs_1hop).intersection(experimental_pairs)
explained_3hop_only = (pairs_3hop - pairs_1hop - pairs_2hop).intersection(experimental_pairs)

all_explained = pairs_1hop.union(pairs_2hop).union(pairs_3hop).intersection(experimental_pairs)
unexplained = experimental_pairs - all_explained

print("Exclusive Coverage (with priority logic):")
print(f"1-hop only: {len(explained_1hop_only):,} pairs ({len(explained_1hop_only) / total_experimental_pairs * 100:.2f}%)")
print(f"2-hop only: {len(explained_2hop_only):,} pairs ({len(explained_2hop_only) / total_experimental_pairs * 100:.2f}%)")
print(f"3-hop only: {len(explained_3hop_only):,} pairs ({len(explained_3hop_only) / total_experimental_pairs * 100:.2f}%)")
print(f"Total explained: {len(all_explained):,} pairs ({len(all_explained) / total_experimental_pairs * 100:.2f}%)")
print(f"Unexplained: {len(unexplained):,} pairs ({len(unexplained) / total_experimental_pairs * 100:.2f}%)")

# ----------------------------------------------------------------------
# PER-PERTURBATION COVERAGE ANALYSIS
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("PER-PERTURBATION COVERAGE ANALYSIS")
print("=" * 80)

perturbation_coverage = []
for gene_name, stats in perturbation_stats.items():
    targets = stats['targets']
    total_targets = len(targets)
    if total_targets == 0:
        continue

    gene_pairs = {(gene_name, target) for target in targets}

    explained_1hop = len(gene_pairs.intersection(pairs_1hop))
    explained_2hop = len(gene_pairs.intersection(pairs_2hop))
    explained_3hop = len(gene_pairs.intersection(pairs_3hop))

    explained_1hop_excl = len(gene_pairs.intersection(pairs_1hop))
    explained_2hop_excl = len(gene_pairs.intersection(pairs_2hop - pairs_1hop))
    explained_3hop_excl = len(gene_pairs.intersection(pairs_3hop - pairs_1hop - pairs_2hop))

    total_explained = explained_1hop_excl + explained_2hop_excl + explained_3hop_excl

    perturbation_coverage.append({
        'perturbation': gene_name,
        'total_targets': total_targets,
        'explained_1hop': explained_1hop,
        'explained_2hop': explained_2hop,
        'explained_3hop': explained_3hop,
        'explained_1hop_excl': explained_1hop_excl,
        'explained_2hop_excl': explained_2hop_excl,
        'explained_3hop_excl': explained_3hop_excl,
        'total_explained': total_explained,
        'coverage_1hop': (explained_1hop / total_targets) * 100,
        'coverage_2hop': (explained_2hop / total_targets) * 100,
        'coverage_3hop': (explained_3hop / total_targets) * 100,
        'coverage_total': (total_explained / total_targets) * 100,
        'unexplained': total_targets - total_explained
    })

coverage_df = pd.DataFrame(perturbation_coverage)

print("Per-perturbation coverage statistics:")
print(f"Mean total coverage: {coverage_df['coverage_total'].mean():.2f}%")
print(f"Mean 1-hop coverage: {coverage_df['coverage_1hop'].mean():.2f}%")
print(f"Mean 2-hop coverage: {coverage_df['coverage_2hop'].mean():.2f}%")
print(f"Mean 3-hop coverage: {coverage_df['coverage_3hop'].mean():.2f}%")

print(f"\nTop 10 perturbations by total coverage:")
print(coverage_df.nlargest(10, 'coverage_total')[
    ['perturbation', 'total_targets', 'coverage_total', 'coverage_1hop', 'coverage_2hop', 'coverage_3hop']]
      .to_string(index=False, float_format='%.1f'))

print(f"\nBottom 10 perturbations by total coverage:")
print(coverage_df.nsmallest(10, 'coverage_total')[
    ['perturbation', 'total_targets', 'coverage_total', 'coverage_1hop', 'coverage_2hop', 'coverage_3hop']]
      .to_string(index=False, float_format='%.1f'))

# ----------------------------------------------------------------------
# COMPREHENSIVE SUMMARY TABLE
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("CREATING COMPREHENSIVE COVERAGE SUMMARY TABLE")
print("=" * 80)

summary_data = {
    'Coverage_Type': [
        'Individual_1hop', 'Individual_2hop', 'Individual_3hop',
        'Exclusive_1hop_only', 'Exclusive_2hop_only', 'Exclusive_3hop_only',
        'Total_Explained', 'Unexplained'
    ],
    'Source_Target_Pairs': [
        len(pairs_1hop.intersection(experimental_pairs)),
        len(pairs_2hop.intersection(experimental_pairs)),
        len(pairs_3hop.intersection(experimental_pairs)),
        len(explained_1hop_only),
        len(explained_2hop_only),
        len(explained_3hop_only),
        len(all_explained),
        len(unexplained)
    ],
    'Percentage': [
        coverage_1hop,
        coverage_2hop,
        coverage_3hop,
        (len(explained_1hop_only) / total_experimental_pairs) * 100,
        (len(explained_2hop_only) / total_experimental_pairs) * 100,
        (len(explained_3hop_only) / total_experimental_pairs) * 100,
        (len(all_explained) / total_experimental_pairs) * 100,
        (len(unexplained) / total_experimental_pairs) * 100
    ]
}
summary_table = pd.DataFrame(summary_data)

print("COMPREHENSIVE COVERAGE SUMMARY TABLE:")
print(f"Total Experimental Source-Target Pairs: {total_experimental_pairs:,}")
print(f"Perturbations Analyzed: {len(perturbation_stats):,}")
print()
print(summary_table.to_string(index=False, float_format='%.2f'))

# ----------------------------------------------------------------------
# SAVE RESULTS
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

summary_table.to_csv('pathway_coverage_summary.csv', index=False)
coverage_df.to_csv('per_perturbation_coverage.csv', index=False)

pair_analysis = {
    'experimental_pairs': list(experimental_pairs),
    'explained_1hop_only': list(explained_1hop_only),
    'explained_2hop_only': list(explained_2hop_only),
    'explained_3hop_only': list(explained_3hop_only),
    'unexplained_pairs': list(unexplained)
}

detailed_pairs = []
for pair_type, pairs_list in pair_analysis.items():
    for source, target in pairs_list:
        detailed_pairs.append({
            'source': source,
            'target': target,
            'classification': pair_type
        })

detailed_df = pd.DataFrame(detailed_pairs)
detailed_df.to_csv('detailed_pair_classification.csv', index=False)

print("\n" + "=" * 80)
print("COVERAGE ANALYSIS COMPLETE!")
print("=" * 80)

print("Files generated:")
print("1. pathway_coverage_summary.csv - Main coverage statistics")
print("2. per_perturbation_coverage.csv - Coverage for each perturbation")
print("3. detailed_pair_classification.csv - Classification of all source-target pairs")

print(f"\nKey Results:")
print(f"- Individual pathway type coverage ranges from {coverage_3hop:.1f}% to {coverage_2hop:.1f}%")
print(f"- Combined coverage of all pathway types: {len(all_explained) / total_experimental_pairs * 100:.1f}%")
print(f"- Unexplained experimental pairs: {len(unexplained) / total_experimental_pairs * 100:.1f}%")
print(f"- Mean per-perturbation coverage: {coverage_df['coverage_total'].mean():.1f}%")
