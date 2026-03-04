import pandas as pd

# Load the main dataset
df = pd.read_csv("/Users/prashammarfatia/Downloads/_new_2hop_mesh_filtered.csv")

print("=" * 80)
print("ORIGINAL DATASET")
print("=" * 80)
print(f"Shape: {df.shape}")
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"Columns: {df.columns.tolist()}\n")

# Load endothelial genes
endothelial_df = pd.read_csv("/Users/prashammarfatia/Downloads/endothelial_present_plus_manual.csv")
endothelial_genes_original = set(endothelial_df['gene'].tolist())

# GWAS genes list
gwas_genes = {
    'BCAR1', 'BMP1', 'CALCRL', 'CCM2', 'CDKN1A', 'CDKN2B', 'CFDP1', 'COL4A1',
    'COL4A2', 'EXOC3L2', 'FBN2', 'FGD6', 'FLT1', 'FURIN', 'GDPD5', 'GGT5',
    'GOSR2', 'IBTK', 'LAMB2', 'LOX', 'MORF4L1', 'N4BP2L2', 'NOS3', 'PALLD',
    'PECAM1', 'PGF', 'PLPP3', 'PREX1', 'PRKAR1A', 'SCUBE1', 'SERPINH1',
    'SH3PXD2A', 'SLK', 'SMAD3', 'SPRY4', 'SVIL', 'SWAP70', 'TFPI', 'TLNRD1',
    'TSPAN14', 'ZEB2'
}

print("=" * 80)
print("GWAS GENES")
print("=" * 80)
print(f"Number of GWAS genes: {len(gwas_genes)}")
print(f"GWAS genes: {sorted(gwas_genes)}\n")

# Check overlap and REMOVE GWAS genes from endothelial list
overlap = gwas_genes.intersection(endothelial_genes_original)
print("=" * 80)
print("OVERLAP ANALYSIS")
print("=" * 80)
print(f"GWAS genes found in endothelial list: {len(overlap)} out of {len(gwas_genes)}")
print(f"Overlap genes: {sorted(overlap)}")

# Remove GWAS genes from endothelial list
endothelial_genes = endothelial_genes_original - gwas_genes
print(f"\nAfter removing GWAS genes from endothelial list:")
print(f"  Original endothelial genes: {len(endothelial_genes_original):,}")
print(f"  GWAS genes removed: {len(overlap)}")
print(f"  Final endothelial genes (excluding GWAS): {len(endothelial_genes):,}\n")

# Statistics BEFORE filtering
print("=" * 80)
print("ORIGINAL DATASET STATISTICS")
print("=" * 80)
print(f"Unique genes in 'source': {df['source'].nunique():,}")
print(f"Unique genes in 'intermediate': {df['intermediate'].nunique():,}")
print(f"Unique genes in 'target': {df['target'].nunique():,}")
print(
    f"\nTotal unique genes across all positions: {pd.concat([df['source'], df['intermediate'], df['target']]).nunique():,}\n")

# STEP 1: Filter by endothelial genes (intermediate ONLY)
print("=" * 80)
print("STEP 1: FILTER BY ENDOTHELIAL INTERMEDIATES")
print("=" * 80)
mask_intermediate = df['intermediate'].isin(endothelial_genes)
endothelial_filtered_df = df[mask_intermediate].copy()

print(f"Rows with intermediate in endothelial list: {len(endothelial_filtered_df):,} out of {len(df):,}")
print(f"Percentage retained: {len(endothelial_filtered_df) / len(df) * 100:.2f}%")
print(f"Rows removed: {len(df) - len(endothelial_filtered_df):,}\n")

# Statistics on endothelial-filtered dataset
print(f"Endothelial-filtered dataset statistics:")
print(f"  Unique genes in 'source': {endothelial_filtered_df['source'].nunique():,}")
print(f"  Unique genes in 'intermediate': {endothelial_filtered_df['intermediate'].nunique():,}")
print(f"  Unique genes in 'target': {endothelial_filtered_df['target'].nunique():,}\n")

# GWAS gene presence in endothelial-filtered dataset
print("=" * 80)
print("GWAS GENE PRESENCE IN ENDOTHELIAL-FILTERED DATASET")
print("=" * 80)
gwas_mask_in_endo = (
    endothelial_filtered_df['source'].isin(gwas_genes) |
    endothelial_filtered_df['intermediate'].isin(gwas_genes) |
    endothelial_filtered_df['target'].isin(gwas_genes)
)
rows_with_gwas = gwas_mask_in_endo.sum()
percent_with_gwas = (rows_with_gwas / len(endothelial_filtered_df)) * 100

print(f"Total endothelial-filtered rows: {len(endothelial_filtered_df):,}")
print(f"Rows containing at least one GWAS gene: {rows_with_gwas:,}")
print(f"Percentage of paths with GWAS genes: {percent_with_gwas:.2f}%")
print(f"Rows without any GWAS genes: {len(endothelial_filtered_df) - rows_with_gwas:,}\n")

# SAVE OUTPUT 1: Endothelial-filtered dataset
output_file_1 = "/Users/prashammarfatia/Downloads/2hop_endothelial_filtered.csv"
endothelial_filtered_df.to_csv(output_file_1, index=False)
print(f"✓ OUTPUT 1 SAVED: {output_file_1}")
print(f"  Shape: {endothelial_filtered_df.shape}\n")

# Add absolute logfoldchange for sorting
endothelial_filtered_df['abs_logfoldchange'] = endothelial_filtered_df['logfoldchange'].abs()

# STEP 2: Get top 100 by absolute logfoldchange
print("=" * 80)
print("STEP 2: SELECT TOP 100 BY ABSOLUTE LOGFOLDCHANGE")
print("=" * 80)
endothelial_sorted = endothelial_filtered_df.sort_values('abs_logfoldchange', ascending=False)
top_100 = endothelial_sorted.head(100).copy()
top_100_indices = set(top_100.index)

print(f"Selected top 100 rows by absolute logfoldchange")
print(f"  Logfoldchange range: {top_100['logfoldchange'].min():.4f} to {top_100['logfoldchange'].max():.4f}")
print(
    f"  Absolute logfoldchange range: {top_100['abs_logfoldchange'].min():.4f} to {top_100['abs_logfoldchange'].max():.4f}")

# Check if any GWAS genes in top 100
top_100_gwas_mask = (
    top_100['source'].isin(gwas_genes) |
    top_100['intermediate'].isin(gwas_genes) |
    top_100['target'].isin(gwas_genes)
)
print(f"  Rows with GWAS genes in top 100: {top_100_gwas_mask.sum()}")
print(f"  Rows without GWAS genes in top 100: {(~top_100_gwas_mask).sum()}\n")

# STEP 3: Find GWAS genes in the endothelial-filtered dataset
print("=" * 80)
print("STEP 3: FIND GWAS GENES IN ENDOTHELIAL-FILTERED DATASET")
print("=" * 80)
mask_gwas = (
    endothelial_filtered_df['source'].isin(gwas_genes) |
    endothelial_filtered_df['intermediate'].isin(gwas_genes) |
    endothelial_filtered_df['target'].isin(gwas_genes)
)
gwas_rows = endothelial_filtered_df[mask_gwas].copy()
gwas_indices = set(gwas_rows.index)

print(f"Total rows containing GWAS genes in endothelial-filtered data: {len(gwas_rows):,}")
print(f"  GWAS in source: {endothelial_filtered_df['source'].isin(gwas_genes).sum():,}")
print(f"  GWAS in intermediate: {endothelial_filtered_df['intermediate'].isin(gwas_genes).sum():,}")
print(f"  GWAS in target: {endothelial_filtered_df['target'].isin(gwas_genes).sum():,}")

# Unique GWAS genes found
gwas_sources_found = set(endothelial_filtered_df[endothelial_filtered_df['source'].isin(gwas_genes)]['source'].unique())
gwas_intermediates_found = set(
    endothelial_filtered_df[endothelial_filtered_df['intermediate'].isin(gwas_genes)]['intermediate'].unique())
gwas_targets_found = set(endothelial_filtered_df[endothelial_filtered_df['target'].isin(gwas_genes)]['target'].unique())
gwas_found = gwas_sources_found | gwas_intermediates_found | gwas_targets_found

print(f"\n  Unique GWAS genes present in any position: {len(gwas_found)} out of {len(gwas_genes)}")
print(f"  GWAS genes found: {sorted(gwas_found)}")
print(f"    As source: {len(gwas_sources_found)} genes - {sorted(gwas_sources_found)}")
print(f"    As intermediate: {len(gwas_intermediates_found)} genes - {sorted(gwas_intermediates_found)}")
print(f"    As target: {len(gwas_targets_found)} genes - {sorted(gwas_targets_found)}")

gwas_not_found = gwas_genes - gwas_found
if gwas_not_found:
    print(f"\n  GWAS genes NOT found in endothelial-filtered data: {sorted(gwas_not_found)}")

# Find overlap between top 100 and GWAS
overlap_indices = top_100_indices.intersection(gwas_indices)
print(f"\n  Rows in BOTH top 100 AND GWAS: {len(overlap_indices)}")

# Combine top 100 with GWAS rows
combined = pd.concat([top_100, gwas_rows]).drop_duplicates()
print(f"  Combined dataset (top 100 + GWAS rows): {len(combined):,} rows")
print(f"  New rows added from GWAS: {len(combined) - len(top_100):,}\n")

# Add selection_reason column
combined['selection_reason'] = combined.index.map(
    lambda idx: 'Top 100 + GWAS Gene' if idx in overlap_indices
    else ('Top 100' if idx in top_100_indices
          else 'GWAS Gene')
)

# STEP 4: Apply uniqueness constraint per GWAS gene
print("=" * 80)
print("STEP 4: APPLY UNIQUENESS CONSTRAINT PER GWAS GENE")
print("=" * 80)
final_rows = []

# Process each GWAS gene individually
for gwas_gene in sorted(gwas_genes):
    # Find rows containing this specific GWAS gene
    gene_mask = (
        (combined['source'] == gwas_gene) |
        (combined['intermediate'] == gwas_gene) |
        (combined['target'] == gwas_gene)
    )
    gene_rows = combined[gene_mask].copy()

    if len(gene_rows) > 0:
        # Count positions
        as_source = (gene_rows['source'] == gwas_gene).sum()
        as_intermediate = (gene_rows['intermediate'] == gwas_gene).sum()
        as_target = (gene_rows['target'] == gwas_gene).sum()

        # Group by (source, target) and keep row with highest abs_logfoldchange
        unique_gene_rows = gene_rows.sort_values('abs_logfoldchange', ascending=False).groupby(
            ['source', 'target'], as_index=False
        ).first()

        final_rows.append(unique_gene_rows)

        reduction = len(gene_rows) - len(unique_gene_rows)
        print(
            f"  {gwas_gene}: {len(gene_rows):,} rows → {len(unique_gene_rows):,} unique (removed {reduction}) | S:{as_source} I:{as_intermediate} T:{as_target}")

# Get non-GWAS rows from top 100
non_gwas_mask = ~(
    combined['source'].isin(gwas_genes) |
    combined['intermediate'].isin(gwas_genes) |
    combined['target'].isin(gwas_genes)
)
non_gwas_rows = combined[non_gwas_mask].copy()
print(f"\n  Non-GWAS rows from top 100: {len(non_gwas_rows):,} rows (no uniqueness applied)")

# Combine all final rows
if final_rows:
    final_gwas_df = pd.concat(final_rows)
    duplicates_before = len(final_gwas_df)
    final_gwas_df = final_gwas_df.drop_duplicates()
    duplicates_removed = duplicates_before - len(final_gwas_df)

    print(f"\n  Combined GWAS rows: {len(final_gwas_df):,}")
    print(f"  Duplicates removed (rows with multiple GWAS genes): {duplicates_removed:,}")

    final_df = pd.concat([final_gwas_df, non_gwas_rows]).drop_duplicates()
else:
    final_df = non_gwas_rows

# Sort by absolute logfoldchange
final_df = final_df.sort_values('abs_logfoldchange', ascending=False)

# STEP 5: Add directionality column
print("\n" + "=" * 80)
print("STEP 5: ADD DIRECTIONALITY COLUMN")
print("=" * 80)
final_df['directionality'] = 'No'
final_df.loc[
    ((final_df['stmt_type_2'] == 'IncreaseAmount') & (final_df['logfoldchange'] < 0)) |
    ((final_df['stmt_type_2'] == 'DecreaseAmount') & (final_df['logfoldchange'] > 0)),
    'directionality'
] = 'Yes'

directionality_counts = final_df['directionality'].value_counts()
print(f"Directionality breakdown:")
print(
    f"  Yes (Agreement): {directionality_counts.get('Yes', 0):,} ({directionality_counts.get('Yes', 0) / len(final_df) * 100:.1f}%)")
print(
    f"  No (Disagreement/Other): {directionality_counts.get('No', 0):,} ({directionality_counts.get('No', 0) / len(final_df) * 100:.1f}%)")

# Show examples
print(f"\nExamples of 'Yes' directionality:")
yes_examples = final_df[final_df['directionality'] == 'Yes'][
    ['source', 'intermediate', 'target', 'stmt_type_2', 'logfoldchange', 'directionality']].head(5)
if len(yes_examples) > 0:
    print(yes_examples.to_string(index=False))
else:
    print("  No examples found")

# Final statistics
print("\n" + "=" * 80)
print("FINAL DATASET STATISTICS")
print("=" * 80)
print(f"Total rows: {len(final_df):,}")

selection_counts = final_df['selection_reason'].value_counts()
print(f"\nSelection breakdown:")
for reason in ['Top 100', 'Top 100 + GWAS Gene', 'GWAS Gene']:
    count = selection_counts.get(reason, 0)
    print(f"  {reason}: {count:,}")

print(f"\nUnique genes:")
print(f"  Source: {final_df['source'].nunique():,}")
print(f"  Intermediate: {final_df['intermediate'].nunique():,}")
print(f"  Target: {final_df['target'].nunique():,}")

print(f"\nLogfoldchange statistics:")
print(f"  Min: {final_df['logfoldchange'].min():.4f}")
print(f"  Max: {final_df['logfoldchange'].max():.4f}")
print(f"  Min (absolute): {final_df['abs_logfoldchange'].min():.4f}")
print(f"  Max (absolute): {final_df['abs_logfoldchange'].max():.4f}")

# Drop temporary column and reorder
final_df = final_df.drop('abs_logfoldchange', axis=1)

# Move selection_reason and directionality to the end
cols = final_df.columns.tolist()
cols.remove('selection_reason')
cols.remove('directionality')
cols.append('selection_reason')
cols.append('directionality')
final_df = final_df[cols]

# SAVE OUTPUT 2: Final top 100 + GWAS dataset
output_file_2 = "/Users/prashammarfatia/Downloads/_top100_endothelial_gwas_priority.csv"
final_df.to_csv(output_file_2, index=False)

print("\n" + "=" * 80)
print("OUTPUTS SAVED")
print("=" * 80)
print(f"✓ OUTPUT 1: Endothelial-filtered dataset")
print(f"  File: {output_file_1}")
print(f"  Shape: {endothelial_filtered_df.shape}")
print(f"  Description: All rows with endothelial intermediates (GWAS genes removed from endothelial list)")
print()
print(f"✓ OUTPUT 2: Final top 100 + GWAS dataset")
print(f"  File: {output_file_2}")
print(f"  Shape: {final_df.shape}")
print(f"  Description: Top 100 by logFC + GWAS genes with uniqueness constraint")
print(f"  Columns: {final_df.columns.tolist()}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ Removed {len(overlap)} GWAS genes from endothelial list")
print(f"✓ Filtered by endothelial intermediates: {len(endothelial_filtered_df):,} rows")
print(f"✓ GWAS genes in endothelial-filtered data: {len(gwas_found)} out of {len(gwas_genes)}")
print(f"✓ Percent of endothelial paths with GWAS genes: {percent_with_gwas:.2f}%")
print(f"✓ Selected top 100 by absolute logfoldchange from endothelial-filtered data")
print(f"✓ Added GWAS genes found in endothelial-filtered data")
print(f"✓ Applied uniqueness constraint per GWAS gene")
print(f"✓ Final output: {len(final_df):,} rows")
print(f"✓ Saved 2 CSV files")
print("=" * 80)