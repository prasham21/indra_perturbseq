import pandas as pd

# Load the filtered dataset
df = pd.read_csv("/Users/prashammarfatia/Downloads/2hop_mesh_filtered_endothelial_gwas.csv")

print("=" * 80)
print("CREATING TOP 100 LOGFOLDCHANGE FILE WITH GWAS PRIORITY")
print("=" * 80)
print(f"Starting with filtered dataset: {len(df):,} rows\n")

# GWAS genes list
gwas_genes = {
    'BCAR1', 'BMP1', 'CALCRL', 'CCM2', 'CDKN1A', 'CDKN2B', 'CFDP1', 'COL4A1',
    'COL4A2', 'EXOC3L2', 'FBN2', 'FGD6', 'FLT1', 'FURIN', 'GDPD5', 'GGT5',
    'GOSR2', 'IBTK', 'LAMB2', 'LOX', 'MORF4L1', 'N4BP2L2', 'NOS3', 'PALLD',
    'PECAM1', 'PGF', 'PLPP3', 'PREX1', 'PRKAR1A', 'SCUBE1', 'SERPINH1',
    'SH3PXD2A', 'SLK', 'SMAD3', 'SPRY4', 'SVIL', 'SWAP70', 'TFPI', 'TLNRD1',
    'TSPAN14', 'ZEB2'
}

# Add absolute logfoldchange column for sorting
df['abs_logfoldchange'] = df['logfoldchange'].abs()

print("=" * 80)
print("STEP 1: TOP 100 BY ABSOLUTE LOGFOLDCHANGE")
print("=" * 80)
# Step 1: Get top 100 by absolute logfoldchange
df_sorted = df.sort_values('abs_logfoldchange', ascending=False)
top_100 = df_sorted.head(100).copy()
top_100_indices = set(top_100.index)
print(f"Selected top 100 rows by absolute logfoldchange")
print(f"  Logfoldchange range: {top_100['logfoldchange'].min():.4f} to {top_100['logfoldchange'].max():.4f}")
print(
    f"  Absolute logfoldchange range: {top_100['abs_logfoldchange'].min():.4f} to {top_100['abs_logfoldchange'].max():.4f}")

# Check how many GWAS genes in top 100
top_100_gwas_mask = (
    top_100['source'].isin(gwas_genes) |
    top_100['intermediate'].isin(gwas_genes) |
    top_100['target'].isin(gwas_genes)
)
print(f"  Rows with GWAS genes in top 100: {top_100_gwas_mask.sum()}")
print(f"  Rows without GWAS genes in top 100: {(~top_100_gwas_mask).sum()}")

print("\n" + "=" * 80)
print("STEP 2: FIND ALL ROWS CONTAINING GWAS GENES")
print("=" * 80)
# Step 2: Get all rows containing any GWAS gene
mask_gwas = (
    df['source'].isin(gwas_genes) |
    df['intermediate'].isin(gwas_genes) |
    df['target'].isin(gwas_genes)
)
gwas_rows = df[mask_gwas].copy()
gwas_indices = set(gwas_rows.index)
print(f"Total rows containing GWAS genes: {len(gwas_rows):,}")

# Unique GWAS genes found
gwas_found = set()
for col in ['source', 'intermediate', 'target']:
    gwas_found.update(gwas_rows[col][gwas_rows[col].isin(gwas_genes)].unique())
print(f"  Unique GWAS genes present: {len(gwas_found)} out of {len(gwas_genes)}")

# Find overlap between top 100 and GWAS
overlap_indices = top_100_indices.intersection(gwas_indices)
print(f"\n  Rows in BOTH top 100 AND GWAS: {len(overlap_indices)}")

# Combine top 100 with GWAS rows (remove duplicates)
combined = pd.concat([top_100, gwas_rows]).drop_duplicates()
print(f"  Combined dataset (top 100 + all GWAS rows): {len(combined):,} rows")

# Add selection_reason column before uniqueness constraint
combined['selection_reason'] = combined.index.map(
    lambda idx: 'Top 100 + GWAS Gene' if idx in overlap_indices
    else ('Top 100' if idx in top_100_indices
          else 'GWAS Gene')
)

print("\n" + "=" * 80)
print("STEP 3: APPLY UNIQUENESS CONSTRAINT PER GWAS GENE")
print("=" * 80)
# Step 3: Apply uniqueness constraint for each GWAS gene
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

# Get non-GWAS rows from top 100 (these don't need uniqueness constraint)
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
    # Remove any duplicates that might occur if same row has multiple GWAS genes
    duplicates_before = len(final_gwas_df)
    final_gwas_df = final_gwas_df.drop_duplicates()
    duplicates_removed = duplicates_before - len(final_gwas_df)

    print(f"\n  Combined GWAS rows: {len(final_gwas_df):,}")
    print(f"  Duplicates removed (rows with multiple GWAS genes): {duplicates_removed:,}")

    # Combine with non-GWAS rows
    final_df = pd.concat([final_gwas_df, non_gwas_rows]).drop_duplicates()
else:
    final_df = non_gwas_rows

# Sort by absolute logfoldchange for final output (DESCENDING - highest first)
final_df = final_df.sort_values('abs_logfoldchange', ascending=False)

print("\n" + "=" * 80)
print("STEP 4: ADD DIRECTIONALITY COLUMN")
print("=" * 80)
# Add directionality column
# Yes = Agreement between INDRA statement and observed effect (given source knockdown)
# Logic:
#   - stmt_type_2 == "IncreaseAmount" AND logfoldchange < 0 → Yes (intermediate increases target, knockdown causes decrease)
#   - stmt_type_2 == "DecreaseAmount" AND logfoldchange > 0 → Yes (intermediate decreases target, knockdown causes increase)
#   - Otherwise → No

final_df['directionality'] = 'No'
final_df.loc[
    ((final_df['stmt_type_2'] == 'IncreaseAmount') & (final_df['logfoldchange'] < 0)) |
    ((final_df['stmt_type_2'] == 'DecreaseAmount') & (final_df['logfoldchange'] > 0)),
    'directionality'
] = 'Yes'

# Count directionality
directionality_counts = final_df['directionality'].value_counts()
print(f"Directionality breakdown:")
print(
    f"  Yes (Agreement): {directionality_counts.get('Yes', 0):,} ({directionality_counts.get('Yes', 0) / len(final_df) * 100:.1f}%)")
print(
    f"  No (Disagreement/Other): {directionality_counts.get('No', 0):,} ({directionality_counts.get('No', 0) / len(final_df) * 100:.1f}%)")

# Show some examples
print(f"\nExamples of 'Yes' directionality:")
yes_examples = final_df[final_df['directionality'] == 'Yes'][
    ['source', 'intermediate', 'target', 'stmt_type_2', 'logfoldchange', 'directionality']].head(5)
if len(yes_examples) > 0:
    print(yes_examples.to_string(index=False))
else:
    print("  No examples found")

print(f"\nExamples of 'No' directionality:")
no_examples = final_df[final_df['directionality'] == 'No'][
    ['source', 'intermediate', 'target', 'stmt_type_2', 'logfoldchange', 'directionality']].head(5)
if len(no_examples) > 0:
    print(no_examples.to_string(index=False))

print("\n" + "=" * 80)
print("FINAL DATASET STATISTICS")
print("=" * 80)
print(f"Total rows: {len(final_df):,}")

# Count by selection reason
selection_counts = final_df['selection_reason'].value_counts()
print(f"\nSelection breakdown:")
for reason in ['Top 100', 'Top 100 + GWAS Gene', 'GWAS Gene']:
    count = selection_counts.get(reason, 0)
    print(f"  {reason}: {count:,}")

# Logfoldchange statistics
print(f"\nLogfoldchange statistics:")
print(f"  Min (absolute): {final_df['abs_logfoldchange'].min():.4f}")
print(f"  Max (absolute): {final_df['abs_logfoldchange'].max():.4f}")

# Drop the temporary abs_logfoldchange column
final_df = final_df.drop('abs_logfoldchange', axis=1)

# Move selection_reason and directionality to the LAST positions (in that order)
cols = final_df.columns.tolist()
cols.remove('selection_reason')
cols.remove('directionality')
cols.append('selection_reason')
cols.append('directionality')
final_df = final_df[cols]

# Save the final dataset
output_file = "/Users/prashammarfatia/Downloads/top100_logfc_gwas_priority__.csv"
final_df.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)
print(f"Saved to: {output_file}")
print(f"Final shape: {final_df.shape}")
print(f"\nColumn order (last two columns):")
for i, col in enumerate(final_df.columns, 1):
    marker = ""
    if col == 'selection_reason':
        marker = " ← SELECTION REASON (2nd to last)"
    elif col == 'directionality':
        marker = " ← DIRECTIONALITY (last)"
    print(f"  {i}. {col}{marker}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ Sorted final output by absolute logfoldchange (descending)")
print(f"✓ Added 'directionality' column (Yes = agreement with INDRA statement)")
print(f"✓ Added 'selection_reason' column")
print(f"✓ Both new columns placed at the end (selection_reason, then directionality)")
print(f"✓ Final output: {len(final_df):,} rows")
print("=" * 80)