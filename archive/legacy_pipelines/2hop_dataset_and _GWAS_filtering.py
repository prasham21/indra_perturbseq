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
endothelial_genes = set(endothelial_df['gene'].tolist())

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

# Check overlap - keep GWAS genes in endothelial list
overlap = gwas_genes.intersection(endothelial_genes)
print("=" * 80)
print("OVERLAP ANALYSIS")
print("=" * 80)
print(f"GWAS genes found in endothelial list: {len(overlap)} out of {len(gwas_genes)}")
print(f"Overlap genes: {sorted(overlap)}")
print(f"\nKeeping all {len(endothelial_genes):,} endothelial genes (including GWAS genes)\n")

# Statistics BEFORE filtering
print("=" * 80)
print("ORIGINAL DATASET STATISTICS")
print("=" * 80)
print(f"Unique genes in 'source': {df['source'].nunique():,}")
print(f"Unique genes in 'intermediate': {df['intermediate'].nunique():,}")
print(f"Unique genes in 'target': {df['target'].nunique():,}")
print(
    f"\nTotal unique genes across all positions: {pd.concat([df['source'], df['intermediate'], df['target']]).nunique():,}\n")

# STEP 1: Filter by endothelial genes (intermediate ONLY) - INCLUDING GWAS genes
print("=" * 80)
print("STEP 1: FILTER BY ENDOTHELIAL INTERMEDIATES (INCLUDING GWAS)")
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

# GWAS genes found in endothelial-filtered dataset
gwas_sources_found = set(endothelial_filtered_df[endothelial_filtered_df['source'].isin(gwas_genes)]['source'].unique())
gwas_intermediates_found = set(
    endothelial_filtered_df[endothelial_filtered_df['intermediate'].isin(gwas_genes)]['intermediate'].unique())
gwas_targets_found = set(endothelial_filtered_df[endothelial_filtered_df['target'].isin(gwas_genes)]['target'].unique())
gwas_found_all = gwas_sources_found | gwas_intermediates_found | gwas_targets_found

print(f"GWAS genes by position in endothelial-filtered dataset:")
print(f"  As source: {len(gwas_sources_found)} genes - {sorted(gwas_sources_found)}")
print(f"  As intermediate: {len(gwas_intermediates_found)} genes - {sorted(gwas_intermediates_found)}")
print(f"  As target: {len(gwas_targets_found)} genes - {sorted(gwas_targets_found)}")
print(f"  Total unique GWAS genes found: {len(gwas_found_all)} out of {len(gwas_genes)}")

gwas_not_found = gwas_genes - gwas_found_all
if gwas_not_found:
    print(f"  GWAS genes NOT found: {sorted(gwas_not_found)}\n")

# SAVE OUTPUT 1: Endothelial-filtered dataset
output_file_1 = "/Users/prashammarfatia/Downloads/2hop_endothelial_filtered.csv"
endothelial_filtered_df.to_csv(output_file_1, index=False)
print(f"✓ OUTPUT 1 SAVED: {output_file_1}")
print(f"  Shape: {endothelial_filtered_df.shape}\n")

# Add absolute logfoldchange for sorting
endothelial_filtered_df['abs_logfoldchange'] = endothelial_filtered_df['logfoldchange'].abs()

# STEP 2: Create TOP 100 CSV (with source-target uniqueness first) - CORRECTED
print("=" * 80)
print("STEP 2: CREATE TOP 100 CSV (WITH SOURCE-TARGET UNIQUENESS) - CORRECTED")
print("=" * 80)

# Apply source-target uniqueness to entire endothelial-filtered dataset
print(f"Applying source-target uniqueness to endothelial-filtered dataset...")
print(f"  Before uniqueness: {len(endothelial_filtered_df):,} rows")
print(f"  Unique source-target pairs: {endothelial_filtered_df.groupby(['source', 'target']).ngroups:,}")

# CORRECTED: Use sort + drop_duplicates to keep highest abs_logfoldchange per source-target pair
unique_endo_df = (endothelial_filtered_df
                  .sort_values('abs_logfoldchange', ascending=False)
                  .drop_duplicates(subset=['source', 'target'], keep='first'))

print(f"  After uniqueness: {len(unique_endo_df):,} rows")
print(f"  Rows removed: {len(endothelial_filtered_df) - len(unique_endo_df):,}")
print(
    f"  LogFC range after uniqueness: {unique_endo_df['abs_logfoldchange'].min():.4f} to {unique_endo_df['abs_logfoldchange'].max():.4f}\n")

# Now take top 100
top_100 = unique_endo_df.head(100).copy()

print(f"Selected top 100 from unique source-target pairs:")
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
print(f"  Rows without GWAS genes in top 100: {(~top_100_gwas_mask).sum()}")

# Add directionality for top 100
top_100['directionality'] = 'No'
top_100.loc[
    ((top_100['stmt_type_2'] == 'IncreaseAmount') & (top_100['logfoldchange'] < 0)) |
    ((top_100['stmt_type_2'] == 'DecreaseAmount') & (top_100['logfoldchange'] > 0)),
    'directionality'
] = 'Yes'

directionality_top100 = top_100['directionality'].value_counts()
print(f"\nDirectionality in top 100:")
print(
    f"  Yes (Agreement): {directionality_top100.get('Yes', 0):,} ({directionality_top100.get('Yes', 0) / len(top_100) * 100:.1f}%)")
print(
    f"  No (Disagreement/Other): {directionality_top100.get('No', 0):,} ({directionality_top100.get('No', 0) / len(top_100) * 100:.1f}%)")

# Show top 10 to verify
print(f"\nTop 10 rows to verify:")
print(top_100[['source', 'intermediate', 'target', 'logfoldchange', 'abs_logfoldchange']].head(10).to_string())

# Reorder columns and save top 100
top_100_final = top_100.drop('abs_logfoldchange', axis=1)
cols = top_100_final.columns.tolist()

# Reorder: source, intermediate, target first
priority_cols = ['source', 'intermediate', 'target']
other_cols = [c for c in cols if c not in priority_cols and c != 'directionality']
top_100_final = top_100_final[priority_cols + other_cols + ['directionality']]

output_file_top100 = "/Users/prashammarfatia/Downloads/top100_endothelial.csv"
top_100_final.to_csv(output_file_top100, index=False)
print(f"\n✓ TOP 100 CSV SAVED: {output_file_top100}")
print(f"  Shape: {top_100_final.shape}")
print(f"  Columns: {top_100_final.columns.tolist()}\n")

# STEP 3: Create GWAS CSV
print("=" * 80)
print("STEP 3: CREATE GWAS CSV")
print("=" * 80)

# Find all rows with GWAS genes
gwas_rows_all = endothelial_filtered_df[gwas_mask_in_endo].copy()
print(f"Total rows containing GWAS genes: {len(gwas_rows_all):,}")

# Separate by GWAS position
gwas_as_source = endothelial_filtered_df[endothelial_filtered_df['source'].isin(gwas_genes)].copy()
gwas_as_intermediate = endothelial_filtered_df[endothelial_filtered_df['intermediate'].isin(gwas_genes)].copy()
gwas_as_target = endothelial_filtered_df[endothelial_filtered_df['target'].isin(gwas_genes)].copy()

print(f"\nGWAS rows by position (before uniqueness):")
print(f"  GWAS as source: {len(gwas_as_source):,} rows")
print(f"  GWAS as intermediate: {len(gwas_as_intermediate):,} rows")
print(f"  GWAS as target: {len(gwas_as_target):,} rows")

# Apply uniqueness logic:
# 1. For source/target: apply source-target uniqueness
# 2. For intermediate: keep all (no uniqueness)

print(f"\nApplying GWAS uniqueness logic...")
print(f"  Logic: Unique source-target pairs for GWAS as source/target")
print(f"         Keep ALL paths where GWAS is intermediate\n")

# Process each GWAS gene individually
gwas_final_rows = []
gwas_stats = {}

for gwas_gene in sorted(gwas_genes):
    gene_mask = (
        (endothelial_filtered_df['source'] == gwas_gene) |
        (endothelial_filtered_df['intermediate'] == gwas_gene) |
        (endothelial_filtered_df['target'] == gwas_gene)
    )
    gene_rows = endothelial_filtered_df[gene_mask].copy()

    if len(gene_rows) > 0:
        # Separate by position
        as_source = gene_rows[gene_rows['source'] == gwas_gene].copy()
        as_intermediate = gene_rows[gene_rows['intermediate'] == gwas_gene].copy()
        as_target = gene_rows[gene_rows['target'] == gwas_gene].copy()

        # For source: apply source-target uniqueness - CORRECTED
        if len(as_source) > 0:
            as_source_unique = (as_source
                                .sort_values('abs_logfoldchange', ascending=False)
                                .drop_duplicates(subset=['source', 'target'], keep='first'))
        else:
            as_source_unique = as_source

        # For target: apply source-target uniqueness - CORRECTED
        if len(as_target) > 0:
            as_target_unique = (as_target
                                .sort_values('abs_logfoldchange', ascending=False)
                                .drop_duplicates(subset=['source', 'target'], keep='first'))
        else:
            as_target_unique = as_target

        # For intermediate: keep all (no uniqueness)
        as_intermediate_unique = as_intermediate

        # Combine and remove duplicates (in case a row has GWAS in multiple positions)
        gene_combined = pd.concat([as_source_unique, as_intermediate_unique, as_target_unique]).drop_duplicates()

        gwas_final_rows.append(gene_combined)

        # Track stats
        gwas_stats[gwas_gene] = {
            'total_before': len(gene_rows),
            'total_after': len(gene_combined),
            'source_before': len(as_source),
            'source_after': len(as_source_unique),
            'intermediate_before': len(as_intermediate),
            'intermediate_after': len(as_intermediate_unique),
            'target_before': len(as_target),
            'target_after': len(as_target_unique),
        }

        print(f"  {gwas_gene}:")
        print(
            f"    Total: {len(gene_rows):,} → {len(gene_combined):,} rows (removed {len(gene_rows) - len(gene_combined):,})")
        print(f"    Source: {len(as_source):,} → {len(as_source_unique):,}")
        print(f"    Intermediate: {len(as_intermediate):,} → {len(as_intermediate_unique):,} (no uniqueness)")
        print(f"    Target: {len(as_target):,} → {len(as_target_unique):,}")

# Combine all GWAS rows
if gwas_final_rows:
    gwas_df = pd.concat(gwas_final_rows).drop_duplicates()
else:
    gwas_df = pd.DataFrame()

print(f"\n{'=' * 80}")
print(f"GWAS CSV STATISTICS")
print(f"{'=' * 80}")
print(f"Total GWAS rows after uniqueness: {len(gwas_df):,}")

# Count by position in final GWAS dataset
gwas_final_sources = gwas_df[gwas_df['source'].isin(gwas_genes)]
gwas_final_intermediates = gwas_df[gwas_df['intermediate'].isin(gwas_genes)]
gwas_final_targets = gwas_df[gwas_df['target'].isin(gwas_genes)]

print(f"\nFinal GWAS rows by position:")
print(f"  GWAS as source: {len(gwas_final_sources):,} rows")
print(f"  GWAS as intermediate: {len(gwas_final_intermediates):,} rows")
print(f"  GWAS as target: {len(gwas_final_targets):,} rows")

# Unique GWAS genes in final dataset
gwas_sources_final = set(gwas_final_sources['source'].unique())
gwas_intermediates_final = set(gwas_final_intermediates['intermediate'].unique())
gwas_targets_final = set(gwas_final_targets['target'].unique())
gwas_all_final = gwas_sources_final | gwas_intermediates_final | gwas_targets_final

print(f"\nUnique GWAS genes in final GWAS CSV:")
print(f"  As source: {len(gwas_sources_final)} out of {len(gwas_genes)} - {sorted(gwas_sources_final)}")
print(
    f"  As intermediate: {len(gwas_intermediates_final)} out of {len(gwas_genes)} - {sorted(gwas_intermediates_final)}")
print(f"  As target: {len(gwas_targets_final)} out of {len(gwas_genes)} - {sorted(gwas_targets_final)}")
print(f"  Total unique GWAS genes: {len(gwas_all_final)} out of {len(gwas_genes)}")


# Add GWAS_genes_in_path column
def get_gwas_in_path(row):
    gwas_in_path = []
    if row['source'] in gwas_genes:
        gwas_in_path.append(row['source'])
    if row['intermediate'] in gwas_genes:
        gwas_in_path.append(row['intermediate'])
    if row['target'] in gwas_genes:
        gwas_in_path.append(row['target'])
    return ', '.join(sorted(set(gwas_in_path)))


gwas_df['GWAS_genes_in_path'] = gwas_df.apply(get_gwas_in_path, axis=1)

# Add directionality
gwas_df['directionality'] = 'No'
gwas_df.loc[
    ((gwas_df['stmt_type_2'] == 'IncreaseAmount') & (gwas_df['logfoldchange'] < 0)) |
    ((gwas_df['stmt_type_2'] == 'DecreaseAmount') & (gwas_df['logfoldchange'] > 0)),
    'directionality'
] = 'Yes'

directionality_gwas = gwas_df['directionality'].value_counts()
print(f"\nDirectionality in GWAS paths:")
print(
    f"  Yes (Agreement): {directionality_gwas.get('Yes', 0):,} ({directionality_gwas.get('Yes', 0) / len(gwas_df) * 100:.1f}%)")
print(
    f"  No (Disagreement/Other): {directionality_gwas.get('No', 0):,} ({directionality_gwas.get('No', 0) / len(gwas_df) * 100:.1f}%)")

# Check overlap with top 100 - CORRECTED
# Need to compare actual indices between the two dataframes
overlap_indices = set(gwas_df.index).intersection(set(top_100.index))
print(f"\nOverlap with top 100:")
print(f"  GWAS rows also in top 100: {len(overlap_indices):,}")

# Reorder columns and save GWAS CSV
gwas_df_final = gwas_df.drop('abs_logfoldchange', axis=1)
cols = gwas_df_final.columns.tolist()

# Reorder: source, intermediate, target first, then GWAS_genes_in_path and directionality at end
priority_cols = ['source', 'intermediate', 'target']
end_cols = ['GWAS_genes_in_path', 'directionality']
middle_cols = [c for c in cols if c not in priority_cols and c not in end_cols]
gwas_df_final = gwas_df_final[priority_cols + middle_cols + end_cols]

# Sort by absolute logfoldchange
gwas_df_final['abs_logfoldchange_temp'] = gwas_df_final['logfoldchange'].abs()
gwas_df_final = gwas_df_final.sort_values('abs_logfoldchange_temp', ascending=False)
gwas_df_final = gwas_df_final.drop('abs_logfoldchange_temp', axis=1)

output_file_gwas = "/Users/prashammarfatia/Downloads/gwas_endothelial_paths.csv"
gwas_df_final.to_csv(output_file_gwas, index=False)
print(f"\n✓ GWAS CSV SAVED: {output_file_gwas}")
print(f"  Shape: {gwas_df_final.shape}")
print(f"  Columns: {gwas_df_final.columns.tolist()}")

# Show top 10 GWAS rows to verify
print(f"\nTop 10 GWAS rows by abs(logFC):")
print(gwas_df_final[['source', 'intermediate', 'target', 'logfoldchange', 'GWAS_genes_in_path']].head(10).to_string())

# FINAL SUMMARY
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"✓ OUTPUT 1: Endothelial-filtered dataset")
print(f"  File: {output_file_1}")
print(f"  Rows: {len(endothelial_filtered_df):,}")
print()
print(f"✓ OUTPUT 2: Top 100 (with source-target uniqueness - CORRECTED)")
print(f"  File: {output_file_top100}")
print(f"  Rows: {len(top_100_final):,}")
print(f"  GWAS genes present: {top_100_gwas_mask.sum():,} rows")
print(f"  LogFC range: {top_100['abs_logfoldchange'].min():.4f} to {top_100['abs_logfoldchange'].max():.4f}")
print()
print(f"✓ OUTPUT 3: GWAS paths (with corrected uniqueness)")
print(f"  File: {output_file_gwas}")
print(f"  Rows: {len(gwas_df_final):,}")
print(f"  GWAS genes covered: {len(gwas_all_final)} out of {len(gwas_genes)}")
print(f"  Overlap with top 100: {len(overlap_indices):,} rows")
print()
print(f"Key statistics:")
print(f"  Total endothelial-filtered rows: {len(endothelial_filtered_df):,}")
print(f"  Percent with GWAS genes: {percent_with_gwas:.2f}%")
print(f"  GWAS genes as intermediates: {len(gwas_intermediates_final)} genes")
print("=" * 80)