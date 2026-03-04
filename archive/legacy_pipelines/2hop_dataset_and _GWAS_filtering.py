"""Legacy script: 2hop_dataset_and _GWAS_filtering."""
from __future__ import annotations

import argparse
import pandas as pd

import logging


logger = logging.getLogger(__name__)
# Load the main dataset
def get_gwas_in_path(row):
    gwas_in_path = []
    if row['source'] in gwas_genes:
        gwas_in_path.append(row['source'])
    if row['intermediate'] in gwas_genes:
        gwas_in_path.append(row['intermediate'])
    if row['target'] in gwas_genes:
        gwas_in_path.append(row['target'])
    return ', '.join(sorted(set(gwas_in_path)))




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-2hop-mesh-filtered", default="_new_2hop_mesh_filtered.csv", help="Path: _new_2hop_mesh_filtered.csv")
    ap.add_argument("--endothelial-present-plus-manua", default="endothelial_present_plus_manual.csv", help="Path: endothelial_present_plus_manual.csv")
    ap.add_argument("--2hop-endothelial-filtered", default="2hop_endothelial_filtered.csv", help="Path: 2hop_endothelial_filtered.csv")
    ap.add_argument("--top100-endothelia", default="top100_endothelial.csv", help="Path: top100_endothelial.csv")
    ap.add_argument("--gwas-endothelial-path", default="gwas_endothelial_paths.csv", help="Path: gwas_endothelial_paths.csv")
    args = ap.parse_args()

    df = pd.read_csv("_new_2hop_mesh_filtered.csv")

    logger.info("=" * 80)
    logger.info("ORIGINAL DATASET")
    logger.info("=" * 80)
    logger.info("Shape: %s", df.shape)
    logger.info("Total rows: %d", len(df))
    logger.info("Total columns: %s", len(df.columns))
    logger.info("Columns: %s\n", df.columns.tolist())

    # Load endothelial genes
    endothelial_df = pd.read_csv("endothelial_present_plus_manual.csv")
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

    logger.info("=" * 80)
    logger.info("GWAS GENES")
    logger.info("=" * 80)
    logger.info("Number of GWAS genes: %s", len(gwas_genes))
    logger.info("GWAS genes: %s\n", sorted(gwas_genes))

    # Check overlap - keep GWAS genes in endothelial list
    overlap = gwas_genes.intersection(endothelial_genes)
    logger.info("=" * 80)
    logger.info("OVERLAP ANALYSIS")
    logger.info("=" * 80)
    logger.info("GWAS genes found in endothelial list: %s out of %s", len(overlap), len(gwas_genes))
    logger.info("Overlap genes: %s", sorted(overlap))
    logger.info("\nKeeping all %d endothelial genes (including GWAS genes)\n", len(endothelial_genes))

    # Statistics BEFORE filtering
    logger.info("=" * 80)
    logger.info("ORIGINAL DATASET STATISTICS")
    logger.info("=" * 80)
    logger.info("Unique genes in 'source': %d", df['source'].nunique())
    logger.info("Unique genes in 'intermediate': %d", df['intermediate'].nunique())
    logger.info("Unique genes in 'target': %d", df['target'].nunique())
    logger.info(f"\nTotal unique genes across all positions: {pd.concat([df['source'], df['intermediate'], df['target']]).nunique():,}\n"))

    logger.info("=" * 80)
    logger.info("STEP 1: FILTER BY ENDOTHELIAL INTERMEDIATES (INCLUDING GWAS)")
    logger.info("=" * 80)
    mask_intermediate = df['intermediate'].isin(endothelial_genes)
    endothelial_filtered_df = df[mask_intermediate].copy()

    logger.info("Rows with intermediate in endothelial list: %d out of %d", len(endothelial_filtered_df), len(df))
    logger.info("Percentage retained: %.2f%%", len(endothelial_filtered_df) / len(df) * 100)
    logger.info("Rows removed: %d\n", len(df) - len(endothelial_filtered_df))

    # Statistics on endothelial-filtered dataset
    logger.info("Endothelial-filtered dataset statistics:")
    logger.info("  Unique genes in 'source': %d", endothelial_filtered_df['source'].nunique())
    logger.info("  Unique genes in 'intermediate': %d", endothelial_filtered_df['intermediate'].nunique())
    logger.info("  Unique genes in 'target': %d\n", endothelial_filtered_df['target'].nunique())

    # GWAS gene presence in endothelial-filtered dataset
    logger.info("=" * 80)
    logger.info("GWAS GENE PRESENCE IN ENDOTHELIAL-FILTERED DATASET")
    logger.info("=" * 80)
    gwas_mask_in_endo = (
        endothelial_filtered_df['source'].isin(gwas_genes) |
        endothelial_filtered_df['intermediate'].isin(gwas_genes) |
        endothelial_filtered_df['target'].isin(gwas_genes)
    )
    rows_with_gwas = gwas_mask_in_endo.sum()
    percent_with_gwas = (rows_with_gwas / len(endothelial_filtered_df)) * 100

    logger.info("Total endothelial-filtered rows: %d", len(endothelial_filtered_df))
    logger.info("Rows containing at least one GWAS gene: %d", rows_with_gwas)
    logger.info("Percentage of paths with GWAS genes: %.2f%%", percent_with_gwas)
    logger.info("Rows without any GWAS genes: %d\n", len(endothelial_filtered_df) - rows_with_gwas)

    # GWAS genes found in endothelial-filtered dataset
    gwas_sources_found = set(endothelial_filtered_df[endothelial_filtered_df['source'].isin(gwas_genes)]['source'].unique())
    gwas_intermediates_found = set(
        endothelial_filtered_df[endothelial_filtered_df['intermediate'].isin(gwas_genes)]['intermediate'].unique())
    gwas_targets_found = set(endothelial_filtered_df[endothelial_filtered_df['target'].isin(gwas_genes)]['target'].unique())
    gwas_found_all = gwas_sources_found | gwas_intermediates_found | gwas_targets_found

    logger.info("GWAS genes by position in endothelial-filtered dataset:")
    logger.info("  As source: %s genes - %s", len(gwas_sources_found), sorted(gwas_sources_found))
    logger.info("  As intermediate: %s genes - %s", len(gwas_intermediates_found), sorted(gwas_intermediates_found))
    logger.info("  As target: %s genes - %s", len(gwas_targets_found), sorted(gwas_targets_found))
    logger.info("  Total unique GWAS genes found: %s out of %s", len(gwas_found_all), len(gwas_genes))

    gwas_not_found = gwas_genes - gwas_found_all
    if gwas_not_found:
        logger.info("  GWAS genes NOT found: %s\n", sorted(gwas_not_found))

    # SAVE OUTPUT 1: Endothelial-filtered dataset
    output_file_1 = "2hop_endothelial_filtered.csv"
    endothelial_filtered_df.to_csv(output_file_1, index=False)
    logger.info(" OUTPUT 1 SAVED: %s", output_file_1)
    logger.info("  Shape: %s\n", endothelial_filtered_df.shape)

    # Add absolute logfoldchange for sorting
    endothelial_filtered_df['abs_logfoldchange'] = endothelial_filtered_df['logfoldchange'].abs()

    logger.info("=" * 80)
    logger.info("STEP 2: CREATE TOP 100 CSV (WITH SOURCE-TARGET UNIQUENESS) - CORRECTED")
    logger.info("=" * 80)

    # Apply source-target uniqueness to entire endothelial-filtered dataset
    logger.info("Applying source-target uniqueness to endothelial-filtered dataset...")
    logger.info("  Before uniqueness: %d rows", len(endothelial_filtered_df))
    logger.info("  Unique source-target pairs: %d", endothelial_filtered_df.groupby(['source', 'target']).ngroups)

    # CORRECTED: Use sort + drop_duplicates to keep highest abs_logfoldchange per source-target pair
    unique_endo_df = (endothelial_filtered_df
                      .sort_values('abs_logfoldchange', ascending=False)
                      .drop_duplicates(subset=['source', 'target'], keep='first'))

    logger.info("  After uniqueness: %d rows", len(unique_endo_df))
    logger.info("  Rows removed: %d", len(endothelial_filtered_df) - len(unique_endo_df))
    logger.info(f"  LogFC range after uniqueness: {unique_endo_df['abs_logfoldchange'].min():.4f} to {unique_endo_df['abs_logfoldchange'].max():.4f}\n"))

    # Now take top 100
    top_100 = unique_endo_df.head(100).copy()

    logger.info("Selected top 100 from unique source-target pairs:")
    logger.info("  Logfoldchange range: %.4f to %.4f", top_100['logfoldchange'].min(), top_100['logfoldchange'].max())
    logger.info(f"  Absolute logfoldchange range: {top_100['abs_logfoldchange'].min():.4f} to {top_100['abs_logfoldchange'].max():.4f}"))

    # Check if any GWAS genes in top 100
    top_100_gwas_mask = (
        top_100['source'].isin(gwas_genes) |
        top_100['intermediate'].isin(gwas_genes) |
        top_100['target'].isin(gwas_genes)
    )
    logger.info("  Rows with GWAS genes in top 100: %s", top_100_gwas_mask.sum())
    logger.info("  Rows without GWAS genes in top 100: %s", (~top_100_gwas_mask).sum())

    # Add directionality for top 100
    top_100['directionality'] = 'No'
    top_100.loc[
        ((top_100['stmt_type_2'] == 'IncreaseAmount') & (top_100['logfoldchange'] < 0)) |
        ((top_100['stmt_type_2'] == 'DecreaseAmount') & (top_100['logfoldchange'] > 0)),
        'directionality'
    ] = 'Yes'

    directionality_top100 = top_100['directionality'].value_counts()
    logger.info("\nDirectionality in top 100:")
    logger.info(f"  Yes (Agreement): {directionality_top100.get('Yes', 0):,} ({directionality_top100.get('Yes', 0) / len(top_100) * 100:.1f}%)"))
    logger.info(f"  No (Disagreement/Other): {directionality_top100.get('No', 0):,} ({directionality_top100.get('No', 0) / len(top_100) * 100:.1f}%)"))

    # Show top 10 to verify
    logger.info("\nTop 10 rows to verify:")
    logger.info(top_100[['source', 'intermediate', 'target', 'logfoldchange', 'abs_logfoldchange']].head(10).to_string())

    # Reorder columns and save top 100
    top_100_final = top_100.drop('abs_logfoldchange', axis=1)
    cols = top_100_final.columns.tolist()

    # Reorder: source, intermediate, target first
    priority_cols = ['source', 'intermediate', 'target']
    other_cols = [c for c in cols if c not in priority_cols and c != 'directionality']
    top_100_final = top_100_final[priority_cols + other_cols + ['directionality']]

    output_file_top100 = "top100_endothelial.csv"
    top_100_final.to_csv(output_file_top100, index=False)
    logger.info("\n TOP 100 CSV SAVED: %s", output_file_top100)
    logger.info("  Shape: %s", top_100_final.shape)
    logger.info("  Columns: %s\n", top_100_final.columns.tolist())

    logger.info("=" * 80)
    logger.info("STEP 3: CREATE GWAS CSV")
    logger.info("=" * 80)

    # Find all rows with GWAS genes
    gwas_rows_all = endothelial_filtered_df[gwas_mask_in_endo].copy()
    logger.info("Total rows containing GWAS genes: %d", len(gwas_rows_all))

    # Separate by GWAS position
    gwas_as_source = endothelial_filtered_df[endothelial_filtered_df['source'].isin(gwas_genes)].copy()
    gwas_as_intermediate = endothelial_filtered_df[endothelial_filtered_df['intermediate'].isin(gwas_genes)].copy()
    gwas_as_target = endothelial_filtered_df[endothelial_filtered_df['target'].isin(gwas_genes)].copy()

    logger.info("\nGWAS rows by position (before uniqueness):")
    logger.info("  GWAS as source: %d rows", len(gwas_as_source))
    logger.info("  GWAS as intermediate: %d rows", len(gwas_as_intermediate))
    logger.info("  GWAS as target: %d rows", len(gwas_as_target))

    # Apply uniqueness logic:

    logger.info("\nApplying GWAS uniqueness logic...")
    logger.info("  Logic: Unique source-target pairs for GWAS as source/target")
    logger.info("         Keep ALL paths where GWAS is intermediate\n")

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

            logger.info("  %s:", gwas_gene)
            logger.info(f"    Total: {len(gene_rows):,} → {len(gene_combined):,} rows (removed {len(gene_rows) - len(gene_combined):,})")
            logger.info("    Source: %d → %d", len(as_source), len(as_source_unique))
            logger.info("    Intermediate: %d → %d (no uniqueness)", len(as_intermediate), len(as_intermediate_unique))
            logger.info("    Target: %d → %d", len(as_target), len(as_target_unique))

    # Combine all GWAS rows
    if gwas_final_rows:
        gwas_df = pd.concat(gwas_final_rows).drop_duplicates()
    else:
        gwas_df = pd.DataFrame()

    logger.info("\n%s", '=' * 80)
    logger.info("GWAS CSV STATISTICS")
    logger.info("%s", '=' * 80)
    logger.info("Total GWAS rows after uniqueness: %d", len(gwas_df))

    # Count by position in final GWAS dataset
    gwas_final_sources = gwas_df[gwas_df['source'].isin(gwas_genes)]
    gwas_final_intermediates = gwas_df[gwas_df['intermediate'].isin(gwas_genes)]
    gwas_final_targets = gwas_df[gwas_df['target'].isin(gwas_genes)]

    logger.info("\nFinal GWAS rows by position:")
    logger.info("  GWAS as source: %d rows", len(gwas_final_sources))
    logger.info("  GWAS as intermediate: %d rows", len(gwas_final_intermediates))
    logger.info("  GWAS as target: %d rows", len(gwas_final_targets))

    # Unique GWAS genes in final dataset
    gwas_sources_final = set(gwas_final_sources['source'].unique())
    gwas_intermediates_final = set(gwas_final_intermediates['intermediate'].unique())
    gwas_targets_final = set(gwas_final_targets['target'].unique())
    gwas_all_final = gwas_sources_final | gwas_intermediates_final | gwas_targets_final

    logger.info("\nUnique GWAS genes in final GWAS CSV:")
    logger.info("  As source: %s out of %s - %s", len(gwas_sources_final), len(gwas_genes), sorted(gwas_sources_final))
    logger.info(f"  As intermediate: {len(gwas_intermediates_final)} out of {len(gwas_genes)} - {sorted(gwas_intermediates_final)}"))
    logger.info("  As target: %s out of %s - %s", len(gwas_targets_final), len(gwas_genes), sorted(gwas_targets_final))
    logger.info("  Total unique GWAS genes: %s out of %s", len(gwas_all_final), len(gwas_genes))


    # Add GWAS_genes_in_path column
    gwas_df['GWAS_genes_in_path'] = gwas_df.apply(get_gwas_in_path, axis=1)

    # Add directionality
    gwas_df['directionality'] = 'No'
    gwas_df.loc[
        ((gwas_df['stmt_type_2'] == 'IncreaseAmount') & (gwas_df['logfoldchange'] < 0)) |
        ((gwas_df['stmt_type_2'] == 'DecreaseAmount') & (gwas_df['logfoldchange'] > 0)),
        'directionality'
    ] = 'Yes'

    directionality_gwas = gwas_df['directionality'].value_counts()
    logger.info("\nDirectionality in GWAS paths:")
    logger.info(f"  Yes (Agreement): {directionality_gwas.get('Yes', 0):,} ({directionality_gwas.get('Yes', 0) / len(gwas_df) * 100:.1f}%)"))
    logger.info(f"  No (Disagreement/Other): {directionality_gwas.get('No', 0):,} ({directionality_gwas.get('No', 0) / len(gwas_df) * 100:.1f}%)"))

    # Check overlap with top 100 - CORRECTED
    # Need to compare actual indices between the two dataframes
    overlap_indices = set(gwas_df.index).intersection(set(top_100.index))
    logger.info("\nOverlap with top 100:")
    logger.info("  GWAS rows also in top 100: %d", len(overlap_indices))

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

    output_file_gwas = "gwas_endothelial_paths.csv"
    gwas_df_final.to_csv(output_file_gwas, index=False)
    logger.info("\n GWAS CSV SAVED: %s", output_file_gwas)
    logger.info("  Shape: %s", gwas_df_final.shape)
    logger.info("  Columns: %s", gwas_df_final.columns.tolist())

    # Show top 10 GWAS rows to verify
    logger.info("\nTop 10 GWAS rows by abs(logFC):")
    logger.info(gwas_df_final[['source', 'intermediate', 'target', 'logfoldchange', 'GWAS_genes_in_path']].head(10).to_string())

    # FINAL SUMMARY
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(" OUTPUT 1: Endothelial-filtered dataset")
    logger.info("  File: %s", output_file_1)
    logger.info("  Rows: %d", len(endothelial_filtered_df))
    logger.info(" OUTPUT 2: Top 100 (with source-target uniqueness - CORRECTED)")
    logger.info("  File: %s", output_file_top100)
    logger.info("  Rows: %d", len(top_100_final))
    logger.info("  GWAS genes present: %d rows", top_100_gwas_mask.sum())
    logger.info("  LogFC range: %.4f to %.4f", top_100['abs_logfoldchange'].min(), top_100['abs_logfoldchange'].max())
    logger.info(" OUTPUT 3: GWAS paths (with corrected uniqueness)")
    logger.info("  File: %s", output_file_gwas)
    logger.info("  Rows: %d", len(gwas_df_final))
    logger.info("  GWAS genes covered: %s out of %s", len(gwas_all_final), len(gwas_genes))
    logger.info("  Overlap with top 100: %d rows", len(overlap_indices))
    logger.info("Key statistics:")
    logger.info("  Total endothelial-filtered rows: %d", len(endothelial_filtered_df))
    logger.info("  Percent with GWAS genes: %.2f%%", percent_with_gwas)
    logger.info("  GWAS genes as intermediates: %s genes", len(gwas_intermediates_final))
    logger.info("=" * 80)



if __name__ == "__main__":
    main()
