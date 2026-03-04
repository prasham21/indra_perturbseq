"""Legacy script: find_2hop_overlap_with_endothelial_genes."""
from __future__ import annotations

import argparse
import pandas as pd

import logging


logger = logging.getLogger(__name__)
# Load the main dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-2hop-mesh-filtered", default="_new_2hop_mesh_filtered.csv", help="Path: _new_2hop_mesh_filtered.csv")
    ap.add_argument("--endothelial-present-plus-manua", default="endothelial_present_plus_manual.csv", help="Path: endothelial_present_plus_manual.csv")
    ap.add_argument("--2hop-endothelial-filtered", default="2hop_endothelial_filtered.csv", help="Path: 2hop_endothelial_filtered.csv")
    ap.add_argument("--top100-endothelial-gwas-priority", default="_top100_endothelial_gwas_priority.csv", help="Path: _top100_endothelial_gwas_priority.csv")
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

    logger.info("=" * 80)
    logger.info("GWAS GENES")
    logger.info("=" * 80)
    logger.info("Number of GWAS genes: %s", len(gwas_genes))
    logger.info("GWAS genes: %s\n", sorted(gwas_genes))

    # Check overlap and REMOVE GWAS genes from endothelial list
    overlap = gwas_genes.intersection(endothelial_genes_original)
    logger.info("=" * 80)
    logger.info("OVERLAP ANALYSIS")
    logger.info("=" * 80)
    logger.info("GWAS genes found in endothelial list: %s out of %s", len(overlap), len(gwas_genes))
    logger.info("Overlap genes: %s", sorted(overlap))

    # Remove GWAS genes from endothelial list
    endothelial_genes = endothelial_genes_original - gwas_genes
    logger.info("\nAfter removing GWAS genes from endothelial list:")
    logger.info("  Original endothelial genes: %d", len(endothelial_genes_original))
    logger.info("  GWAS genes removed: %s", len(overlap))
    logger.info("  Final endothelial genes (excluding GWAS): %d\n", len(endothelial_genes))

    # Statistics BEFORE filtering
    logger.info("=" * 80)
    logger.info("ORIGINAL DATASET STATISTICS")
    logger.info("=" * 80)
    logger.info("Unique genes in 'source': %d", df['source'].nunique())
    logger.info("Unique genes in 'intermediate': %d", df['intermediate'].nunique())
    logger.info("Unique genes in 'target': %d", df['target'].nunique())
    logger.info(f"\nTotal unique genes across all positions: {pd.concat([df['source'], df['intermediate'], df['target']]).nunique():,}\n"))

    logger.info("=" * 80)
    logger.info("STEP 1: FILTER BY ENDOTHELIAL INTERMEDIATES")
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

    # SAVE OUTPUT 1: Endothelial-filtered dataset
    output_file_1 = "2hop_endothelial_filtered.csv"
    endothelial_filtered_df.to_csv(output_file_1, index=False)
    logger.info(" OUTPUT 1 SAVED: %s", output_file_1)
    logger.info("  Shape: %s\n", endothelial_filtered_df.shape)

    # Add absolute logfoldchange for sorting
    endothelial_filtered_df['abs_logfoldchange'] = endothelial_filtered_df['logfoldchange'].abs()

    logger.info("=" * 80)
    logger.info("STEP 2: SELECT TOP 100 BY ABSOLUTE LOGFOLDCHANGE")
    logger.info("=" * 80)
    endothelial_sorted = endothelial_filtered_df.sort_values('abs_logfoldchange', ascending=False)
    top_100 = endothelial_sorted.head(100).copy()
    top_100_indices = set(top_100.index)

    logger.info("Selected top 100 rows by absolute logfoldchange")
    logger.info("  Logfoldchange range: %.4f to %.4f", top_100['logfoldchange'].min(), top_100['logfoldchange'].max())
    logger.info(f"  Absolute logfoldchange range: {top_100['abs_logfoldchange'].min():.4f} to {top_100['abs_logfoldchange'].max():.4f}"))

    # Check if any GWAS genes in top 100
    top_100_gwas_mask = (
        top_100['source'].isin(gwas_genes) |
        top_100['intermediate'].isin(gwas_genes) |
        top_100['target'].isin(gwas_genes)
    )
    logger.info("  Rows with GWAS genes in top 100: %s", top_100_gwas_mask.sum())
    logger.info("  Rows without GWAS genes in top 100: %s\n", (~top_100_gwas_mask).sum())

    logger.info("=" * 80)
    logger.info("STEP 3: FIND GWAS GENES IN ENDOTHELIAL-FILTERED DATASET")
    logger.info("=" * 80)
    mask_gwas = (
        endothelial_filtered_df['source'].isin(gwas_genes) |
        endothelial_filtered_df['intermediate'].isin(gwas_genes) |
        endothelial_filtered_df['target'].isin(gwas_genes)
    )
    gwas_rows = endothelial_filtered_df[mask_gwas].copy()
    gwas_indices = set(gwas_rows.index)

    logger.info("Total rows containing GWAS genes in endothelial-filtered data: %d", len(gwas_rows))
    logger.info("  GWAS in source: %d", endothelial_filtered_df['source'].isin(gwas_genes).sum())
    logger.info("  GWAS in intermediate: %d", endothelial_filtered_df['intermediate'].isin(gwas_genes).sum())
    logger.info("  GWAS in target: %d", endothelial_filtered_df['target'].isin(gwas_genes).sum())

    # Unique GWAS genes found
    gwas_sources_found = set(endothelial_filtered_df[endothelial_filtered_df['source'].isin(gwas_genes)]['source'].unique())
    gwas_intermediates_found = set(
        endothelial_filtered_df[endothelial_filtered_df['intermediate'].isin(gwas_genes)]['intermediate'].unique())
    gwas_targets_found = set(endothelial_filtered_df[endothelial_filtered_df['target'].isin(gwas_genes)]['target'].unique())
    gwas_found = gwas_sources_found | gwas_intermediates_found | gwas_targets_found

    logger.info("\n  Unique GWAS genes present in any position: %s out of %s", len(gwas_found), len(gwas_genes))
    logger.info("  GWAS genes found: %s", sorted(gwas_found))
    logger.info("    As source: %s genes - %s", len(gwas_sources_found), sorted(gwas_sources_found))
    logger.info("    As intermediate: %s genes - %s", len(gwas_intermediates_found), sorted(gwas_intermediates_found))
    logger.info("    As target: %s genes - %s", len(gwas_targets_found), sorted(gwas_targets_found))

    gwas_not_found = gwas_genes - gwas_found
    if gwas_not_found:
        logger.info("\n  GWAS genes NOT found in endothelial-filtered data: %s", sorted(gwas_not_found))

    # Find overlap between top 100 and GWAS
    overlap_indices = top_100_indices.intersection(gwas_indices)
    logger.info("\n  Rows in BOTH top 100 AND GWAS: %s", len(overlap_indices))

    # Combine top 100 with GWAS rows
    combined = pd.concat([top_100, gwas_rows]).drop_duplicates()
    logger.info("  Combined dataset (top 100 + GWAS rows): %d rows", len(combined))
    logger.info("  New rows added from GWAS: %d\n", len(combined) - len(top_100))

    # Add selection_reason column
    combined['selection_reason'] = combined.index.map(
        lambda idx: 'Top 100 + GWAS Gene' if idx in overlap_indices
        else ('Top 100' if idx in top_100_indices
              else 'GWAS Gene')
    )

    logger.info("=" * 80)
    logger.info("STEP 4: APPLY UNIQUENESS CONSTRAINT PER GWAS GENE")
    logger.info("=" * 80)
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
            logger.info(f"  {gwas_gene}: {len(gene_rows):,} rows → {len(unique_gene_rows):,} unique (removed {reduction}) | S:{as_source} I:{as_intermediate} T:{as_target}")

    # Get non-GWAS rows from top 100
    non_gwas_mask = ~(
        combined['source'].isin(gwas_genes) |
        combined['intermediate'].isin(gwas_genes) |
        combined['target'].isin(gwas_genes)
    )
    non_gwas_rows = combined[non_gwas_mask].copy()
    logger.info("\n  Non-GWAS rows from top 100: %d rows (no uniqueness applied)", len(non_gwas_rows))

    # Combine all final rows
    if final_rows:
        final_gwas_df = pd.concat(final_rows)
        duplicates_before = len(final_gwas_df)
        final_gwas_df = final_gwas_df.drop_duplicates()
        duplicates_removed = duplicates_before - len(final_gwas_df)

        logger.info("\n  Combined GWAS rows: %d", len(final_gwas_df))
        logger.info("  Duplicates removed (rows with multiple GWAS genes): %d", duplicates_removed)

        final_df = pd.concat([final_gwas_df, non_gwas_rows]).drop_duplicates()
    else:
        final_df = non_gwas_rows

    # Sort by absolute logfoldchange
    final_df = final_df.sort_values('abs_logfoldchange', ascending=False)

    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: ADD DIRECTIONALITY COLUMN")
    logger.info("=" * 80)
    final_df['directionality'] = 'No'
    final_df.loc[
        ((final_df['stmt_type_2'] == 'IncreaseAmount') & (final_df['logfoldchange'] < 0)) |
        ((final_df['stmt_type_2'] == 'DecreaseAmount') & (final_df['logfoldchange'] > 0)),
        'directionality'
    ] = 'Yes'

    directionality_counts = final_df['directionality'].value_counts()
    logger.info("Directionality breakdown:")
    logger.info(f"  Yes (Agreement): {directionality_counts.get('Yes', 0):,} ({directionality_counts.get('Yes', 0) / len(final_df) * 100:.1f}%)"))
    logger.info(f"  No (Disagreement/Other): {directionality_counts.get('No', 0):,} ({directionality_counts.get('No', 0) / len(final_df) * 100:.1f}%)"))

    # Show examples
    logger.info("\nExamples of 'Yes' directionality:")
    yes_examples = final_df[final_df['directionality'] == 'Yes'][
        ['source', 'intermediate', 'target', 'stmt_type_2', 'logfoldchange', 'directionality']].head(5)
    if len(yes_examples) > 0:
        logger.info(yes_examples.to_string(index=False))
    else:
        logger.info("  No examples found")

    # Final statistics
    logger.info("\n" + "=" * 80)
    logger.info("FINAL DATASET STATISTICS")
    logger.info("=" * 80)
    logger.info("Total rows: %d", len(final_df))

    selection_counts = final_df['selection_reason'].value_counts()
    logger.info("\nSelection breakdown:")
    for reason in ['Top 100', 'Top 100 + GWAS Gene', 'GWAS Gene']:
        count = selection_counts.get(reason, 0)
        logger.info("  %s: %d", reason, count)

    logger.info("\nUnique genes:")
    logger.info("  Source: %d", final_df['source'].nunique())
    logger.info("  Intermediate: %d", final_df['intermediate'].nunique())
    logger.info("  Target: %d", final_df['target'].nunique())

    logger.info("\nLogfoldchange statistics:")
    logger.info("  Min: %.4f", final_df['logfoldchange'].min())
    logger.info("  Max: %.4f", final_df['logfoldchange'].max())
    logger.info("  Min (absolute): %.4f", final_df['abs_logfoldchange'].min())
    logger.info("  Max (absolute): %.4f", final_df['abs_logfoldchange'].max())

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
    output_file_2 = "_top100_endothelial_gwas_priority.csv"
    final_df.to_csv(output_file_2, index=False)

    logger.info("\n" + "=" * 80)
    logger.info("OUTPUTS SAVED")
    logger.info("=" * 80)
    logger.info(" OUTPUT 1: Endothelial-filtered dataset")
    logger.info("  File: %s", output_file_1)
    logger.info("  Shape: %s", endothelial_filtered_df.shape)
    logger.info("  Description: All rows with endothelial intermediates (GWAS genes removed from endothelial list)")
    logger.info(" OUTPUT 2: Final top 100 + GWAS dataset")
    logger.info("  File: %s", output_file_2)
    logger.info("  Shape: %s", final_df.shape)
    logger.info("  Description: Top 100 by logFC + GWAS genes with uniqueness constraint")
    logger.info("  Columns: %s", final_df.columns.tolist())

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(" Removed %s GWAS genes from endothelial list", len(overlap))
    logger.info(" Filtered by endothelial intermediates: %d rows", len(endothelial_filtered_df))
    logger.info(" GWAS genes in endothelial-filtered data: %s out of %s", len(gwas_found), len(gwas_genes))
    logger.info(" Percent of endothelial paths with GWAS genes: %.2f%%", percent_with_gwas)
    logger.info(" Selected top 100 by absolute logfoldchange from endothelial-filtered data")
    logger.info(" Added GWAS genes found in endothelial-filtered data")
    logger.info(" Applied uniqueness constraint per GWAS gene")
    logger.info(" Final output: %d rows", len(final_df))
    logger.info(" Saved 2 CSV files")
    logger.info("=" * 80)



if __name__ == "__main__":
    main()
