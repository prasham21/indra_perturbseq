import pandas as pd
import os
from indra_cogex.analysis.source_targets_explanation import get_valid_gene_ids
from indra.databases.hgnc_client import get_current_hgnc_id


def calculate_3hop_coverage():
    print("Calculating 3-hop pathway coverage for completed genes only...")

    # Load the 3-hop results
    results_file = "partial_3hop_results.csv"

    if not os.path.exists(results_file):
        print(f"Results file '{results_file}' not found!")
        return

    results_df = pd.read_csv(results_file)
    print(f"Loaded 3-hop results: {len(results_df)} pathways")

    # Get unique source-target pairs from results
    results_pairs = results_df[['source', 'target']].drop_duplicates()
    found_pairs = len(results_pairs)
    print(f"Unique source-target pairs found in 3-hop results: {found_pairs}")

    # Get completed source genes from results (these are our 90 genes)
    completed_sources = results_df['source'].unique()
    completed_sources = sorted(completed_sources)  # Sort for consistent ordering
    print(f"Completed source genes: {len(completed_sources)}")
    print(f"Completed genes: {', '.join(completed_sources[:10])}...")  # Show first 10

    # Calculate total possible pairs for ONLY these 90 completed genes
    total_possible_pairs = 0
    gene_pair_counts = {}

    print(f"\nCalculating total possible pairs for {len(completed_sources)} completed genes...")

    for i, source_gene in enumerate(completed_sources):
        print(f"Processing {i + 1}/{len(completed_sources)}: {source_gene}...")

        # Load DEG file for this gene
        deg_path = f"/Users/prashammarfatia/Downloads/de_results_per_gene/{source_gene}_vs_control.csv"

        if os.path.exists(deg_path):
            try:
                df = pd.read_csv(deg_path)
                df = df[df["pvals"] < 0.05]  # Same filtering as in analysis

                gene_symbols = df["names"].dropna().unique().tolist()
                converted = get_valid_gene_ids(gene_symbols)
                valid_targets = len([v for v in converted if v])  # Count valid HGNC conversions

                total_possible_pairs += valid_targets
                gene_pair_counts[source_gene] = valid_targets
                print(f"  {source_gene}: {valid_targets} possible targets")

            except Exception as e:
                print(f"  Error processing {source_gene}: {e}")
                gene_pair_counts[source_gene] = 0
        else:
            print(f"  DEG file not found for {source_gene}")
            gene_pair_counts[source_gene] = 0

    # Calculate coverage for completed genes only
    if total_possible_pairs > 0:
        coverage_percent = (found_pairs / total_possible_pairs) * 100

        print("\n" + "=" * 70)
        print("3-HOP PATHWAY COVERAGE ANALYSIS (COMPLETED GENES ONLY)")
        print("=" * 70)
        print(f"Analysis scope: First {len(completed_sources)} completed perturbations")
        print(f"Total possible source-target pairs: {total_possible_pairs:,}")
        print(f"Source-target pairs with 3-hop pathways: {found_pairs:,}")
        print(f"Coverage percentage: {coverage_percent:.2f}%")
        print("=" * 70)

        # Additional statistics
        avg_pathways_per_pair = len(results_df) / found_pairs if found_pairs > 0 else 0
        print(f"Total 3-hop pathways found: {len(results_df):,}")
        print(f"Average pathways per source-target pair: {avg_pathways_per_pair:.1f}")

        # Show top/bottom performers
        print(f"\nTop 5 genes by possible targets:")
        top_genes = sorted(gene_pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for gene, count in top_genes:
            source_results = results_df[results_df['source'] == gene]
            actual_pairs = len(source_results[['source', 'target']].drop_duplicates())
            gene_coverage = (actual_pairs / count * 100) if count > 0 else 0
            print(f"  {gene}: {actual_pairs}/{count} pairs ({gene_coverage:.1f}% coverage)")

        print(f"\nBottom 5 genes by possible targets:")
        bottom_genes = sorted(gene_pair_counts.items(), key=lambda x: x[1])[:5]
        for gene, count in bottom_genes:
            source_results = results_df[results_df['source'] == gene]
            actual_pairs = len(source_results[['source', 'target']].drop_duplicates())
            gene_coverage = (actual_pairs / count * 100) if count > 0 else 0
            print(f"  {gene}: {actual_pairs}/{count} pairs ({gene_coverage:.1f}% coverage)")

    else:
        print("No possible pairs found - check data files")


if __name__ == "__main__":
    calculate_3hop_coverage()