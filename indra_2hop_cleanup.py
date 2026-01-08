import pandas as pd
import numpy as np


def clean_csv_data(input_file, output_file):
    """
    Clean CSV data by:
    1. Removing rows with non-human genes in intermediate column
    2. Keeping only the row with highest mean evidence for duplicate source-intermediate-target triplets
    """

    print("Reading CSV file...")
    # Read the CSV file
    df = pd.read_csv(input_file)
    print(f"Original data shape: {df.shape}")

    # Step 1: Remove rows with non-human genes in intermediate column
    print("\nStep 1: Filtering out non-human genes from intermediate column...")

    # Define patterns for non-human genes to remove
    non_human_patterns = ['mesh:', 'uniprot:', 'chebi:', 'go:', 'UP:', 'MESH:', 'CHEBI:', 'GO:']

    # Create a mask for human genes (rows to keep)
    human_mask = True
    for pattern in non_human_patterns:
        human_mask = human_mask & (~df['intermediate'].str.contains(pattern, na=False))

    # Apply the filter
    df_filtered = df[human_mask].copy()
    print(f"After removing non-human genes: {df_filtered.shape}")
    print(f"Removed {df.shape[0] - df_filtered.shape[0]} rows")

    # Step 2: Calculate mean evidence and keep only highest for each source-intermediate-target triplet
    print("\nStep 2: Calculating mean evidence and filtering duplicates...")

    # Calculate mean evidence for each row
    df_filtered['mean_evidence'] = (df_filtered['evidence_1'] + df_filtered['evidence_2']) / 2

    # Group by source-intermediate-target triplets and find the index with max mean evidence
    print("Finding rows with highest mean evidence for each source-intermediate-target triplet...")
    idx_to_keep = df_filtered.groupby(['source', 'intermediate', 'target'])['mean_evidence'].idxmax()

    # Keep only these rows
    df_final = df_filtered.loc[idx_to_keep].copy()

    # Remove the temporary mean_evidence column
    df_final = df_final.drop('mean_evidence', axis=1)

    print(f"After removing duplicate source-intermediate-target triplets: {df_final.shape}")
    print(f"Removed {df_filtered.shape[0] - df_final.shape[0]} duplicate triplets")

    # Save the cleaned data
    print(f"\nSaving cleaned data to {output_file}...")
    df_final.to_csv(output_file, index=False)

    print("Data cleaning complete!")
    print(f"Final data shape: {df_final.shape}")

    return df_final


def preview_changes(input_file, sample_size=10):
    """
    Preview the changes that will be made without actually processing the full file
    """
    print("Previewing changes on a sample of the data...")

    # Read a sample
    df_sample = pd.read_csv(input_file, nrows=sample_size * 10)  # Read more to ensure we have examples
    print(f"Sample data shape: {df_sample.shape}")

    # Show examples of intermediates that will be removed
    non_human_patterns = ['mesh:', 'uniprot:', 'chebi:', 'go:', 'UP:', 'MESH:', 'CHEBI:', 'GO:']

    print("\nExamples of intermediates that will be REMOVED:")
    for pattern in non_human_patterns:
        examples = df_sample[df_sample['intermediate'].str.contains(pattern, na=False)]['intermediate'].unique()[:3]
        if len(examples) > 0:
            print(f"  {pattern} examples: {list(examples)}")

    # Show examples of intermediates that will be kept
    human_mask = True
    for pattern in non_human_patterns:
        human_mask = human_mask & (~df_sample['intermediate'].str.contains(pattern, na=False))

    human_intermediates = df_sample[human_mask]['intermediate'].unique()[:5]
    print(f"\nExamples of intermediates that will be KEPT: {list(human_intermediates)}")

    # Show duplicate source-intermediate-target triplets
    df_human = df_sample[human_mask].copy()
    df_human['mean_evidence'] = (df_human['evidence_1'] + df_human['evidence_2']) / 2
    duplicates = df_human.groupby(['source', 'intermediate', 'target']).size()
    duplicate_triplets = duplicates[duplicates > 1]

    if len(duplicate_triplets) > 0:
        print(f"\nFound {len(duplicate_triplets)} duplicate source-intermediate-target triplets in sample:")
        for (source, intermediate, target), count in duplicate_triplets.head(3).items():
            print(f"  {source} -> {intermediate} -> {target}: {count} rows")
            subset = df_human[(df_human['source'] == source) &
                              (df_human['intermediate'] == intermediate) &
                              (df_human['target'] == target)]
            print(f"    Mean evidence values: {list(subset['mean_evidence'].round(3))}")
            print(f"    Will keep row with mean evidence: {subset['mean_evidence'].max():.3f}")
    else:
        print("\nNo duplicate source-intermediate-target triplets found in sample.")


if __name__ == "__main__":
    # File paths
    input_file = "/Users/prashammarfatia/Downloads/indra_2hop_all_perturbations.csv"
    output_file = "/Users/prashammarfatia/Downloads/cleaned_indra_2hop_all_perturbations.csv"

    # Preview changes first (optional)
    print("=" * 50)
    print("PREVIEW MODE")
    print("=" * 50)
    preview_changes(input_file, sample_size=100)

    # Ask for confirmation before processing
    proceed = input("\nDo you want to proceed with the full data cleaning? (y/n): ")

    if proceed.lower() == 'y':
        print("\n" + "=" * 50)
        print("PROCESSING FULL FILE")
        print("=" * 50)

        # Process the full file
        cleaned_df = clean_csv_data(input_file, output_file)

        # Show some statistics
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Cleaned data saved to: {output_file}")
        print(
            f"Unique source-intermediate-target triplets: {cleaned_df.groupby(['source', 'intermediate', 'target']).size().shape[0]}")
        print(f"Unique sources: {cleaned_df['source'].nunique()}")
        print(f"Unique targets: {cleaned_df['target'].nunique()}")
        print(f"Unique intermediates: {cleaned_df['intermediate'].nunique()}")

    else:
        print("Processing cancelled.")