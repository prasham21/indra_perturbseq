"""Legacy script: indra_2hop_cleanup."""
from __future__ import annotations

import argparse
import pandas as pd
import numpy as np

import logging


logger = logging.getLogger(__name__)

def clean_csv_data(input_file, output_file):
    """Remove non-human intermediate genes and deduplicate triplets by highest evidence."""

    logger.info("Reading CSV file...")
    # Read the CSV file
    df = pd.read_csv(input_file)
    logger.info("Original data shape: %s", df.shape)

    logger.info("Filtering out non-human genes from intermediate column...")

    # Define patterns for non-human genes to remove
    non_human_patterns = ['mesh:', 'uniprot:', 'chebi:', 'go:', 'UP:', 'MESH:', 'CHEBI:', 'GO:']

    # Create a mask for human genes (rows to keep)
    human_mask = True
    for pattern in non_human_patterns:
        human_mask = human_mask & (~df['intermediate'].str.contains(pattern, na=False))

    # Apply the filter
    df_filtered = df[human_mask].copy()
    logger.info("After removing non-human genes: %s", df_filtered.shape)
    logger.info("Removed %s rows", df.shape[0] - df_filtered.shape[0])

    logger.info("Calculating mean evidence and filtering duplicates...")

    # Calculate mean evidence for each row
    df_filtered['mean_evidence'] = (df_filtered['evidence_1'] + df_filtered['evidence_2']) / 2

    # Group by source-intermediate-target triplets and find the index with max mean evidence
    logger.info("Finding rows with highest mean evidence for each source-intermediate-target triplet...")
    idx_to_keep = df_filtered.groupby(['source', 'intermediate', 'target'])['mean_evidence'].idxmax()

    # Keep only these rows
    df_final = df_filtered.loc[idx_to_keep].copy()

    # Remove the temporary mean_evidence column
    df_final = df_final.drop('mean_evidence', axis=1)

    logger.info("After removing duplicate source-intermediate-target triplets: %s", df_final.shape)
    logger.info("Removed %s duplicate triplets", df_filtered.shape[0] - df_final.shape[0])

    # Save the cleaned data
    logger.info("\nSaving cleaned data to %s...", output_file)
    df_final.to_csv(output_file, index=False)

    logger.info("Data cleaning complete!")
    logger.info("Final data shape: %s", df_final.shape)

    return df_final


def preview_changes(input_file, sample_size=10):
    """
    Preview the changes that will be made without actually processing the full file
    """
    logger.info("Previewing changes on a sample of the data...")

    # Read a sample
    df_sample = pd.read_csv(input_file, nrows=sample_size * 10)  # Read more to ensure we have examples
    logger.info("Sample data shape: %s", df_sample.shape)

    # Show examples of intermediates that will be removed
    non_human_patterns = ['mesh:', 'uniprot:', 'chebi:', 'go:', 'UP:', 'MESH:', 'CHEBI:', 'GO:']

    logger.info("\nExamples of intermediates that will be REMOVED:")
    for pattern in non_human_patterns:
        examples = df_sample[df_sample['intermediate'].str.contains(pattern, na=False)]['intermediate'].unique()[:3]
        if len(examples) > 0:
            logger.info("  %s examples: %s", pattern, list(examples))

    # Show examples of intermediates that will be kept
    human_mask = True
    for pattern in non_human_patterns:
        human_mask = human_mask & (~df_sample['intermediate'].str.contains(pattern, na=False))

    human_intermediates = df_sample[human_mask]['intermediate'].unique()[:5]
    logger.info("\nExamples of intermediates that will be KEPT: %s", list(human_intermediates))

    # Show duplicate source-intermediate-target triplets
    df_human = df_sample[human_mask].copy()
    df_human['mean_evidence'] = (df_human['evidence_1'] + df_human['evidence_2']) / 2
    duplicates = df_human.groupby(['source', 'intermediate', 'target']).size()
    duplicate_triplets = duplicates[duplicates > 1]

    if len(duplicate_triplets) > 0:
        logger.info("\nFound %s duplicate source-intermediate-target triplets in sample:", len(duplicate_triplets))
        for (source, intermediate, target), count in duplicate_triplets.head(3).items():
            logger.info("  %s -> %s -> %s: %s rows", source, intermediate, target, count)
            subset = df_human[(df_human['source'] == source) &
                              (df_human['intermediate'] == intermediate) &
                              (df_human['target'] == target)]
            logger.info("    Mean evidence values: %s", list(subset['mean_evidence'].round(3)))
            logger.info("    Will keep row with mean evidence: %.3f", subset['mean_evidence'].max())
    else:
        logger.info("\nNo duplicate source-intermediate-target triplets found in sample.")




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indra-2hop-all-perturbation", default="indra_2hop_all_perturbations.csv", help="Path: indra_2hop_all_perturbations.csv")
    ap.add_argument("--cleaned-indra-2hop-all-perturbation", default="cleaned_indra_2hop_all_perturbations.csv", help="Path: cleaned_indra_2hop_all_perturbations.csv")
    args = ap.parse_args()



if __name__ == "__main__":
    main()
