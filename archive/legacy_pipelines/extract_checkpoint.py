"""Legacy script: extract_checkpoint."""
from __future__ import annotations

import pickle
import pandas as pd
import os
from datetime import datetime

import logging


logger = logging.getLogger(__name__)
# Load checkpoint file


def main():
    checkpoint_file = "3hop_optimized_checkpoint.pkl"

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)

        results = data['results']
        processed_genes = data['processed_genes']
        timestamp = data['timestamp']

        # Print summary
        logger.info("=" * 60)
        logger.info("3-HOP ANALYSIS CHECKPOINT SUMMARY")
        logger.info("=" * 60)
        logger.info("Checkpoint saved: %s", datetime.fromtimestamp(timestamp))
        logger.info("Genes completed: %s out of 358", len(processed_genes))
        logger.info("Progress: %.1f%%", len(processed_genes) / 358 * 100)
        logger.info("Total 3-hop pathways found: %s", len(results))
        logger.info("Average pathways per gene: %.1f", len(results) / len(processed_genes))

        # Show completed genes
        logger.info("\nCompleted genes:")
        gene_list = sorted(list(processed_genes))
        for i in range(0, len(gene_list), 10):
            logger.info(", ".join(gene_list[i:i + 10]))

        # Convert to DataFrame and save
        if results:
            df = pd.DataFrame(results)
            output_file = "partial_3hop_results.csv"
            df.to_csv(output_file, index=False)

            logger.info("\n" + "=" * 6")
            logger.info("RESULTS EXPORTED")
            logger.info("=" * 60)
            logger.info("CSV saved to: %s", output_file)
            logger.info("CSV shape: %s (rows × columns)", df.shape)

            # Show column info
            logger.info("\nColumns: %s", list(df.columns))

            # Show sample results
            logger.info("\nSample 3-hop pathways:")
            logger.info("-" * 80)
            for i, row in df.head(5).iterrows():
                logger.info("%s → %s → %s → %s", row['source'], row['intermediate_1'], row['intermediate_2'], row['target'])
                logger.info("  Types: %s → %s → %s", row['stmt_type_1'], row['stmt_type_2'], row['stmt_type_3'])
                logger.info("  Beliefs: %.2f → %.2f → %.2f", row['belief_1'], row['belief_2'], row['belief_3'])
                logger.info()

            # Show some statistics
            logger.info("PATHWAY STATISTICS:")
            logger.info("Unique source genes: %s", df['source'].nunique())
            logger.info("Unique target genes: %s", df['target'].nunique())
            logger.info("Unique intermediate_1 genes: %s", df['intermediate_1'].nunique())
            logger.info("Unique intermediate_2 genes: %s", df['intermediate_2'].nunique())

            # Belief score distribution
            avg_belief_1 = df['belief_1'].mean()
            avg_belief_2 = df['belief_2'].mean()
            avg_belief_3 = df['belief_3'].mean()
            logger.info("Average belief scores: %.3f → %.3f → %.3f", avg_belief_1, avg_belief_2, avg_belief_3)

        else:
            logger.info("No results found in checkpoint")

    else:
        logger.info("Checkpoint file '%s' not found", checkpoint_file)
        logger.info("Make sure you're in the correct directory")



if __name__ == "__main__":
    main()
