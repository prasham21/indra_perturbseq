"""Legacy script: separate_self_taget_genes."""
from __future__ import annotations

import argparse
import pandas as pd
import os

import logging


logger = logging.getLogger(__name__)

def separate_self_targeting_3hop():
    """
    Separate 3-hop dataset into self-targeting and non-self-targeting files
    """
    # File path for 3-hop data
    input_file = "indra_2hop_with_evidence_statements__range_replaced.csv"

    logger.info("=" * 50)
    logger.info("SEPARATING SELF-TARGETING GENES FROM 3-HOP DATA")
    logger.info("=" * 50)

    logger.info("Processing %s...", input_file)

    # Check if file exists
    if not os.path.exists(input_file):
        logger.info(" File not found: %s", input_file)
        return

    try:
        # Read the input file
        df = pd.read_csv(input_file)
        logger.info(" File loaded successfully")
        logger.info(" Total rows: %d", len(df))

        # Check column names
        logger.info(" Columns: %s", list(df.columns))

        # Separate self-targeting vs non-self-targeting
        self_targeting = df[df['source'] == df['target']].copy()
        non_self_targeting = df[df['source'] != df['target']].copy()

        logger.info("\n Results:")
        logger.info("   Self-targeting rows: %d", len(self_targeting))
        logger.info("   Non-self-targeting rows: %d", len(non_self_targeting))

        # Calculate percentage
        self_percentage = (len(self_targeting) / len(df)) * 100 if len(df) > 0 else 0
        logger.info("   Self-targeting percentage: %.2f%%", self_percentage)

        # Generate output file paths
        input_dir = os.path.dirname(input_file)
        base_name = os.path.splitext(os.path.basename(input_file))[0]

        main_output = os.path.join(input_dir, f"{base_name}_main.csv")
        self_output = os.path.join(input_dir, f"{base_name}_self_targeting.csv")

        # Save the files
        logger.info("\n Saving files...")
        non_self_targeting.to_csv(main_output, index=False)
        self_targeting.to_csv(self_output, index=False)

        logger.info(" Main file saved: %s", main_output)
        logger.info(" Self-targeting file saved: %s", self_output)

        # Show some examples if self-targeting genes exist
        if len(self_targeting) > 0:
            logger.info("\n Sample self-targeting genes:")
            sample_genes = self_targeting['source'].value_counts().head(5)
            for gene, count in sample_genes.items():
                logger.info("   %s: %s pathway(s)", gene, count)
        else:
            logger.info("\n  No self-targeting genes found in this dataset")

        logger.info("\n Separation completed successfully!")

    except Exception as e:
        logger.info(" Error processing file: %s", e)
        import traceback
        traceback.print_exc()




def main():
    ap = argparse.ArgumentParser()



if __name__ == "__main__":
    main()
