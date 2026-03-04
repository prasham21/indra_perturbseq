"""Legacy script: omnipath_dataset_3hop."""
from __future__ import annotations

import argparse
import pandas as pd
import numpy as np

import logging


logger = logging.getLogger(__name__)

def create_synthetic_omnipath_from_indra():
    """
    Create a realistic synthetic OmniPath dataset from INDRA 3-hop data
    10% identical, 25% partial overlap, 65% different
    """
    # Load your INDRA dataset
    indra_df = pd.read_csv('indra_3hop_no_hgnc_prefix.csv')

    # Filter and get top 100 by p-value
    indra_clean = indra_df[np.isfinite(indra_df['pval']) & np.isfinite(indra_df['logfoldchange'])].copy()
    top_100 = indra_clean.nsmallest(100, 'pval')

    logger.info("Creating synthetic OmniPath from top 100 INDRA pathways")

    # Collect all unique intermediates
    all_intermediates = set()
    for col in ['intermediate_1', 'intermediate_2']:
        all_intermediates.update(indra_clean[col].dropna().unique())
    all_intermediates = list(all_intermediates)

    logger.info("Pool of %s intermediates available for swapping", len(all_intermediates))

    # Create synthetic OmniPath dataset
    omnipath_data = []

    for idx, row in top_100.iterrows():
        # Decide pathway type with new percentages
        rand = np.random.random()

        if rand < 0.10:  # 10% identical
            pathway_type = 'identical'
            int1 = row['intermediate_1']
            int2 = row['intermediate_2']
        elif rand < 0.35:  # 25% partial overlap
            pathway_type = 'partial'
            if np.random.random() < 0.5:
                int1 = row['intermediate_1']
                int2 = np.random.choice(all_intermediates)
            else:
                int1 = np.random.choice(all_intermediates)
                int2 = row['intermediate_2']
        else:  # 65% completely different
            pathway_type = 'different'
            int1 = np.random.choice(all_intermediates)
            int2 = np.random.choice(all_intermediates)

        # Create OmniPath entry
        omnipath_entry = {
            'source': row['source'],
            'intermediate_1': int1,
            'intermediate_2': int2,
            'target': row['target'],
            'stmt_type_1': row['stmt_type_1'],
            'stmt_type_2': row['stmt_type_2'],
            'stmt_type_3': row['stmt_type_3'],
            'belief_1': np.clip(row['belief_1'] + np.random.uniform(-0.1, 0.1), 0, 1),
            'belief_2': np.clip(row['belief_2'] + np.random.uniform(-0.1, 0.1), 0, 1),
            'belief_3': np.clip(row['belief_3'] + np.random.uniform(-0.1, 0.1), 0, 1),
            'evidence_1': max(1, int(row['evidence_1'] + np.random.randint(-2, 3))),
            'evidence_2': max(1, int(row['evidence_2'] + np.random.randint(-2, 3))),
            'evidence_3': max(1, int(row['evidence_3'] + np.random.randint(-2, 3))),
            'logfoldchange': row['logfoldchange'],
            'pval': row['pval'],
            'pathway_type': pathway_type
        }

        omnipath_data.append(omnipath_entry)

    omnipath_df = pd.DataFrame(omnipath_data)

    # Save
    output_path = 'omnipath_3hop_synthetic.csv'
    omnipath_df.to_csv(output_path, index=False)

    logger.info("\nSynthetic OmniPath dataset created: %s", output_path)
    logger.info("\nPathway overlap distribution:")
    logger.info("  Identical: %s", sum(omnipath_df['pathway_type'] == 'identical'))
    logger.info("  Partial: %s", sum(omnipath_df['pathway_type'] == 'partial'))
    logger.info("  Different: %s", sum(omnipath_df['pathway_type'] == 'different'))

    return omnipath_df




def main():
    ap = argparse.ArgumentParser()



if __name__ == "__main__":
    main()
