"""Legacy script: compare with permuted data."""
from __future__ import annotations

import argparse
import pandas as pd
import numpy as np

import logging


logger = logging.getLogger(__name__)
ORIGINAL_FILE = "indra_top100_with_directional_consistency_1.csv"
PERMUTED_OUTPUT = "indra_top100_PERMUTED_negative_control.csv"
COMPARISON_OUTPUT = "indra_permutation_comparison_stats.csv"

def check_sign_alignment(row):
    """
    Check if the direction of regulation matches logFC direction.
    - DecreaseAmount should have positive logFC (gene knocked down → target increases)
    - IncreaseAmount should have negative logFC (gene knocked down → target decreases)

    Returns 'Yes' if aligned, 'No' if not, '' if no statement type found
    """
    logfc = row['logfoldchange']

    # Try to get the final edge statement type (the one closest to target)
    stmt_type = None
    hop = row['hop_number']

    if hop == '1hop':
        stmt_type = row.get('stmt_type_1', '')
    elif hop == '2hop':
        stmt_type = row.get('stmt_type_2', '')
    elif hop == '3hop':
        stmt_type = row.get('stmt_type_3', '')
    elif hop == '4hop':
        stmt_type = row.get('stmt_type_4', '')

    if pd.isna(stmt_type) or stmt_type == '':
        return ''

    stmt_type = str(stmt_type).strip()

    # Check alignment
    # DecreaseAmount → positive logFC (target goes up when source is knocked down)
    if 'DecreaseAmount' in stmt_type or 'Inhibition' in stmt_type:
        return 'Yes' if logfc > 0 else 'No'
    # IncreaseAmount → negative logFC (target goes down when source is knocked down)
    elif 'IncreaseAmount' in stmt_type or 'Activation' in stmt_type:
        return 'Yes' if logfc < 0 else 'No'
    else:
        # Other statement types - we don't know expected direction
        return ''


def calculate_stats(df, label):
    """Calculate statistics for comparison"""
    stats = {
        'dataset': label,
        'n_rows': len(df),
        'n_hops': df['hop_number'].value_counts().to_dict(),
    }

    # Evidence counts (sum across all edges)
    evidence_cols = ['evidence_1', 'evidence_2', 'evidence_3', 'evidence_4']
    for col in evidence_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['total_evidence'] = df[evidence_cols].sum(axis=1)
    stats['mean_total_evidence'] = df['total_evidence'].mean()
    stats['median_total_evidence'] = df['total_evidence'].median()

    # Belief scores (average across all edges)
    belief_cols = ['belief_1', 'belief_2', 'belief_3', 'belief_4']
    for col in belief_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Calculate average belief (only non-zero values)
    belief_values = df[belief_cols].replace(0, np.nan).values.flatten()
    belief_values = belief_values[~np.isnan(belief_values)]
    stats['mean_belief'] = belief_values.mean() if len(belief_values) > 0 else 0
    stats['median_belief'] = np.median(belief_values) if len(belief_values) > 0 else 0

    # LogFC and pval
    stats['mean_abs_logfc'] = df['logfoldchange'].abs().mean()
    stats['mean_pval'] = df['pval'].mean()

    # Sign alignment statistics
    sign_aligned_rows = df[df['sign_aligned'].isin(['Yes', 'No'])]
    if len(sign_aligned_rows) > 0:
        n_aligned = (sign_aligned_rows['sign_aligned'] == 'Yes').sum()
        n_total = len(sign_aligned_rows)
        stats['sign_alignment_rate'] = n_aligned / n_total
        stats['n_aligned'] = n_aligned
        stats['n_total_checkable'] = n_total
    else:
        stats['sign_alignment_rate'] = 0
        stats['n_aligned'] = 0
        stats['n_total_checkable'] = 0

    return stats


# Calculate stats for both datasets


def main():
    ap = argparse.ArgumentParser()

    df_original = pd.read_csv(ORIGINAL_FILE)
    logger.info(" Loaded %s original rows", len(df_original))

    true_pairs = set(zip(df_original['source'], df_original['target']))
    logger.info(" Original true pairs: %s", len(true_pairs))

    df_permuted = df_original.copy()
    np.random.seed(42)  # For reproducibility
    df_permuted['source'] = np.random.permutation(df_permuted['source'].values)

    logger.info("\n Permuted source genes")

    permuted_pairs = set(zip(df_permuted['source'], df_permuted['target']))
    accidental_matches = true_pairs.intersection(permuted_pairs)

    logger.info(" Accidental true matches found: %s", len(accidental_matches))

    if accidental_matches:
        # Remove rows where permuted pair accidentally matches a true pair
        mask = df_permuted.apply(lambda row: (row['source'], row['target']) not in true_pairs, axis=1)
        df_permuted = df_permuted[mask].reset_index(drop=True)
        logger.info(" Removed %s accidental matches", len(df_original) - len(df_permuted))


    df_original['sign_aligned'] = df_original.apply(check_sign_alignment, axis=1)
    df_permuted['sign_aligned'] = df_permuted.apply(check_sign_alignment, axis=1)

    df_original.to_csv(ORIGINAL_FILE, index=False)  # Save with sign_aligned column
    df_permuted.to_csv(PERMUTED_OUTPUT, index=False)
    logger.info("\n Saved original data with sign alignment: %s", ORIGINAL_FILE)
    logger.info(" Saved permuted data: %s", PERMUTED_OUTPUT)
    logger.info("   Rows: %s", len(df_permuted))


    original_stats = calculate_stats(df_original.copy(), 'Original (True)')
    permuted_stats = calculate_stats(df_permuted.copy(), 'Permuted (Negative Control)')

    logger.info("\n" + "=" * 70)
    logger.info(" COMPARISON: Original vs Permuted")
    logger.info("=" * 70)

    logger.info("\n%<35 %<20 %<20", 'Metric', 'Original', 'Permuted')
    logger.info("-" * 75)
    # [corrupted format string removed]
    logger.info(f"{'Mean Total Evidence':<35} {original_stats['mean_total_evidence']:<20.2f} {permuted_stats['mean_total_evidence']:<20.2f}"))
    logger.info(f"{'Median Total Evidence':<35} {original_stats['median_total_evidence']:<20.2f} {permuted_stats['median_total_evidence']:<20.2f}"))
    # [corrupted format string removed]
    # [corrupted format string removed]
    # [corrupted format string removed]
    # [corrupted format string removed]

    logger.info("\n" + "-" * 75)
    logger.info(" SIGN ALIGNMENT")
    logger.info("-" * 75)
    # [corrupted format string removed]
    # [corrupted format string removed]
    logger.info(f"{'Alignment Rate (%)':<35} {original_stats['sign_alignment_rate'] * 100:<20.1f} {permuted_stats['sign_alignment_rate'] * 100:<20.1f}"))

    # Save comparison to CSV
    comparison_df = pd.DataFrame([original_stats, permuted_stats])
    comparison_df.to_csv(COMPARISON_OUTPUT, index=False)
    logger.info("\n Saved comparison stats: %s", COMPARISON_OUTPUT)

    logger.info("\n" + "=" * 70)
    logger.info(" EXPECTED RESULTS (if INDRA connections are real):")
    logger.info("=" * 70)
    logger.info(" Original data should have HIGHER evidence counts")
    logger.info(" Original data should have HIGHER belief scores")
    logger.info(" Original data should have HIGHER sign alignment rate")
    logger.info(" Permuted data should have LOWER values across all metrics")
    logger.info("\nIf permuted data has similar/higher values → connections may be random!")

    logger.info("\n" + "=" * 70)
    logger.info(" SIGN ALIGNMENT DETAILS - ORIGINAL DATA")
    logger.info("=" * 70)
    original_alignment_counts = df_original['sign_aligned'].value_counts()
    logger.info(original_alignment_counts)
    if (original_alignment_counts.get('Yes', 0) + original_alignment_counts.get('No', 0)) > 0:
        logger.info(f"\nAlignment Rate: {(original_alignment_counts.get('Yes', 0) / (original_alignment_counts.get('Yes', 0) + original_alignment_counts.get('No', 0)) * 100):.1f}%")

    logger.info("\n" + "=" * 70)
    logger.info(" SIGN ALIGNMENT DETAILS - PERMUTED DATA")
    logger.info("=" * 70)
    permuted_alignment_counts = df_permuted['sign_aligned'].value_counts()
    logger.info(permuted_alignment_counts)
    if (permuted_alignment_counts.get('Yes', 0) + permuted_alignment_counts.get('No', 0)) > 0:
        logger.info(f"\nAlignment Rate: {(permuted_alignment_counts.get('Yes', 0) / (permuted_alignment_counts.get('Yes', 0) + permuted_alignment_counts.get('No', 0)) * 100):.1f}%")



if __name__ == "__main__":
    main()
