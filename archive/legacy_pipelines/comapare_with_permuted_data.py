import pandas as pd
import numpy as np

# === CONFIG ===
ORIGINAL_FILE = "/Users/prashammarfatia/Downloads/indra_top100_with_directional_consistency_1.csv"
PERMUTED_OUTPUT = "/Users/prashammarfatia/Downloads/indra_top100_PERMUTED_negative_control.csv"
COMPARISON_OUTPUT = "/Users/prashammarfatia/Downloads/indra_permutation_comparison_stats.csv"

# === LOAD ORIGINAL DATA ===
df_original = pd.read_csv(ORIGINAL_FILE)
print(f"✅ Loaded {len(df_original)} original rows")

# Store the original true pairs
true_pairs = set(zip(df_original['source'], df_original['target']))
print(f"📋 Original true pairs: {len(true_pairs)}")

# === STEP 1: PERMUTE SOURCE GENES ===
df_permuted = df_original.copy()
np.random.seed(42)  # For reproducibility
df_permuted['source'] = np.random.permutation(df_permuted['source'].values)

print(f"\n🔀 Permuted source genes")

# === STEP 2: REMOVE ACCIDENTAL TRUE MATCHES ===
permuted_pairs = set(zip(df_permuted['source'], df_permuted['target']))
accidental_matches = true_pairs.intersection(permuted_pairs)

print(f"⚠️ Accidental true matches found: {len(accidental_matches)}")

if accidental_matches:
    # Remove rows where permuted pair accidentally matches a true pair
    mask = df_permuted.apply(lambda row: (row['source'], row['target']) not in true_pairs, axis=1)
    df_permuted = df_permuted[mask].reset_index(drop=True)
    print(f"✓ Removed {len(df_original) - len(df_permuted)} accidental matches")


# === FUNCTION TO CHECK SIGN ALIGNMENT ===
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

    # Handle empty or NaN statement types
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


# === ADD SIGN ALIGNMENT COLUMN TO BOTH DATASETS ===
df_original['sign_aligned'] = df_original.apply(check_sign_alignment, axis=1)
df_permuted['sign_aligned'] = df_permuted.apply(check_sign_alignment, axis=1)

# === SAVE BOTH DATASETS ===
df_original.to_csv(ORIGINAL_FILE, index=False)  # Save with sign_aligned column
df_permuted.to_csv(PERMUTED_OUTPUT, index=False)
print(f"\n✅ Saved original data with sign alignment: {ORIGINAL_FILE}")
print(f"✅ Saved permuted data: {PERMUTED_OUTPUT}")
print(f"   Rows: {len(df_permuted)}")


# === COMPARE STATISTICS ===
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
original_stats = calculate_stats(df_original.copy(), 'Original (True)')
permuted_stats = calculate_stats(df_permuted.copy(), 'Permuted (Negative Control)')

# === PRINT COMPARISON ===
print("\n" + "=" * 70)
print("📊 COMPARISON: Original vs Permuted")
print("=" * 70)

print(f"\n{'Metric':<35} {'Original':<20} {'Permuted':<20}")
print("-" * 75)
print(f"{'Rows':<35} {original_stats['n_rows']:<20} {permuted_stats['n_rows']:<20}")
print(
    f"{'Mean Total Evidence':<35} {original_stats['mean_total_evidence']:<20.2f} {permuted_stats['mean_total_evidence']:<20.2f}")
print(
    f"{'Median Total Evidence':<35} {original_stats['median_total_evidence']:<20.2f} {permuted_stats['median_total_evidence']:<20.2f}")
print(f"{'Mean Belief Score':<35} {original_stats['mean_belief']:<20.4f} {permuted_stats['mean_belief']:<20.4f}")
print(f"{'Median Belief Score':<35} {original_stats['median_belief']:<20.4f} {permuted_stats['median_belief']:<20.4f}")
print(f"{'Mean |logFC|':<35} {original_stats['mean_abs_logfc']:<20.4f} {permuted_stats['mean_abs_logfc']:<20.4f}")
print(f"{'Mean p-value':<35} {original_stats['mean_pval']:<20.6f} {permuted_stats['mean_pval']:<20.6f}")

print("\n" + "-" * 75)
print("🎯 SIGN ALIGNMENT")
print("-" * 75)
print(f"{'Checkable Rows':<35} {original_stats['n_total_checkable']:<20} {permuted_stats['n_total_checkable']:<20}")
print(f"{'Correctly Aligned':<35} {original_stats['n_aligned']:<20} {permuted_stats['n_aligned']:<20}")
print(
    f"{'Alignment Rate (%)':<35} {original_stats['sign_alignment_rate'] * 100:<20.1f} {permuted_stats['sign_alignment_rate'] * 100:<20.1f}")

# Save comparison to CSV
comparison_df = pd.DataFrame([original_stats, permuted_stats])
comparison_df.to_csv(COMPARISON_OUTPUT, index=False)
print(f"\n✅ Saved comparison stats: {COMPARISON_OUTPUT}")

print("\n" + "=" * 70)
print("🎯 EXPECTED RESULTS (if INDRA connections are real):")
print("=" * 70)
print("✓ Original data should have HIGHER evidence counts")
print("✓ Original data should have HIGHER belief scores")
print("✓ Original data should have HIGHER sign alignment rate")
print("✓ Permuted data should have LOWER values across all metrics")
print("\nIf permuted data has similar/higher values → connections may be random!")

# === DETAILED SIGN ALIGNMENT BREAKDOWN ===
print("\n" + "=" * 70)
print("📋 SIGN ALIGNMENT DETAILS - ORIGINAL DATA")
print("=" * 70)
original_alignment_counts = df_original['sign_aligned'].value_counts()
print(original_alignment_counts)
if (original_alignment_counts.get('Yes', 0) + original_alignment_counts.get('No', 0)) > 0:
    print(
        f"\nAlignment Rate: {(original_alignment_counts.get('Yes', 0) / (original_alignment_counts.get('Yes', 0) + original_alignment_counts.get('No', 0)) * 100):.1f}%")

print("\n" + "=" * 70)
print("📋 SIGN ALIGNMENT DETAILS - PERMUTED DATA")
print("=" * 70)
permuted_alignment_counts = df_permuted['sign_aligned'].value_counts()
print(permuted_alignment_counts)
if (permuted_alignment_counts.get('Yes', 0) + permuted_alignment_counts.get('No', 0)) > 0:
    print(
        f"\nAlignment Rate: {(permuted_alignment_counts.get('Yes', 0) / (permuted_alignment_counts.get('Yes', 0) + permuted_alignment_counts.get('No', 0)) * 100):.1f}%")