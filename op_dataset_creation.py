"""
Create a synthetic OmniPath 3-hop dataset from INDRA results
------------------------------------------------------------
✅ Guarantees all 7 of top 20 pairs are represented
✅ Keeps global size ≈ 18% of INDRA
✅ Applies 20% identical, 35% partial, 45% different intermediate logic
✅ Ensures both intermediates are always filled
✅ Mimics real-world scenario where OmniPath covers fewer pairs & smaller network
"""

import pandas as pd
import numpy as np

# -------------------------
# CONFIGURATION
# -------------------------
INDRA_PATH = '/Users/prashammarfatia/Downloads/indra_3hop_no_hgnc_prefix.csv'
OUTPUT_PATH = '/Users/prashammarfatia/Downloads/omnipath_3hop_final.csv'

OP_COVERAGE = 0.18          # ~18% of INDRA
OP_PAIR_FRACTION = 7 / 20   # ~7 of top 20 pairs

RATIO_IDENTICAL = 0.20
RATIO_PARTIAL = 0.35
RATIO_DIFFERENT = 0.45

# -------------------------
# MAIN FUNCTION
# -------------------------
def create_synthetic_omnipath():
    indra_df = pd.read_csv(INDRA_PATH)
    print(f"Loaded {len(indra_df)} INDRA rows")

    # Keep rows with valid pval and logfoldchange
    indra_clean = indra_df[np.isfinite(indra_df['pval']) & np.isfinite(indra_df['logfoldchange'])].copy()

    # Select top 20 source→target pairs by low pval & high |logFC|
    top_pairs = (
        indra_clean.assign(abs_logfc=lambda x: np.abs(x['logfoldchange']))
        .sort_values(['pval', 'abs_logfc'], ascending=[True, False])
        .drop_duplicates(subset=['source', 'target'])
        .head(20)
    )
    print(f"Top 20 INDRA pairs selected: {len(top_pairs)}")

    # Choose ~7 random pairs to appear in OmniPath
    selected_pairs = top_pairs.sample(n=7, random_state=42)[['source', 'target']]
    selected_pairs_set = set(zip(selected_pairs['source'], selected_pairs['target']))

    print("\nSelected 7 pairs for OmniPath representation:")
    for s, t in selected_pairs_set:
        print(f"  {s} → {t}")

    # Collect all unique intermediates from INDRA
    all_intermediates = set(indra_clean['intermediate_1'].dropna().unique()) | \
                        set(indra_clean['intermediate_2'].dropna().unique())
    all_intermediates = list(all_intermediates)
    np.random.shuffle(all_intermediates)

    # Restrict OmniPath’s universe of intermediates (~25% of INDRA)
    allowed_intermediates = np.random.choice(all_intermediates,
                                             size=int(len(all_intermediates) * 0.25),
                                             replace=False)

    # Build OmniPath rows
    op_rows = []
    rng = np.random.default_rng(42)

    # Add all rows for selected 7 pairs first
    for _, row in indra_clean.iterrows():
        src, tgt = row['source'], row['target']
        if (src, tgt) not in selected_pairs_set:
            continue

        rand = rng.random()
        if rand < RATIO_IDENTICAL:
            int1, int2 = row['intermediate_1'], row['intermediate_2']
            ptype = 'identical'
        elif rand < RATIO_IDENTICAL + RATIO_PARTIAL:
            # partial overlap — keep one original, change the other
            if rng.random() < 0.5:
                int1 = row['intermediate_1']
                int2 = rng.choice(allowed_intermediates)
            else:
                int1 = rng.choice(allowed_intermediates)
                int2 = row['intermediate_2']
            ptype = 'partial'
        else:
            # completely different intermediates
            int1 = rng.choice(allowed_intermediates)
            int2 = rng.choice(allowed_intermediates)
            ptype = 'different'

        op_rows.append({
            'source': src,
            'intermediate_1': int1,
            'intermediate_2': int2,
            'target': tgt,
            'logfoldchange': row['logfoldchange'],
            'pval': row['pval'],
            'pathway_type': ptype
        })

    # Supplement with random rows to reach ~18% coverage
    target_size = int(len(indra_clean) * OP_COVERAGE)
    current_size = len(op_rows)
    remaining_needed = max(0, target_size - current_size)

    if remaining_needed > 0:
        random_extra = indra_clean.sample(n=remaining_needed, random_state=99)
        for _, row in random_extra.iterrows():
            int1 = rng.choice(allowed_intermediates)
            int2 = rng.choice(allowed_intermediates)
            op_rows.append({
                'source': row['source'],
                'intermediate_1': int1,
                'intermediate_2': int2,
                'target': row['target'],
                'logfoldchange': row['logfoldchange'],
                'pval': row['pval'],
                'pathway_type': 'background'
            })

    omnipath_df = pd.DataFrame(op_rows)
    omnipath_df.to_csv(OUTPUT_PATH, index=False)

    # -------------------------
    # SUMMARY
    # -------------------------
    print(f"\n✅ Synthetic OmniPath dataset created: {OUTPUT_PATH}")
    print(f"  Rows: {len(omnipath_df)} (≈ {100*len(omnipath_df)/len(indra_clean):.1f}% of INDRA)")
    print(f"  Unique intermediates: {len(pd.unique(omnipath_df[['intermediate_1','intermediate_2']].values.ravel()))}")
    print(f"  Covered top-20 pairs: {len(selected_pairs_set)} (shown below)")
    for s, t in selected_pairs_set:
        print(f"   • {s} → {t}")
    print("\n  Pathway-type distribution:")
    print(omnipath_df['pathway_type'].value_counts())

    return omnipath_df


if __name__ == "__main__":
    create_synthetic_omnipath()
