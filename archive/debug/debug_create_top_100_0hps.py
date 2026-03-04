import pandas as pd
import numpy as np
import scanpy as sc
import os

def load_cell_counts(adata_path):
    """Load AnnData and compute cell counts per perturbation (source gene)."""
    print("Loading AnnData to compute cell counts...")
    adata = sc.read_h5ad(adata_path)
    print(f"  ✓ Loaded AnnData with shape: {adata.shape}")

    # Count cells per perturbation
    cell_counts = adata.obs['Gene'].value_counts().to_dict()
    print(f"  ✓ Computed cell counts for {len(cell_counts)} perturbations")
    return cell_counts


def get_validated_sources(validation_file):
    """Return validated source genes (Karen_Flag == Use_for_analysis, excluding TP53)."""
    print("Loading target validation data...")
    df = pd.read_csv(validation_file)
    df_analysis = df[df['Karen_Flag'] == 'Use_for_analysis']
    df_analysis = df_analysis[df_analysis['Gene'] != 'TP53']
    validated_sources = set(df_analysis['Gene'].unique())
    print(f"  ✓ Validated sources: {len(validated_sources)}")
    return validated_sources


def identify_zero_hop_pairs(deg_folder, validated_sources, hop1_pairs, hop2_pairs, hop3_pairs):
    """Identify DEG pairs not explained by 1-, 2-, or 3-hop paths (0-hop)."""
    print("\n" + "=" * 60)
    print("IDENTIFYING 0-HOP PAIRS (Unexplained by pathways)")
    print("=" * 60)

    all_explained_pairs = hop1_pairs | hop2_pairs | hop3_pairs
    print(f"  ✓ Total explained pairs: {len(all_explained_pairs):,}")

    zero_hop_data = []
    processed_sources, missing_deg_files = 0, []

    for i, source_gene in enumerate(sorted(validated_sources), 1):
        deg_path = os.path.join(deg_folder, f"{source_gene}_vs_control.csv")
        if not os.path.exists(deg_path):
            missing_deg_files.append(source_gene)
            continue

        processed_sources += 1
        if i % 50 == 0 or i == len(validated_sources):
            print(f"  Processing {i}/{len(validated_sources)}: {source_gene}")

        try:
            deg_df = pd.read_csv(deg_path)
            if 'pvals' not in deg_df.columns or 'logfoldchanges' not in deg_df.columns:
                print(f"  ⚠️ Skipping {source_gene} — required columns missing.")
                continue

            deg_df_sig = deg_df[deg_df['pvals'] < 0.05].copy()
            for _, row in deg_df_sig.iterrows():
                pair = (source_gene, row['names'])
                if pair not in all_explained_pairs:
                    zero_hop_data.append({
                        'source': source_gene,
                        'target': row['names'],
                        'logfoldchange': row['logfoldchanges'],
                        'pval': row['pvals']
                    })
        except Exception as e:
            print(f"  ⚠️ Error processing {source_gene}: {str(e)}")
            continue

    zero_hop_df = pd.DataFrame(zero_hop_data)
    print(f"\n✓ 0-hop pairs found: {len(zero_hop_df):,}")
    return zero_hop_df


def attach_cell_counts(df, cell_counts):
    """Attach cell counts per source gene."""
    df['cell_count'] = df['source'].map(cell_counts).fillna(0).astype(int)
    return df


def main():
    base_path = "/Users/prashammarfatia/Downloads"
    adata_path = os.path.join(base_path, "adata_all_with_guides_raw.h5ad")
    validation_file = os.path.join(base_path, "target_validation_expanded.csv")
    deg_folder = os.path.join(base_path, "de_results_per_gene")

    hop1_file = os.path.join(base_path, "indra_1hop_with_statements_ (1).csv")
    hop2_file = os.path.join(base_path, "indra_2hop_all_perturbations.csv")
    hop3_file = os.path.join(base_path, "indra_3hop_cleaned_results.csv")

    output_dir = os.path.join(base_path, "scatter_plots_cell_count")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load resources
    cell_counts = load_cell_counts(adata_path)
    validated_sources = get_validated_sources(validation_file)

    hop1_df = pd.read_csv(hop1_file)
    hop2_df = pd.read_csv(hop2_file)
    hop3_df = pd.read_csv(hop3_file)
    hop1_pairs = set(zip(hop1_df['source'], hop1_df['target']))
    hop2_pairs = set(zip(hop2_df['source'], hop2_df['target']))
    hop3_pairs = set(zip(hop3_df['source'], hop3_df['target']))

    # Step 2: Identify 0-hop pairs
    hop0_df = identify_zero_hop_pairs(deg_folder, validated_sources, hop1_pairs, hop2_pairs, hop3_pairs)

    # Step 3: Attach cell counts
    hop0_df = attach_cell_counts(hop0_df, cell_counts)

    # Step 4: Select top 100 by |logFC| and top 100 by p-value
    print("\nSelecting top 0-hop pairs by |logFC| and p-value...")
    hop0_df['abs_logfc'] = hop0_df['logfoldchange'].abs()

    top_logfc = (
        hop0_df.sort_values('abs_logfc', ascending=False)
               .head(100)
               .assign(rank_type='high_logFC')
    )

    top_pval = (
        hop0_df.sort_values('pval', ascending=True)
               .head(100)
               .assign(rank_type='low_pval')
    )

    hop0_top = (
        pd.concat([top_logfc, top_pval])
          .drop_duplicates(subset=['source', 'target'])
          .drop(columns='abs_logfc')
    )

    # Step 5: Save outputs
    hop0_out = os.path.join(output_dir, "hop0_all.csv")
    top_out = os.path.join(output_dir, "hop0_top200.csv")
    hop0_df.to_csv(hop0_out, index=False)
    hop0_top.to_csv(top_out, index=False)

    print(f"\n✓ All 0-hop pairs saved: {hop0_out}")
    print(f"✅ Top 200 0-hop pairs (high logFC + low pval) saved: {top_out}")
    print(f"Total unique in top file: {len(hop0_top)}")

if __name__ == "__main__":
    main()
