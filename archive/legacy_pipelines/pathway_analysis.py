import scanpy as sc
import pandas as pd
import numpy as np
import os

# === Step 1: Load and prepare data ===
adata_path = "/Users/prashammarfatia/Downloads/adata_singlets_backup.h5ad"
adata = sc.read_h5ad(adata_path)

# Clean gene names for consistent grouping
adata.obs['Gene_Clean'] = adata.obs['Gene'].astype(str)

# Step 2: Compute per-gene cell counts
cell_counts = adata.obs['Gene_Clean'].value_counts().reset_index()
cell_counts.columns = ['Gene', 'Cell_Count']

# Step 3: Filter for perturbations with ≥100 cells, excluding controls
controls = ['safe-targeting', 'negative-control']
robust_analysis_genes = cell_counts[
    (cell_counts['Cell_Count'] >= 100) & (~cell_counts['Gene'].isin(controls))
]['Gene'].tolist()

# Step 4: Subset AnnData object for DE
adata_de = adata[adata.obs['Gene_Clean'].isin(robust_analysis_genes + controls)].copy()
adata_de.obs['is_control'] = adata_de.obs['Gene_Clean'].isin(controls)

# Step 5: Set up combined control label
adata_de.obs['group_label'] = adata_de.obs['Gene_Clean'].apply(
    lambda x: 'control' if x in controls else x
)

# Step 6: Run DE analysis (can take ~1 hour)
print("Running DE analysis...")
sc.tl.rank_genes_groups(
    adata_de,
    groupby="group_label",
    method="wilcoxon",
    reference="control",
    pts=True
)
print("DE analysis complete.")

# Step 7: Analyze target knockdown success
results = []
for gene in robust_analysis_genes:
    try:
        result_df = sc.get.rank_genes_groups_df(adata_de, group=gene)
        target_row = result_df[result_df['names'] == gene]

        if not target_row.empty:
            logfc = target_row.iloc[0]['logfoldchanges']
            raw_pval = target_row.iloc[0]['pvals']
            adj_pval = target_row.iloc[0]['pvals_adj']

            # Classify result
            if logfc < 0:
                status = "Successful_Knockdown"
                flag = "Use_for_analysis"
            elif logfc > 0:
                status = "Potential_Feedback_Loop"
                flag = "Flag_and_set_aside"
            else:
                status = "No_Change"
                flag = "Unclear"

            significant = raw_pval < 0.05
        else:
            logfc = raw_pval = adj_pval = None
            status = "Target_Not_Detected"
            flag = "Experiment_failed"
            significant = False

        results.append({
            'Gene': gene,
            'Cell_Count': (adata_de.obs['Gene_Clean'] == gene).sum(),
            'Target_LogFC': logfc,
            'Raw_Pvalue': raw_pval,
            'FDR_Pvalue': adj_pval,
            'Significant_Raw': significant,
            'Experiment_Result': status,
            'Karen_Flag': flag
        })

    except Exception as e:
        print(f"Error processing {gene}: {e}")
        results.append({
            'Gene': gene,
            'Cell_Count': (adata_de.obs['Gene_Clean'] == gene).sum(),
            'Target_LogFC': None,
            'Raw_Pvalue': None,
            'FDR_Pvalue': None,
            'Significant_Raw': False,
            'Experiment_Result': "Error",
            'Karen_Flag': "Error"
        })

# Step 8: Save results
final_df = pd.DataFrame(results)
output_path = "/Users/prashammarfatia/Downloads/target_validation_expanded.csv"
final_df.to_csv(output_path, index=False)
print(f"\n Saved expanded target validation to: {output_path}")
print(f"Total genes analyzed: {len(final_df)}")
print(final_df['Karen_Flag'].value_counts())
