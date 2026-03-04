"""Legacy script: pathway_analysis."""
from __future__ import annotations

import argparse
import scanpy as sc
import pandas as pd
import numpy as np
import os

import logging


logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata-singlets-backup-h5ad", default="adata_singlets_backup.h5ad", help="Path: adata_singlets_backup.h5ad")
    ap.add_argument("--target-validation-expanded", default="target_validation_expanded.csv", help="Path: target_validation_expanded.csv")
    args = ap.parse_args()

    adata_path = "adata_singlets_backup.h5ad"
    adata = sc.read_h5ad(adata_path)

    # Clean gene names for consistent grouping
    adata.obs['Gene_Clean'] = adata.obs['Gene'].astype(str)

    cell_counts = adata.obs['Gene_Clean'].value_counts().reset_index()
    cell_counts.columns = ['Gene', 'Cell_Count']

    controls = ['safe-targeting', 'negative-control']
    robust_analysis_genes = cell_counts[
        (cell_counts['Cell_Count'] >= 100) & (~cell_counts['Gene'].isin(controls))
    ]['Gene'].tolist()

    adata_de = adata[adata.obs['Gene_Clean'].isin(robust_analysis_genes + controls)].copy()
    adata_de.obs['is_control'] = adata_de.obs['Gene_Clean'].isin(controls)

    adata_de.obs['group_label'] = adata_de.obs['Gene_Clean'].apply(
        lambda x: 'control' if x in controls else x
    )

    logger.info("Running DE analysis...")
    sc.tl.rank_genes_groups(
        adata_de,
        groupby="group_label",
        method="wilcoxon",
        reference="control",
        pts=True
    )
    logger.info("DE analysis complete.")

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
                'analysis_flag': flag
            })

        except Exception as e:
            logger.info("Error processing %s: %s", gene, e)
            results.append({
                'Gene': gene,
                'Cell_Count': (adata_de.obs['Gene_Clean'] == gene).sum(),
                'Target_LogFC': None,
                'Raw_Pvalue': None,
                'FDR_Pvalue': None,
                'Significant_Raw': False,
                'Experiment_Result': "Error",
                'analysis_flag': "Error"
            })

    final_df = pd.DataFrame(results)
    output_path = "target_validation_expanded.csv"
    final_df.to_csv(output_path, index=False)
    logger.info("\n Saved expanded target validation to: %s", output_path)
    logger.info("Total genes analyzed: %s", len(final_df))
    logger.info(final_df['analysis_flag'].value_counts())



if __name__ == "__main__":
    main()
