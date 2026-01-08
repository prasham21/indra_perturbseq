import scanpy as sc
import pandas as pd
import os

# Load AnnData
adata = sc.read_h5ad("/Users/prashammarfatia/Downloads/adata_singlets_backup.h5ad")

# Prepare control & group labels
adata.obs['Gene_Clean'] = adata.obs['Gene'].str.replace('-TSS2', '', regex=False)
control_genes = ['negative-control', 'safe-targeting']
adata.obs['Gene_Group'] = adata.obs['Gene_Clean'].where(~adata.obs['Gene_Clean'].isin(control_genes), 'control')

# Run DE
sc.tl.rank_genes_groups(
    adata,
    groupby="Gene_Group",
    reference="control",
    method="wilcoxon",
    pts=True
)

# Save DE results (only)
output_dir = "/Users/prashammarfatia/Downloads/de_results_per_gene"
os.makedirs(output_dir, exist_ok=True)

all_groups = adata.uns['rank_genes_groups']['names'].dtype.names

for gene in all_groups:
    df = sc.get.rank_genes_groups_df(adata, group=gene)
    df.to_csv(f"{output_dir}/{gene}_vs_control.csv", index=False)

print(f" Saved DE result CSVs for {len(all_groups)} genes to: {output_dir}")
