import pandas as pd
import scanpy as sc

adata = sc.read_h5ad("/Users/prashammarfatia/Downloads/adata_all_with_guides_raw.h5ad")
cell_counts = adata.obs['Gene'].value_counts().to_dict()

hop2 = pd.read_csv("/Users/prashammarfatia/Downloads/indra_2hop_all_perturbations.csv")
hop2['cell_count'] = hop2['source'].map(cell_counts).fillna(0).astype(int)

# Find sources with low/missing cell counts
low_counts = hop2[hop2['cell_count'] < 200]['source'].unique()
print(f"Sources with cell_count < 200: {len(low_counts)}")
print(f"Examples: {low_counts[:10]}")

# Check if these exist in cell_counts at all
for source in low_counts[:5]:
    if source in cell_counts:
        print(f"  {source}: {cell_counts[source]} cells")
    else:
        print(f"  {source}: NOT FOUND in AnnData")