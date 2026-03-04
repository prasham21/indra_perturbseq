import scanpy as sc
import numpy as np
import pandas as pd

adata_path = "/Users/prashammarfatia/Downloads/adata_all_with_guides_raw.h5ad"
adata = sc.read_h5ad(adata_path)
print(f"Loaded: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

# --- 1️⃣ Define canonical endothelial marker genes ---
markers = ["PECAM1", "CDH5", "KDR", "VWF", "CLDN5"]

# Check which of these are present in your dataset
available_markers = [g for g in markers if g in adata.var_names]
print(f"Available endothelial markers: {available_markers}")

# --- 2️⃣ Compute endothelial "score" per cell (average normalized expression) ---
# This will be a simple average of expression of available markers
adata.raw = adata  # keep raw counts for safety
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

adata.obs["endothelial_score"] = np.array(adata[:, available_markers].X.mean(axis=1)).ravel()

# --- 3️⃣ Define endothelial cells as those with high score (top 10–15%) ---
threshold = np.percentile(adata.obs["endothelial_score"], 85)
adata.obs["is_endothelial"] = adata.obs["endothelial_score"] >= threshold
n_endo = adata.obs["is_endothelial"].sum()

print(f"Identified {n_endo:,} endothelial-like cells "
      f"({n_endo/adata.n_obs*100:.2f}% of total)")

# --- 4️⃣ Subset to endothelial cells ---
adata_endo = adata[adata.obs["is_endothelial"]]
print(f"Subset shape: {adata_endo.n_obs:,} endothelial cells × {adata_endo.n_vars:,} genes")

# --- 5️⃣ Compute gene presence within endothelial subset ---
expr_mask = adata_endo.X > 0
cells_per_gene = np.array(expr_mask.sum(axis=0)).ravel()
percent_cells = (cells_per_gene / adata_endo.n_obs) * 100

# Threshold: ≥3 cells and ≥0.5% of endothelial cells
min_cells = 3
min_percent = 0.5
genes_present = np.where((cells_per_gene >= min_cells) & (percent_cells >= min_percent))[0]

print(f"\nGenes expressed in ≥{min_cells} cells and ≥{min_percent}% "
      f"of endothelial cells: {len(genes_present):,}")

present_gene_names = adata_endo.var_names[genes_present]
print("Top 10 expressed genes in endothelial cells:", list(present_gene_names[:10]))
