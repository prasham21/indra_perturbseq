import scanpy as sc
import pandas as pd

# ---------------------------
# Load AnnData + Cell counts
# ---------------------------
adata_path = "/Users/prashammarfatia/Downloads/adata_all_with_guides_raw.h5ad"
adata = sc.read_h5ad(adata_path)
cell_counts = adata.obs["Gene"].value_counts().to_dict()

# ---------------------------
# Load Validation CSV
# ---------------------------
val_path = "/Users/prashammarfatia/Downloads/target_validation_expanded.csv"
df = pd.read_csv(val_path)

val_genes = df["Gene"].astype(str).tolist()
karen_map = df.set_index("Gene")["Karen_Flag"].to_dict()

# ---------------------------
# Define Genes
# ---------------------------
gwas_genes = [
    "BCAR1","BMP1","CALCRL","CCM2","CDKN1A","CDKN2B","CFDP1","COL4A1","COL4A2",
    "EXOC3L2","FBN2","FGD6","FLT1","FURIN","GDPD5","GGT5","GOSR2","IBTK","LAMB2",
    "LOX","MORF4L1","N4BP2L2","NOS3","PALLD","PECAM1","PGF","PLPP3","PREX1",
    "PRKAR1A","SCUBE1","SERPINH1","SH3PXD2A","SLK","SMAD3","SPRY4","SVIL",
    "SWAP70","TFPI","TLNRD1","TSPAN14","ZEB2"
]

non_gwas_genes = [
    "PRDM16", "PLPP3", "NOS3", "JCAD", "FLT1", "PECAM1", "EDN1", "ARHGEF26"
]

# ---------------------------
# Helper: create full report
# ---------------------------
def create_full_report(gene_list):
    rows = []
    for g in gene_list:
        rows.append({
            "Gene": g,
            "Cell_Count": cell_counts.get(g, 0),
            "Present_in_CSV": g in val_genes,
            "Karen_Flag": karen_map.get(g, "NOT_PRESENT")
        })
    return pd.DataFrame(rows).sort_values("Cell_Count", ascending=False)

# ---------------------------
# Create reports
# ---------------------------
gwas_report = create_full_report(gwas_genes)
non_gwas_report = create_full_report(non_gwas_genes)

# ---------------------------
# Print everything clearly
# ---------------------------
print("\n==================== FULL GWAS REPORT ====================")
print(gwas_report.to_string(index=False))

print("\n==================== FULL NON-GWAS REPORT ====================")
print(non_gwas_report.to_string(index=False))
