import pandas as pd

from debug_cell_count import present_gene_names

# 1️⃣ Load the automatically detected endothelial-present genes
present_genes = list(present_gene_names)  # from your previous run (13,824 genes)

# 2️⃣ Manually curated endothelial/vascular-relevant genes
manual_genes = [
    "BCAR1", "BMP1", "CALCRL", "CCM2", "CDKN1A", "CDKN2B", "CFDP1", "COL4A1", "COL4A2",
    "EXOC3L2", "FBN2", "FGD6", "FLT1", "FURIN", "GDPD5", "GGT5", "GOSR2", "IBTK", "LAMB2",
    "LOX", "MORF4L1", "N4BP2L2", "NOS3", "PALLD", "PECAM1", "PGF", "PLPP3", "PREX1", "PRKAR1A",
    "SCUBE1", "SERPINH1", "SH3PXD2A", "SLK", "SMAD3", "SPRY4", "SVIL", "SWAP70", "TFPI",
    "TLNRD1", "TSPAN14", "ZEB2"
]

# 3️⃣ Merge and deduplicate
combined_genes = sorted(set(present_genes + manual_genes))
print(f"Total unique endothelial-present genes after adding manual list: {len(combined_genes):,}")

# 4️⃣ Save to CSV
output_path = "/Users/prashammarfatia/Downloads/endothelial_present_plus_manual.csv"
pd.Series(combined_genes, name="gene").to_csv(output_path, index=False)
print(f"✅ Saved combined gene list to: {output_path}")

# 5️⃣ Optional preview
print("\nExample genes:", combined_genes[:20])

# Convert to sets
present_set = set(present_genes)
manual_set = set(manual_genes)

# 1️⃣ Overlap genes
overlap_genes = sorted(present_set.intersection(manual_set))

# 2️⃣ Missing genes
missing_genes = sorted(manual_set - present_set)

print(f"Total manual/GWAS genes: {len(manual_set)}")
print(f"Present in endothelial list: {len(overlap_genes)}")
print(f"Missing from endothelial list: {len(missing_genes)}")

print("\n✔ Overlapping genes:")
print(overlap_genes)

print("\n❌ Missing genes:")
print(missing_genes)
