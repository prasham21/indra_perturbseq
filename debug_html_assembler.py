from pathlib import Path

# 1) Point this to the folder you were in when you ran the terminal loop
base_dir = Path("/Users/prashammarfatia/Downloads/de_results_per_gene")

print("base_dir =", base_dir)
print("base_dir exists?", base_dir.exists())
print("num csv files in base_dir:", len(list(base_dir.glob("*.csv"))))
print("sample files:", [p.name for p in list(base_dir.glob("*_vs_control.csv"))[:10]])

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

def exists(gene: str) -> bool:
    return (base_dir / f"{gene}_vs_control.csv").exists()

print("\nGWAS genes:")
for g in sorted(gwas_genes):
    print(f"{g:12s} -> {'YES' if exists(g) else 'NO'}")

print("\nNon-GWAS genes:")
for g in sorted(non_gwas_genes):
    print(f"{g:12s} -> {'YES' if exists(g) else 'NO'}")
