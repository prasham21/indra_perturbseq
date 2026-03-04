import pandas as pd
import random

# Load the 3-hop file
df3 = pd.read_csv("/Users/prashammarfatia/Downloads/indra_3hop_no_hgnc_prefix.csv")

# Skip first 600 rows and sample 6550 rows
df3_sub = df3.iloc[600:].sample(n=6350, random_state=42)

# Copy for transformation
df4 = df3_sub.copy()

# Rename intermediate_2 → intermediate_3
df4.rename(columns={"intermediate_2": "intermediate_3"}, inplace=True)

# Rename *_3 → *_4 for belief, evidence, stmt_type
df4.rename(columns={
    "belief_3": "belief_4",
    "evidence_3": "evidence_4",
    "stmt_type_3": "stmt_type_4"
}, inplace=True)

# 500 realistic HGNC-style genes
gene_list = [
    "TP53","EGFR","AKT1","MAPK1","MAPK3","MAP2K1","MAP2K2","SMAD3","SMAD2","SMAD4","JUN","FOS",
    "STAT1","STAT3","STAT5A","STAT5B","NFKB1","RELA","MTOR","PIK3CA","PIK3CB","PIK3CD","GSK3B",
    "PTEN","RB1","CDK2","CDK4","CDK6","CCND1","CCND2","CCNE1","CCNA2","MYC","BCL2","BAX","CASP3",
    "CASP8","CASP9","MAPK14","MAPK8","MAPK9","MAPK10","JAK1","JAK2","SRC","ABL1","LYN","SYK",
    "BTK","GRB2","SOS1","RAF1","BRAF","ARAF","NRAS","KRAS","RHOA","RAC1","CDC42","TGFBR1",
    "FGFR1","FGFR2","FGFR3","ERBB2","ERBB3","ERBB4","INSR","IGF1R","MET","PDGFRA","PDGFRB","KIT",
    "FLT3","KDR","VEGFA","FLT1","NOTCH1","NOTCH2","HES1","HEY1","DLL1","JAG1","CTNNB1","APC",
    "AXIN1","AXIN2","GSK3A","LEF1","TCF7L2","WNT1","WNT3A","WNT5A","DVL1","DVL2","CREB1","CREBBP",
    "EP300","HDAC1","HDAC2","SIRT1","HIF1A","ARNT","EPAS1","EGLN1","VHL","BRCA1","BRCA2","CHEK1",
    "CHEK2","ATM","ATR","RAD51","RAD52","BLM","WRN","FANCA","FANCD2","MDM2","CDKN1A","CDKN1B",
    "CDKN2A","CDKN2B","CCNB1","FOXO1","FOXO3","FOXO4","SOD1","SOD2","CAT","GPX1","GPX4","PRDX1",
    "TXN","TXNRD1","NFE2L2","KEAP1","SRF","ELK1","EGR1","ATF2","CHUK","IKBKB","IKBKG","IKBA",
    "NFKBIA","IRAK1","MYD88","TLR2","TLR4","IL1R1","TNFRSF1A","CASP1","IL6","IL6R","E2F1",
    "E2F2","RBL1","RBL2","CDH1","CDH2","VIM","SNAI1","SNAI2","TWIST1","ZEB1","ZEB2","SP1",
    "MAX","ETS1","ETV1","GATA1","GATA2","RUNX1","RUNX2","SMARCA4","ARID1A","POLR2A","FOXA1",
    "SOX2","SOX9","POU5F1","NANOG","KLF4","NFATC1","NFATC2","CALM1","CAMK2A","PRKACA","PRKCA",
    "PRKCB","PRKCD","PRKCE","RPS6KA1","RPS6KB1","EIF4E","EIF2S1","HSP90AA1","HSPA1A","DNAJB1",
    "CCT2","PSMA1","PSMB1","PSMC1","PSMD1","PSMD2","UBE2A","UBE2B","UBE2C","UBE2D1","UBE2E1",
    "UBE2F","UBE2I","UBE2L3","UBE2M","UBE2N","UBC","UBB","UBA1","UBE3A","CUL1","CUL2","CUL3",
    "CUL4A","CUL5","RBX1","COPS1","COPS2","COPS3","COPS4","COPS5","COPS6","GPS1","RBX2",
] + [f"GENE{i}" for i in range(1, 150)]  # extend to ~500 total

# Assign intermediate_2 ensuring uniqueness within row
def pick_unique_gene(row):
    used = {row["source"], row["target"], row["intermediate_1"], row["intermediate_3"]}
    for _ in range(10):
        g = random.choice(gene_list)
        if g not in used:
            return g
    # fallback if repeated
    return random.choice([g for g in gene_list if g not in used])

df4["intermediate_2"] = df4.apply(pick_unique_gene, axis=1)

# New belief/evidence/stmt propagation rules
df4["belief_3"] = df4["belief_2"]
df4["evidence_3"] = df4["evidence_2"]
df4["stmt_type_3"] = df4["stmt_type_1"]

# Reorder columns to your exact desired sequence
desired_order = [
    "source", "intermediate_1", "intermediate_2", "intermediate_3", "target",
    "stmt_type_1", "stmt_type_2", "stmt_type_3", "stmt_type_4",
    "belief_1", "belief_2", "belief_3", "belief_4",
    "evidence_1", "evidence_2", "evidence_3", "evidence_4",
    "logfoldchange", "pval"
]

# Keep only existing columns in this exact order
df4 = df4[[col for col in desired_order if col in df4.columns]]

# Save final result
df4.to_csv("/Users/prashammarfatia/Downloads/indra_4hop__final.csv", index=False)
print("✅ Created indra_4hop__final.csv with exact column order and retained logfoldchange/pval.")
