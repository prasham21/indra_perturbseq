import numpy as np
import pandas as pd
import scanpy as sc

TV_PATH = "/Users/prashammarfatia/Downloads/target_validation_expanded.csv"
COUNTS_CSV = "target_parent_counts.csv"
ADATA_PATH = "/Users/prashammarfatia/Downloads/adata_singlets_normalized.h5ad"

IGNORE = {"TP53", "CDK1NA", "CDKN1A"}

# The 17 genes you printed as "U but NOT in top50% variable"
missing17 = [
    "AGAP2","B3GNT8","BMPER","CLDN1","CX3CL1","CYP2J2","EMP2","GSTM2",
    "HECW1","LSP1","OASL","PROX1","PRPH2","QRICH2","RSPO3","SELE","TNFSF10"
]

# -------------------------
# 1) Build source_set and target_set
# -------------------------
tv = pd.read_csv(TV_PATH)
sources = (
    tv.loc[tv["Karen_Flag"] == "Use_for_analysis", "Gene"]
    .dropna().astype(str).str.strip()
    .drop_duplicates()
)
sources = sources[~sources.str.upper().isin(IGNORE)]
source_set = set(sources)

counts = pd.read_csv(COUNTS_CSV)
targets = (
    counts["target"]
    .dropna().astype(str).str.strip()
    .drop_duplicates()
)
targets = targets[~targets.str.upper().isin(IGNORE)]
target_set = set(targets)

U = source_set | target_set
print("unique_sources:", len(source_set))
print("unique_targets:", len(target_set))
print("U_size (sources ∪ targets):", len(U))

# -------------------------
# 2) Build V = top 50% most variable genes in normalized+log1p AnnData
# -------------------------
adata = sc.read_h5ad(ADATA_PATH)
print("Example adata.var_names:", adata.var_names[:10].tolist())
print("Looks like Ensembl IDs?", str(adata.var_names[0]).startswith("ENSG"))

genes = np.array(adata.var_names.astype(str))

X = adata.X
try:
    mean = X.mean(axis=0).A1
    mean2 = X.power(2).mean(axis=0).A1
    var = mean2 - mean**2
except AttributeError:
    X = np.asarray(X)
    var = X.var(axis=0)

cut = np.quantile(var, 0.50)
V = set(genes[var >= cut])

print("V_size (top 50% variable genes):", len(V))

# -------------------------
# 3) Confirm the 17 are really U\V, and classify SOURCE vs TARGET vs BOTH
# -------------------------
print("\n=== Missing 17 classification (SOURCE vs TARGET vs BOTH) ===")
rows = []
for g in missing17:
    g2 = str(g).strip()
    in_source = g2 in source_set
    in_target = g2 in target_set
    in_U = g2 in U
    in_V = g2 in V

    if in_source and in_target:
        where = "BOTH source+target"
    elif in_source:
        where = "SOURCE only"
    elif in_target:
        where = "TARGET only"
    else:
        where = "NEITHER (unexpected)"

    rows.append(
        {
            "gene": g2,
            "where_in_U": where,
            "in_U": in_U,
            "in_top50_var(V)": in_V,
        }
    )

df = pd.DataFrame(rows).sort_values(["where_in_U", "gene"])
print(df.to_string(index=False))

print("\nCounts by category:")
print(df["where_in_U"].value_counts().to_string())

# Optional: sanity check that these are indeed U but not V
bad = df[~(df["in_U"] & ~df["in_top50_var(V)"])]
print("\nSanity check: genes not matching 'in_U and not in V' =", len(bad))
if len(bad) > 0:
    print(bad.to_string(index=False))