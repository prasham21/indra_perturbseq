import os
from collections import defaultdict

import numpy as np
import pandas as pd

TV_PATH = "/Users/prashammarfatia/Downloads/target_validation_expanded.csv"
DEG_DIR = "/Users/prashammarfatia/Downloads/de_results_per_gene"
OUT_CSV = "target_parent_counts.csv"

PVAL_THRESHOLD = 0.05

IGNORE = {"TP53", "CDK1NA", "CDKN1A"}  # ignore these targets

# 1) sources (parents)
tv = pd.read_csv(TV_PATH)
sources = (
    tv.loc[tv["Karen_Flag"] == "Use_for_analysis", "Gene"]
    .dropna()
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .tolist()
)

# 2) build target -> set(parents)
parents = defaultdict(set)
missing_deg = 0

for src in sources:
    deg_path = os.path.join(DEG_DIR, f"{src}_vs_control.csv")
    if not os.path.exists(deg_path):
        missing_deg += 1
        continue

    deg = pd.read_csv(deg_path, usecols=["names", "pvals"])
    deg = deg.dropna(subset=["names", "pvals"])
    deg["pvals"] = pd.to_numeric(deg["pvals"], errors="coerce")
    deg = deg[deg["pvals"] < PVAL_THRESHOLD]

    for tgt in deg["names"].astype(str).str.strip().unique():
        if tgt and tgt.upper() not in IGNORE:
            parents[tgt].add(src)

# 3) write output
out = pd.DataFrame(
    {
        "target": list(parents.keys()),
        "n_parents": [len(v) for v in parents.values()],
        "parents": [";".join(sorted(v)) for v in parents.values()],
    }
).sort_values(["n_parents", "target"], ascending=[False, True])

out.to_csv(OUT_CSV, index=False)

# 4) simple logs (outliers + low/high)
arr = out["n_parents"].to_numpy()
q1, q3 = np.quantile(arr, [0.25, 0.75])
iqr = q3 - q1
hi_cut = q3 + 1.5 * iqr

print(f"Sources used: {len(sources)} (missing DEG files: {missing_deg})")
print(f"Targets: {len(out)}")
print(f"n_parents min/median/mean/max: {arr.min()} / {np.median(arr):.1f} / {arr.mean():.2f} / {arr.max()}")
print(f"High-outlier cutoff (Q3+1.5*IQR): {hi_cut:.1f}")
print("Targets with 1 parent:", int((out["n_parents"] == 1).sum()))
print("Targets with 2 parents:", int((out["n_parents"] == 2).sum()))

print("\nTop 20 most parents:")
print(out.head(20)[["target", "n_parents"]].to_string(index=False))

high_outliers = out[out["n_parents"] > hi_cut]
print(f"\nHigh outliers: {len(high_outliers)}")
if len(high_outliers) > 0:
    print(high_outliers.head(30)[["target", "n_parents"]].to_string(index=False))

print("\nBottom 20 least parents:")
print(out.tail(20)[["target", "n_parents"]].to_string(index=False))

# 5) quick CSV inspection
print("\n=== CSV inspection ===")
df = pd.read_csv(OUT_CSV)
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print("dtypes:\n", df.dtypes)
print("missing:\n", df.isna().sum())
print("\nhead:\n", df.head(5).to_string(index=False))

# sanity: n_parents matches parents list
lens = df["parents"].fillna("").apply(lambda s: 0 if s.strip() == "" else len([x for x in s.split(";") if x.strip()]))
bad = df[lens != df["n_parents"]]
print("rows where len(parents)!=n_parents:", len(bad))