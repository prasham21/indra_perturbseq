import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- EDIT THESE PATHS ----
REAL_1HOP_CSV = "/Users/prashammarfatia/Downloads/indra_1hop_with_statements__main (2).csv"
PERM_1HOP_CSV = "/Users/prashammarfatia/Downloads/1hop_PERMUTED.csv"

# Thresholds to test (you can change this list)
THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]

# If True: count unique (source,target) pairs instead of raw rows
COUNT_UNIQUE_PAIRS = False


def find_pval_col(df: pd.DataFrame) -> str:
    for c in ["pval", "pvals", "PVAL", "PVALS"]:
        if c in df.columns:
            return c
    raise ValueError(f"No p-value column found. Columns are: {df.columns.tolist()}")


def count_paths(df: pd.DataFrame, pval_col: str, thr: float) -> int:
    sub = df[df[pval_col] <= thr]
    if COUNT_UNIQUE_PAIRS and ("source" in sub.columns) and ("target" in sub.columns):
        return sub.drop_duplicates(subset=["source", "target"]).shape[0]
    return sub.shape[0]


def main():
    real = pd.read_csv(REAL_1HOP_CSV)
    perm = pd.read_csv(PERM_1HOP_CSV)

    real_pcol = find_pval_col(real)
    perm_pcol = find_pval_col(perm)

    # Ensure numeric pvals
    real[real_pcol] = pd.to_numeric(real[real_pcol], errors="coerce")
    perm[perm_pcol] = pd.to_numeric(perm[perm_pcol], errors="coerce")

    real_counts = []
    perm_counts = []

    for thr in THRESHOLDS:
        real_counts.append(count_paths(real, real_pcol, thr))
        perm_counts.append(count_paths(perm, perm_pcol, thr))

    out = pd.DataFrame(
        {"pval_threshold": THRESHOLDS, "real_1hop_paths": real_counts, "permuted_1hop_paths": perm_counts}
    )
    print(out)

    plt.figure(figsize=(7, 4.5))
    plt.plot(THRESHOLDS, real_counts, marker="o", linewidth=2, label="Real 1-hop")
    plt.plot(THRESHOLDS, perm_counts, marker="o", linewidth=2, label="Permuted 1-hop")
    plt.xlabel("DEG p-value threshold (keep rows with pval ≤ threshold)")
    plt.ylabel("Number of 1-hop paths" + (" (unique source-target)" if COUNT_UNIQUE_PAIRS else " (rows)"))
    plt.title("1-hop paths vs DEG p-value threshold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("1hop_paths_vs_pval.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()