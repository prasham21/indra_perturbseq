import os
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────
BULK_DIR  = "/Users/prashammarfatia/Downloads/de_results_bulk_6genes"
SCRNA_DIR = "/Users/prashammarfatia/Downloads/de_results_scrna_validation6_top50var"
ENDO_LIST = "/Users/prashammarfatia/Downloads/endothelial_present_plus_manual.csv"

OUT_CSV   = "/Users/prashammarfatia/Downloads/bulk_vs_scrna_overlap_3genes_.csv"

GENES = ["ITGB1BP1", "CCM2", "MAP2K5"]
FDR = 0.05
# ─────────────────────────────────────────────────────────────────────

endo = pd.read_csv(ENDO_LIST, low_memory=False)
endo_set = set(endo["gene"].astype(str).str.strip())
endo_set.discard("")

rows = []
for g in GENES:
    bulk_path  = os.path.join(BULK_DIR,  f"{g}_vs_control.csv")
    scrna_path = os.path.join(SCRNA_DIR, f"{g}_vs_control.csv")

    bulk  = pd.read_csv(bulk_path, low_memory=False)
    scrna = pd.read_csv(scrna_path, low_memory=False)

    # Clean + restrict to endothelial universe
    bulk["names"]  = bulk["names"].astype(str).str.strip()
    scrna["names"] = scrna["names"].astype(str).str.strip()

    bulk  = bulk[bulk["names"].isin(endo_set)].copy()
    scrna = scrna[scrna["names"].isin(endo_set)].copy()

    # Numeric pvals
    bulk["pvals_adj"]  = pd.to_numeric(bulk["pvals_adj"], errors="coerce")
    scrna["pvals_adj"] = pd.to_numeric(scrna["pvals_adj"], errors="coerce")

    # "Overall DEGs" in this context = genes with a valid FDR value (tested + not NaN)
    bulk_tested  = set(bulk.loc[bulk["pvals_adj"].notna(), "names"])
    scrna_tested = set(scrna.loc[scrna["pvals_adj"].notna(), "names"])

    tested_inter = bulk_tested & scrna_tested
    tested_union = bulk_tested | scrna_tested

    # Significant sets (FDR < 0.05)
    bulk_sig  = set(bulk.loc[bulk["pvals_adj"] < FDR, "names"])
    scrna_sig = set(scrna.loc[scrna["pvals_adj"] < FDR, "names"])

    sig_inter = bulk_sig & scrna_sig
    sig_union = bulk_sig | scrna_sig

    rows.append({
        "gene": g,

        # Overall (tested) within endothelial universe
        "bulk_tested_endo": len(bulk_tested),
        "scrna_tested_endo": len(scrna_tested),
        "tested_intersection": len(tested_inter),
        "tested_union": len(tested_union),
        "tested_jaccard": (len(tested_inter) / len(tested_union)) if tested_union else np.nan,

        # Significant at FDR threshold
        "fdr_threshold": FDR,
        "bulk_sig_fdr": len(bulk_sig),
        "scrna_sig_fdr": len(scrna_sig),
        "sig_intersection": len(sig_inter),
        "sig_union": len(sig_union),
        "sig_jaccard": (len(sig_inter) / len(sig_union)) if sig_union else np.nan,
        "sig_overlap_frac_of_bulk": (len(sig_inter) / len(bulk_sig)) if bulk_sig else np.nan,
        "sig_overlap_frac_of_scrna": (len(sig_inter) / len(scrna_sig)) if scrna_sig else np.nan,
    })

summary = pd.DataFrame(rows).sort_values("gene")
summary.to_csv(OUT_CSV, index=False)
print("Wrote:", OUT_CSV)
print(summary.to_string(index=False))