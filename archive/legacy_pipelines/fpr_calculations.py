
import os
import pandas as pd
from indra.databases import hgnc_client

# --------- EDIT PATHS IF NEEDED ----------
REAL_FP_1HOP = "/Users/prashammarfatia/Downloads/fp_1hop_paths_p005.csv"
REAL_FP_2HOP = "/Users/prashammarfatia/Downloads/fp_2hop_paths_p005.csv"

GENES_CSV = "/Users/prashammarfatia/Downloads/target_validation_expanded.csv"
DE_DIR    = "/Users/prashammarfatia/Downloads/de_results_per_gene/"
P_THRESHOLD = 0.05

KAREN_FLAG_COL = "Karen_Flag"
KAREN_FLAG_VALUE = "Use_for_analysis"
GENE_COL = "Gene"

OUT_SUMMARY = "/Users/prashammarfatia/Downloads/real_fp_fpr_summary_p005.csv"

def normalize_hgnc_symbol(symbol: str):
    if symbol is None:
        return None
    s = str(symbol).strip()
    if not s:
        return None
    hid = hgnc_client.get_current_hgnc_id(s.upper())
    if not hid:
        return s.upper()
    if isinstance(hid, (list, tuple, set)):
        hid = sorted(list(hid))[0] if hid else None
        if not hid:
            return s.upper()
    name = hgnc_client.get_hgnc_name(hid) if hid else None
    return name or s.upper()

def pick_sig_column(df: pd.DataFrame) -> str:
    for c in ["pvals", "pval", "p_value", "p_val", "pvals_adj", "padj", "qval", "fdr", "p_adj"]:
        if c in df.columns:
            return c
    raise ValueError(f"No p-value column found. Columns: {df.columns.tolist()}")

def load_pairs(csv_path: str):
    df = pd.read_csv(csv_path, low_memory=False)
    if not {"source","target"}.issubset(df.columns):
        raise ValueError(f"{csv_path} missing source/target columns. Has: {df.columns.tolist()}")
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df = df[df["source"] != df["target"]].copy()
    return set(zip(df["source"], df["target"]))

def total_negative_pairs_tested():
    genes_df = pd.read_csv(GENES_CSV, low_memory=False)
    if KAREN_FLAG_COL in genes_df.columns:
        genes_df = genes_df[genes_df[KAREN_FLAG_COL] == KAREN_FLAG_VALUE].copy()
    sources_raw = [str(x).strip() for x in genes_df[GENE_COL].dropna().tolist() if str(x).strip()]

    total_neg = 0
    per_source = []

    for raw_src in sources_raw:
        src = normalize_hgnc_symbol(raw_src)
        if not src:
            continue

        deg_path = os.path.join(DE_DIR, f"{raw_src}_vs_control.csv")
        if not os.path.exists(deg_path):
            continue

        d = pd.read_csv(deg_path, low_memory=False)
        if "names" not in d.columns:
            continue

        pcol = pick_sig_column(d)
        d[pcol] = pd.to_numeric(d[pcol], errors="coerce")

        # Universe = all names in file (HGNC-normalized)
        all_set = set()
        for t in d["names"].dropna().astype(str).tolist():
            tn = normalize_hgnc_symbol(t)
            if tn:
                all_set.add(tn)

        # Positives = p < threshold
        pos_set = set()
        dd = d[d[pcol] < P_THRESHOLD].copy()
        for t in dd["names"].dropna().astype(str).tolist():
            tn = normalize_hgnc_symbol(t)
            if tn:
                pos_set.add(tn)

        # remove self
        all_set.discard(src)
        pos_set.discard(src)

        neg_set = all_set - pos_set
        nneg = len(neg_set)
        total_neg += nneg

        per_source.append((src, len(all_set), len(pos_set), nneg))

    return total_neg, pd.DataFrame(per_source, columns=["source","universe_targets","positive_targets","negative_targets"])

print("Loading REAL FP result CSVs...")
real1 = load_pairs(REAL_FP_1HOP)
real2 = load_pairs(REAL_FP_2HOP)
real_u = real1 | real2

print("Computing total negative pairs tested from DEG control files...")
TOTAL_NEG, per_src_df = total_negative_pairs_tested()
print("Total negative pairs tested (summed over sources):", TOTAL_NEG)

rows = []
def add(hop, fp_pairs):
    rows.append({
        "dataset": "real",
        "hop": hop,
        "fp_unique_pairs": len(fp_pairs),
        "total_negative_pairs_tested": TOTAL_NEG,
        "fpr": (len(fp_pairs) / TOTAL_NEG) if TOTAL_NEG else 0.0
    })

add("1hop", real1)
add("2hop", real2)
add("<=2hop_union", real_u)

out = pd.DataFrame(rows)
print("\nSUMMARY")
print(out.to_string(index=False))

out.to_csv(OUT_SUMMARY, index=False)
print("\nWrote:", OUT_SUMMARY)

per_src_out = OUT_SUMMARY.replace(".csv", "_per_source_neg_counts.csv")
per_src_df.to_csv(per_src_out, index=False)
print("Wrote:", per_src_out)
