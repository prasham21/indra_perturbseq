import os
import pandas as pd
from indra.databases import hgnc_client

# --------- EDIT THESE 4 FILE PATHS ----------
REAL_1HOP = "/Users/prashammarfatia/Downloads/indra_1hop_NETWORK_EXPORT_main.csv"
REAL_2HOP = "/Users/prashammarfatia/Downloads/2hop_network_export_main.csv"
PERM_1HOP = "/Users/prashammarfatia/Downloads/permuted_1hop_paths_seed44.csv"
PERM_2HOP = "/Users/prashammarfatia/Downloads/permuted_2hop_paths_seed44.csv"

# --------- EDIT THESE INPUT PATHS ----------
GENES_CSV = "/Users/prashammarfatia/Downloads/target_validation_expanded.csv"
DE_DIR    = "/Users/prashammarfatia/Downloads/de_results_per_gene/"

P_THRESHOLD = 0.05
KAREN_FLAG_COL = "Karen_Flag"
KAREN_FLAG_VALUE = "Use_for_analysis"
GENE_COL = "Gene"

OUT_SUMMARY = "/Users/prashammarfatia/Downloads/tp_tpr_summary_seed42.csv"

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
    # tries common columns you use
    for c in ["pvals", "pval", "p_value", "p_val", "pvals_adj", "padj", "qval", "fdr", "p_adj"]:
        if c in df.columns:
            return c
    raise ValueError(f"No p-value column found. Columns: {df.columns.tolist()}")

def load_pairs_from_results(csv_path: str):
    df = pd.read_csv(csv_path, low_memory=False)
    if not {"source","target"}.issubset(df.columns):
        raise ValueError(f"{csv_path} missing source/target columns. Has: {df.columns.tolist()}")
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df = df[df["source"] != df["target"]].copy()
    pairs = set(zip(df["source"], df["target"]))
    return pairs, df

def build_positive_pairs_from_degs():
    genes_df = pd.read_csv(GENES_CSV, low_memory=False)
    if KAREN_FLAG_COL in genes_df.columns:
        genes_df = genes_df[genes_df[KAREN_FLAG_COL] == KAREN_FLAG_VALUE].copy()
    sources_raw = [str(x).strip() for x in genes_df[GENE_COL].dropna().tolist() if str(x).strip()]

    pos_pairs = set()
    per_source_counts = {}

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
        d = d[d[pcol] < P_THRESHOLD].copy()
        if d.empty:
            continue
        targets = []
        for t in d["names"].dropna().astype(str).tolist():
            tn = normalize_hgnc_symbol(t)
            if tn and tn != src:
                targets.append(tn)
        targets = sorted(set(targets))
        per_source_counts[src] = len(targets)
        for tn in targets:
            pos_pairs.add((src, tn))

    return pos_pairs, per_source_counts

def summarize(tp_pairs: set, total_pairs: int):
    tpr = (len(tp_pairs) / total_pairs) if total_pairs else 0.0
    return len(tp_pairs), total_pairs, tpr

print("Loading result CSVs...")
real1_pairs, _ = load_pairs_from_results(REAL_1HOP)
real2_pairs, _ = load_pairs_from_results(REAL_2HOP)
perm1_pairs, _ = load_pairs_from_results(PERM_1HOP)
perm2_pairs, _ = load_pairs_from_results(PERM_2HOP)

print("Building positive-pair denominator from DEG files...")
pos_pairs, per_source = build_positive_pairs_from_degs()
TOTAL = len(pos_pairs)
print("Total positive pairs (p<0.05, self removed):", TOTAL)

# Compute unions (<=2 hops)
real_u = real1_pairs | real2_pairs
perm_u = perm1_pairs | perm2_pairs

rows = []
rows.append({"dataset":"real", "hop":"1hop", "tp_pairs":len(real1_pairs), "total_pos_pairs":TOTAL, "tpr": len(real1_pairs)/TOTAL if TOTAL else 0.0})
rows.append({"dataset":"real", "hop":"2hop", "tp_pairs":len(real2_pairs), "total_pos_pairs":TOTAL, "tpr": len(real2_pairs)/TOTAL if TOTAL else 0.0})
rows.append({"dataset":"real", "hop":"<=2hop_union", "tp_pairs":len(real_u), "total_pos_pairs":TOTAL, "tpr": len(real_u)/TOTAL if TOTAL else 0.0})

rows.append({"dataset":"permuted_seed42", "hop":"1hop", "tp_pairs":len(perm1_pairs), "total_pos_pairs":TOTAL, "tpr": len(perm1_pairs)/TOTAL if TOTAL else 0.0})
rows.append({"dataset":"permuted_seed42", "hop":"2hop", "tp_pairs":len(perm2_pairs), "total_pos_pairs":TOTAL, "tpr": len(perm2_pairs)/TOTAL if TOTAL else 0.0})
rows.append({"dataset":"permuted_seed42", "hop":"<=2hop_union", "tp_pairs":len(perm_u), "total_pos_pairs":TOTAL, "tpr": len(perm_u)/TOTAL if TOTAL else 0.0})

out = pd.DataFrame(rows)
print("\nSUMMARY")
print(out.to_string(index=False))

out.to_csv(OUT_SUMMARY, index=False)
print("\nWrote:", OUT_SUMMARY)