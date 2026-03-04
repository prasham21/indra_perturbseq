import argparse
import json
import os
import pickle
import time
import urllib.request

import numpy as np
import pandas as pd
from indra.databases import hgnc_client

INCDEC = {"IncreaseAmount", "DecreaseAmount"}


def _install_numpy_dtype_shims():
    if "f16" not in np.sctypeDict:
        replacement = np.longdouble if np.dtype(np.longdouble).itemsize == 16 else np.float64
        np.sctypeDict["f16"] = replacement
        if hasattr(np, "typeDict"):
            np.typeDict["f16"] = replacement


def load_graph(pkl_path: str):
    _install_numpy_dtype_shims()
    t0 = time.time()
    with open(pkl_path, "rb") as f:
        g = pickle.load(f)
    return g, time.time() - t0


def is_hgnc_node(G, node):
    return (G.nodes.get(node, {}) or {}).get("ns") == "HGNC"


def normalize_hgnc_symbol(symbol: str):
    """
    Normalize to current HGNC symbol when possible.
    If HGNC returns multiple IDs, pick one deterministically.
    """
    if symbol is None or (isinstance(symbol, float) and pd.isna(symbol)):
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


def pick_sig_column(df: pd.DataFrame, prefer_fdr: bool) -> str:
    fdr_candidates = ["pvals_adj", "padj", "qval", "fdr", "p_adj"]
    p_candidates = ["pvals", "pval", "p_value", "p_val"]
    fdr_col = next((c for c in fdr_candidates if c in df.columns), None)
    p_col = next((c for c in p_candidates if c in df.columns), None)
    if prefer_fdr and fdr_col:
        return fdr_col
    if p_col:
        return p_col
    if fdr_col:
        return fdr_col
    raise ValueError(f"DEG file missing p-value columns. Columns: {df.columns.tolist()}")


def iter_incdec_statements_on_edge(G, src, tgt):
    ed = G.get_edge_data(src, tgt) or {}
    stmts = ed.get("statements", [])
    if not isinstance(stmts, list):
        return
    for s in stmts:
        if not isinstance(s, dict):
            continue
        if s.get("stmt_type") in INCDEC:
            yield s


def fetch_from_hash_json(stmt_hash: int):
    url = f"https://db.indra.bio/statements/from_hash/{stmt_hash}?format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "indra-perturbseq"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def rich_stmt_text_from_hash(stmt_hash: int, cache: dict) -> str:
    """
    Fetch richer statement text for a statement hash from db.indra.bio and cache it.
    Uses the statement payload returned under the 'statements' field.
    """
    if stmt_hash in cache:
        return cache[stmt_hash]

    txt = ""
    try:
        data = fetch_from_hash_json(int(stmt_hash))
        stmt_payload = (data.get("statements", {}) or {}).get(str(stmt_hash))

        if isinstance(stmt_payload, str):
            try:
                stmt_payload = json.loads(stmt_payload)
            except Exception:
                stmt_payload = None

        if isinstance(stmt_payload, dict):
            txt = stmt_payload.get("english", "") or ""
    except Exception:
        txt = ""

    cache[stmt_hash] = txt
    return txt


def main():
    ap = argparse.ArgumentParser(description="1-hop extraction from INDRA network export (local graph).")
    ap.add_argument("--graph-pkl", required=True, help="Path to indranet_dir_graph_fix_corr_weights.pkl")
    ap.add_argument("--perturbations-csv", required=True, help="Path to target_validation_expanded.csv")
    ap.add_argument("--de-dir", required=True, help="Folder containing <GENE>_vs_control.csv files")

    ap.add_argument("--out-csv-main", required=True, help="Output CSV for non-self paths (source != target)")
    ap.add_argument("--out-csv-self", required=True, help="Output CSV for self paths (source == target)")

    ap.add_argument("--karen-flag-col", default="Karen_Flag")
    ap.add_argument("--karen-flag-value", default="Use_for_analysis")
    ap.add_argument("--gene-col", default="Gene")

    ap.add_argument("--p-threshold", type=float, default=0.05)
    ap.add_argument("--prefer-fdr", action="store_true")

    args = ap.parse_args()

    print("Loading network export...")
    G, load_secs = load_graph(args.graph_pkl)
    print(f"Loaded graph in {load_secs/60:.1f} min | nodes={G.number_of_nodes():,} edges={G.number_of_edges():,}")
    print()

    pert = pd.read_csv(args.perturbations_csv, low_memory=False)
    if args.karen_flag_col in pert.columns:
        pert = pert[pert[args.karen_flag_col] == args.karen_flag_value].copy()
    pert = pert.dropna(subset=[args.gene_col])

    genes = [str(x).strip() for x in pert[args.gene_col].tolist() if str(x).strip()]
    print(f"Perturbation genes to process: {len(genes)}")
    print()

    all_rows = []
    t0 = time.time()

    for i, raw_gene in enumerate(genes, start=1):
        gene = normalize_hgnc_symbol(raw_gene)
        if not gene:
            continue

        deg_path = os.path.join(args.de_dir, f"{raw_gene}_vs_control.csv")
        if not os.path.exists(deg_path):
            print(f"[{i}/{len(genes)}] SKIP {raw_gene}: missing DEG file {deg_path}")
            continue

        if gene not in G or not is_hgnc_node(G, gene):
            print(f"[{i}/{len(genes)}] SKIP {raw_gene}->{gene}: source not in graph as HGNC node")
            continue

        df = pd.read_csv(deg_path, low_memory=False)
        if "names" not in df.columns:
            print(f"[{i}/{len(genes)}] SKIP {raw_gene}: DEG missing 'names' column")
            continue

        sig_col = pick_sig_column(df, prefer_fdr=args.prefer_fdr)
        df[sig_col] = pd.to_numeric(df[sig_col], errors="coerce")
        df = df[df[sig_col] < args.p_threshold].copy()
        if df.empty:
            print(f"[{i}/{len(genes)}] {raw_gene}: no significant targets")
            continue

        if "logfoldchanges" in df.columns:
            df["logfoldchanges"] = pd.to_numeric(df["logfoldchanges"], errors="coerce")
        else:
            df["logfoldchanges"] = pd.NA

        df["names"] = df["names"].astype(str)

        targets = [normalize_hgnc_symbol(x) for x in df["names"].dropna().unique().tolist()]
        targets = [t for t in targets if t]

        deg_map = {}
        for _, row in df.iterrows():
            t = normalize_hgnc_symbol(row["names"])
            if not t:
                continue
            p = row.get(sig_col, None)
            lfc = row.get("logfoldchanges", None)
            if t not in deg_map or (p is not None and p < deg_map[t]["pval"]):
                deg_map[t] = {"logfoldchange": lfc, "pval": p}

        found_pairs = 0
        found_rows = 0

        for tgt in targets:
            if tgt not in G or not is_hgnc_node(G, tgt):
                continue
            if not G.has_edge(gene, tgt):
                continue

            found_pairs += 1

            for s in iter_incdec_statements_on_edge(G, gene, tgt):
                found_rows += 1
                all_rows.append({
                    "source": gene,
                    "target": tgt,
                    "stmt_type": s.get("stmt_type"),
                    "english_stmt": "",
                    "belief": s.get("belief"),
                    "evidence_count": s.get("evidence_count"),
                    "logfoldchange": deg_map.get(tgt, {}).get("logfoldchange", None),
                    "pval": deg_map.get(tgt, {}).get("pval", None),
                    "stmt_hash": s.get("stmt_hash"),
                    "source_counts": s.get("source_counts"),
                })

        elapsed = time.time() - t0
        print(f"[{i}/{len(genes)}] {raw_gene}->{gene}: targets={len(targets)} | pairs_with_edge={found_pairs} | rows={found_rows} | elapsed={elapsed/60:.1f}m")

    out_df = pd.DataFrame(all_rows)
    out_df = out_df.reindex(columns=[
        "source", "target", "stmt_type", "english_stmt",
        "belief", "evidence_count", "logfoldchange", "pval",
        "stmt_hash", "source_counts",
    ])

    cache = {}

    def _fill_english(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        s = str(x).strip()
        if not s:
            return ""
        try:
            return rich_stmt_text_from_hash(int(s), cache)
        except Exception:
            return ""

    out_df["english_stmt"] = out_df["stmt_hash"].apply(_fill_english)

    self_df = out_df[out_df["source"] == out_df["target"]].copy()
    main_df = out_df[out_df["source"] != out_df["target"]].copy()

    os.makedirs(os.path.dirname(args.out_csv_main) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv_self) or ".", exist_ok=True)

    main_df.to_csv(args.out_csv_main, index=False)
    self_df.to_csv(args.out_csv_self, index=False)

    print("\nDONE.")
    print(f"- non-self rows: {len(main_df):,} -> {args.out_csv_main}")
    print(f"- self rows:     {len(self_df):,} -> {args.out_csv_self}")


if __name__ == "__main__":
    main()