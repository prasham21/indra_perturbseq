import argparse
import os
import pickle
import time
import math

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


def load_endothelial_gene_set(path: str, gene_col: str = "gene") -> set[str]:
    df = pd.read_csv(path, low_memory=False)
    if gene_col not in df.columns:
        raise ValueError(f"Endothelial list must have column '{gene_col}'. Columns: {df.columns.tolist()}")
    genes = set(df[gene_col].astype(str).str.strip())
    genes.discard("")
    out = set()
    for g in genes:
        out.add(normalize_hgnc_symbol(g) or g)
    out.discard("")
    return out


def pick_sig_column(df: pd.DataFrame) -> str | None:
    # only used for reporting columns; no thresholding is applied
    for c in ["pvals", "pval", "p_value", "p_val", "pvals_adj", "padj", "qval", "fdr", "p_adj"]:
        if c in df.columns:
            return c
    return None


def load_targets_from_deg_as_universe(deg_csv: str) -> tuple[set[str], dict]:
    """
    Returns:
      - universe_targets: set of normalized HGNC symbols seen in DEG file 'names' column
      - stats_map: dict[target] = {"logfoldchange": ..., "pval": ...} (best/lowest pval if available)
    """
    df = pd.read_csv(deg_csv, low_memory=False)
    if "names" not in df.columns:
        raise ValueError(f"{deg_csv} missing 'names' column")

    pcol = pick_sig_column(df)
    if pcol:
        df[pcol] = pd.to_numeric(df[pcol], errors="coerce")

    if "logfoldchanges" in df.columns:
        df["logfoldchanges"] = pd.to_numeric(df["logfoldchanges"], errors="coerce")
    else:
        df["logfoldchanges"] = pd.NA

    universe = set()
    stats = {}

    for _, r in df.iterrows():
        t = normalize_hgnc_symbol(r["names"])
        if not t:
            continue
        universe.add(t)

        p = r[pcol] if pcol else None
        lfc = r.get("logfoldchanges", None)

        if t not in stats:
            stats[t] = {"logfoldchange": lfc, "pval": p}
        else:
            # keep lowest pval if available; otherwise keep first
            oldp = stats[t].get("pval", None)
            if pcol and pd.notna(p) and (oldp is None or (pd.notna(oldp) and p < oldp)):
                stats[t] = {"logfoldchange": lfc, "pval": p}

    return universe, stats


def iter_incdec_statements_on_edge(G, src, tgt):
    ed = G.get_edge_data(src, tgt) or {}
    stmts = ed.get("statements", [])
    if not isinstance(stmts, list):
        return
    for s in stmts:
        if isinstance(s, dict) and s.get("stmt_type") in INCDEC:
            yield s


def best_statement(edge_data, require_incdec: bool):
    stmts = (edge_data or {}).get("statements", [])
    if not isinstance(stmts, list) or not stmts:
        return None

    best = None
    best_key = None
    for s in stmts:
        if not isinstance(s, dict):
            continue
        st = s.get("stmt_type")
        if require_incdec and st not in INCDEC:
            continue

        b = s.get("belief", None)
        ev = s.get("evidence_count", None)
        try:
            bf = float(b) if b is not None else None
        except Exception:
            bf = None
        try:
            evf = int(ev) if ev is not None else 0
        except Exception:
            evf = 0

        if bf is None or not (math.isfinite(bf) and 0.0 <= bf <= 1.0):
            continue

        key = (bf, evf)
        if best_key is None or key > best_key:
            best_key = key
            best = s
    return best


def main():
    ap = argparse.ArgumentParser(
        description="Run outliers only (TP53/CDKN1A) for 1-hop + 2-hop on endothelial target universe (no pval thresholding)."
    )
    ap.add_argument("--graph-pkl", required=True, help="Latest INDRA network export pickle")
    ap.add_argument("--endothelial-list", required=True, help="CSV listing endothelial genes")
    ap.add_argument("--endothelial-gene-col", default="gene")

    ap.add_argument("--sources", nargs="+", required=True, help="e.g. TP53 CDKN1A")
    ap.add_argument("--deg-csvs", nargs="+", required=True, help="Paths to the corresponding <source>_vs_control.csv files")

    ap.add_argument("--out-1hop-csv", required=True)
    ap.add_argument("--out-2hop-csv", required=True)

    args = ap.parse_args()
    if len(args.sources) != len(args.deg_csvs):
        raise SystemExit("ERROR: --sources and --deg-csvs must have the same length")

    print("Loading graph...")
    G, secs = load_graph(args.graph_pkl)
    print(f"Loaded graph in {secs/60:.1f} min | nodes={G.number_of_nodes():,} edges={G.number_of_edges():,}")

    endothelial = load_endothelial_gene_set(args.endothelial_list, gene_col=args.endothelial_gene_col)
    endothelial = {g for g in endothelial if g in G and is_hgnc_node(G, g)}
    print(f"Endothelial genes in-graph (HGNC): {len(endothelial):,}")

    rows_1hop = []
    rows_2hop = []

    for raw_src, deg_csv in zip(args.sources, args.deg_csvs):
        src = normalize_hgnc_symbol(raw_src)
        if not src or src not in G or not is_hgnc_node(G, src):
            print(f"SKIP {raw_src}->{src}: source not in graph as HGNC node")
            continue

        universe, stats = load_targets_from_deg_as_universe(deg_csv)

        targets = (universe & endothelial) - {src}
        targets = {t for t in targets if t in G and is_hgnc_node(G, t)}
        print(f"{raw_src}->{src}: endothelial targets in DEG universe = {len(targets):,}")

        # -------- 1-hop --------
        for tgt in sorted(targets):
            if not G.has_edge(src, tgt):
                continue
            for s in iter_incdec_statements_on_edge(G, src, tgt):
                st = stats.get(tgt, {})
                rows_1hop.append(
                    {
                        "source": src,
                        "target": tgt,
                        "stmt_type": s.get("stmt_type"),
                        "belief": s.get("belief"),
                        "evidence_count": s.get("evidence_count"),
                        "stmt_hash": s.get("stmt_hash"),
                        "logfoldchange": st.get("logfoldchange", None),
                        "pval": st.get("pval", None),
                    }
                )

        # -------- 2-hop --------
        for mid in G.successors(src):
            if mid not in endothelial:
                continue
            hop1_stmt = best_statement(G.get_edge_data(src, mid), require_incdec=False)
            if not hop1_stmt:
                continue

            succs = set(G.successors(mid))
            for tgt in (succs & targets):
                hop2_stmt = best_statement(G.get_edge_data(mid, tgt), require_incdec=True)
                if not hop2_stmt:
                    continue
                st = stats.get(tgt, {})
                rows_2hop.append(
                    {
                        "source": src,
                        "intermediate": mid,
                        "target": tgt,
                        "stmt_type_1": hop1_stmt.get("stmt_type"),
                        "stmt_type_2": hop2_stmt.get("stmt_type"),
                        "belief_1": hop1_stmt.get("belief"),
                        "belief_2": hop2_stmt.get("belief"),
                        "evidence_1": hop1_stmt.get("evidence_count"),
                        "evidence_2": hop2_stmt.get("evidence_count"),
                        "hop1_hash": hop1_stmt.get("stmt_hash"),
                        "hop2_hash": hop2_stmt.get("stmt_hash"),
                        "logfoldchange": st.get("logfoldchange", None),
                        "pval": st.get("pval", None),
                    }
                )

    pd.DataFrame(rows_1hop).to_csv(args.out_1hop_csv, index=False)
    pd.DataFrame(rows_2hop).to_csv(args.out_2hop_csv, index=False)
    print("DONE.")
    print(f"1-hop rows: {len(rows_1hop):,} -> {args.out_1hop_csv}")
    print(f"2-hop rows: {len(rows_2hop):,} -> {args.out_2hop_csv}")


if __name__ == "__main__":
    main()