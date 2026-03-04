#!/usr/bin/env python3
"""
Export FALSE-POSITIVE paths on a NEGATIVE target set, where the target universe
is PER-SOURCE = all genes present in that source's DEG CSV.

For source gene A:
- Universe U(A): all genes in <A>_vs_control.csv (names column)
- Positives P(A): subset of U(A) with p < threshold
- Negatives N(A): U(A) minus P(A) minus {A}

Then run:
- 1-hop FP paths: A -> B where B in N(A) and edge has Inc/Dec stmt
- 2-hop FP paths: A -> M -> B where B in N(A), optional intermediate whitelist

Outputs are PATH CSVs only.
"""

import argparse
import math
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from indra.databases import hgnc_client

INCDEC = {"IncreaseAmount", "DecreaseAmount"}


# -----------------------------
# Graph loading (dtype shim)
# -----------------------------
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


def is_hgnc_node(G, node) -> bool:
    return (G.nodes.get(node, {}) or {}).get("ns") == "HGNC"


# -----------------------------
# HGNC normalization + DEG helpers
# -----------------------------
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


def load_sources_from_target_validation(genes_csv: str, karen_flag_col: str, karen_flag_value: str, gene_col: str):
    df = pd.read_csv(genes_csv, low_memory=False)
    if karen_flag_col in df.columns:
        df = df[df[karen_flag_col] == karen_flag_value].copy()
    genes = [str(x).strip() for x in df[gene_col].dropna().tolist() if str(x).strip()]
    return genes


def load_gene_set_from_csv(path: str, gene_col: str = "gene") -> set[str]:
    df = pd.read_csv(path, low_memory=False)
    if gene_col not in df.columns:
        raise ValueError(f"CSV must have column '{gene_col}'. Columns: {df.columns.tolist()}")
    genes = set(df[gene_col].astype(str).str.strip())
    genes.discard("")
    out = set()
    for g in genes:
        out.add(normalize_hgnc_symbol(g) or g)
    out.discard("")
    return out


def load_deg_universe_and_pos_for_source(de_dir: str, raw_source_gene: str, p_threshold: float, prefer_fdr: bool):
    """
    Reads <raw_source_gene>_vs_control.csv once and returns:
      - all_targets: set of normalized HGNC symbols found in file (names)
      - pos_targets: subset where p < threshold
      - stats_map: dict[target] = {"logfoldchange": ..., "pval": ...} (best/lowest pval per target)
    """
    deg_path = os.path.join(de_dir, f"{raw_source_gene}_vs_control.csv")
    if not os.path.exists(deg_path):
        return None, None, None, f"missing DEG file: {deg_path}"

    df = pd.read_csv(deg_path, low_memory=False)
    if "names" not in df.columns:
        return None, None, None, "DEG missing 'names' column"

    sig_col = pick_sig_column(df, prefer_fdr=prefer_fdr)
    df[sig_col] = pd.to_numeric(df[sig_col], errors="coerce")

    if "logfoldchanges" in df.columns:
        df["logfoldchanges"] = pd.to_numeric(df["logfoldchanges"], errors="coerce")
    else:
        df["logfoldchanges"] = pd.NA

    stats_map = {}
    all_targets = set()
    pos_targets = set()

    for _, row in df.iterrows():
        t = normalize_hgnc_symbol(row["names"])
        if not t:
            continue

        p = row.get(sig_col, None)
        lfc = row.get("logfoldchanges", None)

        all_targets.add(t)

        if t not in stats_map or (p is not None and (stats_map[t]["pval"] is None or p < stats_map[t]["pval"])):
            stats_map[t] = {"logfoldchange": lfc, "pval": p}

        if p is not None and p < p_threshold:
            pos_targets.add(t)

    return all_targets, pos_targets, stats_map, None


# -----------------------------
# Statements on edges
# -----------------------------
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


# -----------------------------
# Optional permutation view
# -----------------------------
class PermutationView:
    def __init__(self, G, hgnc_nodes: list[str], seed: int):
        rng = np.random.default_rng(seed)
        perm = np.array(hgnc_nodes, dtype=object)
        rng.shuffle(perm)
        perm_list = perm.tolist()
        self.phi = dict(zip(hgnc_nodes, perm_list))  # original -> permuted
        self.phi_inv = dict(zip(perm_list, hgnc_nodes))  # permuted -> original
        self.G = G

    def orig_for_label(self, label: str) -> str | None:
        return self.phi_inv.get(label)

    def label_for_orig(self, orig_node: str) -> str | None:
        return self.phi.get(orig_node)


# -----------------------------
# 1-hop / 2-hop on negative targets
# -----------------------------
def run_1hop_for_source(G, src_label: str, neg_targets: list[str], stats_map: dict, pv: PermutationView | None):
    rows = []
    src0 = (src_label if pv is None else pv.orig_for_label(src_label))
    if not src0 or src0 not in G:
        return rows

    for tgt_label in neg_targets:
        if tgt_label == src_label:
            continue
        tgt0 = (tgt_label if pv is None else pv.orig_for_label(tgt_label))
        if not tgt0 or tgt0 not in G:
            continue
        if not G.has_edge(src0, tgt0):
            continue

        for s in iter_incdec_statements_on_edge(G, src0, tgt0):
            st = stats_map.get(tgt_label, {})
            rows.append({
                "source": src_label,
                "target": tgt_label,
                "stmt_type": s.get("stmt_type"),
                "belief": s.get("belief"),
                "evidence_count": s.get("evidence_count"),
                "logfoldchange": st.get("logfoldchange", None),
                "pval": st.get("pval", None),
            })
    return rows


def run_2hop_for_source(G, src_label: str, neg_targets_set: set[str],
                        allowed_intermediate_labels: set[str] | None,
                        stats_map: dict,
                        pv: PermutationView | None):
    rows = []
    src0 = (src_label if pv is None else pv.orig_for_label(src_label))
    if not src0 or src0 not in G:
        return rows

    for mid0 in G.successors(src0):
        if not is_hgnc_node(G, mid0):
            continue

        mid_label = (mid0 if pv is None else pv.label_for_orig(mid0))
        if not mid_label:
            continue

        # same intermediate restriction rule as your TP runs (applied in evaluation label space)
        if allowed_intermediate_labels is not None and mid_label not in allowed_intermediate_labels:
            continue

        hop1_stmt = best_statement(G.get_edge_data(src0, mid0), require_incdec=False)
        if not hop1_stmt:
            continue

        for tgt0 in G.successors(mid0):
            if not is_hgnc_node(G, tgt0):
                continue

            tgt_label = (tgt0 if pv is None else pv.label_for_orig(tgt0))
            if not tgt_label or tgt_label not in neg_targets_set:
                continue
            if tgt_label == src_label:
                continue

            hop2_stmt = best_statement(G.get_edge_data(mid0, tgt0), require_incdec=True)
            if not hop2_stmt:
                continue

            st = stats_map.get(tgt_label, {})
            rows.append({
                "source": src_label,
                "intermediate": mid_label,
                "target": tgt_label,
                "stmt_type_1": hop1_stmt.get("stmt_type"),
                "stmt_type_2": hop2_stmt.get("stmt_type"),
                "belief_1": hop1_stmt.get("belief"),
                "belief_2": hop2_stmt.get("belief"),
                "evidence_1": hop1_stmt.get("evidence_count"),
                "evidence_2": hop2_stmt.get("evidence_count"),
                "logfoldchange": st.get("logfoldchange", None),
                "pval": st.get("pval", None),
            })
    return rows


def run_3hop_for_source(G, src_label: str, neg_targets_set: set[str],
                        allowed_intermediate_labels: set[str] | None,
                        stats_map: dict,
                        pv: PermutationView | None):
    """
    Output columns:
      source, intermediate_1, intermediate_2, target,
      stmt_type_1, stmt_type_2, stmt_type_3,
      belief_1, belief_2, belief_3,
      evidence_1, evidence_2, evidence_3,
      logfoldchange, pval
    """
    rows = []

    src0 = (src_label if pv is None else pv.orig_for_label(src_label))
    if not src0 or src0 not in G:
        return rows

    for m10 in G.successors(src0):
        if not is_hgnc_node(G, m10):
            continue
        m1_label = (m10 if pv is None else pv.label_for_orig(m10))
        if not m1_label:
            continue
        if allowed_intermediate_labels is not None and m1_label not in allowed_intermediate_labels:
            continue

        hop1_stmt = best_statement(G.get_edge_data(src0, m10), require_incdec=False)
        if not hop1_stmt:
            continue

        for m20 in G.successors(m10):
            if not is_hgnc_node(G, m20):
                continue
            m2_label = (m20 if pv is None else pv.label_for_orig(m20))
            if not m2_label:
                continue
            if allowed_intermediate_labels is not None and m2_label not in allowed_intermediate_labels:
                continue

            hop2_stmt = best_statement(G.get_edge_data(m10, m20), require_incdec=False)
            if not hop2_stmt:
                continue

            for tgt0 in G.successors(m20):
                if not is_hgnc_node(G, tgt0):
                    continue

                tgt_label = (tgt0 if pv is None else pv.label_for_orig(tgt0))
                if not tgt_label or tgt_label not in neg_targets_set:
                    continue
                if tgt_label == src_label:
                    continue

                hop3_stmt = best_statement(G.get_edge_data(m20, tgt0), require_incdec=True)
                if not hop3_stmt:
                    continue

                st = stats_map.get(tgt_label, {})
                rows.append({
                    "source": src_label,
                    "intermediate_1": m1_label,
                    "intermediate_2": m2_label,
                    "target": tgt_label,
                    "stmt_type_1": hop1_stmt.get("stmt_type"),
                    "stmt_type_2": hop2_stmt.get("stmt_type"),
                    "stmt_type_3": hop3_stmt.get("stmt_type"),
                    "belief_1": hop1_stmt.get("belief"),
                    "belief_2": hop2_stmt.get("belief"),
                    "belief_3": hop3_stmt.get("belief"),
                    "evidence_1": hop1_stmt.get("evidence_count"),
                    "evidence_2": hop2_stmt.get("evidence_count"),
                    "evidence_3": hop3_stmt.get("evidence_count"),
                    "logfoldchange": st.get("logfoldchange", None),
                    "pval": st.get("pval", None),
                })

    return rows


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Export FP paths using per-source DEG universes (1-hop + 2-hop).")
    ap.add_argument("--graph-pkl", required=True)
    ap.add_argument("--genes-csv", required=True)
    ap.add_argument("--de-dir", required=True)

    ap.add_argument("--allowed-intermediates-csv", default="",
                    help="Optional CSV with gene column defining allowed intermediates")
    ap.add_argument("--allowed-intermediates-gene-col", default="gene")

    ap.add_argument("--p-threshold", type=float, default=0.05)
    ap.add_argument("--prefer-fdr", action="store_true")

    ap.add_argument("--karen-flag-col", default="Karen_Flag")
    ap.add_argument("--karen-flag-value", default="Use_for_analysis")
    ap.add_argument("--gene-col", default="Gene")

    ap.add_argument("--mode", choices=["1hop", "2hop", "3hop", "both", "all"], default="both")

    ap.add_argument("--limit-negatives-per-source", type=int, default=0,
                    help="If >0, sample this many negatives per source (debug/speed). 0 means use all negatives.")
    ap.add_argument("--sample-seed", type=int, default=1)

    ap.add_argument("--permute-seed", type=int, default=0,
                    help="If >0, evaluate on a label-permuted network with this seed; 0 means real network.")

    ap.add_argument("--out-1hop-csv", default="", help="Output CSV for 1-hop FP paths")
    ap.add_argument("--out-2hop-csv", default="", help="Output CSV for 2-hop FP paths")
    ap.add_argument("--out-3hop-csv", default="", help="Output CSV for 3-hop FP paths")

    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    def require(path: str, name: str):
        if not path:
            raise SystemExit(f"ERROR: {name} is required for this --mode")

    if args.mode == "1hop":
        require(args.out_1hop_csv, "--out-1hop-csv")
    elif args.mode == "2hop":
        require(args.out_2hop_csv, "--out-2hop-csv")
    elif args.mode == "3hop":
        require(args.out_3hop_csv, "--out-3hop-csv")
    elif args.mode == "both":
        require(args.out_1hop_csv, "--out-1hop-csv")
        require(args.out_2hop_csv, "--out-2hop-csv")
    elif args.mode == "all":
        require(args.out_1hop_csv, "--out-1hop-csv")
        require(args.out_2hop_csv, "--out-2hop-csv")
        require(args.out_3hop_csv, "--out-3hop-csv")

    print("Loading INDRA export graph...")
    G, load_secs = load_graph(args.graph_pkl)
    print(f"Loaded graph in {load_secs / 60:.1f} min | nodes={G.number_of_nodes():,} edges={G.number_of_edges():,}")

    hgnc_nodes = [n for n in G.nodes if is_hgnc_node(G, n)]
    print(f"HGNC nodes in graph: {len(hgnc_nodes):,}")

    allowed_intermediates = None
    if args.allowed_intermediates_csv:
        allowed_intermediates = load_gene_set_from_csv(args.allowed_intermediates_csv,
                                                       gene_col=args.allowed_intermediates_gene_col)
        allowed_intermediates = {g for g in allowed_intermediates if g in G and is_hgnc_node(G, g)}
        print(f"Allowed intermediates loaded (in-graph HGNC): {len(allowed_intermediates):,}")

    pv = None
    tag = "real"
    if args.permute_seed and args.permute_seed > 0:
        pv = PermutationView(G=G, hgnc_nodes=hgnc_nodes, seed=args.permute_seed)
        tag = f"permuted_seed{args.permute_seed}"
        print(f"Evaluating on PERMUTED network labels (seed={args.permute_seed})")

    sources_raw = load_sources_from_target_validation(
        genes_csv=args.genes_csv,
        karen_flag_col=args.karen_flag_col,
        karen_flag_value=args.karen_flag_value,
        gene_col=args.gene_col,
    )
    print(f"Sources in genes CSV: {len(sources_raw)}")

    rng = np.random.default_rng(args.sample_seed)

    jobs = []
    skipped = 0

    for raw_src in sources_raw:
        src = normalize_hgnc_symbol(raw_src)
        if not src:
            skipped += 1
            continue
        if src not in G or not is_hgnc_node(G, src):
            skipped += 1
            continue

        all_targets, pos_targets, stats_map, err = load_deg_universe_and_pos_for_source(
            args.de_dir, raw_src, args.p_threshold, args.prefer_fdr
        )
        if err:
            skipped += 1
            continue

        # restrict universe + positives to in-graph HGNC
        all_targets = {t for t in all_targets if t in G and is_hgnc_node(G, t)}
        pos_targets = {t for t in pos_targets if t in all_targets}

        # negatives = all - positives - {src}
        neg = sorted(list((all_targets - pos_targets) - {src}))
        if not neg:
            skipped += 1
            continue

        if args.limit_negatives_per_source and args.limit_negatives_per_source > 0 and len(
            neg) > args.limit_negatives_per_source:
            neg = rng.choice(np.array(neg, dtype=object), size=args.limit_negatives_per_source, replace=False).tolist()

        # keep only stats for the subset we might output (negatives)
        # (stats_map already has everything; that's fine)
        jobs.append((src, neg, stats_map))

    print(f"Jobs prepared: {len(jobs)} sources | skipped: {skipped}")
    print(
        f"Mode: {args.mode} | Negatives per source: {'ALL' if args.limit_negatives_per_source == 0 else args.limit_negatives_per_source}")

    onehop_rows = []
    twohop_rows = []
    threehop_rows = []

    def job_fn(job):
        src, neg, stats_map = job
        out1, out2, out3 = [], [], []

        if args.mode in ("1hop", "both", "all"):
            out1 = run_1hop_for_source(G, src, neg, stats_map, pv)
        if args.mode in ("2hop", "both", "all"):
            out2 = run_2hop_for_source(G, src, set(neg), allowed_intermediates, stats_map, pv)
        if args.mode in ("3hop", "all"):
            out3 = run_3hop_for_source(G, src, set(neg), allowed_intermediates, stats_map, pv)

        return out1, out2, out3

    print("Running false-positive pathfinding...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(job_fn, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), start=1):
            out1, out2 = fut.result()
            if out1:
                onehop_rows.extend(out1)
            if out2:
                twohop_rows.extend(out2)
            if i % 25 == 0 or i == len(futs):
                print(f"  progress {i}/{len(futs)} | 1hop_rows={len(onehop_rows):,} | 2hop_rows={len(twohop_rows):,}")

    if args.mode in ("1hop", "both", "all"):
        df1 = pd.DataFrame(onehop_rows)
        if not df1.empty:
            df1 = df1[df1["source"] != df1["target"]].copy()
        df1.to_csv(args.out_1hop_csv, index=False)
        print(f"Wrote 1-hop -> {args.out_1hop_csv} | rows={len(df1):,}")

    if args.mode in ("2hop", "both", "all"):
        df2 = pd.DataFrame(twohop_rows)
        if not df2.empty:
            df2 = df2[df2["source"] != df2["target"]].copy()
        df2.to_csv(args.out_2hop_csv, index=False)
        print(f"Wrote 2-hop -> {args.out_2hop_csv} | rows={len(df2):,}")

    if args.mode in ("3hop", "all"):
        df3 = pd.DataFrame(threehop_rows)
        if not df3.empty:
            df3 = df3[df3["source"] != df3["target"]].copy()
        df3.to_csv(args.out_3hop_csv, index=False)
        print(f"Wrote 3-hop -> {args.out_3hop_csv} | rows={len(df3):,}")

    print("DONE.")


if __name__ == "__main__":
    main()
