#!/usr/bin/env python3
"""Permuted-network pathfinding for TPR comparison (real vs permuted)."""
from __future__ import annotations

import argparse
import math
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from indra.databases import hgnc_client

import logging

logger = logging.getLogger(__name__)


INCDEC = {"IncreaseAmount", "DecreaseAmount"}


def _install_numpy_dtype_shims():
    # Some exported pickles reference numpy dtype aliases (e.g., 'f16') that may not exist locally.
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


def load_sources_from_target_validation(source_genes_csv: str, filter_column: str, filter_value: str, gene_column: str):
    df = pd.read_csv(source_genes_csv, low_memory=False)
    if filter_column in df.columns:
        df = df[df[filter_column] == filter_value].copy()
    genes = [str(x).strip() for x in df[gene_column].dropna().tolist() if str(x).strip()]
    return genes


def load_gene_set_from_csv(path: str, gene_column: str = "gene") -> set[str]:
    df = pd.read_csv(path, low_memory=False)
    if gene_column not in df.columns:
        raise ValueError(f"CSV must have column '{gene_column}'. Columns: {df.columns.tolist()}")
    genes = set(df[gene_column].astype(str).str.strip())
    genes.discard("")
    out = set()
    for g in genes:
        out.add(normalize_hgnc_symbol(g) or g)
    out.discard("")
    return out


def load_deg_targets_for_source(deg_dir: str, raw_source_gene: str, p_threshold: float, prefer_fdr: bool):
    """
    Returns:
      - targets_list: list of normalized HGNC target symbols passing threshold
      - deg_map: dict[target] = {"logfoldchange": ..., "pval": ...} (pval uses chosen sig_col)
    """
    deg_path = os.path.join(deg_dir, f"{raw_source_gene}_vs_control.csv")
    if not os.path.exists(deg_path):
        return None, None, f"missing DEG file: {deg_path}"

    df = pd.read_csv(deg_path, low_memory=False)
    if "names" not in df.columns:
        return None, None, "DEG missing 'names' column"

    sig_col = pick_sig_column(df, prefer_fdr=prefer_fdr)
    df[sig_col] = pd.to_numeric(df[sig_col], errors="coerce")
    df = df[df[sig_col] < p_threshold].copy()
    if df.empty:
        return [], {}, "no significant targets"

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

    return targets, deg_map, None


def iter_incdec_statements_on_edge(G, src, tgt):
    ed = G.get_edge_data(src, tgt) or {}
    stmts = ed.get("statements", [])
    if not isinstance(stmts, list):
        return
    for s in stmts:
        if isinstance(s, dict) and s.get("stmt_type") in INCDEC:
            yield s


def best_statement(edge_data, require_incdec: bool):
    """
    Pick one representative statement from edge_data['statements'].

    Ranking:
    - highest belief
    - tie-break by highest evidence_count
    """
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


class PermutationView:
"""Permuted-label network view via a bijection on HGNC nodes."""

    def __init__(self, G, hgnc_nodes: list[str], seed: int):
        rng = np.random.default_rng(seed)
        perm = np.array(hgnc_nodes, dtype=object)
        rng.shuffle(perm)

        perm_list = perm.tolist()
        self.phi = dict(zip(hgnc_nodes, perm_list))
        self.phi_inv = dict(zip(perm_list, hgnc_nodes))

        self.G = G

    def orig_for_label(self, label: str) -> str | None:
        return self.phi_inv.get(label)

    def label_for_orig(self, orig_node: str) -> str | None:
        return self.phi.get(orig_node)


def run_1hop_for_source_permuted(G, pv: PermutationView, src_label: str, targets: list[str], deg_map: dict):
    rows = []

    src0 = pv.orig_for_label(src_label)
    if not src0 or src0 not in G:
        return rows

    for tgt_label in targets:
        # DROP SELF PATHS (source == target)
        if tgt_label == src_label:
            continue

        tgt0 = pv.orig_for_label(tgt_label)
        if not tgt0 or tgt0 not in G:
            continue

        if not G.has_edge(src0, tgt0):
            continue

        for s in iter_incdec_statements_on_edge(G, src0, tgt0):
            stats = deg_map.get(tgt_label, {})
            rows.append({
                "source": src_label,
                "target": tgt_label,
                "stmt_type": s.get("stmt_type"),
                "belief": s.get("belief"),
                "evidence_count": s.get("evidence_count"),
                "logfoldchange": stats.get("logfoldchange", None),
                "pval": stats.get("pval", None),
            })

    return rows


def run_2hop_for_source_permuted(G, pv: PermutationView, src_label: str, targets_set: set[str], deg_map: dict,
                                allowed_intermediate_labels: set[str] | None):
    rows = []

    src0 = pv.orig_for_label(src_label)
    if not src0 or src0 not in G:
        return rows

    for mid0 in G.successors(src0):
        if not is_hgnc_node(G, mid0):
            continue

        mid_label = pv.label_for_orig(mid0)
        if not mid_label:
            continue

        # IMPORTANT: intermediate restriction applied in PERMUTED LABEL SPACE
        if allowed_intermediate_labels is not None and mid_label not in allowed_intermediate_labels:
            continue

        hop1_stmt = best_statement(G.get_edge_data(src0, mid0), require_incdec=False)
        if not hop1_stmt:
            continue

        for tgt0 in G.successors(mid0):
            if not is_hgnc_node(G, tgt0):
                continue

            tgt_label = pv.label_for_orig(tgt0)
            if not tgt_label or tgt_label not in targets_set:
                continue

            # DROP SELF PATHS (source == target)
            if tgt_label == src_label:
                continue

            hop2_stmt = best_statement(G.get_edge_data(mid0, tgt0), require_incdec=True)
            if not hop2_stmt:
                continue

            stats = deg_map.get(tgt_label, {})
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
                "logfoldchange": stats.get("logfoldchange", None),
                "pval": stats.get("pval", None),
            })

    return rows


def main():
    ap = argparse.ArgumentParser(description="Permuted-network 1-hop/2-hop path CSVs for TPR comparison (no self paths).")
    ap.add_argument("--graph-pkl", required=True)
    ap.add_argument("--source-genes-csv", required=True, help="target_validation_expanded.csv")
    ap.add_argument("--deg-dir", required=True)

    ap.add_argument("--p-threshold", type=float, default=0.05)
    ap.add_argument("--prefer-fdr", action="store_true")

    ap.add_argument("--filter-column", default="analysis_flag")
    ap.add_argument("--filter-value", default="Use_for_analysis")
    ap.add_argument("--gene-column", default="Gene")

    ap.add_argument("--seed", type=int, default=42, help="Permutation seed for HGNC label shuffling")
    ap.add_argument("--mode", choices=["1hop", "2hop", "both"], default="both")

    ap.add_argument("--allowed-intermediates-csv", default="", help="CSV with column 'gene' for allowed intermediates (optional)")
    ap.add_argument("--allowed-intermediates-gene-col", default="gene")

    ap.add_argument("--output-1hop", default="permuted_1hop_paths.csv")
    ap.add_argument("--output-2hop", default="permuted_2hop_paths.csv")

    ap.add_argument("--workers", type=int, default=4)

    args = ap.parse_args()

    logger.info("Loading INDRA export graph...")
    G, load_secs = load_graph(args.graph_pkl)
    logger.info(f"Loaded graph in {load_secs/60:.1f} min | nodes={G.number_of_nodes():,} edges={G.number_of_edges():,}")

    hgnc_nodes = [n for n in G.nodes if is_hgnc_node(G, n)]
    logger.info(f"HGNC nodes in graph: {len(hgnc_nodes):,}")

    logger.info("Creating permutation mapping (HGNC label shuffle; topology unchanged)...")
    pv = PermutationView(G=G, hgnc_nodes=hgnc_nodes, seed=args.seed)
    logger.info(f"Permutation seed: {args.seed}")

    allowed_intermediates = None
    if args.allowed_intermediates_csv:
        allowed_intermediates = load_gene_set_from_csv(
            args.allowed_intermediates_csv,
            gene_column=args.allowed_intermediates_gene_col,
        )
        logger.info(f"Allowed intermediates loaded: {len(allowed_intermediates):,}")

    sources_raw = load_sources_from_target_validation(
        source_genes_csv=args.source_genes_csv,
        filter_column=args.filter_column,
        filter_value=args.filter_value,
        gene_column=args.gene_column,
    )
    logger.info(f"Sources in genes CSV: {len(sources_raw)}")

    # Prepare per-source jobs: build fixed positives from DEG files in label space
    jobs = []
    skipped = 0
    for raw_src in sources_raw:
        src_label = normalize_hgnc_symbol(raw_src)
        if not src_label:
            skipped += 1
            continue

        # Require the source label exists in ORIGINAL graph label space (so it has a permuted image)
        if src_label not in G or not is_hgnc_node(G, src_label):
            skipped += 1
            continue

        targets, deg_map, err = load_deg_targets_for_source(
            deg_dir=args.deg_dir,
            raw_source_gene=raw_src,
            p_threshold=args.p_threshold,
            prefer_fdr=args.prefer_fdr,
        )
        if err:
            skipped += 1
            continue

        # Keep only targets that exist as HGNC nodes in the ORIGINAL graph label space
        targets = [t for t in targets if (t in G and is_hgnc_node(G, t))]
        if not targets:
            skipped += 1
            continue

        # Drop self from the target list up-front (extra safety)
        targets = [t for t in targets if t != src_label]
        if not targets:
            skipped += 1
            continue

        deg_map = {t: deg_map.get(t, {}) for t in targets}
        jobs.append((src_label, targets, deg_map))

    logger.info(f"Jobs prepared: {len(jobs)} sources | skipped: {skipped}")

    onehop_rows = []
    twohop_rows = []

    def job_fn(job):
        src_label, targets, deg_map = job
        out1 = []
        out2 = []
        if args.mode in ("1hop", "both"):
            out1 = run_1hop_for_source_permuted(G, pv, src_label, targets, deg_map)
        if args.mode in ("2hop", "both"):
            out2 = run_2hop_for_source_permuted(
                G, pv, src_label, set(targets), deg_map, allowed_intermediates
            )
        return out1, out2

    logger.info("Running permuted-network pathfinding...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(job_fn, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), start=1):
            out1, out2 = fut.result()
            if out1:
                onehop_rows.extend(out1)
            if out2:
                twohop_rows.extend(out2)
            if i % 25 == 0 or i == len(futs):
                logger.info(f"  progress {i}/{len(futs)} | 1hop_rows={len(onehop_rows):,} | 2hop_rows={len(twohop_rows):,}")

    if args.mode in ("1hop", "both"):
        df1 = pd.DataFrame(onehop_rows)
        df1 = df1[df1["source"] != df1["target"]].copy()  # final safety filter
        df1.to_csv(args.output_1hop, index=False)
        logger.info(f"Wrote {len(df1):,} permuted 1-hop rows -> {args.output_1hop}")

    if args.mode in ("2hop", "both"):
        df2 = pd.DataFrame(twohop_rows)
        df2 = df2[df2["source"] != df2["target"]].copy()  # final safety filter
        df2.to_csv(args.output_2hop, index=False)
        logger.info(f"Wrote {len(df2):,} permuted 2-hop rows -> {args.output_2hop}")

    logger.info("DONE.")


if __name__ == "__main__":
    main()
