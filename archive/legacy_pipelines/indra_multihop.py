"""
INDRA multi-hop path discovery — production-grade CLI tool.

Discovers 1-hop, 2-hop, and/or 3-hop paths from any source gene(s) to
targets within a gene whitelist (e.g. endothelial universe), then annotates
each target with DEG statistics from a pre-computed bulk or single-cell
RNA-seq result file.

No p-value threshold is applied during path discovery — DEG stats are
annotations only. Waterfall exclusion ensures targets explained at a lower
hop are not repeated at higher hops (can be disabled with --no-waterfall).

Run examples
------------
# Minimal — all 3 hops, 6 validation genes
python indra_multihop.py \\
    --graph-pkl  /path/to/indranet_dir_graph_fix_corr_weights.pkl \\
    --endo-list  /path/to/endothelial_present_plus_manual.csv \\
    --deg-dir    /path/to/de_results \\
    --out-dir    /path/to/output \\
    --genes      CCM2 KLF2 MAP2K5

# Read genes from a CSV file instead
python indra_multihop.py \\
    --graph-pkl  /path/to/graph.pkl \\
    --endo-list  /path/to/endo.csv \\
    --deg-dir    /path/to/degs \\
    --out-dir    /path/to/output \\
    --genes-csv  /path/to/gene_list.csv --gene-col Gene

# Only run 1-hop and 2-hop, skip 3-hop
    --hops 1 2

# Turn off waterfall (allow targets to appear at multiple hops)
    --no-waterfall

# Combine all genes into one output file instead of one file per gene
    --combine-output

# Run 4 genes in parallel
    --workers 4

# Allow up to 3 paths per source-target pair in 3-hop
    --max-paths-3hop 3

# Use FDR-adjusted p-value column instead of raw p-value for DEG annotation
    --prefer-fdr
"""

import argparse
import logging
import math
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
from indra.databases import hgnc_client


# ── Constants ─────────────────────────────────────────────────────────────────

INCDEC = {"IncreaseAmount", "DecreaseAmount"}

COLUMN_ORDER = [
    "hop", "source", "intermediate_1", "intermediate_2", "target",
    "stmt_type_1",      "stmt_type_2",      "stmt_type_3",
    "belief_1",         "belief_2",         "belief_3",
    "evidence_count_1", "evidence_count_2", "evidence_count_3",
    "logfoldchange",    "pval",             "pvals_adj",
    "hop1_hash",        "hop2_hash",        "hop3_hash",
    "hop1_indra_url",   "hop2_indra_url",   "hop3_indra_url",
]

log = logging.getLogger("indra_multihop")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="indra_multihop.py",
        description="INDRA multi-hop path discovery from any source gene(s).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required paths ────────────────────────────────────────────────────────
    req = ap.add_argument_group("required paths")
    req.add_argument("--graph-pkl",  required=True, metavar="PATH",
                     help="INDRA network export pickle "
                          "(indranet_dir_graph_fix_corr_weights.pkl)")
    req.add_argument("--endo-list",  required=True, metavar="PATH",
                     help="CSV containing the gene whitelist "
                          "(intermediates + targets must be in this set)")
    req.add_argument("--deg-dir",    required=True, metavar="PATH",
                     help="Folder with <GENE>_vs_control.csv DEG files "
                          "(used for logfoldchange / pval annotation only)")
    req.add_argument("--out-dir",    required=True, metavar="PATH",
                     help="Output folder")

    # ── Gene input (one of the two must be supplied) ──────────────────────────
    gene_src = ap.add_argument_group(
        "source genes (supply exactly one of --genes or --genes-csv)"
    )
    gene_src.add_argument("--genes",     nargs="+", metavar="GENE",
                          help="One or more gene symbols, e.g. CCM2 KLF2 MAP2K5")
    gene_src.add_argument("--genes-csv", metavar="PATH",
                          help="CSV file containing source genes")
    gene_src.add_argument("--gene-col",  default="Gene", metavar="COL",
                          help="Column name in --genes-csv that holds gene symbols")
    gene_src.add_argument("--flag-col",  default=None, metavar="COL",
                          help="Optional column in --genes-csv to filter rows by")
    gene_src.add_argument("--flag-val",  default=None, metavar="VAL",
                          help="Value that --flag-col must equal to include row")

    # ── Whitelist options ─────────────────────────────────────────────────────
    wl = ap.add_argument_group("whitelist options")
    wl.add_argument("--endo-col", default="gene", metavar="COL",
                    help="Column name in --endo-list that holds gene symbols")

    # ── Hop options ───────────────────────────────────────────────────────────
    hop = ap.add_argument_group("hop options")
    hop.add_argument("--hops", nargs="+", type=int, choices=[1, 2, 3],
                     default=[1, 2, 3], metavar="{1,2,3}",
                     help="Which hops to run (e.g. --hops 1 2 skips 3-hop)")
    hop.add_argument("--no-waterfall", action="store_true",
                     help="Disable waterfall exclusion — targets may appear "
                          "at multiple hop levels")
    hop.add_argument("--max-paths-3hop", type=int, default=1, metavar="N",
                     help="Max paths per (source, target) pair in 3-hop. "
                          "Keeps highest last-hop belief path(s). 0 = no limit")

    # ── DEG annotation options ────────────────────────────────────────────────
    deg = ap.add_argument_group("DEG annotation")
    deg.add_argument("--prefer-fdr", action="store_true",
                     help="Prefer FDR-adjusted p-value column over raw p-value "
                          "when both exist in the DEG file")

    # ── Output options ────────────────────────────────────────────────────────
    out = ap.add_argument_group("output options")
    out.add_argument("--combine-output", action="store_true",
                     help="Write a single combined CSV for all genes instead of "
                          "one file per gene")
    out.add_argument("--out-filename", default="all_genes_all_hops.csv",
                     metavar="FILENAME",
                     help="Filename used when --combine-output is set")

    # ── Performance ───────────────────────────────────────────────────────────
    perf = ap.add_argument_group("performance")
    perf.add_argument("--workers", type=int, default=1, metavar="N",
                      help="Number of parallel workers for processing genes. "
                           "Set > 1 only when running many source genes")

    # ── Logging ───────────────────────────────────────────────────────────────
    ap.add_argument("--verbose", action="store_true",
                    help="Enable DEBUG-level logging")

    return ap.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate argument combinations and file existence early."""
    if not args.genes and not args.genes_csv:
        log.error("Supply either --genes or --genes-csv")
        sys.exit(1)
    if args.genes and args.genes_csv:
        log.error("--genes and --genes-csv are mutually exclusive")
        sys.exit(1)
    for label, path in [
        ("--graph-pkl",  args.graph_pkl),
        ("--endo-list",  args.endo_list),
        ("--deg-dir",    args.deg_dir),
    ]:
        if not os.path.exists(path):
            log.error(f"{label} path not found: {path}")
            sys.exit(1)
    if args.genes_csv and not os.path.exists(args.genes_csv):
        log.error(f"--genes-csv not found: {args.genes_csv}")
        sys.exit(1)


# ── Graph helpers ─────────────────────────────────────────────────────────────

def _install_numpy_dtype_shims() -> None:
    if not hasattr(np, "sctypeDict"):
        return
    if "f16" not in np.sctypeDict:
        replacement = (np.longdouble
                       if np.dtype(np.longdouble).itemsize == 16
                       else np.float64)
        np.sctypeDict["f16"] = replacement
        if hasattr(np, "typeDict"):
            np.typeDict["f16"] = replacement


def load_graph(pkl_path: str):
    _install_numpy_dtype_shims()
    t0 = time.time()
    with open(pkl_path, "rb") as fh:
        g = pickle.load(fh)
    elapsed = time.time() - t0
    log.info(f"Graph loaded in {elapsed/60:.1f} min | "
             f"nodes={g.number_of_nodes():,}  edges={g.number_of_edges():,}")
    return g


def is_hgnc_node(G, node: str) -> bool:
    return (G.nodes.get(node, {}) or {}).get("ns") == "HGNC"


def normalize_hgnc_symbol(symbol: str) -> Optional[str]:
    if symbol is None or (isinstance(symbol, float) and pd.isna(symbol)):
        return None
    s = str(symbol).strip()
    if not s:
        return None
    hid = hgnc_client.get_current_hgnc_id(s.upper())
    if not hid:
        return s.upper()
    if isinstance(hid, (list, tuple, set)):
        hid = sorted(hid)[0] if hid else None
        if not hid:
            return s.upper()
    name = hgnc_client.get_hgnc_name(hid) if hid else None
    return name or s.upper()


def load_gene_whitelist(path: str, col: str) -> set:
    """Load gene whitelist from CSV, normalize symbols, return set."""
    df = pd.read_csv(path, low_memory=False)
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found in {path}. "
            f"Available columns: {df.columns.tolist()}. "
            f"Set the correct column with --endo-col."
        )
    genes = set(df[col].astype(str).str.strip())
    genes.discard("")
    normalized = set()
    for g in genes:
        norm = normalize_hgnc_symbol(g)
        if norm:
            normalized.add(norm)
    log.info(f"Whitelist loaded: {len(normalized):,} normalized genes from {path}")
    return normalized


def load_source_genes(args: argparse.Namespace) -> list[tuple[str, str]]:
    """
    Returns list of (raw_symbol, normalized_symbol) tuples.
    Source is either --genes list or --genes-csv file.
    """
    if args.genes:
        raw_genes = [g.strip() for g in args.genes if g.strip()]
    else:
        df = pd.read_csv(args.genes_csv, low_memory=False)
        if args.gene_col not in df.columns:
            log.error(f"Column '{args.gene_col}' not in {args.genes_csv}. "
                      f"Available: {df.columns.tolist()}")
            sys.exit(1)
        if args.flag_col and args.flag_val:
            if args.flag_col in df.columns:
                df = df[df[args.flag_col] == args.flag_val]
            else:
                log.warning(f"--flag-col '{args.flag_col}' not found — ignoring filter")
        raw_genes = [str(x).strip() for x in df[args.gene_col].dropna()
                     if str(x).strip()]

    pairs = [(raw, normalize_hgnc_symbol(raw)) for raw in raw_genes]
    log.info(f"Source genes: {[p[0] for p in pairs]}")
    return pairs


# ── Statement helpers ─────────────────────────────────────────────────────────

def best_statement(edge_data: dict, require_incdec: bool) -> Optional[dict]:
    """Highest belief (tie-break: evidence_count) statement from an edge."""
    stmts = (edge_data or {}).get("statements", [])
    if not isinstance(stmts, list) or not stmts:
        return None
    best, best_key = None, None
    for s in stmts:
        if not isinstance(s, dict):
            continue
        if require_incdec and s.get("stmt_type") not in INCDEC:
            continue
        try:
            bf = float(s.get("belief"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(bf) and 0.0 <= bf <= 1.0):
            continue
        try:
            ev = int(s.get("evidence_count", 0))
        except (TypeError, ValueError):
            ev = 0
        key = (bf, ev)
        if best_key is None or key > best_key:
            best_key, best = key, s
    return best


def indra_url(stmt_hash) -> str:
    if isinstance(stmt_hash, (int, np.integer)):
        return (f"https://db.indra.bio/statements/from_hash/"
                f"{int(stmt_hash)}?format=html")
    return ""


# ── DEG annotation ────────────────────────────────────────────────────────────

def _pick_pval_col(df: pd.DataFrame, prefer_fdr: bool) -> str:
    fdr_cols = ["pvals_adj", "padj", "qval", "fdr", "p_adj"]
    raw_cols  = ["pvals", "pval", "p_value", "p_val"]
    fdr_col = next((c for c in fdr_cols if c in df.columns), None)
    raw_col  = next((c for c in raw_cols  if c in df.columns), None)
    if prefer_fdr and fdr_col:
        return fdr_col
    if raw_col:
        return raw_col
    if fdr_col:
        return fdr_col
    raise ValueError(f"No p-value column found. Columns: {df.columns.tolist()}")


def load_full_deg_map(deg_dir: str, raw_gene: str,
                      prefer_fdr: bool) -> dict:
    """
    Load the complete DEG file for a source gene — all genes, regardless of
    significance. Returns {target: {logfoldchange, pval, pvals_adj}}.
    pval here refers to whichever column is selected (raw or FDR).
    """
    path = os.path.join(deg_dir, f"{raw_gene}_vs_control.csv")
    if not os.path.exists(path):
        log.warning(f"DEG file not found: {path} — targets will have NaN stats")
        return {}

    df = pd.read_csv(path, low_memory=False)
    if "names" not in df.columns:
        log.warning(f"DEG file missing 'names' column: {path}")
        return {}

    try:
        sig_col = _pick_pval_col(df, prefer_fdr)
    except ValueError as e:
        log.warning(str(e))
        sig_col = None

    for col in ("logfoldchanges", "pvals", "pvals_adj", sig_col):
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    deg_map: dict = {}
    for _, row in df.iterrows():
        tgt = normalize_hgnc_symbol(str(row["names"]))
        if not tgt:
            continue
        p    = float(row.get(sig_col, float("nan"))) if sig_col else float("nan")
        padj = float(row.get("pvals_adj",      float("nan")))
        lfc  = float(row.get("logfoldchanges", float("nan")))
        if tgt not in deg_map or (not math.isnan(p) and p < deg_map[tgt]["pval"]):
            deg_map[tgt] = {"logfoldchange": lfc, "pval": p, "pvals_adj": padj}

    log.debug(f"  DEG map for {raw_gene}: {len(deg_map):,} entries")
    return deg_map


# ── Row factory ───────────────────────────────────────────────────────────────

def _deg_annotation(deg_map: dict, target: str) -> dict:
    d = deg_map.get(target, {})
    return {
        "logfoldchange": d.get("logfoldchange", float("nan")),
        "pval":          d.get("pval",          float("nan")),
        "pvals_adj":     d.get("pvals_adj",     float("nan")),
    }


def make_row(hop: int, src: str, tgt: str, deg_map: dict,
             s1=None, s2=None, s3=None,
             mid1: str = "", mid2: str = "") -> dict:
    row = {
        "hop":              hop,
        "source":           src,
        "intermediate_1":   mid1,
        "intermediate_2":   mid2,
        "target":           tgt,
        "stmt_type_1":      s1.get("stmt_type")      if s1 else pd.NA,
        "stmt_type_2":      s2.get("stmt_type")      if s2 else pd.NA,
        "stmt_type_3":      s3.get("stmt_type")      if s3 else pd.NA,
        "belief_1":         s1.get("belief")         if s1 else pd.NA,
        "belief_2":         s2.get("belief")         if s2 else pd.NA,
        "belief_3":         s3.get("belief")         if s3 else pd.NA,
        "evidence_count_1": s1.get("evidence_count") if s1 else pd.NA,
        "evidence_count_2": s2.get("evidence_count") if s2 else pd.NA,
        "evidence_count_3": s3.get("evidence_count") if s3 else pd.NA,
        "hop1_hash":        s1.get("stmt_hash")      if s1 else pd.NA,
        "hop2_hash":        s2.get("stmt_hash")      if s2 else pd.NA,
        "hop3_hash":        s3.get("stmt_hash")      if s3 else pd.NA,
        "hop1_indra_url":   indra_url(s1.get("stmt_hash")) if s1 else "",
        "hop2_indra_url":   indra_url(s2.get("stmt_hash")) if s2 else "",
        "hop3_indra_url":   indra_url(s3.get("stmt_hash")) if s3 else "",
    }
    row.update(_deg_annotation(deg_map, tgt))
    return row


# ── Path extraction ───────────────────────────────────────────────────────────

def run_1hop(G, src: str, whitelist: set,
             deg_map: dict) -> tuple[list[dict], set]:
    rows, found = [], set()
    for tgt in G.successors(src):
        if not is_hgnc_node(G, tgt) or tgt not in whitelist or tgt == src:
            continue
        s1 = best_statement(G.get_edge_data(src, tgt), require_incdec=True)
        if not s1:
            continue
        rows.append(make_row(1, src, tgt, deg_map, s1=s1))
        found.add(tgt)
    return rows, found


def run_2hop(G, src: str, whitelist: set, deg_map: dict,
             excluded: set) -> tuple[list[dict], set]:
    rows, found = [], set()
    for mid in G.successors(src):
        if not is_hgnc_node(G, mid) or mid not in whitelist or mid == src:
            continue
        s1 = best_statement(G.get_edge_data(src, mid), require_incdec=False)
        if not s1:
            continue
        for tgt in G.successors(mid):
            if not is_hgnc_node(G, tgt) or tgt not in whitelist:
                continue
            if tgt in (src, mid) or tgt in excluded:
                continue
            s2 = best_statement(G.get_edge_data(mid, tgt), require_incdec=True)
            if not s2:
                continue
            rows.append(make_row(2, src, tgt, deg_map,
                                 s1=s1, s2=s2, mid1=mid))
            found.add(tgt)
    return rows, found


def run_3hop(G, src: str, whitelist: set, deg_map: dict,
             excluded: set, max_paths_per_pair: int) -> list[dict]:
    """
    Enumerate all qualifying 3-hop paths.
    Per (source, target) pair, keep up to max_paths_per_pair paths ordered by
    descending last-hop belief. If max_paths_per_pair == 0, keep all.
    """
    # target -> list of (belief_3, row), kept sorted descending
    candidates: dict[str, list[tuple[float, dict]]] = {}

    for mid1 in G.successors(src):
        if not is_hgnc_node(G, mid1) or mid1 not in whitelist or mid1 == src:
            continue
        s1 = best_statement(G.get_edge_data(src, mid1), require_incdec=False)
        if not s1:
            continue

        for mid2 in G.successors(mid1):
            if not is_hgnc_node(G, mid2) or mid2 not in whitelist:
                continue
            if mid2 in (src, mid1):
                continue
            s2 = best_statement(G.get_edge_data(mid1, mid2), require_incdec=False)
            if not s2:
                continue

            for tgt in G.successors(mid2):
                if not is_hgnc_node(G, tgt) or tgt not in whitelist:
                    continue
                if tgt in (src, mid1, mid2) or tgt in excluded:
                    continue
                s3 = best_statement(G.get_edge_data(mid2, tgt),
                                    require_incdec=True)
                if not s3:
                    continue

                belief3 = float(s3.get("belief", 0.0))
                row = make_row(3, src, tgt, deg_map,
                               s1=s1, s2=s2, s3=s3, mid1=mid1, mid2=mid2)

                bucket = candidates.setdefault(tgt, [])
                bucket.append((belief3, row))

    # Flatten — keep top-N per target
    rows = []
    for tgt, items in candidates.items():
        items.sort(key=lambda x: x[0], reverse=True)
        keep = items if max_paths_per_pair == 0 else items[:max_paths_per_pair]
        rows.extend(r for _, r in keep)
    return rows


# ── Per-gene orchestration ────────────────────────────────────────────────────

def process_gene(
    raw_gene: str,
    gene: str,
    G,
    whitelist: set,
    args: argparse.Namespace,
) -> Optional[pd.DataFrame]:
    """
    Run requested hops for one source gene.
    Returns a DataFrame of all paths (all hops combined), or None on failure.
    Thread-safe: reads G and whitelist but never modifies them.
    """
    t_gene = time.time()
    log.info(f"[{raw_gene}] Starting  (normalized: {gene})")

    if not gene or gene not in G or not is_hgnc_node(G, gene):
        log.warning(f"[{raw_gene}] SKIP — not in graph as HGNC node")
        return None

    deg_map = load_full_deg_map(args.deg_dir, raw_gene, args.prefer_fdr)

    waterfall = not args.no_waterfall
    excluded: set = set()
    all_rows: list[dict] = []

    # ── 1-hop ─────────────────────────────────────────────────────────────────
    if 1 in args.hops:
        t0 = time.time()
        rows_1, found_1 = run_1hop(G, gene, whitelist, deg_map)
        log.info(f"[{raw_gene}] 1-hop: {len(rows_1):,} paths | "
                 f"{len(found_1):,} unique targets | {time.time()-t0:.1f}s")
        all_rows.extend(rows_1)
        if waterfall:
            excluded |= found_1
    else:
        log.debug(f"[{raw_gene}] 1-hop skipped")

    # ── 2-hop ─────────────────────────────────────────────────────────────────
    if 2 in args.hops:
        t0 = time.time()
        rows_2, found_2 = run_2hop(G, gene, whitelist, deg_map,
                                   excluded=excluded)
        log.info(f"[{raw_gene}] 2-hop: {len(rows_2):,} paths | "
                 f"{len(found_2):,} new targets | {time.time()-t0:.1f}s")
        all_rows.extend(rows_2)
        if waterfall:
            excluded |= found_2
    else:
        log.debug(f"[{raw_gene}] 2-hop skipped")

    # ── 3-hop ─────────────────────────────────────────────────────────────────
    if 3 in args.hops:
        t0 = time.time()
        rows_3 = run_3hop(G, gene, whitelist, deg_map,
                          excluded=excluded,
                          max_paths_per_pair=args.max_paths_3hop)
        found_3 = {r["target"] for r in rows_3}
        log.info(f"[{raw_gene}] 3-hop: {len(rows_3):,} paths | "
                 f"{len(found_3):,} new targets | {time.time()-t0:.1f}s")
        all_rows.extend(rows_3)
    else:
        log.debug(f"[{raw_gene}] 3-hop skipped")

    if not all_rows:
        log.warning(f"[{raw_gene}] No paths found")
        return None

    df = pd.DataFrame(all_rows)
    for col in COLUMN_ORDER:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[COLUMN_ORDER]

    log.info(f"[{raw_gene}] Done — {len(df):,} total paths "
             f"| {time.time()-t_gene:.1f}s elapsed")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    validate_args(args)
    os.makedirs(args.out_dir, exist_ok=True)

    log.info("Loading INDRA graph...")
    G = load_graph(args.graph_pkl)

    log.info("Loading gene whitelist...")
    whitelist_all = load_gene_whitelist(args.endo_list, args.endo_col)
    whitelist = {g for g in whitelist_all if g in G and is_hgnc_node(G, g)}
    log.info(f"Whitelist: {len(whitelist_all):,} genes total | "
             f"{len(whitelist):,} present in graph")

    gene_pairs = load_source_genes(args)
    log.info(f"Hops to run   : {sorted(args.hops)}")
    log.info(f"Waterfall     : {not args.no_waterfall}")
    log.info(f"3-hop max paths per pair: {args.max_paths_3hop} "
             f"(0 = unlimited)")
    log.info(f"Workers       : {args.workers}")

    # ── Process genes (sequentially or in parallel) ───────────────────────────
    results: dict[str, pd.DataFrame] = {}  # raw_gene -> DataFrame

    def _job(raw_gene: str, gene: Optional[str]):
        df = process_gene(raw_gene, gene, G, whitelist, args)
        return raw_gene, df

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_job, raw, norm): raw
                       for raw, norm in gene_pairs}
            for fut in as_completed(futures):
                raw_gene, df = fut.result()
                if df is not None:
                    results[raw_gene] = df
    else:
        for raw_gene, gene in gene_pairs:
            _, df = _job(raw_gene, gene)
            if df is not None:
                results[raw_gene] = df

    if not results:
        log.error("No results produced for any gene. Exiting.")
        sys.exit(1)

    # ── Write output ──────────────────────────────────────────────────────────
    if args.combine_output:
        combined = pd.concat(list(results.values()), ignore_index=True)
        out_path = os.path.join(args.out_dir, args.out_filename)
        combined.to_csv(out_path, index=False)
        log.info(f"Combined output: {len(combined):,} rows → {out_path}")
    else:
        for raw_gene, df in results.items():
            out_path = os.path.join(args.out_dir, f"{raw_gene}_all_hops.csv")
            df.to_csv(out_path, index=False)
            log.info(f"Saved {raw_gene}: {len(df):,} rows → {out_path}")

    log.info("DONE.")


if __name__ == "__main__":
    main()