"""2-hop ONE-STOP PIPELINE (NETWORK EXPORT BACKEND)."""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import time
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
import pandas as pd
from indra.databases import hgnc_client

# MeSH helper (allowed to be CoGEx-dependent per user request)
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client import get_mesh_ids_for_pmids

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


def load_endothelial_gene_set(path: str) -> set[str]:
    df = pd.read_csv(path, low_memory=False)
    if "gene" not in df.columns:
        raise ValueError(f"Endothelial list must have a 'gene' column. Columns: {df.columns.tolist()}")
    genes = set(df["gene"].astype(str).str.strip())
    genes.discard("")
    # normalize to current HGNC symbols where possible
    out = set()
    for g in genes:
        out.add(normalize_hgnc_symbol(g) or g)
    out.discard("")
    return out


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


def format_evidence_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    if text.startswith("Evidence from:") or text.startswith("No evidence") or text.startswith("Error"):
        return text
    pattern = r"(^|\n|; )(\d+\.\s)"
    parts = []
    last_idx = 0
    for match in re.finditer(pattern, text):
        start = match.start(2)
        if start > last_idx:
            parts.append(text[last_idx:start].strip())
        last_idx = start
    parts.append(text[last_idx:].strip())
    return "\n\n".join([re.sub(r"^(\d+)\.\s", r"\1) ", p) for p in parts])


def indra_html_url(stmt_hash: int) -> str:
    return f"https://db.indra.bio/statements/from_hash/{stmt_hash}?format=html"


def best_statement(edge_data, require_incdec: bool):
    """
    Pick one representative statement from edge_data['statements'].
    Ranking: highest belief, tie-break by highest evidence_count.
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


_hash_cache_lock = Lock()
_hash_cache = {}  # stmt_hash -> dict(pmids: list[str], texts: list[str], sources: OrderedDict[str,None])


def fetch_from_hash_json(stmt_hash: int):
    url = f"https://db.indra.bio/statements/from_hash/{stmt_hash}?format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "indra-perturbseq"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def evidence_from_hash(stmt_hash: int, max_texts: int = 20):
    """
    Returns:
      - evidence_text (formatted later)
      - pmids (list[str])
      - sources_seen (OrderedDict for stable ordering)
    """
    with _hash_cache_lock:
        if stmt_hash in _hash_cache:
            c = _hash_cache[stmt_hash]
            return c["evidence_text"], c["pmids"], c["sources_seen"]

    pmids = []
    texts = []
    sources_seen = OrderedDict()

    try:
        data = fetch_from_hash_json(stmt_hash)
        ev_list = (data.get("results", {}) or {}).get(str(stmt_hash), {}).get("evidence", []) or []

        for ev in ev_list:
            if not isinstance(ev, dict):
                continue

            pmid = ev.get("pmid")
            if pmid:
                pmids.append(str(pmid))

            source_api = ev.get("source_api", "") or ""
            source_sub_id = (ev.get("annotations", {}) or {}).get("source_sub_id", "") or ""
            key = f"{source_api}:{source_sub_id}" if source_sub_id else source_api
            if key:
                sources_seen[key] = None

            txt = ev.get("text")
            if txt:
                texts.append(str(txt).strip())

    except Exception:
        pass

    pmids = list(dict.fromkeys(pmids))
    texts = list(dict.fromkeys(texts))[:max_texts]

    if texts:
        evidence_text = "\n".join([f"{i}. {t}" for i, t in enumerate(texts, start=1)])
    elif sources_seen:
        evidence_text = f"Evidence from: {', '.join(sources_seen.keys())}"
    else:
        evidence_text = "No evidence returned (db.indra.bio)"

    with _hash_cache_lock:
        _hash_cache[stmt_hash] = {
            "pmids": pmids,
            "evidence_text": evidence_text,
            "sources_seen": sources_seen,
        }

    return evidence_text, pmids, sources_seen


def build_mesh_id_to_name_map(client: Neo4jClient) -> dict[str, str]:
    """
    Maps MeSH IDs like 'D012345' to human-readable names using CoGEx Neo4j BioEntity nodes.
    """
    query = "MATCH (b:BioEntity) WHERE b.id STARTS WITH 'mesh:' RETURN b.id AS mesh_id, b.name AS mesh_name"
    results = client.query_tx(query)
    mapping = {}
    for rec in results:
        if isinstance(rec, dict):
            mid, name = rec.get("mesh_id"), rec.get("mesh_name")
        else:
            mid, name = rec[0], rec[1]
        if mid and name:
            mapping[str(mid).replace("mesh:", "").upper()] = str(name)
    return mapping


def load_valid_mesh_ids(reference_csv: str) -> set[str]:
    """
    Loads valid MeSH IDs from a reference CSV. Tries to find a column named 'mesh_id'
    (case-insensitive). If not found, uses the first column whose name contains both
    'mesh' and 'id'. Values are normalized to alphanumerics, uppercase.
    """
    ref = pd.read_csv(reference_csv, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
    cols_lower = {c.lower(): c for c in ref.columns}
    mesh_id_col = None
    if "mesh_id" in cols_lower:
        mesh_id_col = cols_lower["mesh_id"]
    else:
        for c in ref.columns:
            cl = c.lower()
            if "mesh" in cl and "id" in cl:
                mesh_id_col = c
                break
    if not mesh_id_col:
        raise ValueError(f"Could not find a MeSH ID column in {reference_csv}. Columns: {ref.columns.tolist()}")

    s = ref[mesh_id_col].astype(str)
    s = s.str.replace(r"\s+", "", regex=True).str.replace(r"[^A-Za-z0-9]", "", regex=True).str.upper()
    return set(s[s.str.startswith("D", na=False)].tolist())


def annotate_mesh(df: pd.DataFrame, mesh_reference_csv: str, mesh_batch_size: int = 200) -> pd.DataFrame:
    """
    Uses CoGEx helper get_mesh_ids_for_pmids to map PMID -> MeSH IDs,
    then filters to a reference set and formats as 'Name (Dxxxxxx)' where possible.
    """
    df = df.copy()
    for col in ["Annotated MeSH terms hop1", "Annotated MeSH terms hop2"]:
        if col not in df.columns:
            df[col] = ""

    valid_mesh_ids = load_valid_mesh_ids(mesh_reference_csv)

    all_pmids = set()
    for col in ["pmids_hop1", "pmids_hop2"]:
        for cell in df[col].fillna("").astype(str).tolist():
            for tok in cell.replace(" ", "").split(";"):
                if tok.isdigit():
                    all_pmids.add(tok)
    all_pmids = sorted(all_pmids, key=lambda x: int(x))

    if not all_pmids:
        return df

    client = Neo4jClient()
    mesh_id_to_name = build_mesh_id_to_name_map(client)

    pmid_to_mesh = {}
    for i in range(0, len(all_pmids), mesh_batch_size):
        batch = all_pmids[i:i + mesh_batch_size]
        pmid_to_mesh.update(get_mesh_ids_for_pmids(batch, client=client))

    def mesh_terms_for_cell(cell: str) -> str:
        pmids = [p for p in str(cell).replace(" ", "").split(";") if p.isdigit()]
        mesh_ids = set()
        for p in pmids:
            mesh_ids.update(pmid_to_mesh.get(p, []) or [])
        kept = []
        for mid in sorted({str(m).upper().strip() for m in mesh_ids}):
            if mid in valid_mesh_ids:
                name = mesh_id_to_name.get(mid)
                if name:
                    kept.append(f"{name} ({mid})")
                else:
                    kept.append(mid)
        return ", ".join(kept)

    df["Annotated MeSH terms hop1"] = df["pmids_hop1"].apply(mesh_terms_for_cell)
    df["Annotated MeSH terms hop2"] = df["pmids_hop2"].apply(mesh_terms_for_cell)
    return df


def run_2hop_for_gene_network(
    G,
    gene: str,
    deg_dir: str,
    p_threshold: float,
    prefer_fdr: bool,
    allowed_intermediates: set[str],
    limit_targets: int = 0,
):
    deg_path = os.path.join(deg_dir, f"{gene}_vs_control.csv")
    if not os.path.exists(deg_path):
        return [], f"SKIP {gene}: missing DEG file"

    src = normalize_hgnc_symbol(gene)
    if not src or src not in G or not is_hgnc_node(G, src):
        return [], f"SKIP {gene}: source not in graph as HGNC"

    df = pd.read_csv(deg_path, low_memory=False)
    if "names" not in df.columns:
        return [], f"SKIP {gene}: DEG missing 'names'"

    sig_col = pick_sig_column(df, prefer_fdr=prefer_fdr)
    df[sig_col] = pd.to_numeric(df[sig_col], errors="coerce")
    df = df[df[sig_col] < p_threshold].copy()
    if df.empty:
        return [], f"{gene}: no significant targets"

    if "logfoldchanges" in df.columns:
        df["logfoldchanges"] = pd.to_numeric(df["logfoldchanges"], errors="coerce")
    else:
        df["logfoldchanges"] = pd.NA

    df["names"] = df["names"].astype(str)

    targets = [normalize_hgnc_symbol(x) for x in df["names"].dropna().unique().tolist()]
    targets = [t for t in targets if t]
    if limit_targets and limit_targets > 0:
        targets = targets[:limit_targets]

    deg_map = {}
    for _, row in df.iterrows():
        t = normalize_hgnc_symbol(row["names"])
        if not t:
            continue
        p = row.get(sig_col, None)
        lfc = row.get("logfoldchanges", None)
        if t not in deg_map or (p is not None and p < deg_map[t]["pval"]):
            deg_map[t] = {"logfoldchange": lfc, "pval": p}

    target_set = {t for t in targets if (t in G and is_hgnc_node(G, t))}

    rows = []

    for mid in G.successors(src):
        if not is_hgnc_node(G, mid):
            continue
        if mid not in allowed_intermediates:
            continue

        hop1_stmt = best_statement(G.get_edge_data(src, mid), require_incdec=False)
        if not hop1_stmt:
            continue

        succs = set(G.successors(mid))
        cand_targets = succs.intersection(target_set)
        for tgt in cand_targets:
            hop2_stmt = best_statement(G.get_edge_data(mid, tgt), require_incdec=True)
            if not hop2_stmt:
                continue

            stats = deg_map.get(tgt, {})
            h1 = hop1_stmt.get("stmt_hash")
            h2 = hop2_stmt.get("stmt_hash")

            rows.append({
                "source": src,
                "intermediate": mid,
                "target": tgt,
                "stmt_type_1": hop1_stmt.get("stmt_type"),
                "stmt_type_2": hop2_stmt.get("stmt_type"),
                "belief_1": hop1_stmt.get("belief"),
                "belief_2": hop2_stmt.get("belief"),
                "evidence_1": hop1_stmt.get("evidence_count"),
                "evidence_2": hop2_stmt.get("evidence_count"),
                "logfoldchange": stats.get("logfoldchange", None),
                "pval": stats.get("pval", None),
                "hop1_hash": h1,
                "hop1_indra_url": indra_html_url(int(h1)) if isinstance(h1, int) else "",
                "hop2_hash": h2,
                "hop2_indra_url": indra_html_url(int(h2)) if isinstance(h2, int) else "",
            })

    return rows, f"{gene}: produced {len(rows)} rows"


def enrich_evidence_and_pmids(df: pd.DataFrame, max_workers: int = 8) -> pd.DataFrame:
    df = df.copy()
    for col in ["evidence_text_hop1", "pmids_hop1", "evidence_text_hop2", "pmids_hop2"]:
        if col not in df.columns:
            df[col] = ""

    def work(i: int):
        r = df.loc[i]
        h1 = r.get("hop1_hash")
        h2 = r.get("hop2_hash")

        ev1_txt, pm1, _ = ("No evidence returned (db.indra.bio)", [], OrderedDict())
        ev2_txt, pm2, _ = ("No evidence returned (db.indra.bio)", [], OrderedDict())

        if isinstance(h1, (int, np.integer)):
            ev1_txt, pm1, _ = evidence_from_hash(int(h1))
        if isinstance(h2, (int, np.integer)):
            ev2_txt, pm2, _ = evidence_from_hash(int(h2))

        return (
            i,
            format_evidence_text(ev1_txt),
            "; ".join(pm1),
            format_evidence_text(ev2_txt),
            "; ".join(pm2),
        )

    idxs = list(df.index)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(work, i) for i in idxs]
        for fut in as_completed(futs):
            i, ev1, pm1, ev2, pm2 = fut.result()
            df.at[i, "evidence_text_hop1"] = ev1
            df.at[i, "pmids_hop1"] = pm1
            df.at[i, "evidence_text_hop2"] = ev2
            df.at[i, "pmids_hop2"] = pm2

    return df


def main():
    ap = argparse.ArgumentParser(description="2-hop one-stop pipeline using INDRA network export (no CoGEx/Cypher for paths).")
    ap.add_argument("--graph-pkl", required=True)
    ap.add_argument("--source-genes-csv", required=True, help="target_validation_expanded.csv")
    ap.add_argument("--deg-dir", required=True)
    ap.add_argument("--endothelial-list", required=True, help="CSV with column 'gene' listing allowed intermediates")
    ap.add_argument("--mesh-reference", required=True, help="MeSH reference CSV (used to filter MeSH terms)")

    ap.add_argument("--output-main", required=True)
    ap.add_argument("--output-self-targets", required=True)

    ap.add_argument("--filter-column", default="analysis_flag")
    ap.add_argument("--filter-value", default="Use_for_analysis")
    ap.add_argument("--gene-column", default="Gene")

    ap.add_argument("--p-threshold", type=float, default=0.05)
    ap.add_argument("--prefer-fdr", action="store_true")
    ap.add_argument("--genes", nargs="+", required=False, help="Optional: list of source genes to run. If omitted, uses all analysis_flag genes.")
    ap.add_argument("--limit-genes", type=int, default=0)
    ap.add_argument("--limit-targets", type=int, default=0)

    ap.add_argument("--path-workers", type=int, default=4)
    ap.add_argument("--evidence-workers", type=int, default=8)
    ap.add_argument("--mesh-batch-size", type=int, default=200)

    args = ap.parse_args()

    logger.info("Loading network export...")
    G, load_secs = load_graph(args.graph_pkl)
    logger.info(f"Loaded graph in {load_secs/60:.1f} min | nodes={G.number_of_nodes():,} edges={G.number_of_edges():,}")

    allowed_intermediates = load_endothelial_gene_set(args.endothelial_list)
    logger.info(f"Loaded endothelial intermediate whitelist: {len(allowed_intermediates):,} genes")

    genes = []
    genes_df = pd.read_csv(args.source_genes_csv, low_memory=False)
    if args.genes:
        genes = [str(x).strip() for x in args.genes if str(x).strip()]
    else:
        if args.filter_column in genes_df.columns:
            genes_df = genes_df[genes_df[args.filter_column] == args.filter_value].copy()
        genes = [str(x).strip() for x in genes_df[args.gene_column].dropna().tolist() if str(x).strip()]

    if args.limit_genes and args.limit_genes > 0:
        genes = genes[:args.limit_genes]

    logger.info(f"Genes to process: {len(genes)}")

    all_rows = []

    def gene_job(g):
        rows, msg = run_2hop_for_gene_network(
            G=G,
            gene=g,
            deg_dir=args.deg_dir,
            p_threshold=args.p_threshold,
            prefer_fdr=args.prefer_fdr,
            allowed_intermediates=allowed_intermediates,
            limit_targets=args.limit_targets,
        )
        return g, rows, msg

    logger.info("Running 2-hop extraction (parallel over genes)...")
    with ThreadPoolExecutor(max_workers=args.path_workers) as ex:
        futs = [ex.submit(gene_job, g) for g in genes]
        for fut in as_completed(futs):
            g, rows, msg = fut.result()
            logger.info(msg)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        logger.info("No rows produced. Exiting.")
        return

    logger.info(f"Extraction complete: rows={len(df):,}")

    logger.info("Enriching evidence + PMIDs (db.indra.bio by stmt_hash; cached)...")
    df = enrich_evidence_and_pmids(df, max_workers=args.evidence_workers)

    logger.info("Annotating MeSH terms (get_mesh_ids_for_pmids; filtered to reference list)...")
    df = annotate_mesh(df, mesh_reference_csv=args.mesh_reference, mesh_batch_size=args.mesh_batch_size)

    df_self = df[df["source"] == df["target"]].copy()
    df_main = df[df["source"] != df["target"]].copy()

    os.makedirs(os.path.dirname(args.output_main) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_self_targets) or ".", exist_ok=True)

    df_main.to_csv(args.output_main, index=False)
    df_self.to_csv(args.output_self_targets, index=False)

    logger.info("DONE.")
    logger.info(f"- non-self rows: {len(df_main):,} -> {args.output_main}")
    logger.info(f"- self rows:     {len(df_self):,} -> {args.output_self_targets}")


if __name__ == "__main__":
    main()
