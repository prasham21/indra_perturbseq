"""3-hop one-stop pipeline using INDRA network export + Neo4j evidence enrichment."""
from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from indra.databases import hgnc_client
from indra.explanation.pathfinding.pathfinding import shortest_simple_paths
from indra_cogex.client import get_mesh_ids_for_pmids
from indra_cogex.client.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

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


def is_hgnc_node(g: nx.DiGraph, node: str) -> bool:
    return (g.nodes.get(node, {}) or {}).get("ns") == "HGNC"


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
        hid = sorted(list(hid))[0] if hid else None
        if not hid:
            return s.upper()

    name = hgnc_client.get_hgnc_name(hid) if hid else None
    return name or s.upper()


def load_endothelial_gene_set(path: str) -> set:
    df = pd.read_csv(path, low_memory=False)
    if "gene" not in df.columns:
        raise ValueError(f"Endothelial list must contain 'gene' column. Columns: {df.columns.tolist()}")
    genes = set(df["gene"].astype(str).str.strip().tolist())
    genes.discard("")
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
    raise ValueError(f"No p-value/FDR column found. Columns: {df.columns.tolist()}")


def best_statement(edge_data: dict, require_incdec: bool = False) -> Optional[dict]:
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


def indra_html_url(stmt_hash: Optional[int]) -> str:
    if isinstance(stmt_hash, (int, np.integer)):
        return f"https://db.indra.bio/statements/from_hash/{int(stmt_hash)}?format=html"
    return ""


def format_evidence_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    if text.startswith("Evidence from:") or text.startswith("No evidence"):
        return text
    pattern = r"(^|\n|; )(\d+\.\s)"
    parts = []
    last_idx = 0
    for m in re.finditer(pattern, text):
        start = m.start(2)
        if start > last_idx:
            parts.append(text[last_idx:start].strip())
        last_idx = start
    parts.append(text[last_idx:].strip())
    return "\n\n".join([re.sub(r"^(\d+)\.\s", r"\1) ", p) for p in parts if p])


def parse_evidence_node(ev_obj) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (text, pmid, source_api_like)."""
    if ev_obj is None:
        return None, None, None

    if isinstance(ev_obj, str):
        return ev_obj.strip(), None, None

    if not isinstance(ev_obj, dict):
        return None, None, None

    if "evidence" in ev_obj and isinstance(ev_obj["evidence"], dict):
        d = ev_obj["evidence"]
    else:
        d = ev_obj

    txt = d.get("text") or ev_obj.get("text")
    pmid = d.get("pmid") or ev_obj.get("pmid")
    source_api = d.get("source_api") or ev_obj.get("source_api")
    annotations = d.get("annotations") if isinstance(d.get("annotations"), dict) else {}
    source_sub_id = annotations.get("source_sub_id") if isinstance(annotations, dict) else None
    source_key = source_api if source_api else None
    if source_key and source_sub_id:
        source_key = f"{source_key}:{source_sub_id}"

    if txt is not None:
        txt = str(txt).strip()
    if pmid is not None:
        pmid = str(pmid).strip()
    if source_key is not None:
        source_key = str(source_key).strip()

    return txt, pmid, source_key


def fetch_evidence_map_from_neo4j_by_hashes(
    hashes: List[int],
    batch_size: int = 2000,


def main():
    ) -> Dict[int, dict]:
        """Batched Neo4j evidence fetch by stmt_hash."""
        client = Neo4jClient()
        out = {}

        query = """
        UNWIND $hashes AS h
        MATCH (e:Evidence {stmt_hash: h})
        RETURN h AS stmt_hash, collect(e) AS ev_nodes



if __name__ == "__main__":
    main()
