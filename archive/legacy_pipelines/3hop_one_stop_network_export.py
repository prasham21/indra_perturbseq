#!/usr/bin/env python3
"""
3-hop ONE-STOP PIPELINE (NETWORK EXPORT BACKEND)

- Uses INDRA network export (.pkl) as backend graph
- Uses INDRA pathfinding function: shortest_simple_paths
- 3-hop paths only: source -> intermediate1 -> intermediate2 -> target
- Constraints:
  * intermediate1/intermediate2 must be HGNC
  * intermediate1/intermediate2 must be in endothelial whitelist
  * final hop (intermediate2 -> target) must include IncreaseAmount/DecreaseAmount
- Saves raw 3-hop rows first (before evidence/PMID/MeSH enrichment)
- Enriches PMIDs/evidence text using Neo4j Evidence nodes by stmt_hash (batched)
- Annotates MeSH using get_mesh_ids_for_pmids and reference list filter
- Splits outputs into non-self and self CSVs
"""

import argparse
import math
import os
import pickle
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

import networkx as nx
import numpy as np
import pandas as pd

from indra.databases import hgnc_client
from indra.explanation.pathfinding.pathfinding import shortest_simple_paths

from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client import get_mesh_ids_for_pmids


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
    """
    Return (text, pmid, source_api_like).
    ev_obj may be:
      - dict with key 'evidence' containing a nested dict
      - dict with keys directly on object
      - string
    """
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
) -> Dict[int, dict]:
    """
    Batched Neo4j fetch:
      UNWIND $hashes AS h
      MATCH (e:Evidence {stmt_hash: h})
      RETURN h, collect(e) AS ev_nodes
    """
    client = Neo4jClient()
    out = {}

    query = """
    UNWIND $hashes AS h
    MATCH (e:Evidence {stmt_hash: h})
    RETURN h AS stmt_hash, collect(e) AS ev_nodes
    """

    total = len(hashes)
    for i in range(0, total, batch_size):
        batch = hashes[i:i + batch_size]
        recs = client.query_tx(query, parameters={"hashes": [int(x) for x in batch]})

        # Initialize empty for all batch hashes so missing hashes are explicit
        for h in batch:
            out[int(h)] = {"evidence_text": "No evidence found (Neo4j)", "pmids": [], "sources": []}

        for r in recs:
            if isinstance(r, dict):
                stmt_hash = r.get("stmt_hash")
                ev_nodes = r.get("ev_nodes", [])
            else:
                stmt_hash = r[0]
                ev_nodes = r[1] if len(r) > 1 else []

            if stmt_hash is None:
                continue

            texts = []
            pmids = []
            sources = OrderedDict()

            for ev in ev_nodes:
                txt, pmid, source_key = parse_evidence_node(ev)
                if txt:
                    texts.append(txt)
                if pmid and pmid.isdigit():
                    pmids.append(pmid)
                if source_key:
                    sources[source_key] = None

            texts = list(dict.fromkeys(texts))
            pmids = list(dict.fromkeys(pmids))
            source_list = list(sources.keys())

            if texts:
                ev_text = "\n".join([f"{idx}. {t}" for idx, t in enumerate(texts, start=1)])
            elif source_list:
                ev_text = f"Evidence from: {', '.join(source_list)}"
            else:
                ev_text = "No evidence found (Neo4j)"

            out[int(stmt_hash)] = {
                "evidence_text": ev_text,
                "pmids": pmids,
                "sources": source_list,
            }

    return out


def build_mesh_id_to_name_map(client: Neo4jClient) -> Dict[str, str]:
    query = "MATCH (b:BioEntity) WHERE b.id STARTS WITH 'mesh:' RETURN b.id AS mesh_id, b.name AS mesh_name"
    recs = client.query_tx(query)
    m = {}
    for r in recs:
        if isinstance(r, dict):
            mid = r.get("mesh_id")
            name = r.get("mesh_name")
        else:
            mid, name = r[0], r[1]
        if mid and name:
            m[str(mid).replace("mesh:", "").upper()] = str(name)
    return m


def load_valid_mesh_ids(reference_csv: str) -> set:
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
        raise ValueError(f"Could not find MeSH ID column in {reference_csv}. Columns: {ref.columns.tolist()}")

    s = ref[mesh_id_col].astype(str)
    s = s.str.replace(r"\s+", "", regex=True).str.replace(r"[^A-Za-z0-9]", "", regex=True).str.upper()
    return set(s[s.str.startswith("D", na=False)].tolist())


def annotate_mesh(df: pd.DataFrame, mesh_reference_csv: str, mesh_batch_size: int = 200) -> pd.DataFrame:
    df = df.copy()
    for c in ["Annotated MeSH terms hop1", "Annotated MeSH terms hop2", "Annotated MeSH terms hop3"]:
        if c not in df.columns:
            df[c] = ""

    valid_mesh_ids = load_valid_mesh_ids(mesh_reference_csv)

    all_pmids = set()
    for col in ["pmids_hop1", "pmids_hop2", "pmids_hop3"]:
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
                kept.append(f"{name} ({mid})" if name else mid)
        return ", ".join(kept)

    df["Annotated MeSH terms hop1"] = df["pmids_hop1"].apply(mesh_terms_for_cell)
    df["Annotated MeSH terms hop2"] = df["pmids_hop2"].apply(mesh_terms_for_cell)
    df["Annotated MeSH terms hop3"] = df["pmids_hop3"].apply(mesh_terms_for_cell)

    return df


def run_3hop_for_gene_network(
    g: nx.DiGraph,
    gene: str,
    de_dir: str,
    p_threshold: float,
    prefer_fdr: bool,
    allowed_intermediates: set,
    limit_targets: int,
    max_paths_per_pair: int,
) -> Tuple[List[dict], str]:
    deg_path = os.path.join(de_dir, f"{gene}_vs_control.csv")
    if not os.path.exists(deg_path):
        return [], f"SKIP {gene}: missing DEG file"

    src = normalize_hgnc_symbol(gene)
    if not src or src not in g or not is_hgnc_node(g, src):
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

    target_set = {t for t in targets if (t in g and is_hgnc_node(g, t))}
    rows = []

    for tgt in target_set:
        # INDRA pathfinding function (as requested)
        try:
            path_gen = shortest_simple_paths(g, src, tgt, weight=None)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        found_for_pair = 0
        try:
            for path in path_gen:
                # We want exactly 3-hop => 4 nodes
                if len(path) < 4:
                    continue
                if len(path) > 4:
                    # shortest_simple_paths is non-decreasing by length
                    break

                _, mid1, mid2, _ = path

                if not is_hgnc_node(g, mid1) or not is_hgnc_node(g, mid2):
                    continue
                if mid1 not in allowed_intermediates or mid2 not in allowed_intermediates:
                    continue

                hop1_stmt = best_statement(g.get_edge_data(src, mid1), require_incdec=False)
                hop2_stmt = best_statement(g.get_edge_data(mid1, mid2), require_incdec=False)
                hop3_stmt = best_statement(g.get_edge_data(mid2, tgt), require_incdec=True)

                if not hop1_stmt or not hop2_stmt or not hop3_stmt:
                    continue

                h1 = hop1_stmt.get("stmt_hash")
                h2 = hop2_stmt.get("stmt_hash")
                h3 = hop3_stmt.get("stmt_hash")

                stats = deg_map.get(tgt, {})
                rows.append({
                    "source": src,
                    "intermediate_1": mid1,
                    "intermediate_2": mid2,
                    "target": tgt,
                    "stmt_type_1": hop1_stmt.get("stmt_type"),
                    "stmt_type_2": hop2_stmt.get("stmt_type"),
                    "stmt_type_3": hop3_stmt.get("stmt_type"),
                    "belief_1": hop1_stmt.get("belief"),
                    "belief_2": hop2_stmt.get("belief"),
                    "belief_3": hop3_stmt.get("belief"),
                    "evidence_1": hop1_stmt.get("evidence_count"),
                    "evidence_2": hop2_stmt.get("evidence_count"),
                    "evidence_3": hop3_stmt.get("evidence_count"),
                    "logfoldchange": stats.get("logfoldchange", None),
                    "pval": stats.get("pval", None),
                    "hop1_hash": h1,
                    "hop2_hash": h2,
                    "hop3_hash": h3,
                    "hop1_indra_url": indra_html_url(h1),
                    "hop2_indra_url": indra_html_url(h2),
                    "hop3_indra_url": indra_html_url(h3),
                })

                found_for_pair += 1
                if max_paths_per_pair > 0 and found_for_pair >= max_paths_per_pair:
                    break
        except nx.NetworkXNoPath:
            pass

    return rows, f"{gene}: produced {len(rows)} rows"


def enrich_evidence_pmids_from_neo4j(df: pd.DataFrame, neo4j_batch_size: int = 2000) -> pd.DataFrame:
    df = df.copy()
    for c in [
        "evidence_text_hop1", "pmids_hop1",
        "evidence_text_hop2", "pmids_hop2",
        "evidence_text_hop3", "pmids_hop3",
    ]:
        if c not in df.columns:
            df[c] = ""

    hashes = set()
    for c in ["hop1_hash", "hop2_hash", "hop3_hash"]:
        vals = pd.to_numeric(df[c], errors="coerce").dropna().astype(np.int64).tolist()
        hashes.update([int(v) for v in vals])

    hashes = sorted(hashes)
    print(f"Evidence fetch (Neo4j): unique stmt_hashes={len(hashes):,}")

    if not hashes:
        return df

    ev_map = fetch_evidence_map_from_neo4j_by_hashes(hashes, batch_size=neo4j_batch_size)

    def fill_hop(row, hop_num: int):
        h = row.get(f"hop{hop_num}_hash")
        if isinstance(h, (int, np.integer)):
            d = ev_map.get(int(h), None)
            if d is None:
                return "No evidence found (Neo4j)", ""
            return format_evidence_text(d["evidence_text"]), "; ".join(d["pmids"])
        return "No evidence found (Neo4j)", ""

    for idx in df.index:
        ev1, pm1 = fill_hop(df.loc[idx], 1)
        ev2, pm2 = fill_hop(df.loc[idx], 2)
        ev3, pm3 = fill_hop(df.loc[idx], 3)
        df.at[idx, "evidence_text_hop1"] = ev1
        df.at[idx, "pmids_hop1"] = pm1
        df.at[idx, "evidence_text_hop2"] = ev2
        df.at[idx, "pmids_hop2"] = pm2
        df.at[idx, "evidence_text_hop3"] = ev3
        df.at[idx, "pmids_hop3"] = pm3

    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    desired = [
        "source", "intermediate_1", "intermediate_2", "target",
        "stmt_type_1", "stmt_type_2", "stmt_type_3",
        "belief_1", "belief_2", "belief_3",
        "evidence_1", "evidence_2", "evidence_3",
        "logfoldchange", "pval",
        "evidence_text_hop1", "pmids_hop1", "Annotated MeSH terms hop1",
        "evidence_text_hop2", "pmids_hop2", "Annotated MeSH terms hop2",
        "evidence_text_hop3", "pmids_hop3", "Annotated MeSH terms hop3",
        # hashes and html links at the end
        "hop1_hash", "hop2_hash", "hop3_hash",
        "hop1_indra_url", "hop2_indra_url", "hop3_indra_url",
    ]
    keep = [c for c in desired if c in df.columns]
    extra = [c for c in df.columns if c not in keep]
    return df[keep + extra]


def main():
    ap = argparse.ArgumentParser(description="3-hop one-stop pipeline using INDRA network export + Neo4j evidence by stmt_hash.")
    ap.add_argument("--graph-pkl", required=True)
    ap.add_argument("--genes-csv", required=True, help="target_validation_expanded.csv")
    ap.add_argument("--de-dir", required=True)
    ap.add_argument("--endothelial-list", required=True, help="CSV with column 'gene'")
    ap.add_argument("--mesh-reference", required=True, help="MeSH reference CSV")

    ap.add_argument("--out-csv-raw", required=True, help="Raw 3-hop rows (before evidence/mesh enrichment)")
    ap.add_argument("--out-csv-main", required=True, help="Final enriched non-self rows")
    ap.add_argument("--out-csv-self", required=True, help="Final enriched self rows")

    ap.add_argument("--karen-flag-col", default="Karen_Flag")
    ap.add_argument("--karen-flag-value", default="Use_for_analysis")
    ap.add_argument("--gene-col", default="Gene")

    ap.add_argument("--p-threshold", type=float, default=0.05)
    ap.add_argument("--prefer-fdr", action="store_true")
    ap.add_argument("--genes", nargs="+", required=False, help="Optional explicit source genes")
    ap.add_argument("--limit-genes", type=int, default=0)
    ap.add_argument("--limit-targets", type=int, default=0)

    ap.add_argument("--max-paths-per-pair", type=int, default=1, help="1 = one 3-hop path per source-target pair; 0 = no limit")
    ap.add_argument("--path-workers", type=int, default=4)
    ap.add_argument("--neo4j-evidence-batch-size", type=int, default=2000)
    ap.add_argument("--mesh-batch-size", type=int, default=200)

    args = ap.parse_args()

    print("Loading network export...")
    g, load_secs = load_graph(args.graph_pkl)
    print(f"Loaded graph in {load_secs/60:.1f} min | nodes={g.number_of_nodes():,} edges={g.number_of_edges():,}")

    allowed_intermediates = load_endothelial_gene_set(args.endothelial_list)
    print(f"Loaded endothelial whitelist: {len(allowed_intermediates):,} genes")

    genes_df = pd.read_csv(args.genes_csv, low_memory=False)
    if args.genes:
        genes = [str(x).strip() for x in args.genes if str(x).strip()]
    else:
        if args.karen_flag_col in genes_df.columns:
            genes_df = genes_df[genes_df[args.karen_flag_col] == args.karen_flag_value].copy()
        genes = [str(x).strip() for x in genes_df[args.gene_col].dropna().tolist() if str(x).strip()]

    if args.limit_genes and args.limit_genes > 0:
        genes = genes[:args.limit_genes]

    print(f"Genes to process: {len(genes)}")

    all_rows = []

    def gene_job(gene):
        return run_3hop_for_gene_network(
            g=g,
            gene=gene,
            de_dir=args.de_dir,
            p_threshold=args.p_threshold,
            prefer_fdr=args.prefer_fdr,
            allowed_intermediates=allowed_intermediates,
            limit_targets=args.limit_targets,
            max_paths_per_pair=args.max_paths_per_pair,
        )

    print("Running 3-hop extraction (parallel over genes)...")
    with ThreadPoolExecutor(max_workers=args.path_workers) as ex:
        futs = {ex.submit(gene_job, gene): gene for gene in genes}
        for fut in as_completed(futs):
            rows, msg = fut.result()
            print(msg)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No rows produced. Exiting.")
        return

    print(f"Extraction complete: rows={len(df):,}")

    os.makedirs(os.path.dirname(args.out_csv_raw) or ".", exist_ok=True)
    df_raw = reorder_columns(df.copy())
    df_raw.to_csv(args.out_csv_raw, index=False)
    print(f"Saved raw 3-hop rows (pre-enrichment): {args.out_csv_raw}")

    print("Enriching evidence + PMIDs (Neo4j Evidence nodes by stmt_hash; batched)...")
    df = enrich_evidence_pmids_from_neo4j(df, neo4j_batch_size=args.neo4j_evidence_batch_size)

    print("Annotating MeSH terms (get_mesh_ids_for_pmids + reference filter)...")
    df = annotate_mesh(df, mesh_reference_csv=args.mesh_reference, mesh_batch_size=args.mesh_batch_size)

    df = reorder_columns(df)
    df_self = df[df["source"] == df["target"]].copy()
    df_main = df[df["source"] != df["target"]].copy()

    os.makedirs(os.path.dirname(args.out_csv_main) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv_self) or ".", exist_ok=True)
    df_main.to_csv(args.out_csv_main, index=False)
    df_self.to_csv(args.out_csv_self, index=False)

    print("\nDONE.")
    print(f"- non-self rows: {len(df_main):,} -> {args.out_csv_main}")
    print(f"- self rows:     {len(df_self):,} -> {args.out_csv_self}")


if __name__ == "__main__":
    main()