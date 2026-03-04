"""Superseded legacy script for 1-hop gene characterization scoring.

Refactored into src/indra_perturbseq/pipelines/.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

from indra_cogex.client.neo4j_client import Neo4jClient
from indra.databases import hgnc_client

logger = logging.getLogger(__name__)

EDGE_EVIDENCE_BATCH_QUERY = """
UNWIND $edges AS e
MATCH (a:BioEntity)-[r:indra_rel {stmt_type: e.stmt_type}]->(b:BioEntity)
WHERE (a.id = e.source_id AND b.id = e.target_id)
   OR (a.id = e.target_id AND b.id = e.source_id)
WITH e, r.stmt_hash AS stmt_hash
MATCH (ev:Evidence {stmt_hash: stmt_hash})
RETURN e.source_id AS source_id, e.target_id AS target_id, e.stmt_type AS stmt_type, ev.evidence AS evidence
"""


def is_nonempty(x) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and pd.isna(x):
        return False
    s = str(x).strip()
    return s != "" and s.lower() not in {"nan", "none"}


def clean_gene_symbol(x: str) -> str:
    """Conservative cleaning for common CSV artifacts."""
    s = str(x)
    s = s.replace("\xa0", " ").strip()
    s = s.replace(" ", "")
    if s.endswith(".00"):
        s = s[:-3]
    if s.endswith(".0"):
        s = s[:-2]
    return s


def parse_evidence_json(evidence_str: str):
    try:
        ev = json.loads(evidence_str)
    except Exception:
        return None, None, None

    pmid = ev.get("pmid")
    source_api = ev.get("source_api")
    annotations = ev.get("annotations") or {}
    source_sub_id = annotations.get("source_sub_id")

    pmid = str(pmid).strip() if pmid is not None else None
    source_api = str(source_api).strip() if source_api is not None else None
    source_sub_id = str(source_sub_id).strip() if source_sub_id is not None else None

    return pmid, source_api, source_sub_id


def evidence_source_identity(
    pmid: str | None, source_api: str | None, source_sub_id: str | None,
) -> str | None:
    if pmid and pmid.isdigit():
        return f"PMID:{pmid}"

    db_id = source_sub_id or source_api
    if db_id and db_id.strip():
        return f"DB:{db_id.strip()}"

    return None


def batched(iterable, batch_size: int):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    ap = argparse.ArgumentParser(description="1-hop gene characterization scoring")
    ap.add_argument("--input-csv", required=True, help="1-hop input CSV")
    ap.add_argument("--out-gene-csv", required=True, help="Gene characterization output CSV")
    ap.add_argument("--out-path-csv", required=True, help="1-hop CSV with characterization columns")
    ap.add_argument("--out-hist", required=True, help="Characterization histogram PNG")

    ap.add_argument("--source-col", default="source")
    ap.add_argument("--target-col", default="target")
    ap.add_argument("--stmt-col", default="stmt_type")

    ap.add_argument("--batch-size", type=int, default=200, help="Batch size for UNWIND edge queries")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv, low_memory=False)

    required = [args.source_col, args.target_col, args.stmt_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Present columns: {df.columns.tolist()}\n\n"
            f"Tip: pass --source-col/--target-col/--stmt-col if names differ."
        )

    edges_sym = set()
    for _, r in df.iterrows():
        a = clean_gene_symbol(r[args.source_col])
        b = clean_gene_symbol(r[args.target_col])
        st = str(r[args.stmt_col]).strip()
        if a and b and st:
            edges_sym.add((a, b, st))

    logger.info("Unique 1-hop edges: %d", len(edges_sym))

    sym_to_hgnc = {}

    def sym_to_hgnc_id(sym: str) -> str | None:
        sym = sym.strip()
        if sym in sym_to_hgnc:
            return sym_to_hgnc[sym]
        hid = hgnc_client.get_current_hgnc_id(sym)
        sym_to_hgnc[sym] = hid
        return hid

    edges = []
    keymap = {}

    for a_sym, b_sym, st in edges_sym:
        a_h = sym_to_hgnc_id(a_sym)
        b_h = sym_to_hgnc_id(b_sym)
        if not a_h or not b_h:
            continue
        a_id = f"hgnc:{a_h}"
        b_id = f"hgnc:{b_h}"
        key = (a_id, b_id, st)
        keymap[(a_sym, b_sym, st)] = key
        edges.append({"source_id": a_id, "target_id": b_id, "stmt_type": st})

    seen = set()
    dedup_edges = []
    for e in edges:
        k = (e["source_id"], e["target_id"], e["stmt_type"])
        if k in seen:
            continue
        seen.add(k)
        dedup_edges.append(e)
    edges = dedup_edges

    logger.info("Neo4j-resolvable 1-hop edges: %d", len(edges))

    client = Neo4jClient()

    edge_sources = defaultdict(set)
    edge_evidence_mentions = defaultdict(int)

    for batch in batched(edges, args.batch_size):
        rows = client.query_tx(EDGE_EVIDENCE_BATCH_QUERY, edges=batch)
        for source_id, target_id, stmt_type, evidence_str in rows:
            pmid, source_api, source_sub_id = parse_evidence_json(evidence_str)
            if pmid is None and source_api is None and source_sub_id is None:
                continue

            edge_evidence_mentions[(source_id, target_id, stmt_type)] += 1

            ident = evidence_source_identity(pmid, source_api, source_sub_id)
            if ident:
                edge_sources[(source_id, target_id, stmt_type)].add(ident)

    logger.info("Edges with any extracted sources: %d", len(edge_sources))

    gene_sources = defaultdict(set)
    gene_evidence_mentions = defaultdict(int)

    for a_sym, b_sym, st in edges_sym:
        key = keymap.get((a_sym, b_sym, st))
        if not key:
            continue

        mcnt = edge_evidence_mentions.get(key, 0)
        if mcnt:
            gene_evidence_mentions[a_sym] += mcnt
            gene_evidence_mentions[b_sym] += mcnt

        srcs = edge_sources.get(key, set())
        if srcs:
            gene_sources[a_sym].update(srcs)
            gene_sources[b_sym].update(srcs)

    all_genes = sorted(
        set(df[args.source_col].astype(str).map(clean_gene_symbol).tolist())
        | set(df[args.target_col].astype(str).map(clean_gene_symbol).tolist())
    )
    for g in all_genes:
        gene_sources[g]
        gene_evidence_mentions[g] += 0

    genes = sorted(gene_sources.keys())
    rows_out = []
    for g in genes:
        srcs = gene_sources[g]
        lit = sum(1 for s in srcs if s.startswith("PMID:"))
        db = sum(1 for s in srcs if s.startswith("DB:"))
        rows_out.append({
            "gene": g,
            "literature_count": lit,
            "database_count": db,
            "characterization_count": lit + db,
            "evidence_mentions_count": int(gene_evidence_mentions.get(g, 0)),
        })

    gene_char = pd.DataFrame(rows_out)
    gene_char.to_csv(args.out_gene_csv, index=False)
    logger.info("Wrote gene characterization -> %s", args.out_gene_csv)

    char_map = gene_char.set_index("gene")["characterization_count"].to_dict()

    src_clean = df[args.source_col].astype(str).map(clean_gene_symbol)
    tgt_clean = df[args.target_col].astype(str).map(clean_gene_symbol)

    new_cols = pd.DataFrame({
        "source_char": src_clean.map(char_map).fillna(0).astype(int),
        "target_char": tgt_clean.map(char_map).fillna(0).astype(int),
    })
    new_cols["min_char"] = new_cols[["source_char", "target_char"]].min(axis=1)

    df_out = pd.concat([df, new_cols], axis=1)
    df_out.to_csv(args.out_path_csv, index=False)
    logger.info("Wrote 1-hop CSV with characterization -> %s", args.out_path_csv)

    plt.figure(figsize=(8, 5))
    plt.hist(gene_char["characterization_count"], bins=50)
    plt.yscale("log")
    plt.xlabel("Unique sources (PMIDs + DBs)")
    plt.ylabel("Number of genes (log scale)")
    plt.title("1-hop gene characterization distribution")
    plt.tight_layout()
    plt.savefig(args.out_hist, dpi=200)
    plt.close()
    logger.info("Wrote histogram -> %s", args.out_hist)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
