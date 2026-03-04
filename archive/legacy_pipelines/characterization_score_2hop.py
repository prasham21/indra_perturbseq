import argparse
import json
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

from indra_cogex.client.neo4j_client import Neo4jClient
from indra.databases import hgnc_client


EDGE_EVIDENCE_BATCH_QUERY = """
UNWIND $edges AS e
MATCH (source:BioEntity {id: e.source_id})-[r:indra_rel {stmt_type: e.stmt_type}]->(target:BioEntity {id: e.target_id})
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


def evidence_source_identity(pmid: str | None, source_api: str | None, source_sub_id: str | None) -> str | None:
    # Karen's rule: count each paper once; DBs only when PMID missing
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--out-gene-csv", required=True)
    ap.add_argument("--out-path-csv", required=True)
    ap.add_argument("--out-hist", required=True)
    ap.add_argument("--batch-size", type=int, default=150, help="Batch size for UNWIND edge queries")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv, low_memory=False)

    required = ["source", "intermediate", "target", "stmt_type_1", "stmt_type_2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present: {df.columns.tolist()}")

    # Build unique edges (symbol-level)
    hop1_edges_sym = set()
    hop2_edges_sym = set()
    for _, r in df.iterrows():
        hop1_edges_sym.add((str(r["source"]).strip(), str(r["intermediate"]).strip(), str(r["stmt_type_1"]).strip()))
        hop2_edges_sym.add((str(r["intermediate"]).strip(), str(r["target"]).strip(), str(r["stmt_type_2"]).strip()))

    print(f"Unique hop1 edges: {len(hop1_edges_sym):,}")
    print(f"Unique hop2 edges: {len(hop2_edges_sym):,}")

    # Cache HGNC symbol -> id and build Neo4j edge maps
    sym_to_hgnc = {}

    def sym_to_hgnc_id(sym: str) -> str | None:
        sym = sym.strip()
        if sym in sym_to_hgnc:
            return sym_to_hgnc[sym]
        hid = hgnc_client.get_current_hgnc_id(sym)
        sym_to_hgnc[sym] = hid
        return hid

    hop1_edges = []
    hop2_edges = []

    # Keep maps from (symbol,symbol,stmt_type) -> (neo4j ids key) so we can join back
    hop1_keymap = {}
    hop2_keymap = {}

    def add_edge(edges_out, keymap_out, a_sym, b_sym, stmt_type):
        a_h = sym_to_hgnc_id(a_sym)
        b_h = sym_to_hgnc_id(b_sym)
        if not a_h or not b_h:
            return
        a_id = f"hgnc:{a_h}"
        b_id = f"hgnc:{b_h}"
        key = (a_id, b_id, stmt_type)
        keymap_out[(a_sym, b_sym, stmt_type)] = key
        edges_out.append({"source_id": a_id, "target_id": b_id, "stmt_type": stmt_type})

    for a, b, st in hop1_edges_sym:
        add_edge(hop1_edges, hop1_keymap, a, b, st)
    for a, b, st in hop2_edges_sym:
        add_edge(hop2_edges, hop2_keymap, a, b, st)

    # Deduplicate Neo4j edges list
    def dedupe_edges(edge_list):
        seen = set()
        out = []
        for e in edge_list:
            k = (e["source_id"], e["target_id"], e["stmt_type"])
            if k in seen:
                continue
            seen.add(k)
            out.append(e)
        return out

    hop1_edges = dedupe_edges(hop1_edges)
    hop2_edges = dedupe_edges(hop2_edges)

    print(f"Neo4j-resolvable hop1 edges: {len(hop1_edges):,}")
    print(f"Neo4j-resolvable hop2 edges: {len(hop2_edges):,}")

    client = Neo4jClient()

    # Query evidence for edges -> edge_key (neo4j) -> set of unique sources (PMID:* or DB:*)
    edge_sources = defaultdict(set)
    edge_evidence_mentions = defaultdict(int)  # (source_id, target_id, stmt_type) -> # Evidence JSON rows

    def fetch_edge_sources(edges):
        for batch in batched(edges, args.batch_size):
            rows = client.query_tx(EDGE_EVIDENCE_BATCH_QUERY, edges=batch)
            for source_id, target_id, stmt_type, evidence_str in rows:
                pmid, source_api, source_sub_id = parse_evidence_json(evidence_str)
                if pmid is None and source_api is None and source_sub_id is None:
                    # JSON parse failed; skip counting
                    continue

                # count every evidence mention (each Evidence JSON item counts once)
                edge_evidence_mentions[(source_id, target_id, stmt_type)] += 1

                ident = evidence_source_identity(pmid, source_api, source_sub_id)
                if not ident:
                    continue
                edge_sources[(source_id, target_id, stmt_type)].add(ident)

    fetch_edge_sources(hop1_edges)
    fetch_edge_sources(hop2_edges)

    print(f"Edges with any extracted sources: {len(edge_sources):,}")

    # Build gene -> unique sources (and evidence mentions) by propagating edge info to endpoint genes
    gene_sources = defaultdict(set)
    gene_evidence_mentions = defaultdict(int)  # gene symbol -> total evidence mentions across its edges

    def add_gene_sources_for_edge(a_sym, b_sym, stmt_type, keymap):
        key = keymap.get((a_sym, b_sym, stmt_type))
        if not key:
            return

        mcnt = edge_evidence_mentions.get(key, 0)
        if mcnt:
            gene_evidence_mentions[a_sym] += mcnt
            gene_evidence_mentions[b_sym] += mcnt

        srcs = edge_sources.get(key, set())
        if not srcs:
            return
        gene_sources[a_sym].update(srcs)
        gene_sources[b_sym].update(srcs)

    for a, b, st in hop1_edges_sym:
        add_gene_sources_for_edge(a, b, st, hop1_keymap)
    for a, b, st in hop2_edges_sym:
        add_gene_sources_for_edge(a, b, st, hop2_keymap)

    # Ensure we include all genes that appear in the CSV (even if 0 sources/mentions)
    all_genes_in_csv = sorted(set(pd.concat([df["source"], df["intermediate"], df["target"]]).astype(str).str.strip()))
    for g in all_genes_in_csv:
        gene_sources[g]  # touch default
        gene_evidence_mentions[g] += 0  # touch default

    # Gene-level output
    genes = sorted(gene_sources.keys())
    literature_count = []
    database_count = []
    characterization_count = []
    evidence_mentions_count = []

    for g in genes:
        srcs = gene_sources[g]
        lit = sum(1 for s in srcs if s.startswith("PMID:"))
        db = sum(1 for s in srcs if s.startswith("DB:"))
        literature_count.append(lit)
        database_count.append(db)
        characterization_count.append(lit + db)
        evidence_mentions_count.append(int(gene_evidence_mentions.get(g, 0)))

    gene_char = pd.DataFrame(
        {
            "gene": genes,
            "literature_count": literature_count,
            "database_count": database_count,
            "characterization_count": characterization_count,
            "evidence_mentions_count": evidence_mentions_count,  # NEW (added at end)
        }
    )
    gene_char.to_csv(args.out_gene_csv, index=False)
    print(f"Wrote gene characterization -> {args.out_gene_csv}")

    # Attach to paths (append new columns at the end; do not modify existing columns)
    char_map = gene_char.set_index("gene")["characterization_count"].to_dict()

    new_cols = pd.DataFrame(
        {
            "source_char": df["source"].astype(str).str.strip().map(char_map).fillna(0).astype(int),
            "intermediate_char": df["intermediate"].astype(str).str.strip().map(char_map).fillna(0).astype(int),
            "target_char": df["target"].astype(str).str.strip().map(char_map).fillna(0).astype(int),
        }
    )
    new_cols["min_char"] = new_cols[["source_char", "intermediate_char", "target_char"]].min(axis=1)

    df_out = pd.concat([df, new_cols], axis=1)
    df_out.to_csv(args.out_path_csv, index=False)
    print(f"Wrote path-level CSV with characterization -> {args.out_path_csv}")

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(gene_char["characterization_count"], bins=50)
    plt.yscale("log")
    plt.xlabel("Unique sources (PMIDs + DBs)")
    plt.ylabel("Number of genes (log scale)")
    plt.title("Gene characterization distribution")
    plt.tight_layout()
    plt.savefig(args.out_hist, dpi=200)
    plt.close()
    print(f"Wrote histogram -> {args.out_hist}")


if __name__ == "__main__":
    main()