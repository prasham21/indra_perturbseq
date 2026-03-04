#!/usr/bin/env python3
"""
Compute what percent of INDRA statements in causal paths have at least one
term from the comprehensive MESH reference list.

Manager's definition:
- Statements used in causal paths across any hop number (1-hop, 2-hop, 3-hop)
- Using only RNA-present genes (endothelial_present_plus_manual.csv)
- Counting only statements that have associated PubMed IDs
- Of those: what percent have at least one of our MESH reference list terms?

A "statement" here is the INDRA triple (subject, predicate, object), i.e.
(from_gene, stmt_type, to_gene).  Activation(A, B) and IncreaseAmount(A, B)
are two distinct statements with separate evidence sets.  The same triple can
appear in many paths; we deduplicate by (from_gene, stmt_type, to_gene) and
merge PMID/MESH information across occurrences.
"""

import csv
import re

# ===== CONFIG =====
RNA_PRESENT_FILE = "/Users/prashammarfatia/Downloads/endothelial_present_plus_manual.csv"

# 1-hop: network-export based (has stmt_hash + indra_url; PMIDs from db.indra.bio)
HOP1_FILE = "/Users/prashammarfatia/Downloads/1hop_final_c14+endo_mesh_terms.csv"

# 2-hop: OLD COGEX-QUERY based (no hop hashes; PMIDs from Neo4j Evidence nodes).
#         The network export version (2hop_network_export_main.csv) exists but has
#         empty pmids_hop1 throughout — evidence enrichment for hop1 was not completed.
HOP2_FILE = "/Users/prashammarfatia/Downloads/2hop_final_c14+endo_mesh_terms.csv"

# 3-hop: older cogex-query based (3hop_network_export_raw.csv exists but has no
#         PMIDs/MESH yet — a fully enriched network-export 3-hop is not yet available)
HOP3_FILE = "/Users/prashammarfatia/Downloads/3hop_mesh_filtered__.csv"

# Note: "Annotated MeSH terms" columns in these files already contain only
# terms from the comprehensive MESH reference list (c14 + endothelial seeds +
# their children), as produced by the reannotation + map_mesh_terms pipeline.


# ===== HELPERS =====

def has_pmids(pmid_str: str) -> bool:
    """Return True if the string contains at least one digit-only PMID token."""
    if not pmid_str or str(pmid_str).strip().lower() in ("", "nan"):
        return False
    return any(tok.strip().isdigit() for tok in re.split(r"[;,\s]+", str(pmid_str)))


def has_mesh(mesh_str: str) -> bool:
    """Return True if the MESH annotation string is non-empty (i.e. has ≥1 term)."""
    if not mesh_str or str(mesh_str).strip().lower() in ("", "nan"):
        return False
    return bool(str(mesh_str).strip())


def load_rna_present_genes(path: str) -> set:
    genes = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = row.get("gene", "").strip().upper()
            if g:
                genes.add(g)
    print(f"RNA-present gene set loaded: {len(genes):,} genes")
    return genes


def all_rna_present(gene_list, rna_set: set) -> bool:
    return all(g.upper() in rna_set for g in gene_list if g and g.strip())


# ===== STATEMENT ACCUMULATOR =====
# key  : (from_gene_upper, stmt_type, to_gene_upper)
# value: dict with keys 'pmids' (bool) and 'mesh' (bool)
#
# Activation(A,B) and IncreaseAmount(A,B) are different INDRA statements and
# get separate keys.  When the same triple appears in multiple paths we OR the
# booleans so that if any occurrence has PMIDs / MESH terms it is credited.

statements: dict = {}


def register_statement(from_gene: str, stmt_type: str, to_gene: str, pmid_str: str, mesh_str: str):
    key = (from_gene.strip().upper(), stmt_type.strip(), to_gene.strip().upper())
    hp = has_pmids(pmid_str)
    hm = has_mesh(mesh_str)
    if key in statements:
        statements[key]["pmids"] = statements[key]["pmids"] or hp
        statements[key]["mesh"]  = statements[key]["mesh"]  or hm
    else:
        statements[key] = {"pmids": hp, "mesh": hm}


# ===== PER-HOP PROCESSORS =====
# Each processor returns a dict of per-hop-level statement stats:
#   { (from_gene, to_gene) -> {"pmids": bool, "mesh": bool} }
# These are also merged into the global `statements` dict.

def _hop_summary(hop_stmts: dict) -> dict:
    """Compute summary stats for a set of hop-level statements."""
    n_total    = len(hop_stmts)
    n_pmids    = sum(1 for v in hop_stmts.values() if v["pmids"])
    n_no_pmids = n_total - n_pmids
    n_hit      = sum(1 for v in hop_stmts.values() if v["pmids"] and v["mesh"])
    n_miss     = n_pmids - n_hit
    pct_pmids  = 100.0 * n_pmids / n_total if n_total else 0.0
    pct_mesh   = 100.0 * n_hit / n_pmids   if n_pmids else 0.0
    return {
        "n_total": n_total, "n_pmids": n_pmids, "n_no_pmids": n_no_pmids,
        "n_hit": n_hit, "n_miss": n_miss,
        "pct_pmids": pct_pmids, "pct_mesh": pct_mesh,
    }


def process_1hop(filepath: str, rna_set: set) -> dict:
    total = rna_filtered = 0
    hop_stmts: dict = {}

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            src  = row.get("source", "").strip()
            tgt  = row.get("target", "").strip()
            stype = row.get("stmt_type", "").strip()
            if not all_rna_present([src, tgt], rna_set):
                continue
            rna_filtered += 1
            pmids = row.get("pmids", "")
            mesh  = row.get("Annotated MeSH terms", "")
            register_statement(src, stype, tgt, pmids, mesh)
            key = (src.upper(), stype, tgt.upper())
            hp, hm = has_pmids(pmids), has_mesh(mesh)
            if key in hop_stmts:
                hop_stmts[key]["pmids"] = hop_stmts[key]["pmids"] or hp
                hop_stmts[key]["mesh"]  = hop_stmts[key]["mesh"]  or hm
            else:
                hop_stmts[key] = {"pmids": hp, "mesh": hm}

    print(f"  1-hop: {total:,} rows  →  {rna_filtered:,} RNA-present rows  →  "
          f"{rna_filtered / total * 100:.1f}% kept")
    return hop_stmts


def process_2hop(filepath: str, rna_set: set) -> tuple:
    """Returns (hop1_stmts, hop2_stmts) as separate dicts."""
    total = rna_filtered = 0
    h1: dict = {}
    h2: dict = {}

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            src = row.get("source", "").strip()
            mid = row.get("intermediate", "").strip()
            tgt = row.get("target", "").strip()
            if not all_rna_present([src, mid, tgt], rna_set):
                continue
            rna_filtered += 1

            for (a, b, stcol, pcol, mcol, store) in [
                (src, mid, "stmt_type_1", "pmids_hop1", "Annotated MeSH terms hop1", h1),
                (mid, tgt, "stmt_type_2", "pmids_hop2", "Annotated MeSH terms hop2", h2),
            ]:
                stype = row.get(stcol, "").strip()
                pmids = row.get(pcol, "")
                mesh  = row.get(mcol, "")
                register_statement(a, stype, b, pmids, mesh)
                key = (a.upper(), stype, b.upper())
                hp, hm = has_pmids(pmids), has_mesh(mesh)
                if key in store:
                    store[key]["pmids"] = store[key]["pmids"] or hp
                    store[key]["mesh"]  = store[key]["mesh"]  or hm
                else:
                    store[key] = {"pmids": hp, "mesh": hm}

    print(f"  2-hop: {total:,} rows  →  {rna_filtered:,} RNA-present rows  →  "
          f"{rna_filtered / total * 100:.1f}% kept")
    return h1, h2


def process_3hop(filepath: str, rna_set: set) -> tuple:
    """Returns (hop1_stmts, hop2_stmts, hop3_stmts) as separate dicts."""
    total = rna_filtered = 0
    h1: dict = {}
    h2: dict = {}
    h3: dict = {}

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            src  = row.get("source", "").strip()
            mid1 = row.get("intermediate_1", "").strip()
            mid2 = row.get("intermediate_2", "").strip()
            tgt  = row.get("target", "").strip()
            if not all_rna_present([src, mid1, mid2, tgt], rna_set):
                continue
            rna_filtered += 1

            for (a, b, stcol, pcol, mcol, store) in [
                (src,  mid1, "stmt_type_1", "pmids_hop1", "Annotated MeSH terms (hop1)", h1),
                (mid1, mid2, "stmt_type_2", "pmids_hop2", "Annotated MeSH terms (hop2)", h2),
                (mid2, tgt,  "stmt_type_3", "pmids_hop3", "Annotated MeSH terms (hop3)", h3),
            ]:
                stype = row.get(stcol, "").strip()
                pmids = row.get(pcol, "")
                mesh  = row.get(mcol, "")
                register_statement(a, stype, b, pmids, mesh)
                key = (a.upper(), stype, b.upper())
                hp, hm = has_pmids(pmids), has_mesh(mesh)
                if key in store:
                    store[key]["pmids"] = store[key]["pmids"] or hp
                    store[key]["mesh"]  = store[key]["mesh"]  or hm
                else:
                    store[key] = {"pmids": hp, "mesh": hm}

    print(f"  3-hop: {total:,} rows  →  {rna_filtered:,} RNA-present rows  →  "
          f"{rna_filtered / total * 100:.1f}% kept")
    return h1, h2, h3


def print_hop_stats(label: str, hop_stmts: dict):
    s = _hop_summary(hop_stmts)
    print(f"\n  {label}")
    print(f"    Unique statements (source, stmt_type, target): {s['n_total']:,}")
    print(f"    With PMIDs:    {s['n_pmids']:,}  ({s['pct_pmids']:.1f}%)")
    print(f"    Without PMIDs: {s['n_no_pmids']:,}  ({100-s['pct_pmids']:.1f}%)")
    if s["n_pmids"]:
        print(f"    Of PMID-backed → with MESH term:  {s['n_hit']:,}  ({s['pct_mesh']:.1f}%)")
        print(f"    Of PMID-backed → no MESH term:    {s['n_miss']:,}  ({100-s['pct_mesh']:.1f}%)")


# ===== MAIN =====

if __name__ == "__main__":
    print("=" * 60)
    print("MESH COVERAGE ANALYSIS")
    print("=" * 60)

    rna_set = load_rna_present_genes(RNA_PRESENT_FILE)

    print("\nLoading and filtering causal path files...")
    h1_stmts              = process_1hop(HOP1_FILE, rna_set)
    h2_hop1, h2_hop2      = process_2hop(HOP2_FILE, rna_set)
    h3_hop1, h3_hop2, h3_hop3 = process_3hop(HOP3_FILE, rna_set)

    # ── Per-path-length breakdown ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS — Per hop number (unique statements by source/stmt_type/target)")
    print("=" * 60)

    print("\n1-HOP paths  (each path = 1 statement)")
    print_hop_stats("Hop 1  (source → target)", h1_stmts)

    print("\n2-HOP paths  (each path = 2 statements)")
    print_hop_stats("Hop 1  (source → intermediate)", h2_hop1)
    print_hop_stats("Hop 2  (intermediate → target)", h2_hop2)
    # Combined unique edges seen across both hop positions of 2-hop paths
    combined_2hop = {}
    for store in (h2_hop1, h2_hop2):
        for k, v in store.items():
            if k in combined_2hop:
                combined_2hop[k]["pmids"] = combined_2hop[k]["pmids"] or v["pmids"]
                combined_2hop[k]["mesh"]  = combined_2hop[k]["mesh"]  or v["mesh"]
            else:
                combined_2hop[k] = dict(v)
    print_hop_stats("All edges in 2-hop paths (combined, deduplicated)", combined_2hop)

    print("\n3-HOP paths  (each path = 3 statements)")
    print_hop_stats("Hop 1  (source → intermediate_1)",        h3_hop1)
    print_hop_stats("Hop 2  (intermediate_1 → intermediate_2)", h3_hop2)
    print_hop_stats("Hop 3  (intermediate_2 → target)",         h3_hop3)
    combined_3hop = {}
    for store in (h3_hop1, h3_hop2, h3_hop3):
        for k, v in store.items():
            if k in combined_3hop:
                combined_3hop[k]["pmids"] = combined_3hop[k]["pmids"] or v["pmids"]
                combined_3hop[k]["mesh"]  = combined_3hop[k]["mesh"]  or v["mesh"]
            else:
                combined_3hop[k] = dict(v)
    print_hop_stats("All edges in 3-hop paths (combined, deduplicated)", combined_3hop)

    print(f"\nUnique statements (source, stmt_type, target) across all hops: {len(statements):,}")

    with_pmids = {k: v for k, v in statements.items() if v["pmids"]}
    without_pmids = {k: v for k, v in statements.items() if not v["pmids"]}

    with_pmids_and_mesh = {k: v for k, v in with_pmids.items() if v["mesh"]}
    with_pmids_no_mesh  = {k: v for k, v in with_pmids.items() if not v["mesh"]}

    n_total   = len(statements)
    n_pmids   = len(with_pmids)
    n_no_pmids = len(without_pmids)
    n_hit     = len(with_pmids_and_mesh)
    n_miss    = len(with_pmids_no_mesh)

    pct_has_pmids = 100.0 * n_pmids / n_total if n_total else 0.0
    pct_mesh      = 100.0 * n_hit / n_pmids   if n_pmids else 0.0

    print("\n" + "=" * 60)
    print("RESULTS — Combined (all hops, deduplicated by source/stmt_type/target)")
    print("=" * 60)
    print(f"Total unique statements in RNA-present paths:  {n_total:,}")
    print(f"  With PMIDs:                                  {n_pmids:,}  ({pct_has_pmids:.1f}%)")
    print(f"  Without PMIDs:                               {n_no_pmids:,}  ({100-pct_has_pmids:.1f}%)")
    print()
    print(f"Of the {n_pmids:,} statements WITH PMIDs:")
    print(f"  Have ≥1 MESH reference-list term:  {n_hit:,}  ({pct_mesh:.1f}%)")
    print(f"  Have no MESH reference-list term:  {n_miss:,}  ({100-pct_mesh:.1f}%)")
    print()
    print(f"==> ANSWER: {pct_mesh:.1f}% of PMID-backed statements have at least one MESH term")
    print("=" * 60)
