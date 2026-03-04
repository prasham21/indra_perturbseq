"""Compute MeSH term coverage across INDRA causal-path statements (1/2/3-hop)."""
from __future__ import annotations

import argparse
import csv
import logging
import re

logger = logging.getLogger(__name__)

RNA_PRESENT_FILE = "endothelial_present_plus_manual.csv"

def has_pmids(pmid_str: str) -> bool:
    """Return True if the string contains at least one digit-only PMID token."""
    if not pmid_str or str(pmid_str).strip().lower() in ("", "nan"):
        return False
    return any(tok.strip().isdigit() for tok in re.split(r"[;,\s]+", str(pmid_str)))


def has_mesh(mesh_str: str) -> bool:
    """Return True if the MESH annotation string is non-empty."""
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
    logger.info("RNA-present gene set loaded: %d genes", len(genes))
    return genes


def all_rna_present(gene_list, rna_set: set) -> bool:
    return all(g.upper() in rna_set for g in gene_list if g and g.strip())


def register_statement(from_gene: str, stmt_type: str, to_gene: str, pmid_str: str, mesh_str: str):
    key = (from_gene.strip().upper(), stmt_type.strip(), to_gene.strip().upper())
    hp = has_pmids(pmid_str)
    hm = has_mesh(mesh_str)
    if key in statements:
        statements[key]["pmids"] = statements[key]["pmids"] or hp
        statements[key]["mesh"] = statements[key]["mesh"] or hm
    else:
        statements[key] = {"pmids": hp, "mesh": hm}


def _hop_summary(hop_stmts: dict) -> dict:
    n_total = len(hop_stmts)
    n_pmids = sum(1 for v in hop_stmts.values() if v["pmids"])
    n_no_pmids = n_total - n_pmids
    n_hit = sum(1 for v in hop_stmts.values() if v["pmids"] and v["mesh"])
    n_miss = n_pmids - n_hit
    pct_pmids = 100.0 * n_pmids / n_total if n_total else 0.0
    pct_mesh = 100.0 * n_hit / n_pmids if n_pmids else 0.0
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
            src = row.get("source", "").strip()
            tgt = row.get("target", "").strip()
            stype = row.get("stmt_type", "").strip()
            if not all_rna_present([src, tgt], rna_set):
                continue
            rna_filtered += 1
            pmids = row.get("pmids", "")
            mesh = row.get("Annotated MeSH terms", "")
            register_statement(src, stype, tgt, pmids, mesh)
            key = (src.upper(), stype, tgt.upper())
            hp, hm = has_pmids(pmids), has_mesh(mesh)
            if key in hop_stmts:
                hop_stmts[key]["pmids"] = hop_stmts[key]["pmids"] or hp
                hop_stmts[key]["mesh"] = hop_stmts[key]["mesh"] or hm
            else:
                hop_stmts[key] = {"pmids": hp, "mesh": hm}

    logger.info(
        "1-hop: %d rows -> %d RNA-present rows -> %.1f%% kept",
        total, rna_filtered, rna_filtered / total * 100,
    )
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
                mesh = row.get(mcol, "")
                register_statement(a, stype, b, pmids, mesh)
                key = (a.upper(), stype, b.upper())
                hp, hm = has_pmids(pmids), has_mesh(mesh)
                if key in store:
                    store[key]["pmids"] = store[key]["pmids"] or hp
                    store[key]["mesh"] = store[key]["mesh"] or hm
                else:
                    store[key] = {"pmids": hp, "mesh": hm}

    logger.info(
        "2-hop: %d rows -> %d RNA-present rows -> %.1f%% kept",
        total, rna_filtered, rna_filtered / total * 100,
    )
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
            src = row.get("source", "").strip()
            mid1 = row.get("intermediate_1", "").strip()
            mid2 = row.get("intermediate_2", "").strip()
            tgt = row.get("target", "").strip()
            if not all_rna_present([src, mid1, mid2, tgt], rna_set):
                continue
            rna_filtered += 1

            for (a, b, stcol, pcol, mcol, store) in [
                (src, mid1, "stmt_type_1", "pmids_hop1", "Annotated MeSH terms (hop1)", h1),
                (mid1, mid2, "stmt_type_2", "pmids_hop2", "Annotated MeSH terms (hop2)", h2),
                (mid2, tgt, "stmt_type_3", "pmids_hop3", "Annotated MeSH terms (hop3)", h3),
            ]:
                stype = row.get(stcol, "").strip()
                pmids = row.get(pcol, "")
                mesh = row.get(mcol, "")
                register_statement(a, stype, b, pmids, mesh)
                key = (a.upper(), stype, b.upper())
                hp, hm = has_pmids(pmids), has_mesh(mesh)
                if key in store:
                    store[key]["pmids"] = store[key]["pmids"] or hp
                    store[key]["mesh"] = store[key]["mesh"] or hm
                else:
                    store[key] = {"pmids": hp, "mesh": hm}

    logger.info(
        "3-hop: %d rows -> %d RNA-present rows -> %.1f%% kept",
        total, rna_filtered, rna_filtered / total * 100,
    )
    return h1, h2, h3


def log_hop_stats(label: str, hop_stmts: dict):
    s = _hop_summary(hop_stmts)
    logger.info("  %s", label)
    logger.info("    Unique statements (source, stmt_type, target): %d", s["n_total"])
    logger.info("    With PMIDs:    %d  (%.1f%%)", s["n_pmids"], s["pct_pmids"])
    logger.info("    Without PMIDs: %d  (%.1f%%)", s["n_no_pmids"], 100 - s["pct_pmids"])
    if s["n_pmids"]:
        logger.info("    Of PMID-backed -> with MESH term:  %d  (%.1f%%)", s["n_hit"], s["pct_mesh"])
        logger.info("    Of PMID-backed -> no MESH term:    %d  (%.1f%%)", s["n_miss"], 100 - s["pct_mesh"])




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--1hop-final-c14-endo-mesh-term", default="1hop_final_c14+endo_mesh_terms.csv", help="Path: 1hop_final_c14+endo_mesh_terms.csv")
    ap.add_argument("--2hop-final-c14-endo-mesh-term", default="2hop_final_c14+endo_mesh_terms.csv", help="Path: 2hop_final_c14+endo_mesh_terms.csv")
    ap.add_argument("--3hop-mesh-filtered", default="3hop_mesh_filtered__.csv", help="Path: 3hop_mesh_filtered__.csv")
    args = ap.parse_args()

    HOP1_FILE = "1hop_final_c14+endo_mesh_terms.csv"
    HOP2_FILE = "2hop_final_c14+endo_mesh_terms.csv"
    HOP3_FILE = "3hop_mesh_filtered__.csv"


    statements: dict = {}




if __name__ == "__main__":
    main()
