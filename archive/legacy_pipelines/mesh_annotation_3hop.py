"""Legacy script: MeSH annotation for 3-hop INDRA results."""
from __future__ import annotations

import argparse
import re
import time

import pandas as pd
from indra_cogex.client import get_mesh_ids_for_pmids
from indra_cogex.client.neo4j_client import Neo4jClient

import logging


logger = logging.getLogger(__name__)

BATCH_SIZE = 100


def extract_all_pmids(df, hop_cols):
    """Extract all unique PMIDs across all hop columns."""
    pmids = set()
    for col in hop_cols:
        for entry in df[col].astype(str):
            for pmid in entry.replace(" ", "").split(";"):
                if pmid.isdigit():
                    pmids.add(pmid)
    return sorted(pmids)


def batch_iterable(iterable, size):
    """Yield successive fixed-size batches."""
    length = len(iterable)
    for i in range(0, length, size):
        yield iterable[i:i + size]


def build_pmid_to_mesh_map(pmids, client):
    """Query Neo4j in batches and build a PMID -> MeSH ID map."""
    pmid_to_mesh = {}
    total_batches = (len(pmids) // BATCH_SIZE) + 1
    logger.info("Fetching MeSH annotations for %d PMIDs in %d batches...", len(pmids), total_batches)
    start = time.time()

    for i, batch in enumerate(batch_iterable(pmids, BATCH_SIZE), start=1):
        mesh_map = get_mesh_ids_for_pmids(batch, client=client)
        pmid_to_mesh.update(mesh_map)
        logger.info("Batch %d/%d processed (%d PMIDs)", i, total_batches, len(batch))

    duration = time.time() - start
    logger.info("Completed in %.1f seconds", duration)
    logger.info("Annotated %d PMIDs with MeSH terms", len([v for v in pmid_to_mesh.values() if v]))

    return pmid_to_mesh


def build_mesh_id_to_name_map(client):
    """Fetch all MeSH node names."""
    logger.info("Fetching MeSH node names...")
    query = "MATCH (b:BioEntity) WHERE b.id STARTS WITH 'mesh:' RETURN b.id AS mesh_id, b.name AS mesh_name"
    results = client.query_tx(query)
    mapping = {}

    for record in results:
        if isinstance(record, dict):
            mesh_id, mesh_name = record.get("mesh_id"), record.get("mesh_name")
        else:
            mesh_id, mesh_name = record[0], record[1]
        if mesh_id and mesh_name:
            norm = mesh_id.replace("mesh:", "").upper()
            mapping[norm] = mesh_name

    logger.info("Loaded %d MeSH name mappings", len(mapping))
    return mapping


def is_valid_mesh_id(mid: str) -> bool:
    """Return True if string looks like a canonical MeSH ID (D####### or C#######)."""
    return bool(re.fullmatch(r"[DC]\d{6,7}", mid))


def annotate_pmids_column(df, pmid_col, pmid_to_mesh, mesh_id_to_name):
    """Return a list of annotated MeSH name+ID strings for a PMID column."""
    annotated_terms = []
    dropped_terms_count = 0

    for _, row in df.iterrows():
        row_pmids = str(row[pmid_col]).replace(" ", "").split(";")
        mesh_terms = set()
        for pmid in row_pmids:
            mesh_terms.update(pmid_to_mesh.get(pmid, []))

        valid_terms = []
        for mid in sorted(mesh_terms):
            if is_valid_mesh_id(mid):
                name = mesh_id_to_name.get(mid.upper())
                if name:
                    valid_terms.append(f"{name} ({mid})")
            else:
                dropped_terms_count += 1

        annotated_terms.append(", ".join(valid_terms) if valid_terms else "")

    logger.info("%s: Dropped %d invalid or unrecognized MeSH entries", pmid_col, dropped_terms_count)
    return annotated_terms


def main():
    ap = argparse.ArgumentParser(description="Annotate 3-hop INDRA results with MeSH terms.")
    ap.add_argument("--input", required=True, help="Input CSV with pmids_hop* columns.")
    ap.add_argument("--output", required=True, help="Output CSV with annotated MeSH terms.")
    args = ap.parse_args()

    logger.info("Loading input file: %s", args.input)
    df = pd.read_csv(args.input)
    logger.info("Loaded %d rows", len(df))

    hop_cols = [c for c in df.columns if c.startswith("pmids_hop")]
    logger.info("Found PMID columns: %s", hop_cols)

    all_pmids = extract_all_pmids(df, hop_cols)
    logger.info("Unique PMIDs found: %d", len(all_pmids))

    client = Neo4jClient()
    pmid_to_mesh = build_pmid_to_mesh_map(all_pmids, client)
    mesh_id_to_name = build_mesh_id_to_name_map(client)

    for col in hop_cols:
        annotated_terms = annotate_pmids_column(df, col, pmid_to_mesh, mesh_id_to_name)
        insert_index = df.columns.get_loc(col) + 1
        df.insert(insert_index, f"Annotated MeSH terms ({col.replace('pmids_', '')})", annotated_terms)

    df.to_csv(args.output, index=False)
    logger.info("Saved enriched file to: %s", args.output)


if __name__ == "__main__":
    main()
