"""Legacy script: MeSH annotation for 2-hop INDRA results."""
from __future__ import annotations

import argparse
import time

import pandas as pd
from indra_cogex.client import get_mesh_ids_for_pmids
from indra_cogex.client.neo4j_client import Neo4jClient

import logging


logger = logging.getLogger(__name__)

BATCH_SIZE = 100


def extract_all_pmids(df, colnames):
    """Extract unique PMIDs from multiple columns."""
    pmids = set()
    for col in colnames:
        for entry in df[col].astype(str):
            for pmid in entry.replace(" ", "").split(";"):
                if pmid.isdigit():
                    pmids.add(pmid)
    return sorted(pmids)


def batch_iterable(iterable, size):
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
    annotated_count = len([v for v in pmid_to_mesh.values() if v])
    logger.info("Completed in %.1f seconds", duration)
    logger.info("Annotated %d PMIDs with MeSH terms", annotated_count)

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
            mapping[mesh_id.replace("mesh:", "").upper()] = mesh_name

    logger.info("Loaded %d MeSH name mappings", len(mapping))
    return mapping


def get_annotated_mesh_terms(pmid_string, pmid_to_mesh, mesh_id_to_name):
    """Return formatted MeSH terms for a semicolon-separated PMID string."""
    mesh_terms = set()
    for pmid in str(pmid_string).replace(" ", "").split(";"):
        mesh_terms.update(pmid_to_mesh.get(pmid, []))
    if not mesh_terms:
        return ""
    formatted = [f"{mesh_id_to_name.get(mid.upper(), mid)} ({mid})" for mid in sorted(mesh_terms)]
    return ", ".join(formatted)


def main():
    ap = argparse.ArgumentParser(description="Annotate 2-hop INDRA results with MeSH terms.")
    ap.add_argument("--input", required=True, help="Input CSV with pmids_hop1/pmids_hop2 columns.")
    ap.add_argument("--output", required=True, help="Output CSV with annotated MeSH terms.")
    args = ap.parse_args()

    logger.info("Loading input file: %s", args.input)
    df = pd.read_csv(args.input)
    logger.info("Loaded %d rows", len(df))

    all_pmids = extract_all_pmids(df, ["pmids_hop1", "pmids_hop2"])
    logger.info("Unique PMIDs found: %d", len(all_pmids))

    client = Neo4jClient()
    pmid_to_mesh = build_pmid_to_mesh_map(all_pmids, client)
    mesh_id_to_name = build_mesh_id_to_name_map(client)

    logger.info("Annotating rows with MeSH names + IDs...")
    df["Annotated MeSH terms hop1"] = df["pmids_hop1"].apply(
        lambda pmids: get_annotated_mesh_terms(pmids, pmid_to_mesh, mesh_id_to_name)
    )
    df["Annotated MeSH terms hop2"] = df["pmids_hop2"].apply(
        lambda pmids: get_annotated_mesh_terms(pmids, pmid_to_mesh, mesh_id_to_name)
    )

    cols = list(df.columns)
    new_order = []
    for col in cols:
        new_order.append(col)
        if col == "pmids_hop1":
            new_order.append("Annotated MeSH terms hop1")
        elif col == "pmids_hop2":
            new_order.append("Annotated MeSH terms hop2")
    df = df[new_order]

    df.to_csv(args.output, index=False)
    logger.info("Saved enriched file to: %s", args.output)


if __name__ == "__main__":
    main()
