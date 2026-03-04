import pandas as pd
import time
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client import get_mesh_ids_for_pmids

# === CONFIG ===
INPUT_FILE = "/Users/prashammarfatia/Downloads/indra_2hop_with_evidence_statements_main.csv"
OUTPUT_FILE = "/Users/prashammarfatia/Downloads/indra_2hop_with_mesh_names_and_ids.csv"
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
    l = len(iterable)
    for i in range(0, l, size):
        yield iterable[i:i + size]


def build_pmid_to_mesh_map(pmids, client):
    """Query Neo4j in batches and build a PMID → MeSH ID map."""
    pmid_to_mesh = {}
    total_batches = (len(pmids) // BATCH_SIZE) + 1
    print(f"\n Fetching MeSH annotations for {len(pmids)} PMIDs in {total_batches} batches...\n")
    start = time.time()

    for i, batch in enumerate(batch_iterable(pmids, BATCH_SIZE), start=1):
        mesh_map = get_mesh_ids_for_pmids(batch, client=client)
        pmid_to_mesh.update(mesh_map)
        print(f"  Batch {i}/{total_batches} processed ({len(batch)} PMIDs)")

    duration = time.time() - start
    annotated_count = len([v for v in pmid_to_mesh.values() if v])
    print(f"\n  Completed in {duration:.1f} seconds")
    print(f" Annotated {annotated_count} PMIDs with MeSH terms\n")

    return pmid_to_mesh


def build_mesh_id_to_name_map(client):
    """Fetch all MeSH node names."""
    print("🔍 Fetching MeSH node names...")
    query = "MATCH (b:BioEntity) WHERE b.id STARTS WITH 'mesh:' RETURN b.id AS mesh_id, b.name AS mesh_name"
    results = client.query_tx(query)
    mapping = {}

    for record in results:
        # handle both list and dict styles
        if isinstance(record, dict):
            mesh_id, mesh_name = record.get("mesh_id"), record.get("mesh_name")
        else:
            mesh_id, mesh_name = record[0], record[1]

        if mesh_id and mesh_name:
            mapping[mesh_id.replace("mesh:", "").upper()] = mesh_name

    print(f" Loaded {len(mapping)} MeSH name mappings")
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
    print(f" Loading input file: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f" Loaded {len(df)} rows")

    # --- Step 1: Extract all unique PMIDs ---
    all_pmids = extract_all_pmids(df, ["pmids_hop1", "pmids_hop2"])
    print(f" Unique PMIDs found: {len(all_pmids)}")

    # --- Step 2: Query Neo4j for MeSH terms ---
    client = Neo4jClient()
    pmid_to_mesh = build_pmid_to_mesh_map(all_pmids, client)

    # --- Step 3: Get MeSH name mappings ---
    mesh_id_to_name = build_mesh_id_to_name_map(client)

    # --- Step 4: Annotate each row ---
    print("\n Annotating rows with MeSH names + IDs...")
    df["Annotated MeSH terms hop1"] = df["pmids_hop1"].apply(
        lambda pmids: get_annotated_mesh_terms(pmids, pmid_to_mesh, mesh_id_to_name)
    )
    df["Annotated MeSH terms hop2"] = df["pmids_hop2"].apply(
        lambda pmids: get_annotated_mesh_terms(pmids, pmid_to_mesh, mesh_id_to_name)
    )

    # --- Step 5: Reorder columns ---
    cols = list(df.columns)
    new_order = []
    for col in cols:
        new_order.append(col)
        if col == "pmids_hop1":
            new_order.append("Annotated MeSH terms hop1")
        elif col == "pmids_hop2":
            new_order.append("Annotated MeSH terms hop2")
    df = df[new_order]

    # --- Step 6: Save output ---
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n Saved enriched file to: {OUTPUT_FILE}")
    print(" Added: 'Annotated MeSH terms hop1' and 'Annotated MeSH terms hop2' (name + ID format)")


if __name__ == "__main__":
    main()
