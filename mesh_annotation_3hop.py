import pandas as pd
import re
import time
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client import get_mesh_ids_for_pmids


# === CONFIG ===
INPUT_FILE = "/Users/prashammarfatia/Downloads/indra_3hop_with_statements__main.csv"
OUTPUT_FILE = "/Users/prashammarfatia/Downloads/indra_3hop_with_mesh_names_and_ids_clean.csv"
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
        print(f"   Batch {i}/{total_batches} processed ({len(batch)} PMIDs)")

    duration = time.time() - start
    print(f"\n⏱  Completed in {duration:.1f} seconds")
    print(f" Annotated {len([v for v in pmid_to_mesh.values() if v])} PMIDs with MeSH terms\n")

    return pmid_to_mesh


def build_mesh_id_to_name_map(client):
    """Fetch all MeSH node names."""
    print("🔍 Fetching MeSH node names...")
    query = "MATCH (b:BioEntity) WHERE b.id STARTS WITH 'mesh:' RETURN b.id AS mesh_id, b.name AS mesh_name"
    results = client.query_tx(query)
    mapping = {}

    for record in results:
        if isinstance(record, dict):
            mesh_id, mesh_name = record.get("mesh_id"), record.get("mesh_name")
        else:
            mesh_id, mesh_name = record[0], record[1]
        if mesh_id and mesh_name:
            norm_id = mesh_id.replace("mesh:", "").upper()
            mapping[norm_id] = mesh_name

    print(f" Loaded {len(mapping)} MeSH name mappings (normalized)")
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

    print(f"   {pmid_col}: Dropped {dropped_terms_count} invalid or unrecognized MeSH entries")
    return annotated_terms


def main():
    print(f" Loading input file: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f" Loaded {len(df)} rows")

    # Identify PMID columns dynamically (e.g., pmids_hop1, pmids_hop2, pmids_hop3)
    hop_cols = [c for c in df.columns if c.startswith("pmids_hop")]
    print(f" Found PMID columns: {hop_cols}")

    # --- Step 1: Extract all unique PMIDs ---
    all_pmids = extract_all_pmids(df, hop_cols)
    print(f" Unique PMIDs found: {len(all_pmids)}")

    # --- Step 2: Query Neo4j for MeSH IDs ---
    client = Neo4jClient()
    pmid_to_mesh = build_pmid_to_mesh_map(all_pmids, client)

    # --- Step 3: Fetch name mappings ---
    mesh_id_to_name = build_mesh_id_to_name_map(client)

    # --- Step 4: Annotate each hop separately ---
    for col in hop_cols:
        annotated_terms = annotate_pmids_column(df, col, pmid_to_mesh, mesh_id_to_name)
        insert_index = df.columns.get_loc(col) + 1
        df.insert(insert_index, f"Annotated MeSH terms ({col.replace('pmids_', '')})", annotated_terms)

    # --- Step 5: Save output ---
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n Saved enriched file to: {OUTPUT_FILE}")
    print(" Columns added: one MeSH annotation column per hop (name + ID format)")


if __name__ == "__main__":
    main()
