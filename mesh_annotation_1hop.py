import pandas as pd
import re
import time
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client import get_mesh_ids_for_pmids


# === CONFIG ===
INPUT_FILE = "/Users/prashammarfatia/Downloads/indra_1hop_with_statements__main (1).csv"
OUTPUT_FILE = "/Users/prashammarfatia/Downloads/indra_1hop_with_mesh_names_and_ids_clean.csv"
BATCH_SIZE = 100


def extract_all_pmids(df):
    """Extract unique PMIDs from the dataframe."""
    pmids = set()
    for entry in df["pmids"].astype(str):
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
    print(f"\n🚀 Fetching MeSH annotations for {len(pmids)} PMIDs in {total_batches} batches...\n")
    start = time.time()

    for i, batch in enumerate(batch_iterable(pmids, BATCH_SIZE), start=1):
        mesh_map = get_mesh_ids_for_pmids(batch, client=client)
        pmid_to_mesh.update(mesh_map)
        print(f"  ✅ Batch {i}/{total_batches} processed ({len(batch)} PMIDs)")

    duration = time.time() - start
    print(f"\n⏱  Completed in {duration:.1f} seconds")
    print(f"📚 Annotated {len([v for v in pmid_to_mesh.values() if v])} PMIDs with MeSH terms\n")

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

    print(f"✅ Loaded {len(mapping)} MeSH name mappings (normalized)")
    return mapping


def is_valid_mesh_id(mid: str) -> bool:
    """Return True if string looks like a canonical MeSH ID (D####### or C#######)."""
    return bool(re.fullmatch(r"[DC]\d{6,7}", mid))


def main():
    print(f"📥 Loading input file: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Loaded {len(df)} rows")

    all_pmids = extract_all_pmids(df)
    print(f"🧾 Unique PMIDs found: {len(all_pmids)}")

    client = Neo4jClient()
    pmid_to_mesh = build_pmid_to_mesh_map(all_pmids, client)
    mesh_id_to_name = build_mesh_id_to_name_map(client)

    annotated_terms = []
    dropped_terms_count = 0

    for _, row in df.iterrows():
        row_pmids = str(row["pmids"]).replace(" ", "").split(";")
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
                dropped_terms_count += 1  # dropped because not a valid MeSH ID

        annotated_terms.append(", ".join(valid_terms) if valid_terms else "")

    pmid_idx = df.columns.get_loc("pmids")
    df.insert(pmid_idx + 1, "Annotated MeSH terms", annotated_terms)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Saved enriched file to: {OUTPUT_FILE}")
    print("✅ Column added: 'Annotated MeSH terms' (only valid name+ID pairs)")
    print(f"🚫 Dropped {dropped_terms_count} invalid or unrecognized MeSH entries")


if __name__ == "__main__":
    main()
