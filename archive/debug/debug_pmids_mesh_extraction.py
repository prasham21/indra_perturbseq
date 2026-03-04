import time
import pytest
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client import get_mesh_ids_for_pmids


@pytest.fixture(scope="module")
def client():
    """Use one Neo4j client for the whole module."""
    return Neo4jClient()


def get_mesh_names_for_ids(mesh_ids, client):
    """
    Map MeSH IDs (e.g., D003324) to their human-readable names
    (e.g., 'Coronary Artery Disease') using lowercase 'mesh:' nodes.
    """
    if not mesh_ids:
        return {}

    # Normalize to lowercase mesh: prefix
    mesh_ids_prefixed = [
        f"mesh:{mid}" if not mid.lower().startswith("mesh:") else mid.lower()
        for mid in mesh_ids
    ]

    query = """
    UNWIND $ids AS mesh_id
    MATCH (b:BioEntity {id: mesh_id})
    RETURN b.id AS mesh_id, b.name AS mesh_name
    """

    result = client.query_tx(query, ids=mesh_ids_prefixed)
    name_map = {}
    for r in result:
        mesh_id = r[0] if isinstance(r, (list, tuple)) else r.get("mesh_id")
        name = r[1] if isinstance(r, (list, tuple)) else r.get("mesh_name")
        if mesh_id and name:
            # normalize back to standard D-code for readability
            mesh_code = mesh_id.replace("mesh:", "").upper()
            name_map[mesh_code] = name
    return name_map


def test_mesh_extraction_summary(client):
    """
    Run a small-batch extraction, report counts of MeSH IDs per PMID,
    and display the corresponding MeSH names for a sample.
    """
    pmids = [
        "23743648",  # AAGAB → AXL evidence
        "29259259",  # AXL → CAV1 evidence
        "30353671",  # AXL → MYC evidence
        "36844467",  # AXL → MYC evidence
    ]

    # --- Execute query ---
    start = time.time()
    mesh_map = get_mesh_ids_for_pmids(pmids, client=client)
    duration = time.time() - start

    # --- Structural validation ---
    assert isinstance(mesh_map, dict)
    assert all(isinstance(k, str) for k in mesh_map.keys())
    assert all(isinstance(v, list) for v in mesh_map.values())

    # --- Stats ---
    total_annotations = sum(len(v) for v in mesh_map.values())
    unique_mesh_ids = {mid for mids in mesh_map.values() for mid in mids}
    avg_per_pmid = total_annotations / len(pmids)

    print(f"\n✅ Extracted MeSH terms for {sum(bool(v) for v in mesh_map.values())}/{len(pmids)} PMIDs")
    print(f"⏱  Query time: {duration:.2f} s\n")
    for pmid, terms in mesh_map.items():
        print(f"PMID {pmid}: {len(terms)} MeSH terms")

    print("\n📊 === MeSH Summary ===")
    print(f"Total MeSH annotations (including duplicates): {total_annotations}")
    print(f"Unique MeSH IDs across all PMIDs: {len(unique_mesh_ids)}")
    print(f"Average MeSH terms per PMID: {avg_per_pmid:.1f}")

    # --- Map a small sample of MeSH IDs to names ---
    sample_mesh_ids = sorted(list(unique_mesh_ids))[:10]
    mesh_name_map = get_mesh_names_for_ids(sample_mesh_ids, client)

    print("\n🧠 Sample MeSH ID → Name mapping:")
    for mid, name in mesh_name_map.items():
        print(f"  {mid} → {name}")

    assert total_annotations > 0, "Expected at least one MeSH annotation overall"
