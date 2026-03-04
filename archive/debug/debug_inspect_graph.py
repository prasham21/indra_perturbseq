import json
import pytest

from indra_cogex.client.neo4j_client import Neo4jClient


@pytest.mark.integration
def test_cogex_evidence_by_stmt_hash_medscan_only():
    stmt_hash = 27166631969242721
    expected_pmid = "28351322"

    client = Neo4jClient()

    query = """
    MATCH (e:Evidence {stmt_hash: $stmt_hash})
    RETURN e.evidence
    """
    rows = client.query_tx(query, stmt_hash=stmt_hash) or []

    # Basic existence check
    assert len(rows) > 0, f"No Evidence nodes found for stmt_hash={stmt_hash}"

    pmids = set()
    source_apis = set()
    text_samples = []

    # rows are typically tuples like (json_string,)
    for (ev_json,) in rows:
        ev = json.loads(ev_json)
        pmid = ev.get("pmid")
        if pmid:
            pmids.add(str(pmid))
        sa = ev.get("source_api")
        if sa:
            source_apis.add(str(sa))
        txt = ev.get("text")
        if txt and len(text_samples) < 3:
            text_samples.append(txt[:160])

    # Assertions matching your observation on the HTML page
    assert expected_pmid in pmids, f"Expected PMID {expected_pmid} not found. PMIDs seen: {sorted(pmids)[:20]}"
    assert "medscan" in {s.lower() for s in source_apis}, f"'medscan' not found in source_api. source_apis={source_apis}"

    # Print a concise debug summary for you to paste back
    print("\n=== Evidence-by-hash debug ===")
    print("stmt_hash:", stmt_hash)
    print("evidence_nodes:", len(rows))
    print("unique_pmids_count:", len(pmids))
    print("pmids_contains_28351322:", expected_pmid in pmids)
    print("source_apis:", sorted(source_apis))
    print("text_samples:", text_samples)