import pandas as pd
from indra_cogex.client.neo4j_client import Neo4jClient
from indra.databases.hgnc_client import get_current_hgnc_id

client = Neo4jClient()

# Test the additional examples
test_cases = [
    ("PVT1", "FN1", "IncreaseAmount"),
    ("PVT1", "PVT1", "DecreaseAmount")
]

query = """
MATCH (source:BioEntity {id: $perturbation})-[r:indra_rel]->(target:BioEntity)
WHERE target.id IN $descendants AND r.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
RETURN source.id, target.id, r.stmt_type, r.belief, r.evidence_count
"""

for source_gene, target_gene, stmt_type in test_cases:
    print(f"\n=== Testing {source_gene} -> {target_gene} ({stmt_type}) ===")

    source_hgnc_id = get_current_hgnc_id(source_gene)
    target_hgnc_id = get_current_hgnc_id(target_gene)

    source_id = f"hgnc:{source_hgnc_id}"
    target_id = f"hgnc:{target_hgnc_id}"

    print(f"HGNC IDs: {source_id} -> {target_id}")

    results = client.query_tx(query, perturbation=source_id, descendants=[target_id])

    print(f"Results found: {len(results)}")
    for r in results:
        if r[2] == stmt_type:
            print(f"  MATCH: {r}")
        else:
            print(f"  Other: {r}")