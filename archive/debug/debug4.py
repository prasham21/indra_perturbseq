from indra_cogex.client.neo4j_client import Neo4jClient

client = Neo4jClient()

source_id = "hgnc:6769"  # SMAD3 normalized lowercase
target_id = "hgnc:12313"  # JMJD1C normalized lowercase

query = """
MATCH (a:BioEntity {id: $source})-[r:indra_rel]->(b:BioEntity {id: $target})
WHERE r.stmt_type IN ["IncreaseAmount", "DecreaseAmount"]
RETURN a.id, b.id, r.stmt_type, r.belief, r.evidence_count, r.stmt_hash
"""

results = client.query_tx(query, source=source_id, target=target_id)

print(f"\n🔍 Custom Cypher: SMAD3 ➝ JMJD1C (causal expression)")
if results:
    for row in results:
        source, target, stmt_type, belief, evidence_count, stmt_hash = row
        print(f"✔️ {source} ➝ {target} | Type: {stmt_type} | Belief: {belief} | Evidence: {evidence_count}")
else:
    print("⚠️ No results found.")
