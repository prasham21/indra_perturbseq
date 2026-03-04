from indra_cogex.client import Neo4jClient


def test_step_by_step():
    client = Neo4jClient()

    # Step 1: Check if nodes exist
    print("Step 1: Checking if nodes exist")
    node1 = client.query_tx("MATCH (a:BioEntity {id: 'hgnc:11850'}) RETURN a.id")
    node2 = client.query_tx("MATCH (a:BioEntity {id: 'uniprot:Q9SZF7'}) RETURN a.id")

    print(f"hgnc:11850 exists: {len(node1) > 0}")
    print(f"uniprot:Q9SZF7 exists: {len(node2) > 0}")

    if len(node1) > 0 and len(node2) > 0:
        # Step 2: Check if any relationship exists
        print("\nStep 2: Checking for any relationships")
        any_rel = client.query_tx("""
            MATCH (a:BioEntity {id: 'hgnc:11850'})-[r:indra_rel]->(b:BioEntity {id: 'uniprot:Q9SZF7'})
            RETURN r.stmt_type, r.belief, r.evidence_count
        """)

        print(f"Found {len(any_rel)} relationships")
        for rel in any_rel:
            stmt_type, belief, evidence = rel
            print(f"  {stmt_type}, belief: {belief}, evidence: {evidence}")

        # Step 3: Check for the specific relationship
        if any_rel:
            print("\nStep 3: Looking for specific Activation relationship")
            specific_rel = client.query_tx("""
                MATCH (a:BioEntity {id: 'hgnc:11850'})-[r:indra_rel]->(b:BioEntity {id: 'uniprot:Q9SZF7'})
                WHERE r.stmt_type = 'Activation'
                RETURN r.belief, r.evidence_count, r.stmt_hash
            """)

            print(f"Found {len(specific_rel)} Activation relationships")
            for rel in specific_rel:
                belief, evidence, stmt_hash = rel
                print(f"  belief: {belief}, evidence: {evidence}, hash: {stmt_hash}")


if __name__ == "__main__":
    test_step_by_step()