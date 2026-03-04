import pandas as pd
from indra_cogex.client.neo4j_client import Neo4jClient
from indra.databases import hgnc_client


def test_specific_2hop_with_stmt_hash():
    """
    Test the specific 2-hop pathway: CX3CL1 -> NLRC4 -> AKR1B1
    Using the modified query that includes stmt_hash values
    """

    client = Neo4jClient()

    # Specific test case
    source_gene = "CX3CL1"
    target_gene = "AKR1B1"
    expected_intermediate = "NLRC4"

    print(f"Testing 2-hop pathway: {source_gene} -> {expected_intermediate} -> {target_gene}")
    print("=" * 70)

    # Get HGNC IDs
    source_hgnc = hgnc_client.get_current_hgnc_id(source_gene)
    target_hgnc = hgnc_client.get_current_hgnc_id(target_gene)

    if not source_hgnc or not target_hgnc:
        print(f"HGNC ID missing - Source: {source_hgnc}, Target: {target_hgnc}")
        return

    source_id = f"hgnc:{source_hgnc}"
    target_id = f"hgnc:{target_hgnc}"

    print(f"HGNC IDs: {source_id} -> {target_id}")

    # Modified 2-hop query with stmt_hash
    modified_2hop_query = """
    MATCH (a:BioEntity {id: $source})-[r1:indra_rel]->(m:BioEntity)-[r2:indra_rel]->(b:BioEntity {id: $target})
    WHERE r2.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
    RETURN a.id, m.id, b.id,
           r1.stmt_type, r2.stmt_type,
           r1.belief, r2.belief,
           r1.evidence_count, r2.evidence_count,
           r1.stmt_hash, r2.stmt_hash
    """

    # Execute the query
    print("\nExecuting 2-hop query with stmt_hash...")
    results = client.query_tx(modified_2hop_query, source=source_id, target=target_id)

    if not results:
        print("No 2-hop pathways found for this source-target pair")
        return

    print(f"Found {len(results)} 2-hop pathway(s)")

    # Process each pathway found
    for i, result in enumerate(results, 1):
        print(f"\n--- PATHWAY {i} ---")

        source_id_result = result[0]
        intermediate_id = result[1]
        target_id_result = result[2]
        r1_stmt_type = result[3]
        r2_stmt_type = result[4]
        r1_belief = result[5]
        r2_belief = result[6]
        r1_evidence_count = result[7]
        r2_evidence_count = result[8]
        r1_stmt_hash = result[9]
        r2_stmt_hash = result[10]

        # Get gene symbols
        intermediate_hgnc_id = intermediate_id.replace("hgnc:", "")
        intermediate_symbol = hgnc_client.get_hgnc_name(intermediate_hgnc_id) or intermediate_id

        print(f"Pathway: {source_gene} --[{r1_stmt_type}]--> {intermediate_symbol} --[{r2_stmt_type}]--> {target_gene}")
        print(f"HOP 1 (r1): {source_gene} -> {intermediate_symbol}")
        print(f"   stmt_type: {r1_stmt_type}")
        print(f"   belief: {r1_belief}")
        print(f"   evidence_count: {r1_evidence_count}")
        print(f"   stmt_hash: {r1_stmt_hash}")

        print(f"HOP 2 (r2): {intermediate_symbol} -> {target_gene}")
        print(f"   stmt_type: {r2_stmt_type}")
        print(f"   belief: {r2_belief}")
        print(f"   evidence_count: {r2_evidence_count}")
        print(f"   stmt_hash: {r2_stmt_hash}")

        # Check if this matches your expected intermediate
        if intermediate_symbol == expected_intermediate:
            print(f"*** MATCHES EXPECTED INTERMEDIATE: {expected_intermediate} ***")

        # Test Evidence nodes for both stmt_hash values
        print(f"\nTesting Evidence nodes:")

        # Test r1 stmt_hash
        if r1_stmt_hash:
            evidence_query = """
            MATCH (e:Evidence {stmt_hash: $stmt_hash})
            RETURN count(e) as evidence_count,
                   collect(e.pmid)[0..3] as sample_pmids,
                   sum(CASE WHEN e.evidence IS NOT NULL THEN 1 ELSE 0 END) as texts_available
            """

            ev_results = client.query_tx(evidence_query, stmt_hash=r1_stmt_hash)
            if ev_results:
                ev_count = ev_results[0][0]
                pmids = [p for p in ev_results[0][1] if p]
                texts = ev_results[0][2]
                print(f"   r1 stmt_hash Evidence: {ev_count} nodes, {len(pmids)} PMIDs, {texts} texts")
                if pmids:
                    print(f"      Sample PMIDs: {pmids}")
            else:
                print(f"   r1 stmt_hash Evidence: No evidence found")
        else:
            print(f"   r1 stmt_hash: NULL")

        # Test r2 stmt_hash
        if r2_stmt_hash:
            ev_results = client.query_tx(evidence_query, stmt_hash=r2_stmt_hash)
            if ev_results:
                ev_count = ev_results[0][0]
                pmids = [p for p in ev_results[0][1] if p]
                texts = ev_results[0][2]
                print(f"   r2 stmt_hash Evidence: {ev_count} nodes, {len(pmids)} PMIDs, {texts} texts")
                if pmids:
                    print(f"      Sample PMIDs: {pmids}")
            else:
                print(f"   r2 stmt_hash Evidence: No evidence found")
        else:
            print(f"   r2 stmt_hash: NULL")

    print(f"\n{'=' * 70}")
    print("SUMMARY:")
    print(f"- Found {len(results)} 2-hop pathway(s) for {source_gene} -> {target_gene}")

    # Check if any pathway matches the expected intermediate
    matching_pathways = []
    for result in results:
        intermediate_id = result[1]
        intermediate_hgnc_id = intermediate_id.replace("hgnc:", "")
        intermediate_symbol = hgnc_client.get_hgnc_name(intermediate_hgnc_id) or intermediate_id
        if intermediate_symbol == expected_intermediate:
            matching_pathways.append(result)

    if matching_pathways:
        print(f"- {len(matching_pathways)} pathway(s) match expected intermediate: {expected_intermediate}")
    else:
        print(f"- No pathways match expected intermediate: {expected_intermediate}")
        print("- Available intermediates:")
        for result in results:
            intermediate_id = result[1]
            intermediate_hgnc_id = intermediate_id.replace("hgnc:", "")
            intermediate_symbol = hgnc_client.get_hgnc_name(intermediate_hgnc_id) or intermediate_id
            print(f"    {intermediate_symbol}")


if __name__ == "__main__":
    test_specific_2hop_with_stmt_hash()