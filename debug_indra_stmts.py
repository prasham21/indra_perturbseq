"""
Check source_api and other evidence fields when PMID is not available
"""

from indra_cogex.client.neo4j_client import Neo4jClient
import json


def check_evidence_details():
    client = Neo4jClient()

    print("Checking evidence details for PPARG → EEF1B2 DecreaseAmount")
    print("=" * 55)

    # Get the evidence for the relationship we found
    query = """
    MATCH (s:BioEntity {id: 'hgnc:9236'})-[r:indra_rel {stmt_type: 'DecreaseAmount'}]->(t:BioEntity {id: 'hgnc:3208'})
    WITH r.stmt_hash as stmt_hash
    MATCH (e:Evidence {stmt_hash: stmt_hash})
    RETURN e.evidence
    """

    results = client.query_tx(query)

    if results:
        print(f"Found {len(results)} evidence records:")

        for i, result in enumerate(results):
            evidence_json_str = result[0]
            print(f"\nEvidence {i + 1}:")
            print(f"Raw JSON: {evidence_json_str}")

            try:
                evidence_data = json.loads(evidence_json_str)

                # Extract all available fields
                print("\nParsed fields:")
                for key, value in evidence_data.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}...")
                    else:
                        print(f"  {key}: {value}")

                # Specifically check the fields you're interested in
                source_api = evidence_data.get('source_api', 'Not found')
                pmid = evidence_data.get('pmid', 'Not found')
                text = evidence_data.get('text', 'Not found')

                print(f"\nKey fields:")
                print(f"  source_api: {source_api}")
                print(f"  pmid: {pmid}")
                print(f"  text: {text}")

            except json.JSONDecodeError as e:
                print(f"  Error parsing JSON: {e}")
    else:
        print("No evidence found")


def enhanced_pmid_extraction(client, source_id, target_id, stmt_type):
    """
    Enhanced version that returns both PMIDs and source_api info
    """
    query = """
    MATCH (source:BioEntity {id: $source_id})-[r:indra_rel {stmt_type: $stmt_type}]->(target:BioEntity {id: $target_id})
    WITH r.stmt_hash as stmt_hash
    MATCH (e:Evidence {stmt_hash: stmt_hash})
    RETURN e.evidence
    """

    try:
        results = client.query_tx(query,
                                  source_id=source_id,
                                  target_id=target_id,
                                  stmt_type=stmt_type)

        pmids = []
        sources = []
        texts = []

        for result in results:
            try:
                evidence_data = json.loads(result[0])

                # Extract PMID if available
                if 'pmid' in evidence_data and evidence_data['pmid']:
                    pmids.append(str(evidence_data['pmid']))

                # Extract source_api
                if 'source_api' in evidence_data:
                    sources.append(evidence_data['source_api'])

                # Extract text if available
                if 'text' in evidence_data and evidence_data['text']:
                    texts.append(evidence_data['text'])

            except json.JSONDecodeError:
                continue

        return {
            'pmids': list(set(pmids)),
            'sources': list(set(sources)),
            'has_text': len(texts) > 0,
            'evidence_count': len(results)
        }

    except Exception as e:
        print(f"Error: {e}")
        return {'pmids': [], 'sources': [], 'has_text': False, 'evidence_count': 0}


if __name__ == "__main__":
    # Check the specific evidence details
    check_evidence_details()

    print("\n" + "=" * 55)
    print("Enhanced extraction example:")

    client = Neo4jClient()
    result = enhanced_pmid_extraction(client, 'hgnc:9236', 'hgnc:3208', 'DecreaseAmount')

    print(f"PMIDs: {result['pmids']}")
    print(f"Sources: {result['sources']}")
    print(f"Has text: {result['has_text']}")
    print(f"Evidence count: {result['evidence_count']}")