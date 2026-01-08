from indra_cogex.analysis.source_targets_explanation import get_valid_gene_ids
from indra_cogex.client.neo4j_client import Neo4jClient
import pandas as pd

# 1. Load DEG result file (replace with your actual file path)
df = pd.read_csv("/Users/prashammarfatia/Downloads/de_results_per_gene/SMAD3_vs_control.csv")
df = df[df['pvals'] < 0.05]
print(f"🧪 Total rows after p < 0.05 filter: {len(df)}")

# 2. Extract gene symbols and convert to HGNC-prefixed IDs
gene_symbols = df['names'].dropna().unique().tolist()
print(f"🧬 Unique gene symbols (pre-HGNC conversion): {len(gene_symbols)}")

converted = get_valid_gene_ids(gene_symbols)
target_ids = [f'hgnc:{v}' for v in converted if v]
print(f"✅ Converted to valid HGNC IDs: {len(target_ids)}")
print(f"🔹 Sample target IDs: {target_ids[:10]}")

# 3. Initialize Neo4j client
client = Neo4jClient()
perturbation_id = "hgnc:6769"  # SMAD3

# 4. Construct Cypher query
query = """
MATCH (source:BioEntity {id: $perturbation})-[r:indra_rel]->(target:BioEntity)
WHERE target.id IN $descendants AND r.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
RETURN source.id, target.id, r.stmt_type, r.belief, r.evidence_count
"""

# 5. Run query
print(f"\n🔍 INDRA 1-hop downstream (causal expression) results for {perturbation_id}:")
try:
    results = client.query_tx(query, perturbation=perturbation_id, descendants=target_ids)
    if not results:
        print("⚠️ No matching causal edges found.")
    else:
        for row in results:
            print(f"✔️ {row[0]} ➝ {row[1]} | {row[2]} | Belief: {row[3]:.2f} | Evidence: {row[4]}")
except Exception as e:
    print(f"❌ Error during query execution: {e}")
