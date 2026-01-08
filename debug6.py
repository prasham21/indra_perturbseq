import pandas as pd
from indra.databases import hgnc_client
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.analysis.source_targets_explanation import get_valid_gene_ids

# === CONFIGURATION ===
DEG_FILE = "/Users/prashammarfatia/Downloads/de_results_per_gene/SMAD3_vs_control.csv"
OUTPUT_FILE = "indra_2-hop_gene_symbol_only_2.csv"
SOURCE_SYMBOL = "SMAD3"

# === 1. Load DEG file and filter ===
df = pd.read_csv(DEG_FILE)
df = df[df['pvals'] < 0.05]
print(f"🧪 Total rows after p < 0.05 filter: {len(df)}")

# Build gene symbol ➝ (lfc, pval) mapping
gene_stats = df.set_index('names')[['logfoldchanges', 'pvals']].to_dict(orient='index')

gene_symbols = df['names'].dropna().unique().tolist()
print(f"🧬 Unique gene symbols (pre-HGNC conversion): {len(gene_symbols)}")

# === 2. Convert to HGNC IDs ===
converted = get_valid_gene_ids(gene_symbols)
symbol_to_hgnc = {k: f"hgnc:{v}" for k, v in zip(gene_symbols, converted) if v}
hgnc_to_symbol = {v: k for k, v in symbol_to_hgnc.items()}
target_ids = list(symbol_to_hgnc.values())

print(f"✅ Valid HGNC IDs: {len(target_ids)}")
print(f"🔹 Sample: {target_ids[:5]}")

# === 3. Get HGNC ID for SMAD3 ===
source_id = hgnc_client.get_current_hgnc_id(SOURCE_SYMBOL)
source_hgnc = f"hgnc:{source_id}"
source_symbol = SOURCE_SYMBOL

# === 4. Prepare query ===
query = """
MATCH (a:BioEntity {id: $source})-[r1:indra_rel]->(m:BioEntity)-[r2:indra_rel]->(b:BioEntity {id: $target})
WHERE r2.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
RETURN a.id, m.id, b.id,
       r1.stmt_type, r2.stmt_type,
       r1.belief, r2.belief,
       r1.evidence_count, r2.evidence_count
"""

# === 5. Initialize Neo4j client ===
client = Neo4jClient()
all_results = []

# === 6. Query per descendant ===
for target_hgnc in target_ids:
    try:
        results = client.query_tx(query, source=source_hgnc, target=target_hgnc)
    except Exception as e:
        print(f"❌ Error querying {target_hgnc}: {e}")
        continue

    target_symbol = hgnc_to_symbol.get(target_hgnc, target_hgnc)
    if results:
        print(f"✔️ Found {len(results)} 2-hop paths for {source_symbol} ➝ {target_symbol}")
        for row in results:
            s, m, t = row[0], row[1], row[2]
            m_sym = hgnc_client.get_hgnc_name(hgnc_client.get_current_hgnc_id(m.replace("hgnc:", ""))) or m
            t_sym = hgnc_to_symbol.get(t, t)

            # Get log fold change and p-value
            stats = gene_stats.get(t_sym, {'logfoldchanges': None, 'pvals': None})
            lfc = stats['logfoldchanges']
            pval = stats['pvals']

            all_results.append({
                "source": source_symbol,
                "intermediate": m_sym,
                "target": t_sym,
                "stmt_type_1": row[3],
                "stmt_type_2": row[4],
                "belief_1": row[5],
                "belief_2": row[6],
                "evidence_1": row[7],
                "evidence_2": row[8],
                "logfoldchange": lfc,
                "pvalue": pval
            })
    else:
        print(f"⚠️ No 2-hop paths found for {source_symbol} ➝ {target_symbol}")

# === 7. Save to CSV ===
results_df = pd.DataFrame(all_results)
results_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n📁 Results saved to {OUTPUT_FILE}")
