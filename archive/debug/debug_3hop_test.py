# run_batch_3hop_test.py
import os
import time
import pandas as pd
import pickle
import concurrent.futures

from indra_cogex.analysis.source_targets_explanation import get_valid_gene_ids
from indra_cogex.client.neo4j_client import Neo4jClient
from indra.databases.hgnc_client import get_hgnc_name, get_current_hgnc_id

# === New Batch Query: LIMIT 1 per target ===
BATCH_3HOP_LIMIT1_QUERY = """
UNWIND $target_list AS target_id
MATCH (a:BioEntity {id: $source})-[r1:indra_rel]->(m1:BioEntity)
      -[r2:indra_rel]->(m2:BioEntity)
      -[r3:indra_rel]->(b:BioEntity {id: target_id})
WHERE r3.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
  AND r1.belief > 0.7 AND r2.belief > 0.7 AND r3.belief > 0.7
WITH target_id, collect({
    a: a.id, m1: m1.id, m2: m2.id, b: b.id,
    stmt1: r1.stmt_type, stmt2: r2.stmt_type, stmt3: r3.stmt_type,
    belief1: r1.belief, belief2: r2.belief, belief3: r3.belief,
    ev1: r1.evidence_count, ev2: r2.evidence_count, ev3: r3.evidence_count
}) AS paths
RETURN target_id, head(paths) AS path
"""


def execute_batch_query(client, source_id, target_ids):
    return client.query_tx(BATCH_3HOP_LIMIT1_QUERY, source=source_id, target_list=target_ids)


def process_single_gene(args):
    gene, total_genes = args
    client = Neo4jClient()
    local_results = []

    print(f"\nProcessing: {gene}")
    hgnc_id = get_current_hgnc_id(gene.upper())
    if not hgnc_id:
        print(f"   ❌ No HGNC ID for {gene}")
        return []
    source_id = f"hgnc:{hgnc_id}"

    deg_path = f"/Users/prashammarfatia/Downloads/de_results_per_gene/{gene}_vs_control.csv"
    if not os.path.exists(deg_path):
        print(f"   ❌ DEG file not found for {gene}")
        return []

    df = pd.read_csv(deg_path)
    df = df[df["pvals"] < 0.05]
    symbols = df["names"].dropna().unique().tolist()
    converted = get_valid_gene_ids(symbols)
    symbol_to_hgnc = {k: f"hgnc:{v}" for k, v in zip(symbols, converted) if v}
    hgnc_to_symbol = {v: k for k, v in symbol_to_hgnc.items()}
    target_ids = list(symbol_to_hgnc.values())
    deg_map = df.set_index("names")[["logfoldchanges", "pvals"]].to_dict("index")

    if not target_ids:
        print(f"   ⚠️ No valid targets for {gene}")
        return []

    print(f"   🔍 Querying {len(target_ids)} targets using batch LIMIT 1 per target...")
    batch_results = execute_batch_query(client, source_id, target_ids)

    for row in batch_results:
        target_id = row["target_id"]
        path = row["path"]
        if not path:
            continue
        target_symbol = hgnc_to_symbol.get(target_id, target_id.replace("hgnc:", ""))

        m1_id = path["m1"].replace("hgnc:", "")
        m2_id = path["m2"].replace("hgnc:", "")
        m1_symbol = get_hgnc_name(get_current_hgnc_id(get_hgnc_name(m1_id))) or path["m1"]
        m2_symbol = get_hgnc_name(get_current_hgnc_id(get_hgnc_name(m2_id))) or path["m2"]

        stats = deg_map.get(target_symbol, {"logfoldchanges": None, "pvals": None})
        local_results.append({
            "source": gene,
            "intermediate_1": m1_symbol,
            "intermediate_2": m2_symbol,
            "target": target_symbol,
            "stmt_type_1": path["stmt1"],
            "stmt_type_2": path["stmt2"],
            "stmt_type_3": path["stmt3"],
            "belief_1": path["belief1"],
            "belief_2": path["belief2"],
            "belief_3": path["belief3"],
            "evidence_1": path["ev1"],
            "evidence_2": path["ev2"],
            "evidence_3": path["ev3"],
            "logfoldchange": stats["logfoldchanges"],
            "pval": stats["pvals"]
        })

    print(f"✅ {gene}: {len(local_results)} results")
    return local_results


def main():
    genes = ["SMAD3", "RAP1A", "STAT5A", "NUDT18"]
    output_file = "batch_limit1_test.csv"

    print("\n🧪 Running batch LIMIT 1 per target test on:", genes)
    start = time.time()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(process_single_gene, (gene, len(genes))): gene for gene in genes}
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"\n✅ Test complete! Total results: {len(results)}")
    print(f"⏱️ Duration: {round((time.time() - start) / 60, 2)} minutes")
    print(f"📄 Saved to: {output_file}")


if __name__ == "__main__":
    main()
