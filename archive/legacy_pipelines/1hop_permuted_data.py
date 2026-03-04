import pandas as pd
import time
from indra_cogex.analysis.source_targets_explanation import get_valid_gene_ids
from indra_cogex.client.neo4j_client import Neo4jClient
from indra.databases.hgnc_client import get_hgnc_name, get_current_hgnc_id

# === LOAD MASTER PERMUTED DATASET ===
permuted_df = pd.read_csv("/Users/prashammarfatia/Downloads/MASTER_permuted_source_target_pairs.csv")
print(f" Loaded {len(permuted_df)} permuted pairs")

# Group by permuted source to batch process
grouped = permuted_df.groupby("source")
print(f" {len(grouped)} unique permuted sources to query\n")

client = Neo4jClient()
all_results = []

start_time = time.time()
for idx, (source_gene, group) in enumerate(grouped):
    print(f"🔀 ({idx + 1}/{len(grouped)}) Processing permuted source: {source_gene}")

    loop_start = time.time()
    try:
        # Get HGNC ID for source
        hgnc_id = get_current_hgnc_id(source_gene.upper())
        if not hgnc_id:
            print(f"   ⚠️ No HGNC ID found for {source_gene}")
            continue
        source_id = f"hgnc:{hgnc_id}"

        # Get targets for this permuted source
        target_symbols = group["target"].dropna().unique().tolist()
        converted = get_valid_gene_ids(target_symbols)
        target_ids = [f"hgnc:{v}" for v in converted if v]

        if not target_ids:
            print(f"    No valid target IDs for {source_gene}")
            continue

        # FIX: Handle duplicate targets by keeping first occurrence
        # This is fine because we just need ONE logFC/pval value per target
        group_dedup = group.drop_duplicates(subset="target", keep="first")

        # Create map of target -> logFC, pval
        deg_map = group_dedup.set_index("target")[["logfoldchange", "pval"]].to_dict("index")

        # Query INDRA for 1-hop connections
        query = """
        MATCH (source:BioEntity {id: $perturbation})-[r:indra_rel]->(target:BioEntity)
        WHERE target.id IN $descendants AND r.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
        RETURN source.id, target.id, r.stmt_type, r.belief, r.evidence_count
        """
        results = client.query_tx(query, perturbation=source_id, descendants=target_ids)

        # Store results
        for r in results:
            _, target_hgnc, stmt_type, belief, ev_count = r
            target_id = target_hgnc.split(":")[1]
            symbol = get_hgnc_name(target_id)
            if symbol and symbol in deg_map:
                all_results.append({
                    "source": source_gene,
                    "target": symbol,
                    "stmt_type": stmt_type,
                    "belief": belief,
                    "evidence_count": ev_count,
                    "logfoldchange": deg_map[symbol]["logfoldchange"],
                    "pval": deg_map[symbol]["pval"]
                })

        loop_time = time.time() - loop_start
        avg_time = (time.time() - start_time) / (idx + 1)
        remaining = avg_time * (len(grouped) - idx - 1)
        print(f"   ✓ {len(results)} edges found | ⏱️ {loop_time:.1f}s | ETA: {remaining / 60:.1f} mins")

    except Exception as e:
        print(f"    Error with {source_gene}: {e}")

# === SAVE PERMUTED 1-HOP RESULTS ===
output_df = pd.DataFrame(all_results)
output_df.to_csv("/Users/prashammarfatia/Downloads/indra_1hop_PERMUTED.csv", index=False)

print("\n" + "=" * 70)
print(f" COMPLETED: Saved permuted 1-hop results")
print(f"   Output file: indra_1hop_PERMUTED.csv")
print(f"   Total edges found: {len(output_df)}")
print(f"   Unique sources: {output_df['source'].nunique()}")
print(f"   Unique targets: {output_df['target'].nunique()}")
print("=" * 70)

# === COMPARISON WITH REAL DATA ===
print("\n Quick comparison:")
print(f"   Permuted 1-hop connections: {len(output_df)}")
print(f"   (Compare this to your real 1-hop file)")