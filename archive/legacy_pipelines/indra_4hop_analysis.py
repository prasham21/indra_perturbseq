import os
import time
import pandas as pd
from indra_cogex.analysis.source_targets_explanation import get_valid_gene_ids
from indra_cogex.client.neo4j_client import Neo4jClient
from indra.databases.hgnc_client import get_hgnc_name, get_current_hgnc_id
from indra.databases import uniprot_client, hgnc_client
import pickle
import concurrent.futures
from threading import Lock
from datetime import datetime

results_lock = Lock()

# =====================================================================
# UPDATED 4-HOP QUERY (HGNC-only intermediates, no belief cutoff)
# =====================================================================
CLEAN_INDIVIDUAL_4HOP_QUERY = """
MATCH (a:BioEntity {id: $source})-[r1:indra_rel]->(m1:BioEntity)
      -[r2:indra_rel]->(m2:BioEntity)
      -[r3:indra_rel]->(m3:BioEntity)
      -[r4:indra_rel]->(b:BioEntity {id: $target})
WHERE r4.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
  AND m1.id STARTS WITH 'hgnc:'
  AND m2.id STARTS WITH 'hgnc:'
  AND m3.id STARTS WITH 'hgnc:'
  AND NOT (
      m1.id CONTAINS 'mesh:' OR m1.id CONTAINS 'chebi:' OR 
      m1.id CONTAINS 'go:' OR m1.id CONTAINS 'uniprot:'
  )
  AND NOT (
      m2.id CONTAINS 'mesh:' OR m2.id CONTAINS 'chebi:' OR 
      m2.id CONTAINS 'go:' OR m2.id CONTAINS 'uniprot:'
  )
  AND NOT (
      m3.id CONTAINS 'mesh:' OR m3.id CONTAINS 'chebi:' OR 
      m3.id CONTAINS 'go:' OR m3.id CONTAINS 'uniprot:'
  )
RETURN a.id, m1.id, m2.id, m3.id, b.id,
       r1.stmt_type, r2.stmt_type, r3.stmt_type, r4.stmt_type,
       r1.belief, r2.belief, r3.belief, r4.belief,
       r1.evidence_count, r2.evidence_count, r3.evidence_count, r4.evidence_count
LIMIT 1
"""

# =====================================================================
# GLOBAL CACHES
# =====================================================================
SYMBOL_CACHE = {}
UNIPROT_CACHE = {}


def convert_intermediate_to_symbol(intermediate_id):
    """Convert HGNC/UniProt/FamPlex identifiers to readable gene symbols."""
    if not isinstance(intermediate_id, str):
        return intermediate_id
    if intermediate_id in SYMBOL_CACHE:
        return SYMBOL_CACHE[intermediate_id]

    result = None
    if intermediate_id.startswith("hgnc:"):
        hgnc_id = intermediate_id.replace("hgnc:", "")
        result = get_hgnc_name(hgnc_id) or intermediate_id
    elif intermediate_id.startswith("uniprot:"):
        uid = intermediate_id.split("uniprot:")[-1]
        hgnc_id = uniprot_client.get_hgnc_id(uid)
        if hgnc_id:
            result = hgnc_client.get_hgnc_name(hgnc_id)
        result = result or intermediate_id
    elif intermediate_id.startswith("fplx:"):
        result = intermediate_id.replace("fplx:", "")
    else:
        result = intermediate_id

    SYMBOL_CACHE[intermediate_id] = result
    return result


def load_source_target_pairs():
    """Load only rows that have UniProt intermediates."""
    input_path = "/Users/prashammarfatia/Downloads/indra_4hop_results_converted.csv"
    df = pd.read_csv(input_path)

    # Keep rows where any intermediate column contains 'uniprot:'
    mask = df[['intermediate_1', 'intermediate_2', 'intermediate_3']].astype(str).apply(
        lambda c: c.str.contains('uniprot:', case=False, na=False)
    ).any(axis=1)
    df_uniprot = df[mask]

    pairs_df = df_uniprot[['source', 'target', 'logfoldchange', 'pval']].drop_duplicates()
    print(f"✅ Loaded {len(pairs_df)} source-target pairs with UniProt intermediates.")
    return pairs_df


def save_checkpoint(results, processed_pairs, checkpoint_file="4hop_checkpoint.pkl"):
    checkpoint_data = {
        "results": results,
        "processed_pairs": processed_pairs,
        "timestamp": time.time(),
        "symbol_cache": SYMBOL_CACHE,
        "uniprot_cache": UNIPROT_CACHE
    }
    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint_data, f)

    if results:
        df_intermediate = pd.DataFrame(results)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_intermediate.to_csv(f"4hop_intermediate_{ts}.csv", index=False)
        print(f"📝 Checkpoint CSV saved: 4hop_intermediate_{ts}.csv")


def load_checkpoint(checkpoint_file="4hop_checkpoint.pkl"):
    global SYMBOL_CACHE, UNIPROT_CACHE
    if not os.path.exists(checkpoint_file):
        print("No checkpoint found. Starting fresh.")
        return [], set()
    with open(checkpoint_file, "rb") as f:
        data = pickle.load(f)
        print(f"Resuming from checkpoint ({len(data['processed_pairs'])} processed)")
        SYMBOL_CACHE.update(data.get("symbol_cache", {}))
        UNIPROT_CACHE.update(data.get("uniprot_cache", {}))
        return data["results"], data["processed_pairs"]


def process_4hop_paths_for_gene(args):
    """Query INDRA for 4-hop paths for one source gene."""
    gene_info, pairs_for_gene = args
    source_gene = gene_info["source"]
    client = Neo4jClient()
    local_results, failed_pairs = [], []

    print(f"Processing 4-hop queries for {source_gene}")
    hgnc_id = get_current_hgnc_id(source_gene.upper())
    if not hgnc_id:
        print(f"⚠️ Skipping {source_gene} (no HGNC ID)")
        return [], []
    source_id = f"hgnc:{hgnc_id}"

    target_symbols = pairs_for_gene["target"].unique().tolist()
    valid_ids = get_valid_gene_ids(target_symbols)
    symbol_to_hgnc = {s: f"hgnc:{v}" for s, v in zip(target_symbols, valid_ids) if v}

    deg_lookup = {
        row["target"]: {"logfoldchange": row["logfoldchange"], "pval": row["pval"]}
        for _, row in pairs_for_gene.iterrows()
    }

    for target_symbol, target_hgnc in symbol_to_hgnc.items():
        try:
            res = client.query_tx(CLEAN_INDIVIDUAL_4HOP_QUERY, source=source_id, target=target_hgnc, timeout=900)
            for r in res:
                if len(r) >= 17:
                    (_, m1, m2, m3, _,
                     s1, s2, s3, s4,
                     b1, b2, b3, b4,
                     e1, e2, e3, e4) = r
                    local_results.append({
                        "source": source_gene,
                        "intermediate_1": convert_intermediate_to_symbol(m1),
                        "intermediate_2": convert_intermediate_to_symbol(m2),
                        "intermediate_3": convert_intermediate_to_symbol(m3),
                        "target": target_symbol,
                        "stmt_type_1": s1, "stmt_type_2": s2, "stmt_type_3": s3, "stmt_type_4": s4,
                        "belief_1": b1, "belief_2": b2, "belief_3": b3, "belief_4": b4,
                        "evidence_1": e1, "evidence_2": e2, "evidence_3": e3, "evidence_4": e4,
                        **deg_lookup.get(target_symbol, {})
                    })
                    break
        except Exception as e:
            failed_pairs.append({"source": source_gene, "target": target_symbol, "error": str(e)})
    return local_results, failed_pairs


def main():
    print("=" * 80)
    print("INDRA 4-HOP RERUN (HGNC-only, from UniProt intermediates)")
    print("=" * 80)

    pairs_df = load_source_target_pairs()
    pairs_by_source = pairs_df.groupby("source")

    if os.path.exists("4hop_checkpoint.pkl"):
        os.remove("4hop_checkpoint.pkl")
        print("🧹 Old checkpoint removed — starting clean.")

    all_results, processed_pairs = [], set()
    print(f"🧬 Sources to process: {len(pairs_by_source.groups)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_to_source = {
            executor.submit(process_4hop_paths_for_gene, ({"source": s}, pairs_by_source.get_group(s))): s
            for s in pairs_by_source.groups.keys()
        }

        for i, future in enumerate(concurrent.futures.as_completed(future_to_source)):
            source = future_to_source[future]
            try:
                res, fails = future.result()
                all_results.extend(res)
                processed_pairs.add(source)
                if i % 5 == 0:
                    save_checkpoint(all_results, processed_pairs)
                print(f"✔️ Completed {source}: {len(res)} paths")
            except Exception as e:
                print(f" {source} failed: {e}")

    df_final = pd.DataFrame(all_results)
    output_file = "/Users/prashammarfatia/Downloads/indra_4hop_results_hgnc_only.csv"
    df_final.to_csv(output_file, index=False)
    print(f"\n Saved final HGNC-only results: {len(df_final)} rows → {output_file}")


if __name__ == "__main__":
    main()
