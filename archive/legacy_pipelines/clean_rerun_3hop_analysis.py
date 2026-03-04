import os
import time
import pandas as pd
from indra_cogex.analysis.source_targets_explanation import get_valid_gene_ids
from indra_cogex.client.neo4j_client import Neo4jClient
from indra.databases.hgnc_client import get_hgnc_name, get_current_hgnc_id
import pickle
import concurrent.futures
from threading import Lock

results_lock = Lock()

# Modified queries to exclude MESH and ChEBI intermediate nodes
CLEAN_INDIVIDUAL_3HOP_QUERY = """
MATCH (a:BioEntity {id: $source})-[r1:indra_rel]->(m1:BioEntity)-[r2:indra_rel]->(m2:BioEntity)-[r3:indra_rel]->(b:BioEntity {id: $target})
WHERE r3.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
  AND r1.belief > 0.7
  AND r2.belief > 0.7
  AND r3.belief > 0.7
  AND NOT (m1.id CONTAINS 'mesh:' OR m1.id CONTAINS 'chebi:')
  AND NOT (m2.id CONTAINS 'mesh:' OR m2.id CONTAINS 'chebi:')
RETURN a.id, m1.id, m2.id, b.id,
       r1.stmt_type, r2.stmt_type, r3.stmt_type,
       r1.belief, r2.belief, r3.belief,
       r1.evidence_count, r2.evidence_count, r3.evidence_count
LIMIT 1
"""


def identify_contaminated_pairs():
    """Get contaminated source-target pairs only"""
    results_path = "/Users/prashammarfatia/Downloads/indra_3hop_optimized_all_perturbations_combined.csv"

    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return None, None

    print("Loading existing results to identify contaminated source-target pairs...")
    df = pd.read_csv(results_path)
    print(f"Total results: {len(df)}")

    # Function to check contamination
    def has_contamination(row):
        def contains_mesh_or_chebi(value):
            if not value or not isinstance(value, str):
                return False
            value_lower = value.lower()
            return 'mesh:' in value_lower or 'chebi:' in value_lower

        return (contains_mesh_or_chebi(row['intermediate_1']) or
                contains_mesh_or_chebi(row['intermediate_2']))

    # Filter contaminated rows
    contaminated_df = df[df.apply(has_contamination, axis=1)]
    clean_df = df[~df.apply(has_contamination, axis=1)]

    print(f"Contaminated rows: {len(contaminated_df)} ({len(contaminated_df) / len(df) * 100:.1f}%)")
    print(f"Clean rows: {len(clean_df)} ({len(clean_df) / len(df) * 100:.1f}%)")

    # Extract contaminated source-target pairs with their DEG data
    contaminated_pairs = contaminated_df[['source', 'target', 'logfoldchange', 'pval']].copy()
    print(f"Contaminated source-target pairs to re-query: {len(contaminated_pairs)}")

    return contaminated_pairs, clean_df


def save_checkpoint(results, processed_pairs, checkpoint_file="clean_rerun_checkpoint.pkl"):
    """Save progress"""
    checkpoint_data = {
        'results': results,
        'processed_pairs': processed_pairs,
        'timestamp': time.time()
    }
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)


def load_checkpoint(checkpoint_file="clean_rerun_checkpoint.pkl"):
    """Load previous progress"""
    try:
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
            saved_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data['timestamp']))
            print(f"Resuming from checkpoint saved at {saved_time}")
            print(f"   - {len(data['results'])} results already collected")
            print(f"   - {len(data['processed_pairs'])} pairs already processed")
            return data['results'], data['processed_pairs']
    except FileNotFoundError:
        print("Starting fresh - no checkpoint found")
        return [], set()


def process_contaminated_pairs_for_gene(args):
    """Process contaminated source-target pairs for one source gene"""
    gene_info, contaminated_pairs_for_gene = args
    source_gene = gene_info["source"]

    client = Neo4jClient()
    local_results = []
    failed_pairs = []

    try:
        print(f"Processing clean queries for: {source_gene}")
        start_time = time.time()

        # Get HGNC ID for source
        hgnc_id = get_current_hgnc_id(source_gene.upper())
        if not hgnc_id:
            print(f"   No HGNC ID for {source_gene}")
            return [], []
        source_id = f"hgnc:{hgnc_id}"

        print(f"    Found {len(contaminated_pairs_for_gene)} contaminated pairs for {source_gene}")

        # Convert target symbols to HGNC IDs
        target_symbols = contaminated_pairs_for_gene['target'].unique().tolist()
        converted = get_valid_gene_ids(target_symbols)
        symbol_to_hgnc = {k: f"hgnc:{v}" for k, v in zip(target_symbols, converted) if v}
        hgnc_to_symbol = {v: k for k, v in symbol_to_hgnc.items()}

        if not symbol_to_hgnc:
            print(f"   No valid target HGNC IDs for {source_gene}")
            return [], []

        print(f"    Querying {len(symbol_to_hgnc)} clean targets with 20min timeout per query...")

        # Create DEG lookup
        deg_lookup = {}
        for _, row in contaminated_pairs_for_gene.iterrows():
            deg_lookup[row['target']] = {
                'logfoldchange': row['logfoldchange'],
                'pval': row['pval']
            }

        # Query each contaminated source-target pair
        for target_symbol, target_hgnc in symbol_to_hgnc.items():
            try:
                results = client.query_tx(
                    CLEAN_INDIVIDUAL_3HOP_QUERY,
                    source=source_id,
                    target=target_hgnc,
                    timeout=1200  # 20 minute timeout per query
                )

                # Process results
                for r in results:
                    if len(r) >= 13:
                        _, m1_hgnc, m2_hgnc, _, stmt1, stmt2, stmt3, belief1, belief2, belief3, ev1, ev2, ev3 = r

                        # Convert intermediate genes
                        try:
                            m1_id = m1_hgnc.replace("hgnc:", "")
                            raw_symbol_1 = get_hgnc_name(m1_id)
                            m1_symbol = get_hgnc_name(
                                get_current_hgnc_id(raw_symbol_1)) if raw_symbol_1 else f"hgnc:{m1_id}"

                            m2_id = m2_hgnc.replace("hgnc:", "")
                            raw_symbol_2 = get_hgnc_name(m2_id)
                            m2_symbol = get_hgnc_name(
                                get_current_hgnc_id(raw_symbol_2)) if raw_symbol_2 else f"hgnc:{m2_id}"
                        except:
                            m1_symbol = m1_hgnc
                            m2_symbol = m2_hgnc

                        # Get DEG stats for this target
                        deg_stats = deg_lookup.get(target_symbol, {"logfoldchange": None, "pval": None})

                        local_results.append({
                            "source": source_gene,
                            "intermediate_1": m1_symbol,
                            "intermediate_2": m2_symbol,
                            "target": target_symbol,
                            "stmt_type_1": stmt1,
                            "stmt_type_2": stmt2,
                            "stmt_type_3": stmt3,
                            "belief_1": belief1,
                            "belief_2": belief2,
                            "belief_3": belief3,
                            "evidence_1": ev1,
                            "evidence_2": ev2,
                            "evidence_3": ev3,
                            "logfoldchange": deg_stats["logfoldchange"],
                            "pval": deg_stats["pval"]
                        })
                        break  # Only take first clean result

            except Exception as e:
                error_msg = str(e)
                if "TransactionTimedOut" in error_msg or "timeout" in error_msg.lower():
                    print(f"      TIMEOUT: {source_gene} → {target_symbol} (>20 min)")
                else:
                    print(f"      FAILED: {source_gene} → {target_symbol}: {e}")

                failed_pairs.append({
                    'source': source_gene,
                    'target': target_symbol,
                    'error': error_msg,
                    'error_type': 'timeout' if 'timeout' in error_msg.lower() else 'other'
                })
                continue

        elapsed = time.time() - start_time
        success_rate = len(local_results) / len(symbol_to_hgnc) * 100 if symbol_to_hgnc else 0
        print(
            f"    {source_gene}: {len(local_results)} clean results, {len(failed_pairs)} failed ({success_rate:.1f}% success) in {elapsed / 60:.1f} min")

        return local_results, failed_pairs

    except Exception as e:
        print(f"    Error with {source_gene}: {e}")
        return [], []


def main():
    print("CLEAN RE-RUN ANALYSIS")
    print("Re-querying only contaminated source-target pairs with MESH/ChEBI exclusion")
    print("=" * 70)

    # Get contaminated source-target pairs
    contaminated_pairs, clean_results_df = identify_contaminated_pairs()

    if contaminated_pairs is None:
        return

    if len(contaminated_pairs) == 0:
        print("No contaminated pairs found! All results are already clean.")
        return

    # Group contaminated pairs by source gene
    contaminated_by_source = contaminated_pairs.groupby('source')
    unique_sources = list(contaminated_by_source.groups.keys())

    print(f"Source genes with contaminated pairs:")
    for source in unique_sources[:10]:
        count = len(contaminated_by_source.get_group(source))
        print(f"  {source}: {count} contaminated targets")
    if len(unique_sources) > 10:
        print(f"  ... and {len(unique_sources) - 10} more genes")

    # Load checkpoint
    checkpoint_file = "clean_rerun_checkpoint.pkl"
    all_results, processed_pairs = load_checkpoint(checkpoint_file)

    # Filter remaining sources
    processed_sources = set([pair.split('->')[0] for pair in processed_pairs if '->' in pair])
    remaining_sources = [s for s in unique_sources if s not in processed_sources]

    print(f"\nProcessing status:")
    print(f"  Total source genes with contamination: {len(unique_sources)}")
    print(f"  Already processed: {len(unique_sources) - len(remaining_sources)}")
    print(f"  Remaining to process: {len(remaining_sources)}")

    if len(remaining_sources) == 0:
        print("All contaminated sources already processed!")
        final_results = clean_results_df.to_dict('records') + all_results
        output_df = pd.DataFrame(final_results)
        output_df.to_csv("indra_3hop_cleaned_results.csv", index=False, header=True)
        print(f"Final cleaned results saved: {len(final_results)} total pathways")
        return

    print(f"\nStarting CLEAN 3-hop analysis...")
    print(f"   Individual queries with 20min timeout per query")
    print(f"   Parallel workers: 4")
    print(f"   Limit per query: 1 pathway")
    print(f"   Filters: NO mesh: or chebi: intermediate nodes")
    print(f"   Belief filtering: > 0.7 for all relationships")
    print(f"   Checkpoint frequency: every 5 genes")
    print("=" * 70)

    # Process remaining sources
    start_time = time.time()
    process_args = []
    for source in remaining_sources:
        source_pairs = contaminated_by_source.get_group(source)
        process_args.append(({"source": source}, source_pairs))

    completed_results = []
    all_failed_pairs = []
    completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_source = {executor.submit(process_contaminated_pairs_for_gene, args): args[0]["source"]
                            for args in process_args}

        for future in concurrent.futures.as_completed(future_to_source):
            source_name = future_to_source[future]
            completed_count += 1

            try:
                source_results, failed_pairs = future.result()
                completed_results.extend(source_results)
                all_failed_pairs.extend(failed_pairs)

                # Mark pairs for this source as processed
                source_pairs = contaminated_by_source.get_group(source_name)
                for _, pair in source_pairs.iterrows():
                    processed_pairs.add(f"{pair['source']}->{pair['target']}")

                elapsed = time.time() - start_time
                avg_time = elapsed / completed_count
                remaining_time = avg_time * (len(remaining_sources) - completed_count)

                print(f"\nProgress: {completed_count}/{len(remaining_sources)} | "
                      f"Clean results: {len(all_results) + len(completed_results)} | "
                      f"Failed pairs: {len(all_failed_pairs)} | "
                      f"ETA: {remaining_time / 60:.1f} min")

                # Checkpoint every 5 genes
                if completed_count % 5 == 0:
                    current_all_results = all_results + completed_results
                    save_checkpoint(current_all_results, processed_pairs, checkpoint_file)
                    print("Checkpoint saved")

            except Exception as e:
                print(f"Failed to process {source_name}: {e}")

    # Save failed pairs log
    if all_failed_pairs:
        failed_df = pd.DataFrame(all_failed_pairs)
        failed_df.to_csv("failed_clean_queries.csv", index=False)
        print(f"\nFailed queries logged to: failed_clean_queries.csv ({len(all_failed_pairs)} pairs)")

        failure_types = failed_df['error_type'].value_counts()
        print("Failure breakdown:")
        for error_type, count in failure_types.items():
            print(f"  {error_type}: {count} pairs")

    # Final combination and save
    final_new_results = all_results + completed_results
    final_combined_results = clean_results_df.to_dict('records') + final_new_results

    output_df = pd.DataFrame(final_combined_results)
    output_df.to_csv("indra_3hop_cleaned_results.csv", index=False, header=True)

    total_time = time.time() - start_time
    print(f"\nClean re-run complete!")
    print(f"    Original clean results: {len(clean_results_df)}")
    print(f"    New clean results: {len(final_new_results)}")
    print(f"    Total final results: {len(final_combined_results)}")
    print(f"    Processing time: {total_time / 3600:.1f} hours")
    print(f"    Average: {total_time / len(remaining_sources) / 60:.1f} min per source gene")
    print(f"    Results saved to: indra_3hop_cleaned_results.csv")

    # Calculate success rate
    original_contaminated = len(contaminated_pairs)
    clean_alternatives_found = len(final_new_results)
    success_rate = (clean_alternatives_found / original_contaminated * 100) if original_contaminated > 0 else 0
    print(
        f"    Success rate: {clean_alternatives_found}/{original_contaminated} ({success_rate:.1f}%) contaminated pairs now have clean alternatives")

    # Cleanup
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


if __name__ == "__main__":
    main()