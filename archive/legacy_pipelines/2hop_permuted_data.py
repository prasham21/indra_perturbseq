import os
import time
import pandas as pd
from indra_cogex.analysis.source_targets_explanation import get_valid_gene_ids
from indra_cogex.client.neo4j_client import Neo4jClient
from indra.databases.hgnc_client import get_hgnc_name, get_current_hgnc_id
import pickle
import concurrent.futures
from threading import Lock

# Thread-safe globals
results_lock = Lock()

# OPTIMIZATION: Limit paths per source-target pair
MAX_PATHS_PER_PAIR = 2


def save_checkpoint(results, processed_genes, checkpoint_file="permuted_2hop_checkpoint.pkl"):
    """Save progress"""
    checkpoint_data = {
        'results': results,
        'processed_genes': processed_genes,
        'timestamp': time.time()
    }
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)


def load_checkpoint(checkpoint_file="permuted_2hop_checkpoint.pkl"):
    """Load previous progress"""
    try:
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
            saved_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data['timestamp']))
            print(f"Resuming from checkpoint saved at {saved_time}")
            print(f"   - {len(data['results'])} results already collected")
            print(f"   - {len(data['processed_genes'])} genes already processed")
            return data['results'], data['processed_genes']
    except FileNotFoundError:
        print(" Starting fresh - no checkpoint found")
        return [], set()


def process_single_source(args):
    """Process one permuted source - designed for parallel execution"""
    source_gene, group, total_sources = args

    # Each thread gets its own client
    client = Neo4jClient()
    local_results = []

    try:
        print(f" Processing: {source_gene}")
        start_time = time.time()

        # Get HGNC ID for source
        hgnc_id = get_current_hgnc_id(source_gene.upper())
        if not hgnc_id:
            print(f"    No HGNC ID for {source_gene}")
            return []
        source_id = f"hgnc:{hgnc_id}"

        # Get targets for this permuted source
        target_symbols = group["target"].dropna().unique().tolist()
        converted = get_valid_gene_ids(target_symbols)
        target_ids = [f"hgnc:{v}" for v in converted if v]

        if not target_ids:
            print(f"   ️ No valid targets for {source_gene}")
            return []

        # Handle duplicate targets
        group_dedup = group.drop_duplicates(subset="target", keep="first")
        deg_map = group_dedup.set_index("target")[["logfoldchange", "pval"]].to_dict("index")

        print(f"   Querying {len(target_ids)} targets...")

        # Process targets in batches
        batch_size = 50
        for i in range(0, len(target_ids), batch_size):
            batch_targets = target_ids[i:i + batch_size]

            # OPTIMIZED 2-HOP QUERY
            # - Batch processing with UNWIND
            # - Limit results per pair
            # - Filter out non-human intermediates (including malformed IDs like hgnc:mesh:D000XXX)
            query = """
            UNWIND $target_list AS target_id
            MATCH (a:BioEntity {id: $source})-[r1:indra_rel]->(m:BioEntity)-[r2:indra_rel]->(b:BioEntity {id: target_id})
            WHERE r2.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
              AND m.id STARTS WITH 'hgnc:'
              AND NOT (
                  toLower(m.id) CONTAINS 'mesh:' OR 
                  toLower(m.id) CONTAINS 'chebi:' OR 
                  toLower(m.id) CONTAINS 'go:' OR 
                  toLower(m.id) CONTAINS 'uniprot:'
              )
            RETURN a.id, m.id, b.id, target_id,
                   r1.stmt_type, r2.stmt_type,
                   r1.belief, r2.belief,
                   r1.evidence_count, r2.evidence_count
            LIMIT $max_results
            """

            # Query with limit
            max_results = len(batch_targets) * MAX_PATHS_PER_PAIR

            try:
                results = client.query_tx(
                    query,
                    source=source_id,
                    target_list=batch_targets,
                    max_results=max_results
                )

                # Process results
                for r in results:
                    _, intermediate_hgnc, _, target_id, stmt1, stmt2, belief1, belief2, ev1, ev2 = r[:10]

                    # Convert to symbols
                    interm_id = intermediate_hgnc.replace("hgnc:", "").replace("HGNC:", "")
                    interm_symbol = get_hgnc_name(interm_id) or intermediate_hgnc

                    target_id_clean = target_id.replace("hgnc:", "").replace("HGNC:", "")
                    target_symbol = get_hgnc_name(target_id_clean) or target_id

                    # Get logFC/pval
                    if target_symbol in deg_map:
                        local_results.append({
                            "source": source_gene,
                            "intermediate": interm_symbol,
                            "target": target_symbol,
                            "stmt_type_1": stmt1,
                            "stmt_type_2": stmt2,
                            "belief_1": belief1,
                            "belief_2": belief2,
                            "evidence_1": ev1,
                            "evidence_2": ev2,
                            "logfoldchange": deg_map[target_symbol]["logfoldchange"],
                            "pval": deg_map[target_symbol]["pval"]
                        })

            except Exception as e:
                print(f"      Batch query failed: {e}")
                continue

        elapsed = time.time() - start_time
        print(f"    {source_gene}: {len(local_results)} results in {elapsed:.1f}s")
        return local_results

    except Exception as e:
        print(f"    Error with {source_gene}: {e}")
        return []


def main():
    print("=" * 70)
    print(" PERMUTED 2-HOP ANALYSIS")
    print("=" * 70)

    # Load master permuted dataset
    permuted_df = pd.read_csv("/Users/prashammarfatia/Downloads/MASTER_permuted_source_target_pairs.csv")
    print(f"\n Loaded {len(permuted_df)} permuted pairs")

    # Group by source
    grouped = permuted_df.groupby("source")
    print(f" {len(grouped)} unique permuted sources")

    # Load checkpoint
    checkpoint_file = "permuted_2hop_checkpoint.pkl"
    all_results, processed_sources = load_checkpoint(checkpoint_file)

    # Filter remaining sources
    remaining_sources = [src for src in grouped.groups.keys() if src not in processed_sources]
    print(f" Remaining to process: {len(remaining_sources)} sources\n")

    if len(remaining_sources) == 0:
        print(" All sources already processed!")
        output_df = pd.DataFrame(all_results)
        output_df.to_csv("/Users/prashammarfatia/Downloads/indra_2hop_PERMUTED.csv", index=False)
        print(f" Saved: indra_2hop_PERMUTED.csv ({len(output_df)} results)")
        return

    print("🚀 Starting optimized processing...")
    print(f"   Max paths per pair: {MAX_PATHS_PER_PAIR}")
    print(f"   Parallel workers: 3")
    print(f"   Batch size: 50 targets")
    print("=" * 70 + "\n")

    # Prepare arguments for parallel processing
    process_args = []
    for src in remaining_sources:
        group = grouped.get_group(src)
        process_args.append((src, group, len(remaining_sources)))

    # Process with parallelization
    start_time = time.time()
    completed_results = []
    completed_count = 0

    # Use 3 workers to be conservative
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all jobs
        future_to_source = {executor.submit(process_single_source, args): args[0]
                            for args in process_args}

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_source):
            source_name = future_to_source[future]
            completed_count += 1

            try:
                source_results = future.result()
                completed_results.extend(source_results)
                processed_sources.add(source_name)

                # Progress update
                elapsed = time.time() - start_time
                avg_time = elapsed / completed_count
                remaining_time = avg_time * (len(remaining_sources) - completed_count)

                print(f"\n Progress: {completed_count}/{len(remaining_sources)} | "
                      f"Results: {len(all_results) + len(completed_results)} | "
                      f"ETA: {remaining_time / 60:.1f} min")

                # Checkpoint every 10 sources
                if completed_count % 10 == 0:
                    current_all_results = all_results + completed_results
                    save_checkpoint(current_all_results, processed_sources, checkpoint_file)
                    print(" Checkpoint saved")

            except Exception as e:
                print(f" Failed to process {source_name}: {e}")

    # Final save
    final_results = all_results + completed_results
    output_df = pd.DataFrame(final_results)
    output_df.to_csv("/Users/prashammarfatia/Downloads/indra_2hop_PERMUTED.csv", index=False)

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(" PROCESSING COMPLETE!")
    print("=" * 70)
    print(f" Output: indra_2hop_PERMUTED.csv")
    print(f" Total paths found: {len(final_results)}")
    print(f"   Unique sources: {output_df['source'].nunique()}")
    print(f"   Unique targets: {output_df['target'].nunique()}")
    print(f"   Unique intermediates: {output_df['intermediate'].nunique()}")
    print(f"️  Total time: {total_time / 3600:.2f} hours")
    print(f"   Average: {total_time / len(remaining_sources):.1f}s per source")
    print("=" * 70)

    # Cleanup checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("️ Checkpoint file removed")


if __name__ == "__main__":
    main()