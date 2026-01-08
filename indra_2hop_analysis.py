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


def save_checkpoint(results, processed_genes, checkpoint_file="speed_optimization_checkpoint.pkl"):
    """Save progress"""
    checkpoint_data = {
        'results': results,
        'processed_genes': processed_genes,
        'timestamp': time.time()
    }
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)


def load_checkpoint(checkpoint_file="speed_optimization_checkpoint.pkl"):
    """Load previous progress"""
    try:
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
            saved_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data['timestamp']))
            print(f" Resuming from checkpoint saved at {saved_time}")
            print(f"   - {len(data['results'])} results already collected")
            print(f"   - {len(data['processed_genes'])} genes already processed")
            return data['results'], data['processed_genes']
    except FileNotFoundError:
        print(" Starting fresh - no checkpoint found")
        return [], set()


def test_batch_query_safety(client, source_id, target_ids):
    """Test if batch queries work and are faster"""
    if len(target_ids) < 5:
        return False, None

    print("🧪 Testing batch query performance...")
    test_targets = target_ids[:min(10, len(target_ids))]

    # Original individual query
    original_query = """
    MATCH (a:BioEntity {id: $source})-[r1:indra_rel]->(m:BioEntity)-[r2:indra_rel]->(b:BioEntity {id: $target})
    WHERE r2.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
    RETURN a.id, m.id, b.id,
           r1.stmt_type, r2.stmt_type,
           r1.belief, r2.belief,
           r1.evidence_count, r2.evidence_count
    """

    # Batch query
    batch_query = """
    UNWIND $target_list AS target_id
    MATCH (a:BioEntity {id: $source})-[r1:indra_rel]->(m:BioEntity)-[r2:indra_rel]->(b:BioEntity {id: target_id})
    WHERE r2.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
    RETURN a.id, m.id, b.id, target_id,
           r1.stmt_type, r2.stmt_type,
           r1.belief, r2.belief,
           r1.evidence_count, r2.evidence_count
    """

    # Test individual queries
    start_time = time.time()
    individual_count = 0
    try:
        for target_id in test_targets:
            results = client.query_tx(original_query, source=source_id, target=target_id)
            individual_count += len(results)
        individual_time = time.time() - start_time
    except Exception as e:
        print(f"    Individual queries failed: {e}")
        return False, None

    # Test batch query
    try:
        start_time = time.time()
        batch_results = client.query_tx(batch_query, source=source_id, target_list=test_targets)
        batch_time = time.time() - start_time
        batch_count = len(batch_results)

        speedup = individual_time / batch_time if batch_time > 0 else 0
        result_similarity = abs(batch_count - individual_count) / max(individual_count, 1)

        print(f"   Individual: {individual_time:.2f}s, {individual_count} results")
        print(f"   Batch: {batch_time:.2f}s, {batch_count} results")
        print(f"   Speedup: {speedup:.1f}x, Result diff: {result_similarity:.1%}")

        # Use batch if it's faster and results are similar (within 10% difference)
        if speedup > 1.2 and result_similarity < 0.1:
            print("    Batch queries look good - will use them")
            return True, batch_query
        else:
            print("   ️ Batch queries not beneficial - sticking with individual")
            return False, None

    except Exception as e:
        print(f"    Batch query failed: {e}")
        return False, None


def execute_queries_smart(client, source_id, target_ids, use_batch, batch_query, batch_size=50):
    """Execute queries using the best method available"""

    original_query = """
    MATCH (a:BioEntity {id: $source})-[r1:indra_rel]->(m:BioEntity)-[r2:indra_rel]->(b:BioEntity {id: $target})
    WHERE r2.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
    RETURN a.id, m.id, b.id,
           r1.stmt_type, r2.stmt_type,
           r1.belief, r2.belief,
           r1.evidence_count, r2.evidence_count
    """

    all_results = []

    if not use_batch:
        # Use individual queries (original approach)
        for target_id in target_ids:
            try:
                results = client.query_tx(original_query, source=source_id, target=target_id)
                # Add target_id to match batch format
                for result in results:
                    extended_result = list(result)
                    extended_result.insert(3, target_id)  # Add target_id at position 3
                    all_results.append(tuple(extended_result))
            except Exception as e:
                print(f"      Query failed for target: {e}")
                continue
        return all_results

    # Use batch queries with fallback
    for i in range(0, len(target_ids), batch_size):
        batch_targets = target_ids[i:i + batch_size]
        try:
            batch_results = client.query_tx(batch_query, source=source_id, target_list=batch_targets)
            all_results.extend(batch_results)
        except Exception as e:
            print(f"      Batch failed, using individual queries: {e}")
            # Fallback to individual queries for this batch
            for target_id in batch_targets:
                try:
                    results = client.query_tx(original_query, source=source_id, target=target_id)
                    for result in results:
                        extended_result = list(result)
                        extended_result.insert(3, target_id)
                        all_results.append(tuple(extended_result))
                except:
                    continue

    return all_results


def process_single_gene(args):
    """Process one gene - designed for parallel execution"""
    gene_info, total_genes, use_batch, batch_query = args
    perturb_gene = gene_info["Gene"]

    # Each thread gets its own client
    client = Neo4jClient()
    local_results = []

    try:
        print(f"🔬 Processing: {perturb_gene}")
        start_time = time.time()

        # Get HGNC ID
        hgnc_id = get_current_hgnc_id(perturb_gene.upper())
        if not hgnc_id:
            print(f"   ️ No HGNC ID for {perturb_gene}")
            return []
        source_id = f"hgnc:{hgnc_id}"

        # Load DEG data
        deg_path = f"/Users/prashammarfatia/Downloads/de_results_per_gene/{perturb_gene}_vs_control.csv"
        if not os.path.exists(deg_path):
            print(f"    DEG file not found for {perturb_gene}")
            return []

        df = pd.read_csv(deg_path)
        df = df[df["pvals"] < 0.05]

        gene_symbols = df["names"].dropna().unique().tolist()
        converted = get_valid_gene_ids(gene_symbols)
        symbol_to_hgnc = {k: f"hgnc:{v}" for k, v in zip(gene_symbols, converted) if v}
        hgnc_to_symbol = {v: k for k, v in symbol_to_hgnc.items()}
        target_ids = list(symbol_to_hgnc.values())
        deg_map = df.set_index("names")[["logfoldchanges", "pvals"]].to_dict("index")

        if not target_ids:
            print(f"   ️ No valid targets for {perturb_gene}")
            return []

        print(f"    Querying {len(target_ids)} targets...")

        # Execute queries
        results = execute_queries_smart(client, source_id, target_ids, use_batch, batch_query)

        # Process results
        for r in results:
            if len(r) >= 10:  # Batch format with target_id
                _, intermediate_hgnc, _, target_id, stmt1, stmt2, belief1, belief2, ev1, ev2 = r[:10]
                target_symbol = hgnc_to_symbol.get(target_id, target_id.replace("hgnc:", ""))
            else:  # Individual query format (fallback)
                continue

            interm_id = intermediate_hgnc.replace("hgnc:", "")
            interm_symbol = get_hgnc_name(interm_id) or intermediate_hgnc

            stats = deg_map.get(target_symbol, {"logfoldchanges": None, "pvals": None})
            local_results.append({
                "source": perturb_gene,
                "intermediate": interm_symbol,
                "target": target_symbol,
                "stmt_type_1": stmt1,
                "stmt_type_2": stmt2,
                "belief_1": belief1,
                "belief_2": belief2,
                "evidence_1": ev1,
                "evidence_2": ev2,
                "logfoldchange": stats["logfoldchanges"],
                "pval": stats["pvals"]
            })

        elapsed = time.time() - start_time
        print(f"    {perturb_gene}: {len(local_results)} results in {elapsed / 60:.1f} min")
        return local_results

    except Exception as e:
        print(f"    Error with {perturb_gene}: {e}")
        return []


def main():
    # Load data
    perturb_df = pd.read_csv("/Users/prashammarfatia/Downloads/target_validation_expanded.csv")
    perturb_df = perturb_df[perturb_df['Karen_Flag'] == "Use_for_analysis"]
    print(f"🔬 Perturbations selected: {len(perturb_df)}")

    # Load checkpoint
    checkpoint_file = "speed_optimization_checkpoint.pkl"
    all_results, processed_genes = load_checkpoint(checkpoint_file)

    # Filter remaining genes
    remaining_df = perturb_df[~perturb_df['Gene'].isin(processed_genes)]
    print(f" Remaining to process: {len(remaining_df)} genes")

    if len(remaining_df) == 0:
        print(" All genes already processed!")
        output_df = pd.DataFrame(all_results)
        output_df.to_csv("indra_2hop_all_perturbations.csv", index=False)
        return

    # Test batch queries on first gene
    use_batch = False
    batch_query = None

    if len(remaining_df) > 0:
        test_gene = remaining_df.iloc[0]
        test_client = Neo4jClient()

        try:
            hgnc_id = get_current_hgnc_id(test_gene["Gene"].upper())
            if hgnc_id:
                source_id = f"hgnc:{hgnc_id}"

                # Get some targets for testing
                deg_path = f"/Users/prashammarfatia/Downloads/de_results_per_gene/{test_gene['Gene']}_vs_control.csv"
                if os.path.exists(deg_path):
                    df = pd.read_csv(deg_path)
                    df = df[df["pvals"] < 0.05]
                    gene_symbols = df["names"].dropna().unique().tolist()[:20]  # Test with first 20
                    converted = get_valid_gene_ids(gene_symbols)
                    target_ids = [f"hgnc:{v}" for v in converted if v]

                    if len(target_ids) >= 5:
                        use_batch, batch_query = test_batch_query_safety(test_client, source_id, target_ids)
        except Exception as e:
            print(f"   Batch test failed: {e}")

        # Let Python handle cleanup automatically

    print(f"\n Starting optimized processing...")
    print(f"   Batch queries: {' Enabled' if use_batch else ' Disabled'}")
    print(f"   Parallel workers: 3 (conservative)")
    print("=" * 60)

    # Process with limited parallelization
    start_time = time.time()
    gene_data = [{"Gene": row["Gene"]} for _, row in remaining_df.iterrows()]
    process_args = [(gene_info, len(remaining_df), use_batch, batch_query) for gene_info in gene_data]

    completed_results = []
    completed_count = 0

    # Use only 3 workers to be conservative
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all jobs
        future_to_gene = {executor.submit(process_single_gene, args): args[0]["Gene"]
                          for args in process_args}

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_gene):
            gene_name = future_to_gene[future]
            completed_count += 1

            try:
                gene_results = future.result()
                completed_results.extend(gene_results)
                processed_genes.add(gene_name)

                # Progress update
                elapsed = time.time() - start_time
                avg_time = elapsed / completed_count
                remaining_time = avg_time * (len(remaining_df) - completed_count)

                print(f"\n Progress: {completed_count}/{len(remaining_df)} | "
                      f"Results: {len(all_results) + len(completed_results)} | "
                      f"ETA: {remaining_time / 60:.1f} min")

                # Checkpoint every 10 genes
                if completed_count % 10 == 0:
                    current_all_results = all_results + completed_results
                    save_checkpoint(current_all_results, processed_genes, checkpoint_file)
                    print(" Checkpoint saved")

            except Exception as e:
                print(f" Failed to process {gene_name}: {e}")

    # Final save
    final_results = all_results + completed_results
    output_df = pd.DataFrame(final_results)
    output_df.to_csv("indra_2hop_all_perturbations.csv", index=False)

    total_time = time.time() - start_time
    print(f"\n Processing complete!")
    print(f"    Total results: {len(final_results)}")
    print(f"   ️ Total time: {total_time / 3600:.1f} hours")
    print(f"    Average: {total_time / len(remaining_df) / 60:.1f} min per gene")
    print(f"    Results saved to: indra_2hop_all_perturbations.csv")

    # Cleanup
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


if __name__ == "__main__":
    main()