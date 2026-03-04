"""Superseded legacy script for 2-hop INDRA analysis with batch/individual query modes.

Refactored into src/indra_perturbseq/pipelines/.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import pickle
import time
from threading import Lock

import pandas as pd
from indra_cogex.analysis.source_targets_explanation import get_valid_gene_ids
from indra_cogex.client.neo4j_client import Neo4jClient
from indra.databases.hgnc_client import get_hgnc_name, get_current_hgnc_id

logger = logging.getLogger(__name__)

results_lock = Lock()


def save_checkpoint(results, processed_genes, checkpoint_file="speed_optimization_checkpoint.pkl"):
    """Save progress."""
    checkpoint_data = {
        "results": results,
        "processed_genes": processed_genes,
        "timestamp": time.time(),
    }
    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint_data, f)


def load_checkpoint(checkpoint_file="speed_optimization_checkpoint.pkl"):
    """Load previous progress."""
    try:
        with open(checkpoint_file, "rb") as f:
            data = pickle.load(f)
            saved_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data["timestamp"]))
            logger.info("Resuming from checkpoint saved at %s", saved_time)
            logger.info("  %d results already collected", len(data["results"]))
            logger.info("  %d genes already processed", len(data["processed_genes"]))
            return data["results"], data["processed_genes"]
    except FileNotFoundError:
        logger.info("Starting fresh - no checkpoint found")
        return [], set()


def test_batch_query_safety(client, source_id, target_ids):
    """Test if batch queries work and are faster."""
    if len(target_ids) < 5:
        return False, None

    logger.info("Testing batch query performance...")
    test_targets = target_ids[:min(10, len(target_ids))]

    original_query = """
    MATCH (a:BioEntity {id: $source})-[r1:indra_rel]->(m:BioEntity)-[r2:indra_rel]->(b:BioEntity {id: $target})
    WHERE r2.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
    RETURN a.id, m.id, b.id,
           r1.stmt_type, r2.stmt_type,
           r1.belief, r2.belief,
           r1.evidence_count, r2.evidence_count
    """

    batch_query = """
    UNWIND $target_list AS target_id
    MATCH (a:BioEntity {id: $source})-[r1:indra_rel]->(m:BioEntity)-[r2:indra_rel]->(b:BioEntity {id: target_id})
    WHERE r2.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
    RETURN a.id, m.id, b.id, target_id,
           r1.stmt_type, r2.stmt_type,
           r1.belief, r2.belief,
           r1.evidence_count, r2.evidence_count
    """

    start_time = time.time()
    individual_count = 0
    try:
        for target_id in test_targets:
            results = client.query_tx(original_query, source=source_id, target=target_id)
            individual_count += len(results)
        individual_time = time.time() - start_time
    except Exception as e:
        logger.warning("Individual queries failed: %s", e)
        return False, None

    try:
        start_time = time.time()
        batch_results = client.query_tx(batch_query, source=source_id, target_list=test_targets)
        batch_time = time.time() - start_time
        batch_count = len(batch_results)

        speedup = individual_time / batch_time if batch_time > 0 else 0
        result_similarity = abs(batch_count - individual_count) / max(individual_count, 1)

        logger.info("  Individual: %.2fs, %d results", individual_time, individual_count)
        logger.info("  Batch: %.2fs, %d results", batch_time, batch_count)
        logger.info("  Speedup: %.1fx, Result diff: %.1f%%", speedup, result_similarity * 100)

        if speedup > 1.2 and result_similarity < 0.1:
            logger.info("Batch queries look good - will use them")
            return True, batch_query
        else:
            logger.info("Batch queries not beneficial - sticking with individual")
            return False, None

    except Exception as e:
        logger.warning("Batch query failed: %s", e)
        return False, None


def execute_queries_smart(client, source_id, target_ids, use_batch, batch_query, batch_size=50):
    """Execute queries using the best method available."""
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
        for target_id in target_ids:
            try:
                results = client.query_tx(original_query, source=source_id, target=target_id)
                for result in results:
                    extended_result = list(result)
                    extended_result.insert(3, target_id)
                    all_results.append(tuple(extended_result))
            except Exception as e:
                logger.debug("Query failed for target: %s", e)
                continue
        return all_results

    for i in range(0, len(target_ids), batch_size):
        batch_targets = target_ids[i:i + batch_size]
        try:
            batch_results = client.query_tx(batch_query, source=source_id, target_list=batch_targets)
            all_results.extend(batch_results)
        except Exception as e:
            logger.warning("Batch failed, using individual queries: %s", e)
            for target_id in batch_targets:
                try:
                    results = client.query_tx(original_query, source=source_id, target=target_id)
                    for result in results:
                        extended_result = list(result)
                        extended_result.insert(3, target_id)
                        all_results.append(tuple(extended_result))
                except Exception:
                    continue

    return all_results


def process_single_gene(args_tuple):
    """Process one gene - designed for parallel execution."""
    gene_info, total_genes, use_batch, batch_query, deg_dir = args_tuple
    perturb_gene = gene_info["Gene"]

    client = Neo4jClient()
    local_results = []

    try:
        logger.info("Processing: %s", perturb_gene)
        start_time = time.time()

        hgnc_id = get_current_hgnc_id(perturb_gene.upper())
        if not hgnc_id:
            logger.warning("No HGNC ID for %s", perturb_gene)
            return []
        source_id = f"hgnc:{hgnc_id}"

        deg_path = os.path.join(deg_dir, f"{perturb_gene}_vs_control.csv")
        if not os.path.exists(deg_path):
            logger.warning("DEG file not found for %s", perturb_gene)
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
            logger.warning("No valid targets for %s", perturb_gene)
            return []

        logger.debug("Querying %d targets...", len(target_ids))

        results = execute_queries_smart(client, source_id, target_ids, use_batch, batch_query)

        for r in results:
            if len(r) >= 10:
                _, intermediate_hgnc, _, target_id, stmt1, stmt2, belief1, belief2, ev1, ev2 = r[:10]
                target_symbol = hgnc_to_symbol.get(target_id, target_id.replace("hgnc:", ""))
            else:
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
                "pval": stats["pvals"],
            })

        elapsed = time.time() - start_time
        logger.info("%s: %d results in %.1f min", perturb_gene, len(local_results), elapsed / 60)
        return local_results

    except Exception as e:
        logger.error("Error with %s: %s", perturb_gene, e)
        return []


def main():
    parser = argparse.ArgumentParser(description="2-hop INDRA analysis")
    parser.add_argument("--perturb-csv", required=True, help="Path to perturbation CSV")
    parser.add_argument("--deg-dir", required=True, help="Directory with DEG CSVs")
    parser.add_argument("--output", default="indra_2hop_all_perturbations.csv", help="Output CSV path")
    parser.add_argument("--checkpoint-file", default="speed_optimization_checkpoint.pkl")
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()

    perturb_df = pd.read_csv(args.perturb_csv)
    perturb_df = perturb_df[perturb_df["analysis_flag"] == "Use_for_analysis"]
    logger.info("Perturbations selected: %d", len(perturb_df))

    all_results, processed_genes = load_checkpoint(args.checkpoint_file)

    remaining_df = perturb_df[~perturb_df["Gene"].isin(processed_genes)]
    logger.info("Remaining to process: %d genes", len(remaining_df))

    if len(remaining_df) == 0:
        logger.info("All genes already processed!")
        output_df = pd.DataFrame(all_results)
        output_df.to_csv(args.output, index=False)
        return

    use_batch = False
    batch_query = None

    if len(remaining_df) > 0:
        test_gene = remaining_df.iloc[0]
        test_client = Neo4jClient()

        try:
            hgnc_id = get_current_hgnc_id(test_gene["Gene"].upper())
            if hgnc_id:
                source_id = f"hgnc:{hgnc_id}"
                deg_path = os.path.join(args.deg_dir, f"{test_gene['Gene']}_vs_control.csv")
                if os.path.exists(deg_path):
                    df = pd.read_csv(deg_path)
                    df = df[df["pvals"] < 0.05]
                    gene_symbols = df["names"].dropna().unique().tolist()[:20]
                    converted = get_valid_gene_ids(gene_symbols)
                    target_ids = [f"hgnc:{v}" for v in converted if v]

                    if len(target_ids) >= 5:
                        use_batch, batch_query = test_batch_query_safety(
                            test_client, source_id, target_ids,
                        )
        except Exception as e:
            logger.warning("Batch test failed: %s", e)

    logger.info("Starting optimized processing...")
    logger.info("  Batch queries: %s", "Enabled" if use_batch else "Disabled")
    logger.info("  Parallel workers: %d (conservative)", args.max_workers)

    start_time = time.time()
    gene_data = [{"Gene": row["Gene"]} for _, row in remaining_df.iterrows()]
    process_args = [
        (gene_info, len(remaining_df), use_batch, batch_query, args.deg_dir)
        for gene_info in gene_data
    ]

    completed_results = []
    completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_gene = {
            executor.submit(process_single_gene, a): a[0]["Gene"]
            for a in process_args
        }

        for future in concurrent.futures.as_completed(future_to_gene):
            gene_name = future_to_gene[future]
            completed_count += 1

            try:
                gene_results = future.result()
                completed_results.extend(gene_results)
                processed_genes.add(gene_name)

                elapsed = time.time() - start_time
                avg_time = elapsed / completed_count
                remaining_time = avg_time * (len(remaining_df) - completed_count)

                logger.info(
                    "Progress: %d/%d | Results: %d | ETA: %.1f min",
                    completed_count, len(remaining_df),
                    len(all_results) + len(completed_results),
                    remaining_time / 60,
                )

                if completed_count % 10 == 0:
                    current_all_results = all_results + completed_results
                    save_checkpoint(current_all_results, processed_genes, args.checkpoint_file)
                    logger.info("Checkpoint saved")

            except Exception as e:
                logger.error("Failed to process %s: %s", gene_name, e)

    final_results = all_results + completed_results
    output_df = pd.DataFrame(final_results)
    output_df.to_csv(args.output, index=False)

    total_time = time.time() - start_time
    logger.info("Processing complete!")
    logger.info("  Total results: %d", len(final_results))
    logger.info("  Total time: %.1f hours", total_time / 3600)
    logger.info("  Average: %.1f min per gene", total_time / len(remaining_df) / 60)
    logger.info("  Results saved to: %s", args.output)

    if os.path.exists(args.checkpoint_file):
        os.remove(args.checkpoint_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
