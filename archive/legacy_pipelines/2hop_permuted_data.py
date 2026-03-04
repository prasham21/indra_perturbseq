"""Superseded legacy script for 2-hop permuted data analysis.

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

MAX_PATHS_PER_PAIR = 2


def save_checkpoint(results, processed_genes, checkpoint_file="permuted_2hop_checkpoint.pkl"):
    """Save progress."""
    checkpoint_data = {
        "results": results,
        "processed_genes": processed_genes,
        "timestamp": time.time(),
    }
    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint_data, f)


def load_checkpoint(checkpoint_file="permuted_2hop_checkpoint.pkl"):
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


def process_single_source(args_tuple):
    """Process one permuted source - designed for parallel execution."""
    source_gene, group, total_sources = args_tuple

    client = Neo4jClient()
    local_results = []

    try:
        logger.info("Processing: %s", source_gene)
        start_time = time.time()

        hgnc_id = get_current_hgnc_id(source_gene.upper())
        if not hgnc_id:
            logger.warning("No HGNC ID for %s", source_gene)
            return []
        source_id = f"hgnc:{hgnc_id}"

        target_symbols = group["target"].dropna().unique().tolist()
        converted = get_valid_gene_ids(target_symbols)
        target_ids = [f"hgnc:{v}" for v in converted if v]

        if not target_ids:
            logger.warning("No valid targets for %s", source_gene)
            return []

        group_dedup = group.drop_duplicates(subset="target", keep="first")
        deg_map = group_dedup.set_index("target")[["logfoldchange", "pval"]].to_dict("index")

        logger.debug("Querying %d targets...", len(target_ids))

        batch_size = 50
        for i in range(0, len(target_ids), batch_size):
            batch_targets = target_ids[i:i + batch_size]

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

            max_results = len(batch_targets) * MAX_PATHS_PER_PAIR

            try:
                results = client.query_tx(
                    query,
                    source=source_id,
                    target_list=batch_targets,
                    max_results=max_results,
                )

                for r in results:
                    _, intermediate_hgnc, _, target_id, stmt1, stmt2, belief1, belief2, ev1, ev2 = r[:10]

                    interm_id = intermediate_hgnc.replace("hgnc:", "").replace("HGNC:", "")
                    interm_symbol = get_hgnc_name(interm_id) or intermediate_hgnc

                    target_id_clean = target_id.replace("hgnc:", "").replace("HGNC:", "")
                    target_symbol = get_hgnc_name(target_id_clean) or target_id

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
                            "pval": deg_map[target_symbol]["pval"],
                        })

            except Exception as e:
                logger.warning("Batch query failed: %s", e)
                continue

        elapsed = time.time() - start_time
        logger.info("%s: %d results in %.1fs", source_gene, len(local_results), elapsed)
        return local_results

    except Exception as e:
        logger.error("Error with %s: %s", source_gene, e)
        return []


def main():
    parser = argparse.ArgumentParser(description="Permuted 2-hop analysis")
    parser.add_argument("--permuted-csv", required=True, help="Master permuted source-target pairs CSV")
    parser.add_argument("--output", default="indra_2hop_PERMUTED.csv", help="Output CSV path")
    parser.add_argument("--checkpoint-file", default="permuted_2hop_checkpoint.pkl")
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()

    logger.info("PERMUTED 2-HOP ANALYSIS")

    permuted_df = pd.read_csv(args.permuted_csv)
    logger.info("Loaded %d permuted pairs", len(permuted_df))

    grouped = permuted_df.groupby("source")
    logger.info("%d unique permuted sources", len(grouped))

    all_results, processed_sources = load_checkpoint(args.checkpoint_file)

    remaining_sources = [src for src in grouped.groups.keys() if src not in processed_sources]
    logger.info("Remaining to process: %d sources", len(remaining_sources))

    if len(remaining_sources) == 0:
        logger.info("All sources already processed!")
        output_df = pd.DataFrame(all_results)
        output_df.to_csv(args.output, index=False)
        logger.info("Saved: %s (%d results)", args.output, len(output_df))
        return

    logger.info("Starting optimized processing...")
    logger.info("  Max paths per pair: %d", MAX_PATHS_PER_PAIR)
    logger.info("  Parallel workers: %d", args.max_workers)
    logger.info("  Batch size: 50 targets")

    process_args = []
    for src in remaining_sources:
        group = grouped.get_group(src)
        process_args.append((src, group, len(remaining_sources)))

    start_time = time.time()
    completed_results = []
    completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_source = {
            executor.submit(process_single_source, a): a[0]
            for a in process_args
        }

        for future in concurrent.futures.as_completed(future_to_source):
            source_name = future_to_source[future]
            completed_count += 1

            try:
                source_results = future.result()
                completed_results.extend(source_results)
                processed_sources.add(source_name)

                elapsed = time.time() - start_time
                avg_time = elapsed / completed_count
                remaining_time = avg_time * (len(remaining_sources) - completed_count)

                logger.info(
                    "Progress: %d/%d | Results: %d | ETA: %.1f min",
                    completed_count, len(remaining_sources),
                    len(all_results) + len(completed_results),
                    remaining_time / 60,
                )

                if completed_count % 10 == 0:
                    current_all_results = all_results + completed_results
                    save_checkpoint(current_all_results, processed_sources, args.checkpoint_file)
                    logger.info("Checkpoint saved")

            except Exception as e:
                logger.error("Failed to process %s: %s", source_name, e)

    final_results = all_results + completed_results
    output_df = pd.DataFrame(final_results)
    output_df.to_csv(args.output, index=False)

    total_time = time.time() - start_time
    logger.info("PROCESSING COMPLETE!")
    logger.info("  Output: %s", args.output)
    logger.info("  Total paths found: %d", len(final_results))
    logger.info("  Unique sources: %d", output_df["source"].nunique())
    logger.info("  Unique targets: %d", output_df["target"].nunique())
    logger.info("  Unique intermediates: %d", output_df["intermediate"].nunique())
    logger.info("  Total time: %.2f hours", total_time / 3600)
    logger.info("  Average: %.1fs per source", total_time / len(remaining_sources))

    if os.path.exists(args.checkpoint_file):
        os.remove(args.checkpoint_file)
        logger.info("Checkpoint file removed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
