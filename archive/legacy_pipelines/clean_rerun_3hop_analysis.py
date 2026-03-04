"""Superseded legacy script for clean re-run of 3-hop analysis.

Re-queries contaminated source-target pairs (MESH/ChEBI intermediates)
with exclusion filters applied.

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


def identify_contaminated_pairs(results_path):
    """Get contaminated source-target pairs only."""
    if not os.path.exists(results_path):
        logger.error("Results file not found: %s", results_path)
        return None, None

    logger.info("Loading existing results to identify contaminated source-target pairs...")
    df = pd.read_csv(results_path)
    logger.info("Total results: %d", len(df))

    def has_contamination(row):
        def contains_mesh_or_chebi(value):
            if not value or not isinstance(value, str):
                return False
            value_lower = value.lower()
            return "mesh:" in value_lower or "chebi:" in value_lower

        return (
            contains_mesh_or_chebi(row["intermediate_1"])
            or contains_mesh_or_chebi(row["intermediate_2"])
        )

    contaminated_df = df[df.apply(has_contamination, axis=1)]
    clean_df = df[~df.apply(has_contamination, axis=1)]

    logger.info("Contaminated rows: %d (%.1f%%)", len(contaminated_df), len(contaminated_df) / len(df) * 100)
    logger.info("Clean rows: %d (%.1f%%)", len(clean_df), len(clean_df) / len(df) * 100)

    contaminated_pairs = contaminated_df[["source", "target", "logfoldchange", "pval"]].copy()
    logger.info("Contaminated source-target pairs to re-query: %d", len(contaminated_pairs))

    return contaminated_pairs, clean_df


def save_checkpoint(results, processed_pairs, checkpoint_file="clean_rerun_checkpoint.pkl"):
    """Save progress."""
    checkpoint_data = {
        "results": results,
        "processed_pairs": processed_pairs,
        "timestamp": time.time(),
    }
    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint_data, f)


def load_checkpoint(checkpoint_file="clean_rerun_checkpoint.pkl"):
    """Load previous progress."""
    try:
        with open(checkpoint_file, "rb") as f:
            data = pickle.load(f)
            saved_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data["timestamp"]))
            logger.info("Resuming from checkpoint saved at %s", saved_time)
            logger.info("  %d results already collected", len(data["results"]))
            logger.info("  %d pairs already processed", len(data["processed_pairs"]))
            return data["results"], data["processed_pairs"]
    except FileNotFoundError:
        logger.info("Starting fresh - no checkpoint found")
        return [], set()


def process_contaminated_pairs_for_gene(args_tuple):
    """Process contaminated source-target pairs for one source gene."""
    gene_info, contaminated_pairs_for_gene = args_tuple
    source_gene = gene_info["source"]

    client = Neo4jClient()
    local_results = []
    failed_pairs = []

    try:
        logger.info("Processing clean queries for: %s", source_gene)
        start_time = time.time()

        hgnc_id = get_current_hgnc_id(source_gene.upper())
        if not hgnc_id:
            logger.warning("No HGNC ID for %s", source_gene)
            return [], []
        source_id = f"hgnc:{hgnc_id}"

        logger.info("  Found %d contaminated pairs for %s", len(contaminated_pairs_for_gene), source_gene)

        target_symbols = contaminated_pairs_for_gene["target"].unique().tolist()
        converted = get_valid_gene_ids(target_symbols)
        symbol_to_hgnc = {k: f"hgnc:{v}" for k, v in zip(target_symbols, converted) if v}
        hgnc_to_symbol = {v: k for k, v in symbol_to_hgnc.items()}

        if not symbol_to_hgnc:
            logger.warning("No valid target HGNC IDs for %s", source_gene)
            return [], []

        logger.info("  Querying %d clean targets with 20min timeout per query...", len(symbol_to_hgnc))

        deg_lookup = {}
        for _, row in contaminated_pairs_for_gene.iterrows():
            deg_lookup[row["target"]] = {
                "logfoldchange": row["logfoldchange"],
                "pval": row["pval"],
            }

        for target_symbol, target_hgnc in symbol_to_hgnc.items():
            try:
                results = client.query_tx(
                    CLEAN_INDIVIDUAL_3HOP_QUERY,
                    source=source_id,
                    target=target_hgnc,
                    timeout=1200,
                )

                for r in results:
                    if len(r) >= 13:
                        _, m1_hgnc, m2_hgnc, _, stmt1, stmt2, stmt3, belief1, belief2, belief3, ev1, ev2, ev3 = r

                        try:
                            m1_id = m1_hgnc.replace("hgnc:", "")
                            raw_symbol_1 = get_hgnc_name(m1_id)
                            m1_symbol = (
                                get_hgnc_name(get_current_hgnc_id(raw_symbol_1))
                                if raw_symbol_1
                                else f"hgnc:{m1_id}"
                            )

                            m2_id = m2_hgnc.replace("hgnc:", "")
                            raw_symbol_2 = get_hgnc_name(m2_id)
                            m2_symbol = (
                                get_hgnc_name(get_current_hgnc_id(raw_symbol_2))
                                if raw_symbol_2
                                else f"hgnc:{m2_id}"
                            )
                        except Exception:
                            m1_symbol = m1_hgnc
                            m2_symbol = m2_hgnc

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
                            "pval": deg_stats["pval"],
                        })
                        break

            except Exception as e:
                error_msg = str(e)
                if "TransactionTimedOut" in error_msg or "timeout" in error_msg.lower():
                    logger.warning("TIMEOUT: %s -> %s (>20 min)", source_gene, target_symbol)
                else:
                    logger.warning("FAILED: %s -> %s: %s", source_gene, target_symbol, e)

                failed_pairs.append({
                    "source": source_gene,
                    "target": target_symbol,
                    "error": error_msg,
                    "error_type": "timeout" if "timeout" in error_msg.lower() else "other",
                })
                continue

        elapsed = time.time() - start_time
        success_rate = len(local_results) / len(symbol_to_hgnc) * 100 if symbol_to_hgnc else 0
        logger.info(
            "  %s: %d clean results, %d failed (%.1f%% success) in %.1f min",
            source_gene, len(local_results), len(failed_pairs), success_rate, elapsed / 60,
        )

        return local_results, failed_pairs

    except Exception as e:
        logger.error("Error with %s: %s", source_gene, e)
        return [], []


def main():
    parser = argparse.ArgumentParser(
        description="Clean re-run of 3-hop analysis excluding MESH/ChEBI intermediates",
    )
    parser.add_argument("--results-csv", required=True, help="Existing 3-hop results CSV")
    parser.add_argument("--output", default="indra_3hop_cleaned_results.csv")
    parser.add_argument("--failed-output", default="failed_clean_queries.csv")
    parser.add_argument("--checkpoint-file", default="clean_rerun_checkpoint.pkl")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    logger.info("CLEAN RE-RUN ANALYSIS")
    logger.info("Re-querying only contaminated source-target pairs with MESH/ChEBI exclusion")

    contaminated_pairs, clean_results_df = identify_contaminated_pairs(args.results_csv)

    if contaminated_pairs is None:
        return

    if len(contaminated_pairs) == 0:
        logger.info("No contaminated pairs found! All results are already clean.")
        return

    contaminated_by_source = contaminated_pairs.groupby("source")
    unique_sources = list(contaminated_by_source.groups.keys())

    logger.info("Source genes with contaminated pairs:")
    for source in unique_sources[:10]:
        count = len(contaminated_by_source.get_group(source))
        logger.info("  %s: %d contaminated targets", source, count)
    if len(unique_sources) > 10:
        logger.info("  ... and %d more genes", len(unique_sources) - 10)

    all_results, processed_pairs = load_checkpoint(args.checkpoint_file)

    processed_sources = {pair.split("->")[0] for pair in processed_pairs if "->" in pair}
    remaining_sources = [s for s in unique_sources if s not in processed_sources]

    logger.info("Processing status:")
    logger.info("  Total source genes with contamination: %d", len(unique_sources))
    logger.info("  Already processed: %d", len(unique_sources) - len(remaining_sources))
    logger.info("  Remaining to process: %d", len(remaining_sources))

    if len(remaining_sources) == 0:
        logger.info("All contaminated sources already processed!")
        final_results = clean_results_df.to_dict("records") + all_results
        output_df = pd.DataFrame(final_results)
        output_df.to_csv(args.output, index=False, header=True)
        logger.info("Final cleaned results saved: %d total pathways", len(final_results))
        return

    logger.info("Starting CLEAN 3-hop analysis...")
    logger.info("  Individual queries with 20min timeout per query")
    logger.info("  Parallel workers: %d", args.max_workers)
    logger.info("  Filters: NO mesh: or chebi: intermediate nodes")
    logger.info("  Belief filtering: > 0.7 for all relationships")

    start_time = time.time()
    process_args = []
    for source in remaining_sources:
        source_pairs = contaminated_by_source.get_group(source)
        process_args.append(({"source": source}, source_pairs))

    completed_results = []
    all_failed_pairs = []
    completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_source = {
            executor.submit(process_contaminated_pairs_for_gene, a): a[0]["source"]
            for a in process_args
        }

        for future in concurrent.futures.as_completed(future_to_source):
            source_name = future_to_source[future]
            completed_count += 1

            try:
                source_results, failed_pairs = future.result()
                completed_results.extend(source_results)
                all_failed_pairs.extend(failed_pairs)

                source_pairs = contaminated_by_source.get_group(source_name)
                for _, pair in source_pairs.iterrows():
                    processed_pairs.add(f"{pair['source']}->{pair['target']}")

                elapsed = time.time() - start_time
                avg_time = elapsed / completed_count
                remaining_time = avg_time * (len(remaining_sources) - completed_count)

                logger.info(
                    "Progress: %d/%d | Clean results: %d | Failed pairs: %d | ETA: %.1f min",
                    completed_count, len(remaining_sources),
                    len(all_results) + len(completed_results),
                    len(all_failed_pairs), remaining_time / 60,
                )

                if completed_count % 5 == 0:
                    current_all_results = all_results + completed_results
                    save_checkpoint(current_all_results, processed_pairs, args.checkpoint_file)
                    logger.info("Checkpoint saved")

            except Exception as e:
                logger.error("Failed to process %s: %s", source_name, e)

    if all_failed_pairs:
        failed_df = pd.DataFrame(all_failed_pairs)
        failed_df.to_csv(args.failed_output, index=False)
        logger.info("Failed queries logged to: %s (%d pairs)", args.failed_output, len(all_failed_pairs))

        failure_types = failed_df["error_type"].value_counts()
        logger.info("Failure breakdown:")
        for error_type, count in failure_types.items():
            logger.info("  %s: %d pairs", error_type, count)

    final_new_results = all_results + completed_results
    final_combined_results = clean_results_df.to_dict("records") + final_new_results

    output_df = pd.DataFrame(final_combined_results)
    output_df.to_csv(args.output, index=False, header=True)

    total_time = time.time() - start_time
    logger.info("Clean re-run complete!")
    logger.info("  Original clean results: %d", len(clean_results_df))
    logger.info("  New clean results: %d", len(final_new_results))
    logger.info("  Total final results: %d", len(final_combined_results))
    logger.info("  Processing time: %.1f hours", total_time / 3600)
    logger.info("  Average: %.1f min per source gene", total_time / len(remaining_sources) / 60)
    logger.info("  Results saved to: %s", args.output)

    original_contaminated = len(contaminated_pairs)
    clean_alternatives_found = len(final_new_results)
    success_rate = (clean_alternatives_found / original_contaminated * 100) if original_contaminated > 0 else 0
    logger.info(
        "  Success rate: %d/%d (%.1f%%) contaminated pairs now have clean alternatives",
        clean_alternatives_found, original_contaminated, success_rate,
    )

    if os.path.exists(args.checkpoint_file):
        os.remove(args.checkpoint_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
