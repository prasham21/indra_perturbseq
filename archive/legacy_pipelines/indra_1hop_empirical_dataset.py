"""Superseded legacy script for 1-hop INDRA empirical endothelial dataset analysis.

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

ONEHOP_QUERY = """
MATCH (source:BioEntity {id: $perturbation})-[r:indra_rel]->(target:BioEntity)
WHERE target.id IN $descendants
  AND r.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
WITH
    source.id AS source_id,
    target.id AS target_id,
    collect(DISTINCT r.stmt_type) AS stmt_types,
    max(r.belief) AS max_belief,
    sum(r.evidence_count) AS total_evidence
RETURN
    source_id,
    target_id,
    stmt_types[0] AS stmt_type,
    max_belief AS belief,
    total_evidence AS evidence_count
"""


def save_checkpoint(results, processed_genes, checkpoint_file):
    """Save progress (results + processed gene names)."""
    checkpoint_data = {
        "results": results,
        "processed_genes": processed_genes,
        "timestamp": time.time(),
    }
    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint_data, f)
    logger.info("Checkpoint saved: %d rows, %d genes", len(results), len(processed_genes))


def load_checkpoint(checkpoint_file):
    """Load previous progress if checkpoint exists."""
    try:
        with open(checkpoint_file, "rb") as f:
            data = pickle.load(f)
        saved_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data["timestamp"]))
        logger.info("Resuming from checkpoint saved at %s", saved_time)
        logger.info("  %d rows already collected", len(data["results"]))
        logger.info("  %d genes already processed", len(data["processed_genes"]))
        return data["results"], set(data["processed_genes"])
    except FileNotFoundError:
        logger.info("Starting fresh - no checkpoint found")
        return [], set()


def process_single_gene(args_tuple):
    """Process one perturbation gene for 1-hop analysis.

    Returns a list of dicts with:
    source, target, stmt_type, belief, evidence_count, logfoldchange, pval
    """
    gene_info, endo_genes, deg_dir = args_tuple
    perturb_gene = gene_info["Gene"]
    local_results = []

    try:
        logger.info("Processing: %s", perturb_gene)
        start_time = time.time()

        client = Neo4jClient()

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
        df = df[df["names"].isin(endo_genes)]
        if df.empty:
            logger.warning("No overlapping endothelial genes in DEG file for %s", perturb_gene)
            return []

        gene_symbols = df["names"].dropna().unique().tolist()
        converted = get_valid_gene_ids(gene_symbols)
        symbol_to_hgnc = {k: f"hgnc:{v}" for k, v in zip(gene_symbols, converted) if v}
        hgnc_to_symbol = {v: k for k, v in symbol_to_hgnc.items()}
        target_ids = list(symbol_to_hgnc.values())

        if not target_ids:
            logger.warning("No valid HGNC targets for %s", perturb_gene)
            return []

        deg_map = df.set_index("names")[["logfoldchanges", "pvals"]].to_dict("index")

        results = client.query_tx(
            ONEHOP_QUERY,
            perturbation=source_id,
            descendants=target_ids,
        )

        for r in results:
            _, target_hgnc, stmt_type, belief, ev_count = r
            target_id = target_hgnc
            target_symbol = hgnc_to_symbol.get(target_id, target_id.split(":", 1)[-1])

            if target_symbol not in deg_map:
                continue

            stats = deg_map[target_symbol]
            local_results.append({
                "source": perturb_gene,
                "target": target_symbol,
                "stmt_type": stmt_type,
                "belief": belief,
                "evidence_count": ev_count,
                "logfoldchange": stats["logfoldchanges"],
                "pval": stats["pvals"],
            })

        elapsed = time.time() - start_time
        logger.info("%s: %d edges in %.2f min", perturb_gene, len(local_results), elapsed / 60)
        return local_results

    except Exception as e:
        logger.error("Error with %s: %s", perturb_gene, e)
        return []


def main():
    parser = argparse.ArgumentParser(
        description="1-hop endothelial dataset analysis with checkpointing",
    )
    parser.add_argument("--endo-file", required=True, help="Endothelial gene list CSV")
    parser.add_argument("--perturb-file", required=True, help="Perturbation CSV")
    parser.add_argument("--deg-dir", required=True, help="Directory with DEG CSVs")
    parser.add_argument(
        "--checkpoint-file", default="indra_1hop_endothelial_checkpoint.pkl",
        help="Checkpoint file path",
    )
    parser.add_argument(
        "--output", default="indra_1hop_all_perturbations_endothelial_dataset.csv",
        help="Output CSV path",
    )
    parser.add_argument("--max-workers", type=int, default=6, help="Number of parallel workers")
    parser.add_argument(
        "--checkpoint-every", type=int, default=100,
        help="Checkpoint after every N new edges",
    )
    args = parser.parse_args()

    endo_df = pd.read_csv(args.endo_file)
    endo_genes = set(endo_df["gene"].astype(str))
    logger.info("Total endothelial-present genes (CSV): %d", len(endo_genes))

    perturb_df = pd.read_csv(args.perturb_file)
    perturb_df = perturb_df[perturb_df["analysis_flag"] == "Use_for_analysis"]
    logger.info("Perturbations selected: %d", len(perturb_df))

    all_results, processed_genes = load_checkpoint(args.checkpoint_file)

    remaining_df = perturb_df[~perturb_df["Gene"].isin(processed_genes)]
    logger.info("Remaining to process: %d genes", len(remaining_df))

    if len(remaining_df) == 0:
        logger.info("All genes already processed from checkpoint!")
        output_df = pd.DataFrame(all_results)
        output_df.to_csv(args.output, index=False)
        logger.info("Results saved to: %s", args.output)
        return

    logger.info("Starting optimized 1-hop endothelial analysis...")
    logger.info("  Parallel workers: %d", args.max_workers)
    logger.info("  Checkpoint frequency: every ~%d new rows", args.checkpoint_every)

    start_time = time.time()
    gene_data = [{"Gene": row["Gene"]} for _, row in remaining_df.iterrows()]
    process_args = [(gene_info, endo_genes, args.deg_dir) for gene_info in gene_data]

    completed_results = []
    completed_count = 0
    last_checkpoint_rows = len(all_results)

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

                total_rows_now = len(all_results) + len(completed_results)
                elapsed = time.time() - start_time
                avg_time = elapsed / max(completed_count, 1)
                remaining_time = avg_time * (len(remaining_df) - completed_count)

                logger.info(
                    "Progress: %d/%d genes | Rows so far: %d | ETA: %.1f min",
                    completed_count, len(remaining_df), total_rows_now,
                    remaining_time / 60,
                )

                if total_rows_now - last_checkpoint_rows >= args.checkpoint_every:
                    current_all_results = all_results + completed_results
                    save_checkpoint(current_all_results, list(processed_genes), args.checkpoint_file)
                    last_checkpoint_rows = total_rows_now

            except Exception as e:
                logger.error("Failed to process %s: %s", gene_name, e)

    final_results = all_results + completed_results
    output_df = pd.DataFrame(final_results)
    output_df.to_csv(args.output, index=False)

    total_time = time.time() - start_time
    logger.info("Optimized 1-hop processing complete!")
    logger.info("  Total results: %d", len(final_results))
    logger.info("  Total time: %.2f hours", total_time / 3600)
    logger.info("  Average: %.2f min per gene", total_time / len(remaining_df) / 60)
    logger.info("  Results saved to: %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
