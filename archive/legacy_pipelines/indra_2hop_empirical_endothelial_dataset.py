"""Superseded legacy script for 2-hop resume/append/dedup endothelial dataset.

Loads an existing checkpoint, processes remaining genes with 2-hop queries
(up to 10 candidates per (source,target)), appends new rows, and writes
a deduplicated CSV output with one row per unique (source, target) pair.

Refactored into src/indra_perturbseq/pipelines/.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import pickle
import time
from threading import local
from typing import Any

import pandas as pd

from indra_cogex.analysis.source_targets_explanation import get_valid_gene_ids
from indra_cogex.client.neo4j_client import Neo4jClient
from indra.databases.hgnc_client import get_hgnc_name, get_current_hgnc_id

logger = logging.getLogger(__name__)

logging.getLogger("indra_cogex").setLevel(logging.ERROR)
logging.getLogger("indra_cogex.analysis.source_targets_explanation").setLevel(logging.ERROR)

_thread_state = local()


def get_thread_client() -> Neo4jClient:
    if not hasattr(_thread_state, "client") or _thread_state.client is None:
        _thread_state.client = Neo4jClient()
    return _thread_state.client


TWOHOP_UPTO10_BATCH_QUERY = """
MATCH (a:BioEntity {id: $source})
MATCH (a)-[r1:indra_rel]->(m:BioEntity)-[r2:indra_rel]->(b:BioEntity)
WHERE b.id IN $target_list
  AND r2.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']

  AND m.id STARTS WITH 'hgnc:'
  AND NOT (
      m.id CONTAINS 'mesh:' OR m.id CONTAINS 'chebi:' OR
      m.id CONTAINS 'go:'   OR m.id CONTAINS 'uniprot:'
  )

WITH a, b,
     collect([
        m.id,
        r1.stmt_type, r2.stmt_type,
        r1.belief, r2.belief,
        r1.evidence_count, r2.evidence_count
     ])[0..10] AS ps
UNWIND ps AS p
RETURN
  a.id AS source_id,
  b.id AS target_id,
  p[0] AS intermediate_id,
  p[1] AS stmt_type_1, p[2] AS stmt_type_2,
  p[3] AS belief_1,   p[4] AS belief_2,
  p[5] AS evidence_1, p[6] AS evidence_2
"""

OUTPUT_COLUMNS = [
    "source",
    "intermediate",
    "target",
    "stmt_type_1",
    "stmt_type_2",
    "belief_1",
    "belief_2",
    "evidence_1",
    "evidence_2",
    "logfoldchange",
    "pval",
]


def save_checkpoint(
    results: list[dict[str, Any]],
    processed_genes: set,
    checkpoint_file: str,
):
    checkpoint_data = {
        "results": results,
        "processed_genes": list(processed_genes),
        "timestamp": time.time(),
    }
    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint_data, f)
    logger.info("Checkpoint saved: %d rows, %d genes", len(results), len(processed_genes))


def load_checkpoint(checkpoint_file: str) -> tuple[list[dict[str, Any]], set]:
    """Load both results and processed_genes, preserving old correct rows."""
    try:
        with open(checkpoint_file, "rb") as f:
            data = pickle.load(f)
        saved_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data["timestamp"]))
        results = data.get("results", [])
        processed = set(data.get("processed_genes", []))
        logger.info("Resuming from checkpoint saved at %s", saved_time)
        logger.info("  %d rows already collected", len(results))
        logger.info("  %d genes already processed", len(processed))
        return results, processed
    except FileNotFoundError:
        logger.info("Starting fresh - no checkpoint found")
        return [], set()


def normalize_row_schema(row: dict[str, Any]) -> dict[str, Any]:
    """Ensure each row has exactly the output columns."""
    return {c: row.get(c, None) for c in OUTPUT_COLUMNS}


def safe_symbol_from_hgnc_curie(curie: str) -> str:
    """Convert 'hgnc:####' to HGNC symbol if possible, else fallback to numeric id."""
    if isinstance(curie, str) and curie.startswith("hgnc:"):
        hid = curie.replace("hgnc:", "")
        sym = get_hgnc_name(hid)
        return sym or hid
    return str(curie)


def process_single_gene(gene_name: str, deg_dir: str, batch_size: int) -> list[dict[str, Any]]:
    """For one perturbation gene, query 2-hop up to 10 candidates per target."""
    gene_name = str(gene_name).strip()
    if not gene_name:
        return []

    try:
        logger.info("Processing: %s", gene_name)
        start_time = time.time()

        src_hgnc_id = get_current_hgnc_id(gene_name.upper())
        if not src_hgnc_id:
            logger.warning("No HGNC ID for %s", gene_name)
            return []
        source_id = f"hgnc:{src_hgnc_id}"

        deg_path = os.path.join(deg_dir, f"{gene_name}_vs_control.csv")
        if not os.path.exists(deg_path):
            logger.warning("DEG file not found for %s", gene_name)
            return []

        df = pd.read_csv(deg_path, low_memory=False)
        if "names" not in df.columns:
            logger.warning("DEG file missing 'names' column for %s", gene_name)
            return []

        df["names"] = df["names"].astype(str).str.strip()
        target_symbols = df["names"].dropna().unique().tolist()
        if not target_symbols:
            logger.warning("No targets in DEG file for %s", gene_name)
            return []

        converted = get_valid_gene_ids(target_symbols)
        symbol_to_hgnc = {sym: f"hgnc:{hid}" for sym, hid in zip(target_symbols, converted) if hid}
        if not symbol_to_hgnc:
            logger.warning("No valid HGNC targets for %s", gene_name)
            return []

        hgnc_to_symbol = {v: k for k, v in symbol_to_hgnc.items()}
        target_ids = list(symbol_to_hgnc.values())

        if "logfoldchanges" in df.columns:
            df["logfoldchanges"] = pd.to_numeric(df["logfoldchanges"], errors="coerce")
        if "pvals" in df.columns:
            df["pvals"] = pd.to_numeric(df["pvals"], errors="coerce")

        cols = []
        if "logfoldchanges" in df.columns:
            cols.append("logfoldchanges")
        if "pvals" in df.columns:
            cols.append("pvals")
        deg_map = df.set_index("names")[cols].to_dict("index") if cols else {}

        client = get_thread_client()
        out_rows: list[dict[str, Any]] = []

        total_batches = (len(target_ids) + batch_size - 1) // batch_size

        for bi, i in enumerate(range(0, len(target_ids), batch_size), start=1):
            batch_targets = target_ids[i:i + batch_size]
            logger.debug("%s: batch %d/%d (targets=%d)", gene_name, bi, total_batches, len(batch_targets))

            rows = client.query_tx(
                TWOHOP_UPTO10_BATCH_QUERY,
                source=source_id,
                target_list=batch_targets,
            )
            logger.debug("%s: batch %d -> returned %d rows", gene_name, bi, len(rows))

            for r in rows:
                source_out, target_id, intermediate_id, stmt1, stmt2, belief1, belief2, ev1, ev2 = r[:9]

                interm_symbol = safe_symbol_from_hgnc_curie(str(intermediate_id))
                target_symbol = hgnc_to_symbol.get(str(target_id), str(target_id).replace("hgnc:", ""))

                stats = deg_map.get(target_symbol, {})

                out_rows.append(normalize_row_schema({
                    "source": gene_name,
                    "intermediate": str(interm_symbol),
                    "target": str(target_symbol),
                    "stmt_type_1": stmt1,
                    "stmt_type_2": stmt2,
                    "belief_1": belief1,
                    "belief_2": belief2,
                    "evidence_1": ev1,
                    "evidence_2": ev2,
                    "logfoldchange": stats.get("logfoldchanges", None),
                    "pval": stats.get("pvals", None),
                }))

        elapsed = time.time() - start_time
        logger.info("%s: wrote %d candidate rows in %.2f min", gene_name, len(out_rows), elapsed / 60)
        return out_rows

    except Exception as e:
        logger.error("Error with %s: %s", gene_name, e)
        return []


def dedup_one_row_per_source_target(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Dedup to one row per unique (source, target) pair. Keeps first occurrence."""
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    norm_rows = [normalize_row_schema(r) for r in rows]

    df = pd.DataFrame(norm_rows, columns=OUTPUT_COLUMNS)
    df["_row_order"] = range(len(df))
    df = df.sort_values("_row_order")
    df = df.drop_duplicates(subset=["source", "target"], keep="first")
    df = df.drop(columns=["_row_order"])
    return df


def main():
    parser = argparse.ArgumentParser(
        description="2-hop resume/append/dedup endothelial dataset",
    )
    parser.add_argument("--endo-file", required=True, help="Endothelial gene list CSV")
    parser.add_argument("--deg-dir", required=True, help="Directory with DEG CSVs")
    parser.add_argument("--perturb-file", required=True, help="Perturbation CSV")
    parser.add_argument(
        "--checkpoint-file", default="indra_2hop_endothelial_checkpoint.pkl",
    )
    parser.add_argument(
        "--output", default="indra_2hop_all_perturbations_DEDUP_same_columns.csv",
    )
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=50000)
    args = parser.parse_args()

    endo_df = pd.read_csv(args.endo_file)
    if "gene" not in endo_df.columns:
        raise ValueError(f"Endothelial file must have a 'gene' column. Columns: {endo_df.columns.tolist()}")
    endo_symbols_set = set(endo_df["gene"].astype(str).str.strip()) - {""}
    logger.info("Total endothelial-present genes (CSV symbols): %d", len(endo_symbols_set))

    perturb_df = pd.read_csv(args.perturb_file)
    perturb_df = perturb_df[perturb_df["analysis_flag"] == "Use_for_analysis"]
    genes = perturb_df["Gene"].astype(str).tolist()
    logger.info("Perturbations selected: %d", len(genes))

    all_results, processed_genes = load_checkpoint(args.checkpoint_file)

    remaining_genes = [g for g in genes if g not in processed_genes]
    logger.info("Remaining to process: %d genes", len(remaining_genes))

    logger.info("Starting 2-hop extraction (NO pval filtering; endothelial filtering later)...")
    logger.info("  Parallel workers: %d", args.max_workers)
    logger.info("  Batch size: %d", args.batch_size)
    logger.info("  Checkpoint frequency: every ~%d new rows", args.checkpoint_every)

    start_time = time.time()
    completed_results: list[dict[str, Any]] = []
    completed_count = 0
    last_checkpoint_rows = len(all_results)

    if remaining_genes:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_gene = {
                executor.submit(process_single_gene, g, args.deg_dir, args.batch_size): g
                for g in remaining_genes
            }

            for future in concurrent.futures.as_completed(future_to_gene):
                gene_name = future_to_gene[future]
                completed_count += 1

                gene_rows = future.result()
                completed_results.extend(gene_rows)
                processed_genes.add(gene_name)

                total_rows_now = len(all_results) + len(completed_results)
                elapsed = time.time() - start_time
                avg_time = elapsed / max(completed_count, 1)
                remaining_time = avg_time * (len(remaining_genes) - completed_count)

                logger.info(
                    "Progress: %d/%d genes | Rows so far: %d | ETA: %.1f min",
                    completed_count, len(remaining_genes), total_rows_now,
                    remaining_time / 60,
                )

                if total_rows_now - last_checkpoint_rows >= args.checkpoint_every:
                    current_all_results = all_results + completed_results
                    save_checkpoint(current_all_results, processed_genes, args.checkpoint_file)
                    last_checkpoint_rows = total_rows_now

    final_results = all_results + completed_results
    save_checkpoint(final_results, processed_genes, args.checkpoint_file)

    dedup_df = dedup_one_row_per_source_target(final_results)
    dedup_df.to_csv(args.output, index=False)

    total_time = time.time() - start_time
    logger.info("DONE.")
    logger.info("  Total raw rows in checkpoint: %d", len(final_results))
    logger.info("  Total dedup rows (1 per source-target): %d", len(dedup_df))
    logger.info("  Dedup output saved to: %s", args.output)
    logger.info("  Total time this run: %.1f min", total_time / 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
