"""Superseded legacy script for 1-hop permuted data analysis.

Refactored into src/indra_perturbseq/pipelines/.
"""
from __future__ import annotations

import argparse
import logging
import time

import pandas as pd
from indra_cogex.analysis.source_targets_explanation import get_valid_gene_ids
from indra_cogex.client.neo4j_client import Neo4jClient
from indra.databases.hgnc_client import get_hgnc_name, get_current_hgnc_id

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="1-hop permuted data analysis")
    parser.add_argument("--permuted-csv", required=True, help="Master permuted source-target pairs CSV")
    parser.add_argument("--output", default="indra_1hop_PERMUTED.csv", help="Output CSV path")
    args = parser.parse_args()

    permuted_df = pd.read_csv(args.permuted_csv)
    logger.info("Loaded %d permuted pairs", len(permuted_df))

    grouped = permuted_df.groupby("source")
    logger.info("%d unique permuted sources to query", len(grouped))

    client = Neo4jClient()
    all_results = []

    start_time = time.time()
    for idx, (source_gene, group) in enumerate(grouped):
        logger.info("(%d/%d) Processing permuted source: %s", idx + 1, len(grouped), source_gene)

        loop_start = time.time()
        try:
            hgnc_id = get_current_hgnc_id(source_gene.upper())
            if not hgnc_id:
                logger.warning("No HGNC ID found for %s", source_gene)
                continue
            source_id = f"hgnc:{hgnc_id}"

            target_symbols = group["target"].dropna().unique().tolist()
            converted = get_valid_gene_ids(target_symbols)
            target_ids = [f"hgnc:{v}" for v in converted if v]

            if not target_ids:
                logger.warning("No valid target IDs for %s", source_gene)
                continue

            group_dedup = group.drop_duplicates(subset="target", keep="first")
            deg_map = group_dedup.set_index("target")[["logfoldchange", "pval"]].to_dict("index")

            query = """
            MATCH (source:BioEntity {id: $perturbation})-[r:indra_rel]->(target:BioEntity)
            WHERE target.id IN $descendants AND r.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
            RETURN source.id, target.id, r.stmt_type, r.belief, r.evidence_count
            """
            results = client.query_tx(query, perturbation=source_id, descendants=target_ids)

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
                        "pval": deg_map[symbol]["pval"],
                    })

            loop_time = time.time() - loop_start
            avg_time = (time.time() - start_time) / (idx + 1)
            remaining = avg_time * (len(grouped) - idx - 1)
            logger.info(
                "  %d edges found | %.1fs | ETA: %.1f mins",
                len(results), loop_time, remaining / 60,
            )

        except Exception as e:
            logger.error("Error with %s: %s", source_gene, e)

    output_df = pd.DataFrame(all_results)
    output_df.to_csv(args.output, index=False)

    logger.info("COMPLETED: Saved permuted 1-hop results")
    logger.info("  Output file: %s", args.output)
    logger.info("  Total edges found: %d", len(output_df))
    logger.info("  Unique sources: %d", output_df["source"].nunique())
    logger.info("  Unique targets: %d", output_df["target"].nunique())

    logger.info("Quick comparison:")
    logger.info("  Permuted 1-hop connections: %d", len(output_df))
    logger.info("  (Compare this to your real 1-hop file)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
