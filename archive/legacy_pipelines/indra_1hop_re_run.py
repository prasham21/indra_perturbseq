"""Superseded legacy script for 1-hop INDRA re-run with readable statements.

Refactored into src/indra_perturbseq/pipelines/.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import pandas as pd
from indra_cogex.analysis.source_targets_explanation import get_valid_gene_ids
from indra_cogex.client.neo4j_client import Neo4jClient
from indra.databases.hgnc_client import get_hgnc_name, get_current_hgnc_id

logger = logging.getLogger(__name__)


def create_readable_statement_text(stmt_data: dict) -> str:
    """Convert INDRA statement JSON to readable structured text."""
    stmt_type = stmt_data.get("type", "Unknown")

    subj = stmt_data.get("subj", {})
    subj_name = subj.get("name", "Unknown")
    subj_mods = subj.get("mods", [])

    obj = stmt_data.get("obj", {})
    obj_name = obj.get("name", "Unknown")
    obj_mods = obj.get("mods", [])

    def _build_description(name, mods):
        if not mods:
            return name
        mod_descriptions = []
        for mod in mods:
            if mod.get("mod_type") == "phosphorylation":
                residue = mod.get("residue")
                position = mod.get("position")
                if residue and position:
                    mod_descriptions.append(f"phosphorylated on {residue}{position}")
                else:
                    mod_descriptions.append("phosphorylated")
            elif mod.get("mod_type") == "modification":
                mod_descriptions.append("modified")
            else:
                mod_descriptions.append(f"{mod.get('mod_type', 'modified')}")
        return f"{name} ({', '.join(mod_descriptions)})"

    subject_desc = _build_description(subj_name, subj_mods)
    object_desc = _build_description(obj_name, obj_mods)

    if stmt_type == "IncreaseAmount":
        return f"{subject_desc} increases the amount of {object_desc}"
    elif stmt_type == "DecreaseAmount":
        return f"{subject_desc} decreases the amount of {object_desc}"
    else:
        return f"{subject_desc} {stmt_type.lower()} {object_desc}"


def main():
    parser = argparse.ArgumentParser(description="1-hop INDRA re-run with readable statements")
    parser.add_argument("--perturb-csv", required=True, help="Path to perturbation CSV")
    parser.add_argument("--deg-dir", required=True, help="Directory with DEG CSVs")
    parser.add_argument(
        "--output", default="indra_1hop_with_readable_statements.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    perturb_df = pd.read_csv(args.perturb_csv)
    perturb_df = perturb_df[perturb_df["analysis_flag"] == "Use_for_analysis"]
    logger.info("Perturbations selected: %d", len(perturb_df))

    client = Neo4jClient()
    all_results = []

    start_time = time.time()
    for idx, row in perturb_df.iterrows():
        perturb_gene = row["Gene"]
        logger.info("(%d/%d) Processing: %s", idx + 1, len(perturb_df), perturb_gene)

        loop_start = time.time()
        try:
            hgnc_id = get_current_hgnc_id(perturb_gene.upper())
            if not hgnc_id:
                logger.warning("No HGNC ID found for %s", perturb_gene)
                continue
            source_id = f"hgnc:{hgnc_id}"

            csv_path = os.path.join(args.deg_dir, f"{perturb_gene}_vs_control.csv")
            if not os.path.exists(csv_path):
                logger.warning("DEG file not found: %s", csv_path)
                continue
            df = pd.read_csv(csv_path)
            df = df[df["pvals"] < 0.05]

            gene_symbols = df["names"].dropna().unique().tolist()
            converted = get_valid_gene_ids(gene_symbols)
            target_ids = [f"hgnc:{v}" for v in converted if v]

            deg_map = df.set_index("names")[["logfoldchanges", "pvals"]].to_dict("index")

            query = """
            MATCH (source:BioEntity {id: $perturbation})-[r:indra_rel]->(target:BioEntity)
            WHERE target.id IN $descendants AND r.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
            RETURN source.id, target.id, r.stmt_type, r.stmt_json, r.belief, r.evidence_count
            """
            results = client.query_tx(query, perturbation=source_id, descendants=target_ids)

            for r in results:
                _, target_hgnc, stmt_type, stmt_json, belief, ev_count = r
                target_id = target_hgnc.split(":")[1]
                symbol = get_hgnc_name(target_id)
                if symbol and symbol in deg_map:
                    try:
                        stmt_data = json.loads(stmt_json)
                        statement_text = create_readable_statement_text(stmt_data)
                    except Exception as e:
                        statement_text = f"Parsing error: {e!s}"

                    all_results.append({
                        "source": perturb_gene,
                        "target": symbol,
                        "stmt_type": stmt_type,
                        "statement_text": statement_text,
                        "belief": belief,
                        "evidence_count": ev_count,
                        "logfoldchange": deg_map[symbol]["logfoldchanges"],
                        "pval": deg_map[symbol]["pvals"],
                    })

            loop_time = time.time() - loop_start
            avg_time = (time.time() - start_time) / (idx + 1)
            remaining = avg_time * (len(perturb_df) - idx - 1)
            logger.info(
                "%d edges found | %.1fs | ETA: %.1f mins",
                len(results), loop_time, remaining / 60,
            )

        except Exception as e:
            logger.error("Error with %s: %s", perturb_gene, e)

    output_df = pd.DataFrame(all_results)
    output_df.to_csv(args.output, index=False)
    logger.info("Saved all results to %s", args.output)
    logger.info("Total results: %d", len(all_results))
    logger.info("Total time: %.1f minutes", (time.time() - start_time) / 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
