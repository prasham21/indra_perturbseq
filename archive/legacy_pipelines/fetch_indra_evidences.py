"""Superseded legacy script for fetching INDRA evidence text for 3-hop results.

Refactored into src/indra_perturbseq/pipelines/.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client.queries import get_statements

logger = logging.getLogger(__name__)

logging.getLogger().setLevel(logging.ERROR)

client = Neo4jClient()


def process_identifier(agent_str, *, for_query=False):
    """Normalize or parse agent identifiers."""
    if not agent_str or pd.isna(agent_str):
        return None if for_query else agent_str

    agent_str = str(agent_str).strip()
    mappings = {
        "hgnc:uniprot.chain:": "UNIPROT",
        "hgnc:uniprot:": "UNIPROT",
        "hgnc:mesh:": "MESH",
        "hgnc:chebi:": "CHEBI",
        "hgnc:fplx:": "FPLX",
        "hgnc:": "HGNC",
    }

    for prefix, label in mappings.items():
        if agent_str.startswith(prefix):
            value = agent_str.replace(prefix, "")
            return (label, value) if for_query else f"{label}:{value}"

    return agent_str if not for_query else agent_str


def fetch_evidence_text(agent1, agent2, stmt_type, max_evidences=20):
    """Fetch up to max_evidences texts for a given edge."""
    try:
        stmts = get_statements(
            agent=process_identifier(agent1, for_query=True),
            other_agent=process_identifier(agent2, for_query=True),
            agent_role="subject",
            other_role="object",
            rel_types=stmt_type,
            limit=max_evidences,
            evidence_limit=max_evidences,
            client=client,
        )

        if not stmts:
            return "Database evidence only"

        evidences = []
        for stmt in stmts:
            if stmt.evidence:
                for idx, ev in enumerate(stmt.evidence[:max_evidences], 1):
                    if ev.text:
                        evidences.append(f"{idx}) {ev.text.strip()}")

        return "\n\n".join(evidences) if evidences else "Database evidence only"
    except Exception as e:
        return f"Error fetching evidence: {e}"


def process_row(idx, row, max_evidences=20):
    """Process a single CSV row: fetch evidences for its 3 edges."""
    try:
        edge1_text = fetch_evidence_text(row["source"], row["intermediate_1"], row["stmt_type_1"], max_evidences)
        edge2_text = fetch_evidence_text(row["intermediate_1"], row["intermediate_2"], row["stmt_type_2"], max_evidences)
        edge3_text = fetch_evidence_text(row["intermediate_2"], row["target"], row["stmt_type_3"], max_evidences)
        return idx, edge1_text, edge2_text, edge3_text
    except Exception as e:
        return idx, f"Error: {e}", "", ""


def main():
    parser = argparse.ArgumentParser(description="Fetch INDRA evidences for 3-hop results")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-evidences", type=int, default=20)
    args = parser.parse_args()

    start_time = time.time()
    df = pd.read_csv(args.input)

    for col in ["edge1_statements", "edge2_statements", "edge3_statements"]:
        if col not in df.columns:
            df[col] = ""

    if os.path.exists(args.output):
        df_existing = pd.read_csv(args.output)
        df.update(df_existing)
        logger.info("Resuming from checkpoint: %s", args.output)

    total_rows = len(df)
    logger.info("Processing %d rows with %d workers...", total_rows, args.max_workers)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_row, idx, row, args.max_evidences): idx
            for idx, row in df.iterrows()
            if pd.isna(df.at[idx, "edge1_statements"]) or df.at[idx, "edge1_statements"] == ""
        }

        completed = 0
        for future in as_completed(futures):
            idx, edge1_text, edge2_text, edge3_text = future.result()

            df.at[idx, "edge1_statements"] = edge1_text
            df.at[idx, "edge2_statements"] = edge2_text
            df.at[idx, "edge3_statements"] = edge3_text

            completed += 1
            if completed % args.checkpoint_every == 0:
                df.to_csv(args.output, index=False)
                elapsed = (time.time() - start_time) / 60
                logger.info(
                    "Checkpoint saved at row %d/%d (%.2f min elapsed)",
                    completed, total_rows, elapsed,
                )

    df["intermediate_1"] = df["intermediate_1"].apply(lambda x: process_identifier(x, for_query=False))
    df["intermediate_2"] = df["intermediate_2"].apply(lambda x: process_identifier(x, for_query=False))

    df.to_csv(args.output, index=False)
    total_time = (time.time() - start_time) / 60
    logger.info("Processing complete")
    logger.info("Total rows processed: %d", total_rows)
    logger.info("Results saved to: %s", args.output)
    logger.info("Total time taken: %.2f minutes", total_time)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
