"""Superseded legacy script for fetching evidence text for 3-hop results.

Refactored into src/indra_perturbseq/pipelines/.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client.queries import get_statements

logger = logging.getLogger(__name__)

logging.getLogger().setLevel(logging.ERROR)

client = Neo4jClient()


def process_identifier(agent_str, *, for_query=False):
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


def fetch_evidence_info(agent1, agent2, stmt_type, max_evidences=30):
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
            return "No evidence found", [], []

        evidences = []
        pmids = set()
        db_sources = OrderedDict()

        for stmt in stmts:
            for idx, ev in enumerate(stmt.evidence or [], 1):
                if ev.text:
                    evidences.append(f"{idx}) {ev.text.strip()}")
                if ev.pmid:
                    pmids.add(str(ev.pmid))
                source_api = ev.source_api or ""
                source_sub_id = ev.annotations.get("source_sub_id", "") if ev.annotations else ""
                key = f"{source_api}:{source_sub_id}" if source_sub_id else source_api
                if key:
                    db_sources[key] = None

        if evidences:
            evidence_text = "\n\n".join(evidences)
        elif db_sources:
            evidence_text = f"Evidence from: {', '.join(db_sources.keys())}"
        else:
            evidence_text = "No evidence found"

        return evidence_text, sorted(pmids), list(db_sources.keys())

    except Exception as e:
        return f"Error: {e}", [], []


def process_row(idx, row, max_evidences=30):
    try:
        ev1, pmid1, _ = fetch_evidence_info(row["source"], row["intermediate_1"], row["stmt_type_1"], max_evidences)
        ev2, pmid2, _ = fetch_evidence_info(row["intermediate_1"], row["intermediate_2"], row["stmt_type_2"], max_evidences)
        ev3, pmid3, _ = fetch_evidence_info(row["intermediate_2"], row["target"], row["stmt_type_3"], max_evidences)
        return {
            "idx": idx,
            "evidence_text_hop1": ev1,
            "evidence_text_hop2": ev2,
            "evidence_text_hop3": ev3,
            "pmids_hop1": "; ".join(pmid1),
            "pmids_hop2": "; ".join(pmid2),
            "pmids_hop3": "; ".join(pmid3),
        }
    except Exception as e:
        return {
            "idx": idx,
            "evidence_text_hop1": f"Error: {e}",
            "evidence_text_hop2": "",
            "evidence_text_hop3": "",
            "pmids_hop1": "",
            "pmids_hop2": "",
            "pmids_hop3": "",
        }


def main():
    parser = argparse.ArgumentParser(description="Fetch evidence text for 3-hop results")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--max-evidences", type=int, default=30)
    args = parser.parse_args()

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    start = time.time()
    df = pd.read_csv(args.input)

    new_cols = [
        "evidence_text_hop1", "evidence_text_hop2", "evidence_text_hop3",
        "pmids_hop1", "pmids_hop2", "pmids_hop3",
    ]
    for col in new_cols:
        if col not in df.columns:
            df[col] = ""

    if args.checkpoint_dir:
        checkpoints = [f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pkl")]
        if checkpoints:
            latest_cp = max(checkpoints, key=lambda x: int(re.search(r"\d+", x).group()))
            cp_path = os.path.join(args.checkpoint_dir, latest_cp)
            logger.info("Resuming from checkpoint: %s", cp_path)
            df = pd.read_pickle(cp_path)
        else:
            logger.info("No checkpoint found, starting fresh.")

    total_rows = len(df)
    logger.info("Total rows: %d -- Using %d threads...", total_rows, args.max_workers)

    pending_rows = [
        (idx, row) for idx, row in df.iterrows()
        if pd.isna(row["evidence_text_hop1"]) or row["evidence_text_hop1"] == ""
    ]

    logger.info("Rows to process: %d", len(pending_rows))

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(process_row, idx, row, args.max_evidences)
            for idx, row in pending_rows
        ]

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            idx = result.pop("idx")
            for k, v in result.items():
                df.at[idx, k] = v

            completed += 1
            if completed % args.log_every == 0:
                logger.info("Processed %d rows...", completed)

            if args.checkpoint_dir and completed % args.checkpoint_every == 0:
                cp_file = os.path.join(args.checkpoint_dir, f"checkpoint_row_{completed}.pkl")
                df.to_pickle(cp_file)
                logger.info("Checkpoint saved: %s", cp_file)

    df.to_csv(args.output, index=False)
    total_time = (time.time() - start) / 60
    logger.info("Completed %d rows in %.2f minutes.", total_rows, total_time)
    logger.info("Final output saved to: %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
