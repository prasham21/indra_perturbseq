"""Superseded legacy script for fetching evidence text for 2-hop results.

Refactored into src/indra_perturbseq/pipelines/.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from threading import local

import pandas as pd

from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client.queries import get_statements
from indra.databases import hgnc_client

logger = logging.getLogger(__name__)

logging.getLogger().setLevel(logging.ERROR)

_thread_state = local()


def get_thread_client():
    """Reuse one Neo4jClient per thread to avoid exhausting the connection pool."""
    if not hasattr(_thread_state, "client") or _thread_state.client is None:
        _thread_state.client = Neo4jClient()
    return _thread_state.client


def normalize_gene_symbol(symbol):
    if not symbol or pd.isna(symbol):
        return symbol
    hgnc_id = hgnc_client.get_current_hgnc_id(symbol)
    if hgnc_id:
        return hgnc_client.get_hgnc_name(hgnc_id) or symbol
    return symbol


def process_identifier(agent_str, *, for_query=False):
    if not agent_str or pd.isna(agent_str):
        return None if for_query else agent_str
    agent_str = str(agent_str).strip()
    return agent_str if for_query else f"HGNC:{agent_str}"


def get_evidence_info(agent1, agent2, stmt_type, client):
    """Collect source APIs and PMIDs via Cypher on stmt_hash -> Evidence."""
    try:
        hgnc_id1 = hgnc_client.get_current_hgnc_id(agent1)
        hgnc_id2 = hgnc_client.get_current_hgnc_id(agent2)
        if not hgnc_id1 or not hgnc_id2:
            return "No evidence found", []

        query = """
        MATCH (source:BioEntity {id: $source_id})-[r:indra_rel {stmt_type: $stmt_type}]->(target:BioEntity {id: $target_id})
        WITH r.stmt_hash as stmt_hash
        MATCH (e:Evidence {stmt_hash: stmt_hash})
        RETURN e.evidence
        """
        results = client.query_tx(
            query,
            source_id=f"hgnc:{hgnc_id1}",
            target_id=f"hgnc:{hgnc_id2}",
            stmt_type=stmt_type,
        )
        if not results:
            return "No evidence found", []

        sources_seen = OrderedDict()
        pmids_seen = set()

        for result in results:
            try:
                evidence_data = json.loads(result[0])
            except json.JSONDecodeError:
                continue

            pmid = evidence_data.get("pmid")
            if pmid:
                pmids_seen.add(str(pmid))

            source_api = evidence_data.get("source_api", "")
            source_sub_id = evidence_data.get("annotations", {}).get("source_sub_id", "")
            key = f"{source_api}:{source_sub_id}" if source_sub_id else source_api
            if key:
                sources_seen[key] = None

        db_info = f"Evidence from: {', '.join(sources_seen.keys())}" if sources_seen else "No evidence found"
        pmids = sorted(pmids_seen, key=lambda x: int(x) if x.isdigit() else x)
        return db_info, pmids
    except Exception:
        return "No evidence found", []


def get_database_source(agent1, agent2, stmt_type, client):
    db_info, _ = get_evidence_info(agent1, agent2, stmt_type, client)
    return db_info


def fetch_evidence_text(agent1, agent2, stmt_type, client):
    """Pull INDRA statements and format numbered evidence; fallback to sources list."""
    try:
        stmts = get_statements(
            agent=process_identifier(agent1, for_query=True),
            other_agent=process_identifier(agent2, for_query=True),
            agent_role="subject",
            other_role="object",
            rel_types=stmt_type,
            limit=20,
            evidence_limit=20,
            client=client,
        )
        if not stmts:
            return get_database_source(agent1, agent2, stmt_type, client)

        evidences = []
        counter = 1
        for stmt in stmts:
            for ev in (stmt.evidence or []):
                if ev.text:
                    evidences.append(f"{counter}. {ev.text.strip()}")
                    counter += 1

        return "\n".join(evidences) if evidences else get_database_source(agent1, agent2, stmt_type, client)
    except Exception as e:
        return f"Error fetching evidence: {e}"


def format_evidence_text(text):
    """Reformat '1. ...' into '1) ...' with blank lines between numbered items."""
    if not isinstance(text, str):
        return text
    if text.startswith("Evidence from:") or text.startswith("No evidence found"):
        return text
    pattern = r"(^|\n|; )(\d+\.\s)"
    parts = []
    last_idx = 0
    for match in re.finditer(pattern, text):
        start = match.start(2)
        if start > last_idx:
            parts.append(text[last_idx:start].strip())
        last_idx = start
    parts.append(text[last_idx:].strip())
    return "\n\n".join([re.sub(r"^(\d+)\.\s", r"\1) ", p) for p in parts])


def process_row(idx, row):
    client = get_thread_client()
    source = normalize_gene_symbol(row["source"])
    intermediate = normalize_gene_symbol(row["intermediate"])
    target = normalize_gene_symbol(row["target"])
    stmt1 = row["stmt_type_1"]
    stmt2 = row["stmt_type_2"]

    ev1 = fetch_evidence_text(source, intermediate, stmt1, client)
    _, pmids1 = get_evidence_info(source, intermediate, stmt1, client)
    ev2 = fetch_evidence_text(intermediate, target, stmt2, client)
    _, pmids2 = get_evidence_info(intermediate, target, stmt2, client)

    return {
        "idx": idx,
        "evidence_text_hop1": format_evidence_text(ev1),
        "pmids_hop1": "; ".join(pmids1),
        "evidence_text_hop2": format_evidence_text(ev2),
        "pmids_hop2": "; ".join(pmids2),
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch evidence text for 2-hop results")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for checkpoints")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--heartbeat-secs", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=None)
    args = parser.parse_args()

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    start = time.time()

    if args.checkpoint_dir:
        checkpoint_files = sorted([f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pkl")])
        if checkpoint_files:
            latest = checkpoint_files[-1]
            df = pd.read_pickle(os.path.join(args.checkpoint_dir, latest))
            logger.info("Resuming from checkpoint: %s", latest)
        else:
            df = pd.read_csv(args.input)
            logger.info("No checkpoint found. Starting from scratch...")
    else:
        df = pd.read_csv(args.input)
        logger.info("Starting from scratch...")

    for col in ["evidence_text_hop1", "pmids_hop1", "evidence_text_hop2", "pmids_hop2"]:
        if col not in df.columns:
            df[col] = ""

    if args.start_index is not None:
        pending = df.loc[args.start_index:]
    else:
        pending = df

    to_process = pending[
        (pending["evidence_text_hop1"].isna()) | (pending["evidence_text_hop2"].isna()) |
        (pending["evidence_text_hop1"] == "") | (pending["evidence_text_hop2"] == "")
    ]

    total_pending = len(to_process)
    logger.info("Processing %d pending rows using %d threads...", total_pending, args.max_workers)

    completed = 0
    in_flight = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for idx in to_process.index:
            in_flight[executor.submit(process_row, idx, df.loc[idx])] = idx

        last_heartbeat = time.time()

        while in_flight:
            done, not_done = wait(in_flight.keys(), timeout=args.heartbeat_secs, return_when=FIRST_COMPLETED)

            if not done:
                logger.debug(
                    "Still working... %d rows in flight (no completion in last %ds)",
                    len(not_done), args.heartbeat_secs,
                )
                last_heartbeat = time.time()
                continue

            for fut in done:
                idx = in_flight.pop(fut)
                try:
                    result = fut.result()
                    row_idx = result.pop("idx")
                    for k, v in result.items():
                        df.at[row_idx, k] = v
                except Exception as e:
                    df.at[idx, "evidence_text_hop1"] = f"Error: {e}"
                    df.at[idx, "evidence_text_hop2"] = f"Error: {e}"
                    df.at[idx, "pmids_hop1"] = ""
                    df.at[idx, "pmids_hop2"] = ""

                completed += 1
                if completed % args.log_every == 0:
                    logger.info("Processed %d/%d rows...", completed, total_pending)

                if args.checkpoint_dir and completed % args.checkpoint_every == 0:
                    checkpoint_file = os.path.join(args.checkpoint_dir, f"checkpoint_row_{completed}.pkl")
                    df.to_pickle(checkpoint_file)
                    logger.info("Checkpoint saved at row %d -> %s", completed, checkpoint_file)

    df["evidence_text_hop1"] = df["evidence_text_hop1"].apply(format_evidence_text)
    df["evidence_text_hop2"] = df["evidence_text_hop2"].apply(format_evidence_text)
    df["pmids_hop1"] = df["pmids_hop1"].astype(str).replace(["nan", "None"], "")
    df["pmids_hop2"] = df["pmids_hop2"].astype(str).replace(["nan", "None"], "")

    cols = df.columns.tolist()
    evidence_cols = ["evidence_text_hop1", "evidence_text_hop2", "pmids_hop1", "pmids_hop2"]
    for col in evidence_cols:
        if col in cols:
            cols.remove(col)
    if "stmt_type_2" in cols:
        insert_at = cols.index("stmt_type_2") + 1
        for i, col in enumerate(evidence_cols):
            cols.insert(insert_at + i, col)
        df = df[cols]

    df.to_csv(args.output, index=False)

    total = len(df)
    pmid_hop1_count = sum(1 for x in df["pmids_hop1"] if str(x).strip())
    pmid_hop2_count = sum(1 for x in df["pmids_hop2"] if str(x).strip())
    db_evidence_hop1_count = sum(
        1 for x in df["evidence_text_hop1"]
        if isinstance(x, str) and (x.startswith("Evidence from:") or x == "No evidence found")
    )
    db_evidence_hop2_count = sum(
        1 for x in df["evidence_text_hop2"]
        if isinstance(x, str) and (x.startswith("Evidence from:") or x == "No evidence found")
    )

    duration = (time.time() - start) / 60
    logger.info("Output saved to: %s", args.output)
    logger.info("Time taken: %.2f minutes", duration)
    logger.info("Summary:")
    logger.info("  Hop 1 (source -> intermediate):")
    logger.info("    Rows with PMIDs: %d/%d (%.1f%%)", pmid_hop1_count, total, (pmid_hop1_count / total) * 100)
    logger.info(
        "    Rows with fallback evidence: %d/%d (%.1f%%)",
        db_evidence_hop1_count, total, (db_evidence_hop1_count / total) * 100,
    )
    logger.info("  Hop 2 (intermediate -> target):")
    logger.info("    Rows with PMIDs: %d/%d (%.1f%%)", pmid_hop2_count, total, (pmid_hop2_count / total) * 100)
    logger.info(
        "    Rows with fallback evidence: %d/%d (%.1f%%)",
        db_evidence_hop2_count, total, (db_evidence_hop2_count / total) * 100,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
