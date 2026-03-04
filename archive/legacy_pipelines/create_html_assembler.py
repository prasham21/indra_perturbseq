"""Legacy script: HTML assembler for GWAS endothelial INDRA statements."""
from __future__ import annotations

import argparse

import pandas as pd
from indra.assemblers.html.assembler import HtmlAssembler
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client.queries import get_statements

import logging


logger = logging.getLogger(__name__)


def find_matching_statement(stmts, target_belief, target_evcnt):
    """
    Identify the statement from INDRA that best matches the CSV info.
    Matching order:
        1. Perfect evidence_count match AND closest belief
        2. Otherwise: closest belief
    """
    best_match = None
    min_belief_diff = float("inf")

    # Pass 1: Evidence count exact match
    for stmt in stmts:
        stmt_evcnt = len(stmt.evidence)
        if stmt_evcnt == target_evcnt:
            diff = abs(stmt.belief - target_belief)
            if diff < min_belief_diff:
                min_belief_diff = diff
                best_match = stmt

    if best_match:
        return best_match

    # Pass 2: No evidence_count match → fallback to closest belief
    for stmt in stmts:
        diff = abs(stmt.belief - target_belief)
        if diff < min_belief_diff:
            min_belief_diff = diff
            best_match = stmt

    return best_match


def main():
    ap = argparse.ArgumentParser(description="Generate INDRA HTML evidence report.")
    ap.add_argument("--input", required=True, help="Input CSV with GWAS endothelial paths.")
    ap.add_argument("--output-csv", required=True, help="Enriched output CSV with INDRA URLs.")
    ap.add_argument("--output-html", required=True, help="Output HTML report path.")
    args = ap.parse_args()

    logger.info("Loading CSV: %s", args.input)
    df = pd.read_csv(args.input)

    client = Neo4jClient()

    hop1_hashes, hop1_urls = [], []
    hop2_hashes, hop2_urls = [], []

    all_statements_for_html = []

    logger.info("Processing rows...")
    for idx, row in df.iterrows():

        src = row["source"]
        mid = row["intermediate"]
        tgt = row["target"]

        stmt1 = row["stmt_type_1"]
        stmt2 = row["stmt_type_2"]

        belief1 = float(row["belief_1"])
        belief2 = float(row["belief_2"])

        evcnt1 = int(row["evidence_1"])
        evcnt2 = int(row["evidence_2"])

        # HOP 1: source → intermediate
        hop1_stmts = get_statements(
            agent=src,
            other_agent=mid,
            rel_types=stmt1,
            evidence_limit=50,
            client=client,
        )

        matched_hop1 = find_matching_statement(hop1_stmts, belief1, evcnt1)

        if matched_hop1:
            h1_hash = matched_hop1.get_hash()
            hop1_hashes.append(h1_hash)
            hop1_urls.append(f"https://db.indra.bio/statements/from_hash/{h1_hash}?format=html")
            all_statements_for_html.append(matched_hop1)
        else:
            hop1_hashes.append("")
            hop1_urls.append("")

        # HOP 2: intermediate → target
        hop2_stmts = get_statements(
            agent=mid,
            other_agent=tgt,
            rel_types=stmt2,
            evidence_limit=50,
            client=client,
        )

        matched_hop2 = find_matching_statement(hop2_stmts, belief2, evcnt2)

        if matched_hop2:
            h2_hash = matched_hop2.get_hash()
            hop2_hashes.append(h2_hash)
            hop2_urls.append(f"https://db.indra.bio/statements/from_hash/{h2_hash}?format=html")
            all_statements_for_html.append(matched_hop2)
        else:
            hop2_hashes.append("")
            hop2_urls.append("")

        if idx % 25 == 0:
            logger.info("Processed %d/%d rows...", idx, len(df))

    df["hop1_hash"] = hop1_hashes
    df["hop1_indra_url"] = hop1_urls
    df["hop2_hash"] = hop2_hashes
    df["hop2_indra_url"] = hop2_urls

    logger.info("Saving enriched CSV: %s", args.output_csv)
    df.to_csv(args.output_csv, index=False)

    logger.info("Generating HtmlAssembler report...")

    uniq = []
    seen = set()
    for s in all_statements_for_html:
        h = s.get_hash()
        if h not in seen:
            uniq.append(s)
            seen.add(h)

    ha = HtmlAssembler(
        statements=uniq,
        title="GWAS 2-Hop INDRA Evidence",
    )

    ha.make_model(grouping_level="statement")
    ha.save_model(args.output_html)

    logger.info("Done! HTML saved to %s", args.output_html)


if __name__ == "__main__":
    main()
