import pandas as pd
import os
import re
import time
import logging
import json
from collections import OrderedDict
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client.queries import get_statements
from indra.databases import hgnc_client

# ==============================
# CONFIGURATION
# ==============================
INPUT_FILE = "/Users/prashammarfatia/Downloads/indra_1hop_gene_names_cleaned.csv"
OUTPUT_FILE = "/Users/prashammarfatia/Downloads/indra_1hop_with_statements_7.csv"

logging.getLogger().setLevel(logging.ERROR)
client = Neo4jClient()


# ==============================
# IDENTIFIER + GENE NAME CLEANUP
# ==============================
def normalize_gene_symbol(symbol):
    if not symbol or pd.isna(symbol):
        return symbol
    hgnc_id = hgnc_client.get_current_hgnc_id(symbol)
    if hgnc_id:
        valid_name = hgnc_client.get_hgnc_name(hgnc_id)
        return valid_name or symbol
    return symbol


def process_identifier(agent_str, *, for_query=False):
    if not agent_str or pd.isna(agent_str):
        return None if for_query else agent_str
    agent_str = str(agent_str).strip()
    return agent_str if for_query else f"HGNC:{agent_str}"


# ==============================
# DATABASE SOURCE + PMID EXTRACTION
# ==============================
def get_evidence_info(agent1, agent2, stmt_type):
    try:
        hgnc_id1 = hgnc_client.get_current_hgnc_id(agent1)
        hgnc_id2 = hgnc_client.get_current_hgnc_id(agent2)

        if not hgnc_id1 or not hgnc_id2:
            return "No evidence found", []

        source_id = f"hgnc:{hgnc_id1}"
        target_id = f"hgnc:{hgnc_id2}"

        query = """
        MATCH (source:BioEntity {id: $source_id})-[r:indra_rel {stmt_type: $stmt_type}]->(target:BioEntity {id: $target_id})
        WITH r.stmt_hash as stmt_hash
        MATCH (e:Evidence {stmt_hash: stmt_hash})
        RETURN e.evidence
        """

        results = client.query_tx(query, source_id=source_id, target_id=target_id, stmt_type=stmt_type)

        if not results:
            return "No evidence found", []

        sources_seen = OrderedDict()
        pmids_seen = set()

        for result in results:
            try:
                evidence_data = json.loads(result[0])
                pmid = evidence_data.get('pmid')
                if pmid:
                    pmids_seen.add(str(pmid))

                source_api = evidence_data.get('source_api', '')
                source_sub_id = evidence_data.get('annotations', {}).get('source_sub_id', '')

                if source_sub_id:
                    key = f"{source_api}:{source_sub_id}"
                elif source_api:
                    key = source_api
                else:
                    key = None

                if key:
                    sources_seen[key] = None

            except json.JSONDecodeError:
                continue

        db_info = (
            f"Evidence from: {', '.join(sources_seen.keys())}"
            if sources_seen else "No evidence found"
        )

        pmids = sorted(pmids_seen, key=lambda x: int(x) if x.isdigit() else x)
        return db_info, pmids

    except Exception:
        return "No evidence found", []


def get_database_source(agent1, agent2, stmt_type):
    db_info, _ = get_evidence_info(agent1, agent2, stmt_type)
    return db_info


# ==============================
# EVIDENCE FETCHING
# ==============================
def fetch_evidence_text(agent1, agent2, stmt_type):
    try:
        stmts = get_statements(
            agent=process_identifier(agent1, for_query=True),
            other_agent=process_identifier(agent2, for_query=True),
            agent_role="subject",
            other_role="object",
            rel_types=stmt_type,
            limit=50,
            evidence_limit=50,
            client=client
        )

        if not stmts:
            return get_database_source(agent1, agent2, stmt_type)

        evidences = []
        for stmt in stmts:
            for idx, ev in enumerate(stmt.evidence or [], 1):
                if ev.text:
                    evidences.append(f"{idx}. {ev.text.strip()}")

        if evidences:
            return "\n".join(evidences)
        else:
            return get_database_source(agent1, agent2, stmt_type)

    except Exception as e:
        return f"Error fetching evidence: {e}"


# ==============================
# TEXT FORMATTING
# ==============================
def format_evidence_text(text):
    if not isinstance(text, str):
        return text

    if text.startswith("Evidence from:") or text.startswith("No evidence found"):
        return text

    pattern = r'(^|\n|; )(\d+\.\s)'
    parts = []
    last_idx = 0

    for match in re.finditer(pattern, text):
        start = match.start(2)
        if start > last_idx:
            parts.append(text[last_idx:start].strip())
        last_idx = start

    parts.append(text[last_idx:].strip())

    formatted = [re.sub(r'^(\d+)\.\s', r'\1) ', part) for part in parts]
    return "\n\n".join(formatted)


# ==============================
# MAIN
# ==============================
def main():
    start_time = time.time()
    print("Loading input...")
    df = pd.read_excel(INPUT_FILE) if INPUT_FILE.endswith(('.xls', '.xlsx')) else pd.read_csv(INPUT_FILE)
    print(f"Total rows: {len(df)}")

    print("Normalizing gene names...")
    df["source"] = df["source"].apply(normalize_gene_symbol)
    df["target"] = df["target"].apply(normalize_gene_symbol)

    if "evidence_text" not in df.columns:
        df["evidence_text"] = ""
    if "pmids" not in df.columns:
        df["pmids"] = ""

    print("Extracting evidence text and PMIDs...")
    for idx, row in df.iterrows():
        if pd.isna(row["evidence_text"]) or not str(row["evidence_text"]).strip():
            evidence_text = fetch_evidence_text(row["source"], row["target"], row["stmt_type"])
            df.at[idx, "evidence_text"] = evidence_text

        if pd.isna(row["pmids"]) or not str(row["pmids"]).strip():
            _, pmids = get_evidence_info(row["source"], row["target"], row["stmt_type"])
            pmids_str = "; ".join(pmids) if pmids else ""
            df.at[idx, "pmids"] = pmids_str

        if idx % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)} rows")

    print("Formatting evidence text...")
    df["evidence_text"] = df["evidence_text"].apply(format_evidence_text)

    # Reorder columns
    if 'statement_text' in df.columns:
        cols = df.columns.tolist()
        stmt_idx = cols.index('statement_text')
        for col in ['evidence_text', 'pmids']:
            if col in cols:
                cols.remove(col)
        cols.insert(stmt_idx + 1, 'evidence_text')
        cols.insert(stmt_idx + 2, 'pmids')
        df = df[cols]

    df['pmids'] = df['pmids'].astype(str).replace('nan', '').replace('None', '')

    df.to_csv(OUTPUT_FILE, index=False)

    total_time = (time.time() - start_time) / 60
    pmid_count = sum(1 for x in df["pmids"] if x.strip())
    db_evidence_count = sum(1 for x in df["evidence_text"] if str(x).startswith("Evidence from:") or str(x) == "No evidence found")

    print(f"\nOutput saved to: {OUTPUT_FILE}")
    print(f"Time elapsed: {total_time:.2f} minutes")
    print("Summary:")
    print(f"   Rows with PMIDs: {pmid_count}/{len(df)} ({pmid_count/len(df)*100:.1f}%)")
    print(f"   Rows with fallback evidence: {db_evidence_count}/{len(df)} ({db_evidence_count/len(df)*100:.1f}%)")


if __name__ == "__main__":
    main()
