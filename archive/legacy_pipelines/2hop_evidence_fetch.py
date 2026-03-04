import pandas as pd
import os
import re
import time
import logging
import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from threading import local

from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client.queries import get_statements
from indra.databases import hgnc_client

# ==============================
# CONFIGURATION
# ==============================
INPUT_FILE = "/Users/prashammarfatia/Downloads/cleaned_indra_2hop_all_perturbations.csv"
OUTPUT_FILE = "/Users/prashammarfatia/Downloads/indra_2hop_with_evidence_statements_.csv"
CHECKPOINT_DIR = "/Users/prashammarfatia/Downloads/2hop_checkpoints/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Concurrency & logging/checkpoint cadence
MAX_WORKERS = 5
HEARTBEAT_SECS = 10
LOG_EVERY = 100
CHECKPOINT_EVERY = 1000

START_INDEX = 8500

logging.getLogger().setLevel(logging.ERROR)

# ==============================
# THREAD-LOCAL NEO4J CLIENT
# ==============================
_thread_state = local()

def get_thread_client():
    """Reuse one Neo4jClient per thread to avoid exhausting the connection pool."""
    if not hasattr(_thread_state, "client") or _thread_state.client is None:
        _thread_state.client = Neo4jClient()
    return _thread_state.client


# ==============================
# SYMBOL NORMALIZATION
# ==============================
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


# ==============================
# EVIDENCE UTILITIES
# ==============================
def get_evidence_info(agent1, agent2, stmt_type, client):
    """Collect source APIs and PMIDs via Cypher on stmt_hash → Evidence."""
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
            stmt_type=stmt_type
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

            pmid = evidence_data.get('pmid')
            if pmid:
                pmids_seen.add(str(pmid))

            source_api = evidence_data.get('source_api', '')
            source_sub_id = evidence_data.get('annotations', {}).get('source_sub_id', '')
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
    """Pull INDRA statements (lighter limits) and format numbered evidence; fallback to sources list."""
    try:
        stmts = get_statements(
            agent=process_identifier(agent1, for_query=True),
            other_agent=process_identifier(agent2, for_query=True),
            agent_role="subject",
            other_role="object",
            rel_types=stmt_type,
            limit=20,            # lighter than 50
            evidence_limit=20,   # lighter than 50
            client=client
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
    """Reformat '1. ...' into '1) ...' with blank lines between numbered items; keep fallback text as-is."""
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
    return "\n\n".join([re.sub(r'^(\d+)\.\s', r'\1) ', p) for p in parts])


# ==============================
# PARALLEL ROW PROCESSOR
# ==============================
def process_row(idx, row):
    client = get_thread_client()  # thread-local reuse
    source = normalize_gene_symbol(row['source'])
    intermediate = normalize_gene_symbol(row['intermediate'])
    target = normalize_gene_symbol(row['target'])
    stmt1 = row['stmt_type_1']
    stmt2 = row['stmt_type_2']

    ev1 = fetch_evidence_text(source, intermediate, stmt1, client)
    _, pmids1 = get_evidence_info(source, intermediate, stmt1, client)
    ev2 = fetch_evidence_text(intermediate, target, stmt2, client)
    _, pmids2 = get_evidence_info(intermediate, target, stmt2, client)

    return {
        'idx': idx,
        'evidence_text_hop1': format_evidence_text(ev1),
        'pmids_hop1': "; ".join(pmids1),
        'evidence_text_hop2': format_evidence_text(ev2),
        'pmids_hop2': "; ".join(pmids2)
    }


# ==============================
# MAIN FUNCTION
# ==============================
def main():
    start = time.time()

    # 1) Load latest checkpoint if present; otherwise CSV
    checkpoint_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pkl")])
    if checkpoint_files:
        latest = checkpoint_files[-1]
        df = pd.read_pickle(os.path.join(CHECKPOINT_DIR, latest))
        print(f"🔁 Resuming from checkpoint: {latest}")
    else:
        df = pd.read_csv(INPUT_FILE)
        print("🆕 No checkpoint found. Starting from scratch...")

    # 2) Ensure result columns exist
    for col in ['evidence_text_hop1', 'pmids_hop1', 'evidence_text_hop2', 'pmids_hop2']:
        if col not in df.columns:
            df[col] = ""

    # 3) Select rows to process (optionally force a start index)
    if START_INDEX is not None:
        pending = df.loc[START_INDEX:]
    else:
        pending = df

    to_process = pending[
        (pending['evidence_text_hop1'].isna()) | (pending['evidence_text_hop2'].isna()) |
        (pending['evidence_text_hop1'] == "") | (pending['evidence_text_hop2'] == "")
    ]

    total_pending = len(to_process)
    print(f"🔧 Processing {total_pending} pending rows using {MAX_WORKERS} threads...")

    # 4) Parallel processing with heartbeat + checkpoints + error marking
    completed = 0
    in_flight = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # submit all tasks
        for idx in to_process.index:
            in_flight[executor.submit(process_row, idx, df.loc[idx])] = idx

        last_heartbeat = time.time()

        while in_flight:
            done, not_done = wait(in_flight.keys(), timeout=HEARTBEAT_SECS, return_when=FIRST_COMPLETED)

            # Heartbeat if nothing completed
            if not done:
                print(f"⏳ Still working... {len(not_done)} rows in flight (no completion in last {HEARTBEAT_SECS}s)")
                last_heartbeat = time.time()
                continue

            # Consume finished futures
            for fut in done:
                idx = in_flight.pop(fut)
                try:
                    result = fut.result()
                    row_idx = result.pop('idx')
                    for k, v in result.items():
                        df.at[row_idx, k] = v
                except Exception as e:
                    # Mark as error so it won't block future runs
                    df.at[idx, 'evidence_text_hop1'] = f"Error: {e}"
                    df.at[idx, 'evidence_text_hop2'] = f"Error: {e}"
                    df.at[idx, 'pmids_hop1'] = ""
                    df.at[idx, 'pmids_hop2'] = ""

                completed += 1
                if completed % LOG_EVERY == 0:
                    print(f"✅ Processed {completed}/{total_pending} rows...")

                if completed % CHECKPOINT_EVERY == 0:
                    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_row_{completed}.pkl")
                    df.to_pickle(checkpoint_file)
                    print(f"💾 Checkpoint saved at row {completed} → {checkpoint_file}")

    # 5) Final clean + format
    df["evidence_text_hop1"] = df["evidence_text_hop1"].apply(format_evidence_text)
    df["evidence_text_hop2"] = df["evidence_text_hop2"].apply(format_evidence_text)
    df["pmids_hop1"] = df["pmids_hop1"].astype(str).replace(['nan', 'None'], '')
    df["pmids_hop2"] = df["pmids_hop2"].astype(str).replace(['nan', 'None'], '')

    # 6) Reorder columns after stmt_type_2
    cols = df.columns.tolist()
    evidence_cols = ['evidence_text_hop1', 'evidence_text_hop2', 'pmids_hop1', 'pmids_hop2']
    for col in evidence_cols:
        if col in cols:
            cols.remove(col)
    if 'stmt_type_2' in cols:
        insert_at = cols.index('stmt_type_2') + 1
        for i, col in enumerate(evidence_cols):
            cols.insert(insert_at + i, col)
        df = df[cols]

    # 7) Save CSV + summary
    df.to_csv(OUTPUT_FILE, index=False)

    total = len(df)
    pmid_hop1_count = sum(1 for x in df["pmids_hop1"] if str(x).strip())
    pmid_hop2_count = sum(1 for x in df["pmids_hop2"] if str(x).strip())
    db_evidence_hop1_count = sum(1 for x in df["evidence_text_hop1"]
                                 if isinstance(x, str) and (x.startswith("Evidence from:") or x == "No evidence found"))
    db_evidence_hop2_count = sum(1 for x in df["evidence_text_hop2"]
                                 if isinstance(x, str) and (x.startswith("Evidence from:") or x == "No evidence found"))

    duration = (time.time() - start) / 60
    print(f"\n✅ Output saved to: {OUTPUT_FILE}")
    print(f"⏱️ Time taken: {duration:.2f} minutes")
    print("📊 Summary:")
    print(f"   Hop 1 (source ➝ intermediate):")
    print(f"     Rows with PMIDs: {pmid_hop1_count}/{total} ({(pmid_hop1_count / total) * 100:.1f}%)")
    print(f"     Rows with fallback evidence: {db_evidence_hop1_count}/{total} ({(db_evidence_hop1_count / total) * 100:.1f}%)")
    print(f"   Hop 2 (intermediate ➝ target):")
    print(f"     Rows with PMIDs: {pmid_hop2_count}/{total} ({(pmid_hop2_count / total) * 100:.1f}%)")
    print(f"     Rows with fallback evidence: {db_evidence_hop2_count}/{total} ({(db_evidence_hop2_count / total) * 100:.1f}%)")


if __name__ == "__main__":
    main()
