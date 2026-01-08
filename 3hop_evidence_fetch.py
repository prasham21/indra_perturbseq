import pandas as pd
import os
import re
import time
import json
import logging
import pickle
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client.queries import get_statements

# ==============================
# CONFIGURATION
# ==============================
INPUT_CSV = "/Users/prashammarfatia/Downloads/indra_3hop_no_hgnc_prefix.csv"
OUTPUT_CSV = "/Users/prashammarfatia/Downloads/indra_3hop_with_statements_full.csv"
CHECKPOINT_DIR = "/Users/prashammarfatia/Downloads/3hop_checkpoints/"
CHECKPOINT_EVERY = 500
LOG_EVERY = 10
MAX_WORKERS = 5
MAX_EVIDENCES = 30

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
client = Neo4jClient()
logging.getLogger().setLevel(logging.ERROR)

# ==============================
# IDENTIFIER PROCESSING
# ==============================
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
        "hgnc:": "HGNC"
    }

    for prefix, label in mappings.items():
        if agent_str.startswith(prefix):
            value = agent_str.replace(prefix, "")
            return (label, value) if for_query else f"{label}:{value}"

    return agent_str if not for_query else agent_str

# ==============================
# EVIDENCE FETCHING
# ==============================
def fetch_evidence_info(agent1, agent2, stmt_type):
    try:
        stmts = get_statements(
            agent=process_identifier(agent1, for_query=True),
            other_agent=process_identifier(agent2, for_query=True),
            agent_role="subject",
            other_role="object",
            rel_types=stmt_type,
            limit=MAX_EVIDENCES,
            evidence_limit=MAX_EVIDENCES,
            client=client
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

# ==============================
# ROW PROCESSING
# ==============================
def process_row(idx, row):
    try:
        ev1, pmid1, _ = fetch_evidence_info(row["source"], row["intermediate_1"], row["stmt_type_1"])
        ev2, pmid2, _ = fetch_evidence_info(row["intermediate_1"], row["intermediate_2"], row["stmt_type_2"])
        ev3, pmid3, _ = fetch_evidence_info(row["intermediate_2"], row["target"], row["stmt_type_3"])
        return {
            'idx': idx,
            'evidence_text_hop1': ev1,
            'evidence_text_hop2': ev2,
            'evidence_text_hop3': ev3,
            'pmids_hop1': "; ".join(pmid1),
            'pmids_hop2': "; ".join(pmid2),
            'pmids_hop3': "; ".join(pmid3),
        }
    except Exception as e:
        return {
            'idx': idx,
            'evidence_text_hop1': f"Error: {e}",
            'evidence_text_hop2': "",
            'evidence_text_hop3': "",
            'pmids_hop1': "",
            'pmids_hop2': "",
            'pmids_hop3': "",
        }

# ==============================
# MAIN
# ==============================
def main():
    start = time.time()
    df = pd.read_csv(INPUT_CSV)

    new_cols = [
        'evidence_text_hop1', 'evidence_text_hop2', 'evidence_text_hop3',
        'pmids_hop1', 'pmids_hop2', 'pmids_hop3'
    ]
    for col in new_cols:
        if col not in df.columns:
            df[col] = ""

    # Resume logic
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pkl')]
    if checkpoints:
        latest_cp = max(checkpoints, key=lambda x: int(re.search(r'\d+', x).group()))
        cp_path = os.path.join(CHECKPOINT_DIR, latest_cp)
        print(f"🧠 Resuming from checkpoint: {cp_path}")
        df = pd.read_pickle(cp_path)
    else:
        print("🧠 No checkpoint found, starting fresh.")

    total_rows = len(df)
    print(f"📊 Total rows: {total_rows} — Using {MAX_WORKERS} threads...")

    # Detect rows that still need processing
    pending_rows = [
        (idx, row) for idx, row in df.iterrows()
        if pd.isna(row['evidence_text_hop1']) or row['evidence_text_hop1'] == ""
    ]

    print(f"🔁 Rows to process: {len(pending_rows)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, idx, row) for idx, row in pending_rows]

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            idx = result.pop("idx")
            for k, v in result.items():
                df.at[idx, k] = v

            completed += 1
            if completed % LOG_EVERY == 0:
                print(f"✅ Processed {completed} rows...")

            if completed % CHECKPOINT_EVERY == 0:
                cp_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_row_{completed}.pkl")
                df.to_pickle(cp_file)
                print(f"💾 Checkpoint saved: {cp_file}")

    df.to_csv(OUTPUT_CSV, index=False)
    total_time = (time.time() - start) / 60
    print(f"\n✅ Completed {total_rows} rows in {total_time:.2f} minutes.")
    print(f"📁 Final output saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
