import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client.queries import get_statements
import os
import time
import logging

# Suppress noisy Cypher logs
logging.getLogger().setLevel(logging.ERROR)

# ==============================
# CONFIGURATION
# ==============================
INPUT_CSV = "/Users/prashammarfatia/Downloads/indra_3hop_optimized_all_perturbations_combined.csv"
OUTPUT_CSV = "/Users/prashammarfatia/Downloads/indra_3hop_with_statements_rf.csv"
CHECKPOINT_EVERY = 10  # Save every N rows
MAX_WORKERS = 4  # Parallel threads
MAX_EVIDENCES = 20  # Limit evidence texts per edge

client = Neo4jClient()


# ==============================
# IDENTIFIER PROCESSING
# ==============================
def process_identifier(agent_str, *, for_query=False):
    """
    Normalize or parse agent identifiers.

    Parameters
    ----------
    agent_str : str
        Raw identifier from the CSV (e.g., 'hgnc:uniprot:P12345').
    for_query : bool
        - If True: returns a tuple like ("UNIPROT", "P12345") for INDRA queries.
        - If False: returns a cleaned string like "UNIPROT:P12345" for CSV output.

    Returns
    -------
    str | tuple | None
    """
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
def fetch_evidence_text(agent1, agent2, stmt_type):
    """Fetch up to MAX_EVIDENCES texts for a given edge."""
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
            return "Database evidence only"

        evidences = []
        for stmt in stmts:
            if stmt.evidence:
                for idx, ev in enumerate(stmt.evidence[:MAX_EVIDENCES], 1):
                    if ev.text:
                        evidences.append(f"{idx}) {ev.text.strip()}")

        return "\n\n".join(evidences) if evidences else "Database evidence only"
    except Exception as e:
        return f"Error fetching evidence: {e}"


# ==============================
# ROW PROCESSING
# ==============================
def process_row(idx, row):
    """Process a single CSV row: fetch evidences for its 3 edges."""
    try:
        edge1_text = fetch_evidence_text(row["source"], row["intermediate_1"], row["stmt_type_1"])
        edge2_text = fetch_evidence_text(row["intermediate_1"], row["intermediate_2"], row["stmt_type_2"])
        edge3_text = fetch_evidence_text(row["intermediate_2"], row["target"], row["stmt_type_3"])
        return idx, edge1_text, edge2_text, edge3_text
    except Exception as e:
        return idx, f"Error: {e}", "", ""


# ==============================
# MAIN EXECUTION
# ==============================
def main():
    start_time = time.time()
    df = pd.read_csv(INPUT_CSV)

    # Add empty columns for new data if not present
    for col in ["edge1_statements", "edge2_statements", "edge3_statements"]:
        if col not in df.columns:
            df[col] = ""

    # Resume if checkpoint exists
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        df.update(df_existing)
        print(f"Resuming from checkpoint: {OUTPUT_CSV}")

    total_rows = len(df)
    print(f"Processing {total_rows} rows with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_row, idx, row): idx
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
            if completed % CHECKPOINT_EVERY == 0:
                df.to_csv(OUTPUT_CSV, index=False)
                elapsed = (time.time() - start_time) / 60
                print(f"Checkpoint saved at row {completed}/{total_rows} ({elapsed:.2f} min elapsed)")

    # Final cleanup: fix intermediate names for readability
    df["intermediate_1"] = df["intermediate_1"].apply(lambda x: process_identifier(x, for_query=False))
    df["intermediate_2"] = df["intermediate_2"].apply(lambda x: process_identifier(x, for_query=False))

    # Final save
    df.to_csv(OUTPUT_CSV, index=False)
    total_time = (time.time() - start_time) / 60
    print("\nProcessing complete")
    print(f"Total rows processed: {total_rows}")
    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"Total time taken: {total_time:.2f} minutes")


if __name__ == "__main__":
    main()
