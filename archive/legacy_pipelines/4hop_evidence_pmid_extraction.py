import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client.queries import get_statements
import os
import time
import logging
import json
import pickle
from datetime import datetime

# Suppress noisy Cypher logs
logging.getLogger().setLevel(logging.ERROR)

# ==============================
# CONFIGURATION
# ==============================
INPUT_CSV = "/Users/prashammarfatia/Downloads/indra_4hop_merged_final.csv"  # Input file
OUTPUT_CSV = "/Users/prashammarfatia/Downloads/indra_4hop_with_evidence_and_pmids_.csv"
CHECKPOINT_FILE = "4-hop_evidence_checkpoint.pkl"
CHECKPOINT_EVERY = 10  # Save every N rows
MAX_WORKERS = 6  # Parallel threads
MAX_EVIDENCES = 20  # Limit evidence texts per edge

client = Neo4jClient()

# Cache for identifier processing
IDENTIFIER_CACHE = {}


# ==============================
# IDENTIFIER PROCESSING
# ==============================
def process_identifier(agent_str, *, for_query=False):
    """
    Normalize or parse agent identifiers with caching.

    Parameters
    ----------
    agent_str : str
        Raw identifier from the CSV (e.g., 'hgnc:uniprot:P12345', 'TP53', 'FPLX:AKT').
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

    # Check cache
    cache_key = f"{agent_str}_{for_query}"
    if cache_key in IDENTIFIER_CACHE:
        return IDENTIFIER_CACHE[cache_key]

    result = None

    # Handle different identifier formats
    mappings = {
        "hgnc:uniprot.chain:": "UNIPROT",
        "hgnc:uniprot:": "UNIPROT",
        "uniprot:": "UNIPROT",
        "hgnc:mesh:": "MESH",
        "mesh:": "MESH",
        "hgnc:chebi:": "CHEBI",
        "chebi:": "CHEBI",
        "hgnc:fplx:": "FPLX",
        "fplx:": "FPLX",
        "hgnc:": "HGNC",
        "go:": "GO"
    }

    # Check if it matches any known prefix
    matched = False
    for prefix, label in mappings.items():
        if agent_str.lower().startswith(prefix):
            value = agent_str[len(prefix):]
            result = (label, value) if for_query else f"{label}:{value}"
            matched = True
            break

    # If no prefix matched, assume it's a gene symbol (HGNC)
    if not matched:
        # Check if it looks like a gene symbol (all caps, alphanumeric)
        if agent_str.replace('_', '').replace('-', '').isalnum():
            result = ("HGNC-SYMBOL", agent_str) if for_query else agent_str
        else:
            result = agent_str if not for_query else ("TEXT", agent_str)

    # Cache the result
    IDENTIFIER_CACHE[cache_key] = result
    return result


# ==============================
# COMBINED EVIDENCE EXTRACTION
# ==============================
def extract_all_evidence(source_agent, target_agent, stmt_type):
    """
    Extract PMIDs, source databases, and evidence text for a given edge.
    Returns a dictionary with all information needed.
    """
    try:
        # Process identifiers for query
        source_query = process_identifier(source_agent, for_query=True)
        target_query = process_identifier(target_agent, for_query=True)

        if not source_query or not target_query:
            return {
                'evidence_text': 'Invalid identifiers',
                'pmids': [],
                'sources': []
            }

        # Query for evidence data from Neo4j
        query = """
        MATCH (source:BioEntity)-[r:indra_rel {stmt_type: $stmt_type}]->(target:BioEntity)
        WHERE (source.id = $source_id OR source.name = $source_name)
          AND (target.id = $target_id OR target.name = $target_name)
        WITH r.stmt_hash as stmt_hash
        MATCH (e:Evidence {stmt_hash: stmt_hash})
        RETURN e.evidence
        LIMIT 100
        """

        results = client.query_tx(
            query,
            stmt_type=stmt_type,
            source_id=source_query[1] if isinstance(source_query, tuple) else "",
            source_name=source_agent if isinstance(source_agent, str) else "",
            target_id=target_query[1] if isinstance(target_query, tuple) else "",
            target_name=target_agent if isinstance(target_agent, str) else ""
        )

        pmids = []
        sources = []
        evidences = []

        for result in results:
            try:
                evidence_data = json.loads(result[0])

                # Extract PMID if available
                if 'pmid' in evidence_data and evidence_data['pmid']:
                    pmid = str(evidence_data['pmid'])
                    if pmid and pmid != 'None':
                        pmids.append(pmid)

                # Extract source_api (database name)
                if 'source_api' in evidence_data and evidence_data['source_api']:
                    sources.append(evidence_data['source_api'])

                # Extract text evidence
                if 'text' in evidence_data and evidence_data['text']:
                    text = evidence_data['text'].strip()
                    if text:
                        evidences.append(text)

            except (json.JSONDecodeError, IndexError):
                continue

        # Build evidence text output
        if evidences:
            # Format numbered evidence texts (limit to MAX_EVIDENCES)
            formatted_evidences = [
                f"{idx}) {text}"
                for idx, text in enumerate(evidences[:MAX_EVIDENCES], 1)
            ]
            evidence_text = "\n\n".join(formatted_evidences)
        else:
            # No text evidence - show database names instead
            if sources:
                unique_sources = list(set(sources))[:5]
                evidence_text = f"Database: {', '.join(unique_sources)}"
            else:
                evidence_text = "Database evidence only"

        return {
            'evidence_text': evidence_text,
            'pmids': list(set(pmids))[:10],  # Limit to 10 unique PMIDs
            'sources': list(set(sources))[:5]  # Keep for internal use
        }

    except Exception as e:
        print(f"Error extracting evidence for {source_agent} -> {target_agent}: {e}")
        return {
            'evidence_text': f"Error: {e}",
            'pmids': [],
            'sources': []
        }


# ==============================
# ROW PROCESSING FOR 4-HOP
# ==============================
def process_4hop_row(idx, row):
    """Process a single 4-hop CSV row: extract all evidence data for its 4 edges."""
    try:
        # Extract all evidence for each edge
        edge1_evidence = extract_all_evidence(row["source"], row["intermediate_1"], row["stmt_type_1"])
        edge2_evidence = extract_all_evidence(row["intermediate_1"], row["intermediate_2"], row["stmt_type_2"])
        edge3_evidence = extract_all_evidence(row["intermediate_2"], row["intermediate_3"], row["stmt_type_3"])
        edge4_evidence = extract_all_evidence(row["intermediate_3"], row["target"], row["stmt_type_4"])

        return {
            'idx': idx,
            'edge1_evidence_text': edge1_evidence['evidence_text'],
            'edge2_evidence_text': edge2_evidence['evidence_text'],
            'edge3_evidence_text': edge3_evidence['evidence_text'],
            'edge4_evidence_text': edge4_evidence['evidence_text'],
            'edge1_pmids': '; '.join(edge1_evidence['pmids']),
            'edge2_pmids': '; '.join(edge2_evidence['pmids']),
            'edge3_pmids': '; '.join(edge3_evidence['pmids']),
            'edge4_pmids': '; '.join(edge4_evidence['pmids'])
        }
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        return {
            'idx': idx,
            'edge1_evidence_text': f"Error: {e}",
            'edge2_evidence_text': "",
            'edge3_evidence_text': "",
            'edge4_evidence_text': "",
            'edge1_pmids': "",
            'edge2_pmids': "",
            'edge3_pmids': "",
            'edge4_pmids': ""
        }


# ==============================
# CHECKPOINT MANAGEMENT
# ==============================
def save_checkpoint(df, processed_indices, checkpoint_file=CHECKPOINT_FILE):
    """Save checkpoint with current progress"""
    checkpoint_data = {
        'dataframe': df,
        'processed_indices': processed_indices,
        'timestamp': time.time(),
        'identifier_cache': IDENTIFIER_CACHE
    }
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)

    # Also save intermediate CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    intermediate_file = f"4hop_evidence_intermediate_{timestamp}.csv"
    df.to_csv(intermediate_file, index=False)
    return intermediate_file


def load_checkpoint(checkpoint_file=CHECKPOINT_FILE):
    """Load checkpoint if exists"""
    global IDENTIFIER_CACHE

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)

            saved_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data['timestamp']))
            print(f"Resuming from checkpoint saved at {saved_time}")

            # Restore cache
            if 'identifier_cache' in data:
                IDENTIFIER_CACHE = data['identifier_cache']
                print(f"  Restored {len(IDENTIFIER_CACHE)} cached identifiers")

            return data['dataframe'], data['processed_indices']
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None, set()
    return None, set()


# ==============================
# MAIN EXECUTION
# ==============================
def main():
    start_time = time.time()

    # Load the 4-hop results
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file {INPUT_CSV} not found!")
        print("Please ensure the input file is in the current directory.")
        return

    print(f"Loading 4-hop results from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # Add columns for evidence and PMIDs if not present
    new_columns = [
        "edge1_evidence_text", "edge2_evidence_text", "edge3_evidence_text", "edge4_evidence_text",
        "edge1_pmids", "edge2_pmids", "edge3_pmids", "edge4_pmids"
    ]

    for col in new_columns:
        if col not in df.columns:
            df[col] = ""

    # Try to load checkpoint
    checkpoint_df, processed_indices = load_checkpoint()
    if checkpoint_df is not None:
        # Update the dataframe with checkpoint data
        df.update(checkpoint_df)
        print(f"  Resumed with {len(processed_indices)} rows already processed")
    else:
        processed_indices = set()
        print("Starting fresh - no checkpoint found")

    total_rows = len(df)
    rows_to_process = [i for i in range(total_rows) if i not in processed_indices]

    print(f"\nProcessing evidence and PMIDs for 4-hop pathways")
    print(f"  Total rows: {total_rows}")
    print(f"  Already processed: {len(processed_indices)}")
    print(f"  Remaining to process: {len(rows_to_process)}")
    print(f"  Workers: {MAX_WORKERS}")
    print(f"  Max evidences per edge: {MAX_EVIDENCES}")
    print("=" * 70)

    if len(rows_to_process) == 0:
        print("All rows already processed!")
        # Reorder columns before final save
        df = reorder_columns(df)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Results saved to: {OUTPUT_CSV}")
        return

    # Process rows in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all unprocessed rows
        futures = {
            executor.submit(process_4hop_row, idx, df.iloc[idx]): idx
            for idx in rows_to_process
        }

        completed = len(processed_indices)
        for future in as_completed(futures):
            result = future.result()
            idx = result['idx']

            # Update dataframe with results
            for key, value in result.items():
                if key != 'idx':
                    df.at[idx, key] = value

            processed_indices.add(idx)
            completed += 1

            # Progress update and checkpoint
            if completed % CHECKPOINT_EVERY == 0 or completed == total_rows:
                elapsed = (time.time() - start_time) / 60
                remaining = total_rows - completed
                rate = completed / (elapsed + 0.001)  # Avoid division by zero
                eta = remaining / rate if rate > 0 else 0

                print(f"\nProgress: {completed}/{total_rows} rows processed")
                print(f"  Elapsed: {elapsed:.1f} min | ETA: {eta:.1f} min")
                print(f"  Cache size: {len(IDENTIFIER_CACHE)} identifiers")

                # Save checkpoint
                intermediate_file = save_checkpoint(df, processed_indices)
                print(f"  Checkpoint saved: {intermediate_file}")

    # Final cleanup: process intermediate names for readability
    print("\nCleaning up intermediate node names...")
    for col in ["intermediate_1", "intermediate_2", "intermediate_3"]:
        df[col] = df[col].apply(lambda x: process_identifier(x, for_query=False))

    # Reorder columns as requested
    print("Reordering columns...")
    df = reorder_columns(df)

    # Final statistics
    print("\nCalculating statistics...")
    total_pmids = 0
    edges_with_pmids = 0
    edges_with_text = 0
    edges_with_db_only = 0

    for edge_num in range(1, 5):
        pmid_col = f"edge{edge_num}_pmids"
        text_col = f"edge{edge_num}_evidence_text"

        has_pmids = df[pmid_col].notna() & (df[pmid_col] != "")
        has_text = df[text_col].notna() & (df[text_col] != "") & (~df[text_col].str.startswith("Database"))
        has_db = df[text_col].str.startswith("Database", na=False)

        edges_with_pmids += has_pmids.sum()
        edges_with_text += has_text.sum()
        edges_with_db_only += has_db.sum()

        # Count total PMIDs
        for pmids in df[pmid_col]:
            if pmids and str(pmids) != "" and str(pmids) != "nan":
                total_pmids += len(str(pmids).split('; '))

    # Final save
    df.to_csv(OUTPUT_CSV, index=False)

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    # Final report
    total_time = (time.time() - start_time) / 60
    print("\n" + "=" * 70)
    print("EVIDENCE AND PMID EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"Processing Statistics:")
    print(f"  Total 4-hop pathways: {total_rows}")
    print(f"  Total edges: {total_rows * 4}")
    print(f"  Edges with text evidence: {edges_with_text}")
    print(f"  Edges with database evidence only: {edges_with_db_only}")
    print(f"  Edges with PMIDs: {edges_with_pmids}")
    print(f"  Total PMIDs extracted: ~{total_pmids}")
    print(f"  Processing time: {total_time:.1f} minutes")
    print(f"\nOutput file: {OUTPUT_CSV}")
    print(f"File size: {os.path.getsize(OUTPUT_CSV) / (1024 * 1024):.1f} MB")


def reorder_columns(df):
    """
    Reorder columns in the desired format:
    source, intermediate_1, intermediate_2, intermediate_3, target,
    stmt_type_1, edge1_evidence_text, edge1_pmids,
    stmt_type_2, edge2_evidence_text, edge2_pmids,
    stmt_type_3, edge3_evidence_text, edge3_pmids,
    stmt_type_4, edge4_evidence_text, edge4_pmids,
    logfoldchange, pval,
    belief_1, belief_2, belief_3, belief_4,
    evidence_1, evidence_2, evidence_3, evidence_4
    """
    column_order = [
        # Pathway nodes
        'source', 'intermediate_1', 'intermediate_2', 'intermediate_3', 'target',
        # Edge 1
        'stmt_type_1', 'edge1_evidence_text', 'edge1_pmids',
        # Edge 2
        'stmt_type_2', 'edge2_evidence_text', 'edge2_pmids',
        # Edge 3
        'stmt_type_3', 'edge3_evidence_text', 'edge3_pmids',
        # Edge 4
        'stmt_type_4', 'edge4_evidence_text', 'edge4_pmids',
        # Gene expression data
        'logfoldchange', 'pval',
        # Belief scores
        'belief_1', 'belief_2', 'belief_3', 'belief_4',
        # Evidence counts
        'evidence_1', 'evidence_2', 'evidence_3', 'evidence_4'
    ]

    # Only include columns that exist in the dataframe
    existing_columns = [col for col in column_order if col in df.columns]

    return df[existing_columns]


if __name__ == "__main__":
    main()