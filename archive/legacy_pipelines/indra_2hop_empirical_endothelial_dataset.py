"""
indra_2hop_resume_append_upto10_then_dedup_same_columns.py

What this does:
1) Loads your EXISTING checkpoint (keeps old results).
2) Processes remaining genes:
   - loads DEG targets (NO pval filtering)
   - queries Neo4j for 2-hop paths with HGNC-only intermediate constraint
   - returns up to 10 candidate paths per (source,target) (implemented as up to 10 rows per target)
   - DOES NOT do endothelial filtering in Neo4j (you will do later)
3) Appends new rows to the checkpoint.
4) Writes a DEDUPED CSV output with EXACTLY the SAME COLUMN NAMES as your previous output:
   source, intermediate, target, stmt_type_1, stmt_type_2, belief_1, belief_2,
   evidence_1, evidence_2, logfoldchange, pval

Dedup rule:
- Keep ONLY ONE row per unique (source, target)
- Keeps the FIRST row encountered for that pair (stable)

Resumes:
- Uses your checkpoint file indra_2hop_endothelial_checkpoint.pkl
- Preserves old rows (assumed correct)
"""

import os
import time
import pickle
import logging
import concurrent.futures
from threading import local
from typing import List, Dict, Any, Tuple

import pandas as pd

from indra_cogex.analysis.source_targets_explanation import get_valid_gene_ids
from indra_cogex.client.neo4j_client import Neo4jClient
from indra.databases.hgnc_client import get_hgnc_name, get_current_hgnc_id

# -------------------- CONFIG -------------------- #
ENDO_FILE = "/Users/prashammarfatia/Downloads/endothelial_present_plus_manual.csv"  # not used for filtering here
DEG_DIR = "/Users/prashammarfatia/Downloads/de_results_per_gene"
PERTURB_FILE = "/Users/prashammarfatia/Downloads/target_validation_expanded.csv"

# MUST be your existing checkpoint to resume from 22 processed genes
CHECKPOINT_FILE = "indra_2hop_endothelial_checkpoint.pkl"

# Deduped output file (same columns as old output)
OUTPUT_FILE_DEDUP = "indra_2hop_all_perturbations_DEDUP_same_columns.csv"

MAX_WORKERS = 3
BATCH_SIZE = 500

CHECKPOINT_EVERY_RESULTS = 50000
# ------------------------------------------------ #

# Silence noisy warnings
logging.getLogger("indra_cogex").setLevel(logging.ERROR)
logging.getLogger("indra_cogex.analysis.source_targets_explanation").setLevel(logging.ERROR)

_thread_state = local()


def get_thread_client() -> Neo4jClient:
    if not hasattr(_thread_state, "client") or _thread_state.client is None:
        _thread_state.client = Neo4jClient()
    return _thread_state.client


# NOTE: returns up to 10 candidates per (a,b) by collecting 10 and UNWINDing them.
# HGNC-only intermediate constraint, hop2 stmt types restricted.
TWOHOP_UPTO10_BATCH_QUERY = """
MATCH (a:BioEntity {id: $source})
MATCH (a)-[r1:indra_rel]->(m:BioEntity)-[r2:indra_rel]->(b:BioEntity)
WHERE b.id IN $target_list
  AND r2.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']

  AND m.id STARTS WITH 'hgnc:'
  AND NOT (
      m.id CONTAINS 'mesh:' OR m.id CONTAINS 'chebi:' OR
      m.id CONTAINS 'go:'   OR m.id CONTAINS 'uniprot:'
  )

WITH a, b,
     collect([
        m.id,
        r1.stmt_type, r2.stmt_type,
        r1.belief, r2.belief,
        r1.evidence_count, r2.evidence_count
     ])[0..10] AS ps
UNWIND ps AS p
RETURN
  a.id AS source_id,
  b.id AS target_id,
  p[0] AS intermediate_id,
  p[1] AS stmt_type_1, p[2] AS stmt_type_2,
  p[3] AS belief_1,   p[4] AS belief_2,
  p[5] AS evidence_1, p[6] AS evidence_2
"""


# EXACT column set we will enforce everywhere (old output schema)
OUTPUT_COLUMNS = [
    "source",
    "intermediate",
    "target",
    "stmt_type_1",
    "stmt_type_2",
    "belief_1",
    "belief_2",
    "evidence_1",
    "evidence_2",
    "logfoldchange",
    "pval",
]


def save_checkpoint(results: List[Dict[str, Any]], processed_genes: set, checkpoint_file: str = CHECKPOINT_FILE):
    checkpoint_data = {
        "results": results,
        "processed_genes": list(processed_genes),
        "timestamp": time.time(),
    }
    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint_data, f)
    print(f"💾 Checkpoint saved: {len(results)} rows, {len(processed_genes)} genes")


def load_checkpoint(checkpoint_file: str = CHECKPOINT_FILE) -> Tuple[List[Dict[str, Any]], set]:
    """
    Loads BOTH results and processed_genes, preserving old correct rows.
    """
    try:
        with open(checkpoint_file, "rb") as f:
            data = pickle.load(f)
        saved_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data["timestamp"]))
        results = data.get("results", [])
        processed = set(data.get("processed_genes", []))
        print(f"Resuming from checkpoint saved at {saved_time}")
        print(f"   - {len(results)} rows already collected")
        print(f"   - {len(processed)} genes already processed")
        return results, processed
    except FileNotFoundError:
        print("Starting fresh - no checkpoint found")
        return [], set()


def normalize_row_schema(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure each row has EXACTLY the old output columns (missing -> None),
    and strip any extra keys (so final CSV columns match old output).
    """
    return {c: row.get(c, None) for c in OUTPUT_COLUMNS}


def safe_symbol_from_hgnc_curie(curie: str) -> str:
    """
    Convert 'hgnc:####' -> HGNC symbol if possible, else fallback to numeric id.
    Ensures output shows gene name (symbol) rather than 'hgnc:####'.
    """
    if isinstance(curie, str) and curie.startswith("hgnc:"):
        hid = curie.replace("hgnc:", "")
        sym = get_hgnc_name(hid)
        return sym or hid
    return str(curie)


def process_single_gene(gene_name: str) -> List[Dict[str, Any]]:
    """
    For one perturbation gene:
    - read DEG targets (NO pval filter)
    - query 2-hop up to 10 candidates per target
    - return rows in OLD OUTPUT SCHEMA (OUTPUT_COLUMNS)
    """
    gene_name = str(gene_name).strip()
    if not gene_name:
        return []

    try:
        print(f"Processing: {gene_name}")
        start_time = time.time()

        # Source HGNC id
        src_hgnc_id = get_current_hgnc_id(gene_name.upper())
        if not src_hgnc_id:
            print(f"   No HGNC ID for {gene_name}")
            return []
        source_id = f"hgnc:{src_hgnc_id}"

        # Load DEG
        deg_path = os.path.join(DEG_DIR, f"{gene_name}_vs_control.csv")
        if not os.path.exists(deg_path):
            print(f"   DEG file not found for {gene_name}")
            return []

        df = pd.read_csv(deg_path, low_memory=False)
        if "names" not in df.columns:
            print(f"   DEG file missing 'names' column for {gene_name}")
            return []

        df["names"] = df["names"].astype(str).str.strip()
        target_symbols = df["names"].dropna().unique().tolist()
        if not target_symbols:
            print(f"   No targets in DEG file for {gene_name}")
            return []

        # Convert target symbols -> HGNC curies
        converted = get_valid_gene_ids(target_symbols)
        symbol_to_hgnc = {sym: f"hgnc:{hid}" for sym, hid in zip(target_symbols, converted) if hid}
        if not symbol_to_hgnc:
            print(f"   No valid HGNC targets for {gene_name}")
            return []

        hgnc_to_symbol = {v: k for k, v in symbol_to_hgnc.items()}
        target_ids = list(symbol_to_hgnc.values())

        # Optional DEG stats (no filtering)
        if "logfoldchanges" in df.columns:
            df["logfoldchanges"] = pd.to_numeric(df["logfoldchanges"], errors="coerce")
        if "pvals" in df.columns:
            df["pvals"] = pd.to_numeric(df["pvals"], errors="coerce")

        cols = []
        if "logfoldchanges" in df.columns:
            cols.append("logfoldchanges")
        if "pvals" in df.columns:
            cols.append("pvals")
        deg_map = df.set_index("names")[cols].to_dict("index") if cols else {}

        client = get_thread_client()
        out_rows: List[Dict[str, Any]] = []

        total_batches = (len(target_ids) + BATCH_SIZE - 1) // BATCH_SIZE

        for bi, i in enumerate(range(0, len(target_ids), BATCH_SIZE), start=1):
            batch_targets = target_ids[i:i + BATCH_SIZE]
            print(f"   {gene_name}: batch {bi}/{total_batches} (targets={len(batch_targets)})")

            rows = client.query_tx(
                TWOHOP_UPTO10_BATCH_QUERY,
                source=source_id,
                target_list=batch_targets,
            )
            print(f"   {gene_name}: batch {bi} -> returned {len(rows)} rows")

            for r in rows:
                source_out, target_id, intermediate_id, stmt1, stmt2, belief1, belief2, ev1, ev2 = r[:9]

                # Convert intermediate to SYMBOL for output (required by you)
                interm_symbol = safe_symbol_from_hgnc_curie(str(intermediate_id))

                # Convert target to SYMBOL for output
                target_symbol = hgnc_to_symbol.get(str(target_id), str(target_id).replace("hgnc:", ""))

                stats = deg_map.get(target_symbol, {})

                out_rows.append(normalize_row_schema({
                    "source": gene_name,
                    "intermediate": str(interm_symbol),
                    "target": str(target_symbol),
                    "stmt_type_1": stmt1,
                    "stmt_type_2": stmt2,
                    "belief_1": belief1,
                    "belief_2": belief2,
                    "evidence_1": ev1,
                    "evidence_2": ev2,
                    "logfoldchange": stats.get("logfoldchanges", None),
                    "pval": stats.get("pvals", None),
                }))

        elapsed = time.time() - start_time
        print(f"   {gene_name}: wrote {len(out_rows)} candidate rows in {elapsed/60:.2f} min")
        return out_rows

    except Exception as e:
        print(f"   Error with {gene_name}: {e}")
        return []


def dedup_one_row_per_source_target(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Dedup to ONE row per unique (source, target) pair.
    Keeps first occurrence (stable).
    Output columns exactly match OUTPUT_COLUMNS.
    """
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Normalize all rows to the exact schema (also strips any extra fields from old checkpoint)
    norm_rows = [normalize_row_schema(r) for r in rows]

    df = pd.DataFrame(norm_rows, columns=OUTPUT_COLUMNS)
    df["_row_order"] = range(len(df))
    df = df.sort_values("_row_order")
    df = df.drop_duplicates(subset=["source", "target"], keep="first")
    df = df.drop(columns=["_row_order"])
    return df


def main():
    # Load endothelial list only to print counts (not filtering here)
    endo_df = pd.read_csv(ENDO_FILE)
    if "gene" not in endo_df.columns:
        raise ValueError(f"Endothelial file must have a 'gene' column. Columns: {endo_df.columns.tolist()}")
    endo_symbols_set = set(endo_df["gene"].astype(str).str.strip()) - {""}
    print(f"Total endothelial-present genes (CSV symbols): {len(endo_symbols_set)}")

    # Load perturbations
    perturb_df = pd.read_csv(PERTURB_FILE)
    perturb_df = perturb_df[perturb_df["Karen_Flag"] == "Use_for_analysis"]
    genes = perturb_df["Gene"].astype(str).tolist()
    print(f"Perturbations selected: {len(genes)}")

    # Load checkpoint (keep old results)
    all_results, processed_genes = load_checkpoint(CHECKPOINT_FILE)

    remaining_genes = [g for g in genes if g not in processed_genes]
    print(f"Remaining to process: {len(remaining_genes)} genes")

    print("\nStarting 2-hop extraction (NO pval filtering; endothelial filtering later)...")
    print(f"   Parallel workers: {MAX_WORKERS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Checkpoint frequency: every ~{CHECKPOINT_EVERY_RESULTS} new rows")
    print("=" * 60)

    start_time = time.time()
    completed_results: List[Dict[str, Any]] = []
    completed_count = 0
    last_checkpoint_rows = len(all_results)

    if remaining_genes:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_gene = {executor.submit(process_single_gene, g): g for g in remaining_genes}

            for future in concurrent.futures.as_completed(future_to_gene):
                gene_name = future_to_gene[future]
                completed_count += 1

                gene_rows = future.result()
                completed_results.extend(gene_rows)
                processed_genes.add(gene_name)

                total_rows_now = len(all_results) + len(completed_results)
                elapsed = time.time() - start_time
                avg_time = elapsed / max(completed_count, 1)
                remaining_time = avg_time * (len(remaining_genes) - completed_count)

                print(
                    f"\nProgress: {completed_count}/{len(remaining_genes)} genes | "
                    f"Rows so far: {total_rows_now} | "
                    f"ETA: {remaining_time/60:.1f} min"
                )

                if total_rows_now - last_checkpoint_rows >= CHECKPOINT_EVERY_RESULTS:
                    current_all_results = all_results + completed_results
                    save_checkpoint(current_all_results, processed_genes, CHECKPOINT_FILE)
                    last_checkpoint_rows = total_rows_now

    # Merge old + new (preserve old)
    final_results = all_results + completed_results

    # Save final checkpoint
    save_checkpoint(final_results, processed_genes, CHECKPOINT_FILE)

    # DEDUP view: one row per (source,target), with IDENTICAL columns to old output
    dedup_df = dedup_one_row_per_source_target(final_results)
    dedup_df.to_csv(OUTPUT_FILE_DEDUP, index=False)

    total_time = time.time() - start_time
    print("\nDONE.")
    print(f"    Total raw rows in checkpoint: {len(final_results)}")
    print(f"    Total dedup rows (1 per source-target): {len(dedup_df)}")
    print(f"    Dedup output saved to: {OUTPUT_FILE_DEDUP}")
    print(f"    Total time this run: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
