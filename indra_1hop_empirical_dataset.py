import os
import time
import pickle
import concurrent.futures
from threading import Lock

import pandas as pd
from indra_cogex.analysis.source_targets_explanation import get_valid_gene_ids
from indra_cogex.client.neo4j_client import Neo4jClient
from indra.databases.hgnc_client import get_hgnc_name, get_current_hgnc_id

# -------------------- CONFIG -------------------- #
ENDO_FILE = "/Users/prashammarfatia/Downloads/endothelial_present_plus_manual.csv"
DEG_DIR = "/Users/prashammarfatia/Downloads/de_results_per_gene"
PERTURB_FILE = "/Users/prashammarfatia/Downloads/target_validation_expanded.csv"
CHECKPOINT_FILE = "indra_1hop_endothelial_checkpoint.pkl"
OUTPUT_FILE = "indra_1hop_all_perturbations_endothelial_dataset.csv"

MAX_WORKERS = 6
CHECKPOINT_EVERY_RESULTS = 100  # checkpoint after every ~100 new edges
# ------------------------------------------------ #

results_lock = Lock()

# ---- 1-hop query: 1 row per (source, target) ---- #
ONEHOP_QUERY = """
MATCH (source:BioEntity {id: $perturbation})-[r:indra_rel]->(target:BioEntity)
WHERE target.id IN $descendants
  AND r.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
WITH 
    source.id AS source_id,
    target.id AS target_id,
    collect(DISTINCT r.stmt_type) AS stmt_types,
    max(r.belief) AS max_belief,
    sum(r.evidence_count) AS total_evidence
RETURN 
    source_id,
    target_id,
    stmt_types[0] AS stmt_type,
    max_belief AS belief,
    total_evidence AS evidence_count
"""


def save_checkpoint(results, processed_genes, checkpoint_file=CHECKPOINT_FILE):
    """Save progress (results + processed gene names)."""
    checkpoint_data = {
        "results": results,
        "processed_genes": processed_genes,
        "timestamp": time.time(),
    }
    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint_data, f)
    print(f"💾 Checkpoint saved: {len(results)} rows, {len(processed_genes)} genes")


def load_checkpoint(checkpoint_file=CHECKPOINT_FILE):
    """Load previous progress if checkpoint exists."""
    try:
        with open(checkpoint_file, "rb") as f:
            data = pickle.load(f)
        saved_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data["timestamp"]))
        print(f"Resuming from checkpoint saved at {saved_time}")
        print(f"   - {len(data['results'])} rows already collected")
        print(f"   - {len(data['processed_genes'])} genes already processed")
        return data["results"], set(data["processed_genes"])
    except FileNotFoundError:
        print("Starting fresh - no checkpoint found")
        return [], set()


def process_single_gene(args):
    """
    Process one perturbation gene for 1-hop analysis.

    Returns a list of dicts with:
    source, target, stmt_type, belief, evidence_count, logfoldchange, pval
    """
    gene_info, endo_genes = args
    perturb_gene = gene_info["Gene"]

    local_results = []

    try:
        print(f"Processing: {perturb_gene}")
        start_time = time.time()

        client = Neo4jClient()

        # --- Map perturbation symbol -> HGNC ID --- #
        hgnc_id = get_current_hgnc_id(perturb_gene.upper())
        if not hgnc_id:
            print(f"   No HGNC ID for {perturb_gene}")
            return []

        source_id = f"hgnc:{hgnc_id}"

        # --- Load DEG file and restrict to endothelial genes --- #
        deg_path = os.path.join(DEG_DIR, f"{perturb_gene}_vs_control.csv")
        if not os.path.exists(deg_path):
            print(f"   DEG file not found for {perturb_gene}")
            return []

        df = pd.read_csv(deg_path)

        # Restrict to endothelial-present genes (gene universe = 13,787)
        df = df[df["names"].isin(endo_genes)]
        if df.empty:
            print(f"   No overlapping endothelial genes in DEG file for {perturb_gene}")
            return []

        # Gene symbols to target
        gene_symbols = df["names"].dropna().unique().tolist()

        # Convert HGNC symbols -> HGNC IDs that exist in CoGEx
        converted = get_valid_gene_ids(gene_symbols)
        symbol_to_hgnc = {k: f"hgnc:{v}" for k, v in zip(gene_symbols, converted) if v}
        hgnc_to_symbol = {v: k for k, v in symbol_to_hgnc.items()}
        target_ids = list(symbol_to_hgnc.values())

        if not target_ids:
            print(f"   No valid HGNC targets for {perturb_gene}")
            return []

        # Map symbol -> (logfoldchange, pval)
        deg_map = df.set_index("names")[["logfoldchanges", "pvals"]].to_dict("index")

        # --- Run 1-hop query (one row per (source, target)) --- #
        results = client.query_tx(
            ONEHOP_QUERY,
            perturbation=source_id,
            descendants=target_ids,
        )

        for r in results:
            # source_id, target_id, stmt_type, belief, evidence_count
            _, target_hgnc, stmt_type, belief, ev_count = r
            target_id = target_hgnc  # e.g., "hgnc:1234"
            target_symbol = hgnc_to_symbol.get(target_id, target_id.split(":", 1)[-1])

            if target_symbol not in deg_map:
                continue

            stats = deg_map[target_symbol]
            local_results.append({
                "source": perturb_gene,
                "target": target_symbol,
                "stmt_type": stmt_type,
                "belief": belief,
                "evidence_count": ev_count,
                "logfoldchange": stats["logfoldchanges"],
                "pval": stats["pvals"],
            })

        elapsed = time.time() - start_time
        print(f"   {perturb_gene}: {len(local_results)} edges in {elapsed / 60:.2f} min")
        return local_results

    except Exception as e:
        print(f"   Error with {perturb_gene}: {e}")
        return []


def main():
    # --- Load endothelial gene universe --- #
    endo_df = pd.read_csv(ENDO_FILE)
    endo_genes = set(endo_df["gene"].astype(str))
    print(f"Total endothelial-present genes (CSV): {len(endo_genes)}")

    # --- Load perturbation list --- #
    perturb_df = pd.read_csv(PERTURB_FILE)
    perturb_df = perturb_df[perturb_df["Karen_Flag"] == "Use_for_analysis"]
    print(f"Perturbations selected: {len(perturb_df)}")

    # --- Load checkpoint if exists --- #
    all_results, processed_genes = load_checkpoint(CHECKPOINT_FILE)

    # Filter for remaining genes
    remaining_df = perturb_df[~perturb_df["Gene"].isin(processed_genes)]
    print(f"Remaining to process: {len(remaining_df)} genes")

    if len(remaining_df) == 0:
        print("All genes already processed from checkpoint!")
        output_df = pd.DataFrame(all_results)
        output_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Results saved to: {OUTPUT_FILE}")
        return

    print("\nStarting optimized 1-hop endothelial analysis...")
    print(f"   Parallel workers: {MAX_WORKERS}")
    print(f"   Checkpoint frequency: every ~{CHECKPOINT_EVERY_RESULTS} new rows")
    print("=" * 60)

    start_time = time.time()
    gene_data = [{"Gene": row["Gene"]} for _, row in remaining_df.iterrows()]
    process_args = [(gene_info, endo_genes) for gene_info in gene_data]

    completed_results = []
    completed_count = 0
    last_checkpoint_rows = len(all_results)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_gene = {
            executor.submit(process_single_gene, args): args[0]["Gene"]
            for args in process_args
        }

        for future in concurrent.futures.as_completed(future_to_gene):
            gene_name = future_to_gene[future]
            completed_count += 1

            try:
                gene_results = future.result()
                completed_results.extend(gene_results)
                processed_genes.add(gene_name)

                total_rows_now = len(all_results) + len(completed_results)
                elapsed = time.time() - start_time
                avg_time = elapsed / max(completed_count, 1)
                remaining_time = avg_time * (len(remaining_df) - completed_count)

                print(
                    f"\nProgress: {completed_count}/{len(remaining_df)} genes | "
                    f"Rows so far: {total_rows_now} | "
                    f"ETA: {remaining_time / 60:.1f} min"
                )

                # Checkpoint based on number of new rows
                if total_rows_now - last_checkpoint_rows >= CHECKPOINT_EVERY_RESULTS:
                    current_all_results = all_results + completed_results
                    save_checkpoint(current_all_results, list(processed_genes), CHECKPOINT_FILE)
                    last_checkpoint_rows = total_rows_now

            except Exception as e:
                print(f"Failed to process {gene_name}: {e}")

    # --- Final save --- #
    final_results = all_results + completed_results
    output_df = pd.DataFrame(final_results)
    output_df.to_csv(OUTPUT_FILE, index=False)

    total_time = time.time() - start_time
    print("\nOptimized 1-hop processing complete!")
    print(f"    Total results: {len(final_results)}")
    print(f"    Total time: {total_time / 3600:.2f} hours")
    print(f"    Average: {total_time / len(remaining_df) / 60:.2f} min per gene")
    print(f"    Results saved to: {OUTPUT_FILE}")

    # Optional: clean up checkpoint
    # if os.path.exists(CHECKPOINT_FILE):
    #     os.remove(CHECKPOINT_FILE)


if __name__ == "__main__":
    main()
