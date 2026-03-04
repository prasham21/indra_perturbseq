import pandas as pd
import time
import logging
from indra.databases.hgnc_client import get_hgnc_name, get_current_hgnc_id
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.analysis.source_targets_explanation import get_valid_gene_ids

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smad3_3hop_protip.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DEG_FILE = "/Users/prashammarfatia/Downloads/de_results_per_gene/SMAD3_vs_control.csv"
OUTPUT_FILE = "indra_3hop_smad3_protip.csv"
SOURCE_SYMBOL = "SMAD3"


def main():
    logger.info("Starting SMAD3 3-hop INDRA analysis with pro tip method")
    start_time = time.time()

    logger.info(f"Loading DEG file: {DEG_FILE}")
    try:
        df = pd.read_csv(DEG_FILE)
        logger.info(f"Total rows in DEG file: {len(df)}")

        df = df[df['pvals'] < 0.05]
        logger.info(f"Rows after p < 0.05 filter: {len(df)}")

        if len(df) == 0:
            logger.error("No significant genes found after p-value filtering")
            return

    except Exception as e:
        logger.error(f"Failed to load DEG file: {e}")
        return

    gene_stats = df.set_index('names')[['logfoldchanges', 'pvals']].to_dict(orient='index')
    gene_symbols = df['names'].dropna().unique().tolist()
    logger.info(f"Unique gene symbols before HGNC conversion: {len(gene_symbols)}")

    logger.info("Converting gene symbols to HGNC IDs")
    try:
        converted = get_valid_gene_ids(gene_symbols)
        symbol_to_hgnc = {k: f"hgnc:{v}" for k, v in zip(gene_symbols, converted) if v}
        hgnc_to_symbol = {v: k for k, v in symbol_to_hgnc.items()}
        target_ids = list(symbol_to_hgnc.values())

        logger.info(f"Valid HGNC IDs obtained: {len(target_ids)}")
        logger.info(f"Conversion success rate: {len(target_ids) / len(gene_symbols) * 100:.1f}%")

    except Exception as e:
        logger.error(f"Failed to convert gene symbols to HGNC IDs: {e}")
        return

    logger.info(f"Getting HGNC ID for source gene: {SOURCE_SYMBOL}")
    try:
        source_hgnc_id = get_current_hgnc_id(SOURCE_SYMBOL)
        if not source_hgnc_id:
            logger.error(f"Could not find HGNC ID for {SOURCE_SYMBOL}")
            return

        source_hgnc = f"hgnc:{source_hgnc_id}"
        logger.info(f"Source HGNC ID: {source_hgnc}")

    except Exception as e:
        logger.error(f"Failed to get HGNC ID for source gene: {e}")
        return

    query = """
    MATCH (a:BioEntity {id: $source})-[r1:indra_rel]->(m1:BioEntity)-[r2:indra_rel]->(m2:BioEntity)-[r3:indra_rel]->(b:BioEntity {id: $target})
    WHERE r3.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
      AND r1.belief > 0.7
      AND r2.belief > 0.7
      AND r3.belief > 0.7
    RETURN a.id, m1.id, m2.id, b.id,
           r1.stmt_type, r2.stmt_type, r3.stmt_type,
           r1.belief, r2.belief, r3.belief,
           r1.evidence_count, r2.evidence_count, r3.evidence_count
    LIMIT 1
    """
    logger.info("3-hop query prepared with LIMIT 100")

    logger.info("Initializing Neo4j client")
    try:
        client = Neo4jClient()
        logger.info("Neo4j client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j client: {e}")
        return

    all_results = []
    successful_queries = 0
    failed_queries = 0
    total_paths = 0

    logger.info(f"Starting queries for {len(target_ids)} target genes")

    for i, target_hgnc in enumerate(target_ids):
        query_start = time.time()

        try:
            results = client.query_tx(query, source=source_hgnc, target=target_hgnc)
            query_time = time.time() - query_start

            target_symbol = hgnc_to_symbol.get(target_hgnc, target_hgnc.replace("hgnc:", ""))

            if results:
                logger.info(f"Query {i + 1}: {SOURCE_SYMBOL} -> {target_symbol}: "
                            f"{len(results)} paths found ({query_time:.2f}s)")

                for row in results:
                    source_id, m1_id, m2_id, target_id = row[0], row[1], row[2], row[3]

                    try:
                        m1_hgnc_id = m1_id.replace("hgnc:", "")
                        raw_symbol_1 = get_hgnc_name(m1_hgnc_id)
                        m1_symbol = get_hgnc_name(
                            get_current_hgnc_id(raw_symbol_1)) if raw_symbol_1 else f"hgnc:{m1_hgnc_id}"

                        m2_hgnc_id = m2_id.replace("hgnc:", "")
                        raw_symbol_2 = get_hgnc_name(m2_hgnc_id)
                        m2_symbol = get_hgnc_name(
                            get_current_hgnc_id(raw_symbol_2)) if raw_symbol_2 else f"hgnc:{m2_hgnc_id}"

                    except Exception as e:
                        logger.warning(f"Failed to convert intermediate genes to symbols: {e}")
                        m1_symbol = m1_id
                        m2_symbol = m2_id

                    stats = gene_stats.get(target_symbol, {'logfoldchanges': None, 'pvals': None})

                    all_results.append({
                        "source": SOURCE_SYMBOL,
                        "intermediate_1": m1_symbol,
                        "intermediate_2": m2_symbol,
                        "target": target_symbol,
                        "stmt_type_1": row[4],
                        "stmt_type_2": row[5],
                        "stmt_type_3": row[6],
                        "belief_1": row[7],
                        "belief_2": row[8],
                        "belief_3": row[9],
                        "evidence_1": row[10],
                        "evidence_2": row[11],
                        "evidence_3": row[12],
                        "logfoldchange": stats['logfoldchanges'],
                        "pvalue": stats['pvals']
                    })

                total_paths += len(results)
                successful_queries += 1

            else:
                if i % 20 == 0:
                    logger.debug(f"No 3-hop paths found for {SOURCE_SYMBOL} -> {target_symbol}")
                successful_queries += 1

        except Exception as e:
            logger.error(f"Query failed for {target_hgnc}: {e}")
            failed_queries += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining_time = avg_time * (len(target_ids) - i - 1)
            logger.info(f"Progress: {i + 1}/{len(target_ids)} queries completed. "
                        f"Total paths so far: {total_paths}. ETA: {remaining_time / 60:.1f} min")

    logger.info(f"Saving results to {OUTPUT_FILE}")
    try:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"Results saved successfully")

        total_time = time.time() - start_time
        logger.info("=== ANALYSIS COMPLETE ===")
        logger.info(f"Total execution time: {total_time / 60:.1f} minutes")
        logger.info(f"Successful queries: {successful_queries}")
        logger.info(f"Failed queries: {failed_queries}")
        logger.info(f"Total 3-hop paths found: {total_paths}")
        logger.info(f"Average paths per successful target: {total_paths / max(successful_queries, 1):.1f}")
        logger.info(f"Results saved to: {OUTPUT_FILE}")

        if len(results_df) > 0:
            logger.info(f"Result file shape: {results_df.shape}")
            logger.info("Sample results:")
            for i, row in results_df.head(3).iterrows():
                logger.info(
                    f"  {row['source']} -> {row['intermediate_1']} -> {row['intermediate_2']} -> {row['target']}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")


if __name__ == "__main__":
    main()