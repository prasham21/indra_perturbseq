import pandas as pd
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client.queries import get_statements
from indra.assemblers.html.assembler import HtmlAssembler


# =========================================================
# CONFIGURATION
# =========================================================
INPUT_CSV = "/Users/prashammarfatia/Downloads/gwas_endothelial_paths.csv"
OUTPUT_CSV = "/Users/prashammarfatia/Downloads/gwas_endothelial_paths_url_enriched.csv"
OUTPUT_HTML = "/Users/prashammarfatia/Downloads/gwas_endothelial_paths_statements.html"


# =========================================================
# MATCH STATEMENT BASED ON BELIEF + EVIDENCE COUNT
# =========================================================
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


# =========================================================
# MAIN WORKFLOW
# =========================================================
def main():
    print(" Loading CSV...")
    df = pd.read_csv(INPUT_CSV)

    client = Neo4jClient()

    hop1_hashes, hop1_urls = [], []
    hop2_hashes, hop2_urls = [], []

    # Will store statements exactly in CSV order
    all_statements_for_html = []

    print(" Processing rows...")
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

        # -----------------------------------------------------
        # HOP 1: source → intermediate
        # -----------------------------------------------------
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

        # -----------------------------------------------------
        # HOP 2: intermediate → target
        # -----------------------------------------------------
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
            print(f"   Processed {idx}/{len(df)} rows...")

    # =========================================================
    # ADD NEW COLUMNS TO CSV
    # =========================================================
    df["hop1_hash"] = hop1_hashes
    df["hop1_indra_url"] = hop1_urls
    df["hop2_hash"] = hop2_hashes
    df["hop2_indra_url"] = hop2_urls

    print(f" Saving enriched CSV → {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)

    # =========================================================
    # GENERATE HTML PAGE — IN EXACT CSV ORDER
    # =========================================================
    print(" Generating HtmlAssembler report...")

    # Deduplicate while preserving original order
    uniq = []
    seen = set()
    for s in all_statements_for_html:
        h = s.get_hash()
        if h not in seen:
            uniq.append(s)
            seen.add(h)

    # HtmlAssembler with statement-level (NO grouping or sorting)
    ha = HtmlAssembler(
        statements=uniq,
        title="GWAS 2-Hop INDRA Evidence",
    )

    ha.make_model(grouping_level="statement")  # <<< KEY FIX
    ha.save_model(OUTPUT_HTML)

    print(f" Done! HTML saved to → {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
