import pandas as pd
import os
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.analysis.source_targets_explanation import get_valid_gene_ids
from indra.databases import hgnc_client
from tqdm import tqdm

# Path to DEG folder
deg_folder = "/Users/prashammarfatia/Downloads/de_results_per_gene"
client = Neo4jClient()

# Load perturbation list from metadata file
meta_df = pd.read_csv("/Users/prashammarfatia/Downloads/target_validation_expanded.csv")
perturb_list = meta_df[meta_df['Karen_Flag'] == True]['Gene'].dropna().unique().tolist()
print(f"📦 Total Karen-flagged perturbations: {len(perturb_list)}")

results = []

for perturb_gene in tqdm(perturb_list, desc="🔍 Processing perturbations"):
    deg_path = os.path.join(deg_folder, f"{perturb_gene}_vs_control.csv")
    if not os.path.exists(deg_path):
        print(f"⚠️ File missing: {deg_path}")
        continue

    try:
        deg_df = pd.read_csv(deg_path)
        deg_df = deg_df[deg_df['pvals'] < 0.05]
    except Exception as e:
        print(f"❌ Error reading {deg_path}: {e}")
        continue

    if deg_df.empty:
        continue

    descendant_symbols = deg_df['names'].dropna().unique().tolist()
    valid_ids = get_valid_gene_ids(descendant_symbols)
    descendant_ids = [f"hgnc:{v}" for v in valid_ids if v]

    pert_id = hgnc_client.get_current_hgnc_id(perturb_gene)
    if not pert_id:
        print(f"⚠️ Skipping {perturb_gene} (invalid HGNC)")
        continue
    source_id = f"hgnc:{pert_id}"

    query = """
    MATCH (source:BioEntity {id: $source})-[r:indra_rel]->(target:BioEntity)
    WHERE target.id IN $descendants AND r.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
    RETURN source.id, target.id, r.stmt_type, r.belief, r.evidence_count
    """

    try:
        res = client.query_tx(query, source=source_id, descendants=descendant_ids)
    except Exception as e:
        print(f"❌ Query error for {perturb_gene}: {e}")
        continue

    if not res:
        continue

    for row in res:
        _, target_id, stmt_type, belief, evidence_count = row
        target_hgnc = target_id.split(":")[1]
        target_symbol = hgnc_client.get_hgnc_name(target_hgnc)
        if not target_symbol:
            continue

        # Match to get logFC and p-value
        match = deg_df[deg_df['names'] == target_symbol]
        if match.empty:
            continue
        logfc = match.iloc[0]['logfoldchanges']
        pval = match.iloc[0]['pvals']

        results.append({
            "source": perturb_gene,
            "target": target_symbol,
            "stmt_type": stmt_type,
            "belief": belief,
            "evidence_count": evidence_count,
            "logfoldchange": logfc,
            "pval": pval
        })
    print(f"✅ {perturb_gene}: {len(res)} edges")

# Save output
out_df = pd.DataFrame(results)
out_df.to_csv("indra_1hop_all_perturbations_FULL.csv", index=False)
print(f"\n🎯 Saved {len(out_df)} results to indra_1hop_all_perturbations_FULL.csv")
