import os
import time
import json
import pandas as pd
from indra_cogex.analysis.source_targets_explanation import get_valid_gene_ids
from indra_cogex.client.neo4j_client import Neo4jClient
from indra.databases.hgnc_client import get_hgnc_name, get_current_hgnc_id


def create_readable_statement_text(stmt_data):
    """Convert INDRA statement JSON to readable structured text"""
    stmt_type = stmt_data.get('type', 'Unknown')

    # Extract subject information
    subj = stmt_data.get('subj', {})
    subj_name = subj.get('name', 'Unknown')
    subj_mods = subj.get('mods', [])

    # Extract object information
    obj = stmt_data.get('obj', {})
    obj_name = obj.get('name', 'Unknown')
    obj_mods = obj.get('mods', [])

    # Build subject description
    if subj_mods:
        mod_descriptions = []
        for mod in subj_mods:
            if mod.get('mod_type') == 'phosphorylation':
                residue = mod.get('residue')
                position = mod.get('position')
                if residue and position:
                    mod_descriptions.append(f"phosphorylated on {residue}{position}")
                else:
                    mod_descriptions.append("phosphorylated")
            elif mod.get('mod_type') == 'modification':
                mod_descriptions.append("modified")
            else:
                mod_descriptions.append(f"{mod.get('mod_type', 'modified')}")

        subject_desc = f"{subj_name} ({', '.join(mod_descriptions)})"
    else:
        subject_desc = subj_name

    # Build object description
    if obj_mods:
        mod_descriptions = []
        for mod in obj_mods:
            if mod.get('mod_type') == 'phosphorylation':
                residue = mod.get('residue')
                position = mod.get('position')
                if residue and position:
                    mod_descriptions.append(f"phosphorylated on {residue}{position}")
                else:
                    mod_descriptions.append("phosphorylated")
            elif mod.get('mod_type') == 'modification':
                mod_descriptions.append("modified")
            else:
                mod_descriptions.append(f"{mod.get('mod_type', 'modified')}")

        object_desc = f"{obj_name} ({', '.join(mod_descriptions)})"
    else:
        object_desc = obj_name

    # Create readable statement
    if stmt_type == 'IncreaseAmount':
        return f"{subject_desc} increases the amount of {object_desc}"
    elif stmt_type == 'DecreaseAmount':
        return f"{subject_desc} decreases the amount of {object_desc}"
    else:
        return f"{subject_desc} {stmt_type.lower()} {object_desc}"


# Load all perturbations
perturb_df = pd.read_csv("/Users/prashammarfatia/Downloads/target_validation_expanded.csv")
perturb_df = perturb_df[perturb_df['Karen_Flag'] == "Use_for_analysis"]
print(f"Perturbations selected: {len(perturb_df)}")

client = Neo4jClient()
all_results = []

start_time = time.time()
for idx, row in perturb_df.iterrows():
    perturb_gene = row["Gene"]
    print(f"\n({idx + 1}/{len(perturb_df)}) Processing: {perturb_gene}")

    loop_start = time.time()
    try:
        hgnc_id = get_current_hgnc_id(perturb_gene.upper())
        if not hgnc_id:
            print(f"No HGNC ID found for {perturb_gene}")
            continue
        source_id = f"hgnc:{hgnc_id}"

        csv_path = f"/Users/prashammarfatia/Downloads/de_results_per_gene/{perturb_gene}_vs_control.csv"
        if not os.path.exists(csv_path):
            print(f"DEG file not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df = df[df["pvals"] < 0.05]

        gene_symbols = df["names"].dropna().unique().tolist()
        converted = get_valid_gene_ids(gene_symbols)
        target_ids = [f"hgnc:{v}" for v in converted if v]

        deg_map = df.set_index("names")[["logfoldchanges", "pvals"]].to_dict("index")

        query = """
        MATCH (source:BioEntity {id: $perturbation})-[r:indra_rel]->(target:BioEntity)
        WHERE target.id IN $descendants AND r.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
        RETURN source.id, target.id, r.stmt_type, r.stmt_json, r.belief, r.evidence_count
        """
        results = client.query_tx(query, perturbation=source_id, descendants=target_ids)

        for r in results:
            _, target_hgnc, stmt_type, stmt_json, belief, ev_count = r
            target_id = target_hgnc.split(":")[1]
            symbol = get_hgnc_name(target_id)
            if symbol and symbol in deg_map:
                # Create readable statement text
                try:
                    stmt_data = json.loads(stmt_json)
                    statement_text = create_readable_statement_text(stmt_data)
                except Exception as e:
                    statement_text = f"Parsing error: {str(e)}"

                all_results.append({
                    "source": perturb_gene,
                    "target": symbol,
                    "stmt_type": stmt_type,
                    "statement_text": statement_text,
                    "belief": belief,
                    "evidence_count": ev_count,
                    "logfoldchange": deg_map[symbol]["logfoldchanges"],
                    "pval": deg_map[symbol]["pvals"]
                })

        loop_time = time.time() - loop_start
        avg_time = (time.time() - start_time) / (idx + 1)
        remaining = avg_time * (len(perturb_df) - idx - 1)
        print(f"{len(results)} edges found | {loop_time:.1f}s | ETA: {remaining / 60:.1f} mins")

    except Exception as e:
        print(f"Error with {perturb_gene}: {e}")

# Save final CSV
output_df = pd.DataFrame(all_results)
output_df.to_csv("indra_1hop_with_readable_statements.csv", index=False)
print(f"\nSaved all results to indra_1hop_with_readable_statements.csv")
print(f"Total results: {len(all_results)}")
print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")