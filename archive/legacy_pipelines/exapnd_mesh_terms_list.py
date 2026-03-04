import pandas as pd
from indra_cogex.client.neo4j_client import Neo4jClient

# ========= CONFIG =========
EXISTING_CSV = "/Users/prashammarfatia/Downloads/comprehensive_mesh_list_with_origin.csv"
OUTPUT_CSV = "/Users/prashammarfatia/Downloads/comprehensive_mesh_list_EXPANDED_.csv"
CHILD_DEPTH = 4
NEW_ORIGIN = "CAD Mechanistic"

# Broad terms to drop if they ever show up as children
BROAD_STOPLIST = {
    "D006801",  # Humans
    "D000818",  # Animals
    "D002477",  # Cells
    "D005260",  # Enzymes
    "D010801",  # Proteins
    "D000740",  # Anatomy
}

# 26 mechanistic MeSH terms (ID → Name)
MECHANISTIC_SEEDS = {
    "D058506": "Coronary Artery Disease",
    "D001161": "Atherosclerosis",
    "D058226": "Plaque, Atherosclerotic",
    "D004730": "Endothelium, Vascular",
    "D056669": "Endothelial Dysfunction",
    "D014652": "Vascular Permeability",
    "D006439": "Hemodynamics",
    "D012711": "Shear Stress",
    "D004730": "Endothelial Cells",
    "D000293": "Angiogenesis",
    "D015815": "Cell Adhesion Molecules",
    "D017207": "Extracellular Matrix",
    "D001446": "Basement Membrane",
    "D022001": "Focal Adhesions",
    "D000199": "Actins",
    "D003562": "Cytoskeleton",
    "D015398": "Signal Transduction",
    "D000076325": "Cerebral Cavernous Malformations",
    "D051741": "Kruppel-Like Transcription Factors",
    "D020928": "Mitogen-Activated Protein Kinases",
    "D020741": "Rho GTP-Binding Proteins",
    "D040542": "Mechanotransduction, Cellular",
    "D007365": "Intercellular Junctions",
    "D002462": "Cell Migration",
    "D002452": "Cell Shape",
    "D042461": "Vascular Endothelial Growth Factor A",
}


def get_children_for_mesh_id(mesh_id: str, client: Neo4jClient, depth: int):
    """Return (child_id, child_name) for MeSH descendants up to given depth."""
    parent_curie = f"mesh:{mesh_id}"
    query = f"""
    MATCH (c:BioEntity)-[:isa|partof*1..{depth}]->(p:BioEntity {{id: $parent_id}})
    WHERE c.id STARTS WITH 'mesh:'
    RETURN DISTINCT c.id AS mesh_id, c.name AS mesh_name
    """
    results = client.query_tx(query, parent_id=parent_curie)

    children = set()
    for rec in results:
        if isinstance(rec, dict):
            mid, name = rec.get("mesh_id"), rec.get("mesh_name")
        else:
            mid, name = rec[0], rec[1]

        if not mid or not name:
            continue

        bare = mid.replace("mesh:", "").upper()

        if bare in BROAD_STOPLIST:
            continue

        children.add((bare, name))

    return sorted(children, key=lambda x: x[0])


def pretty_print_table(df: pd.DataFrame):
    """Print dataframe as aligned text table."""
    cols = ["mesh_id", "mesh_name", "origin", "child_of"]
    df = df.fillna("").astype(str)

    widths = {c: max(len(c), df[c].map(len).max()) for c in cols}
    fmt = "  ".join(f"{{:{widths[c]}}}" for c in cols)

    print(fmt.format(*cols))
    print("  ".join("-" * widths[c] for c in cols))

    for _, row in df.iterrows():
        print(fmt.format(row["mesh_id"], row["mesh_name"],
                         row["origin"], row["child_of"]))


def main():
    print(f"📥 Loading existing list: {EXISTING_CSV}")
    df_old = pd.read_csv(EXISTING_CSV, dtype=str).fillna("")
    print(f"   → {len(df_old)} rows")

    client = Neo4jClient()

    print(f"\n🔎 Expanding the 26 mechanistic seeds (depth={CHILD_DEPTH})…\n")
    new_rows = []

    for mesh_id, mesh_name in MECHANISTIC_SEEDS.items():

        # Parent
        new_rows.append({
            "mesh_id": mesh_id,
            "mesh_name": mesh_name,
            "origin": NEW_ORIGIN,
            "child_of": "None"
        })

        # Children
        children = get_children_for_mesh_id(mesh_id, client, CHILD_DEPTH)
        print(f"  {mesh_name} ({mesh_id}) → {len(children)} children")

        for cid, cname in children:
            new_rows.append({
                "mesh_id": cid,
                "mesh_name": cname,
                "origin": NEW_ORIGIN,
                "child_of": mesh_name
            })

    df_new = pd.DataFrame(new_rows).drop_duplicates()
    print(f"\n🆕 New rows (seeds + children): {len(df_new)}")

    # Merge
    combined = pd.concat([df_old, df_new], ignore_index=True)
    combined_unique = combined.drop_duplicates(
        subset=["mesh_id", "mesh_name", "origin", "child_of"]
    ).reset_index(drop=True)

    print("\n📊 Final statistics:")
    print(f"   Old rows      : {len(df_old)}")
    print(f"   New rows      : {len(df_new)}")
    print(f"   Combined uniq : {len(combined_unique)}")

    # PRINT nicely
    print("\n📋 Final Combined Table:\n")
    pretty_print_table(combined_unique)

    # WRITE new CSV
    combined_unique.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Saved expanded comprehensive list to:\n   → {OUTPUT_CSV}\n")


if __name__ == "__main__":
    main()
