"""Legacy script: expand MeSH terms list with mechanistic seeds."""
from __future__ import annotations

import argparse

import pandas as pd
from indra_cogex.client.neo4j_client import Neo4jClient

import logging


logger = logging.getLogger(__name__)

CHILD_DEPTH = 4
NEW_ORIGIN = "CAD Mechanistic"

BROAD_STOPLIST = {
    "D006801",
    "D000818",
    "D002477",
    "D005260",
    "D010801",
    "D000740",
}

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

    logger.info(fmt.format(*cols))
    logger.info("  ".join("-" * widths[c] for c in cols))

    for _, row in df.iterrows():
        logger.info(fmt.format(row["mesh_id"], row["mesh_name"],
                               row["origin"], row["child_of"]))


def main():
    ap = argparse.ArgumentParser(description="Expand MeSH terms list with mechanistic seeds.")
    ap.add_argument("--existing-csv", required=True, help="Path to existing comprehensive MeSH list CSV.")
    ap.add_argument("--output-csv", required=True, help="Path for expanded output CSV.")
    ap.add_argument("--child-depth", type=int, default=CHILD_DEPTH, help="Depth for child expansion.")
    args = ap.parse_args()

    logger.info("Loading existing list: %s", args.existing_csv)
    df_old = pd.read_csv(args.existing_csv, dtype=str).fillna("")
    logger.info("Existing rows: %d", len(df_old))

    client = Neo4jClient()

    logger.info("Expanding mechanistic seeds (depth=%d)...", args.child_depth)
    new_rows = []

    for mesh_id, mesh_name in MECHANISTIC_SEEDS.items():
        new_rows.append({
            "mesh_id": mesh_id,
            "mesh_name": mesh_name,
            "origin": NEW_ORIGIN,
            "child_of": "None"
        })

        children = get_children_for_mesh_id(mesh_id, client, args.child_depth)
        logger.info("  %s (%s) -> %d children", mesh_name, mesh_id, len(children))

        for cid, cname in children:
            new_rows.append({
                "mesh_id": cid,
                "mesh_name": cname,
                "origin": NEW_ORIGIN,
                "child_of": mesh_name
            })

    df_new = pd.DataFrame(new_rows).drop_duplicates()
    logger.info("New rows (seeds + children): %d", len(df_new))

    combined = pd.concat([df_old, df_new], ignore_index=True)
    combined_unique = combined.drop_duplicates(
        subset=["mesh_id", "mesh_name", "origin", "child_of"]
    ).reset_index(drop=True)

    logger.info("Final statistics:")
    logger.info("  Old rows      : %d", len(df_old))
    logger.info("  New rows      : %d", len(df_new))
    logger.info("  Combined uniq : %d", len(combined_unique))

    pretty_print_table(combined_unique)

    combined_unique.to_csv(args.output_csv, index=False)
    logger.info("Saved expanded comprehensive list to: %s", args.output_csv)


if __name__ == "__main__":
    main()
