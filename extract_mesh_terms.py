import time
import pandas as pd
from typing import Iterable, List, Tuple, Set, Dict, Optional
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.representation import norm_id
from indra_cogex.client import get_mesh_ids_for_pmids


# =========================
# CONFIG
# =========================
OUTPUT_FILE = "/Users/prashammarfatia/Downloads/comprehensive_mesh_list_with_origin.csv"
PAPER_PMID = "38326615"  # Convergence of CAD genes onto endothelial cell programs
CHILD_DEPTH = 3  # Expand children up to this depth


# === Stoplist: filter broad MeSH terms ===
BROAD_STOPLIST = {
    "D006801",  # Humans
    "D000818",  # Animals
    "D002477",  # Cells
    "D005260",  # Enzymes
    "D010801",  # Proteins
    "D000740",  # Anatomy
    "D009944",  # Metabolism
    "D012679",  # Signal Transduction
    "D004958",  # Genes
    "D013964",  # Subcellular Fractions
}


# =========================
# SEED FAMILIES
# =========================
CAD_SEEDS = [
    ("D003324", "Coronary Artery Disease"),
    ("D001161", "Atherosclerosis"),
    ("D009203", "Myocardial Ischemia"),
    ("D009204", "Myocardial Infarction"),
    ("D006331", "Ischemic Cardiomyopathy"),
    ("D006333", "Heart Failure, Ischemic"),
    ("D014659", "Coronary Vasospasm"),
    ("D000787", "Angina Pectoris"),
    ("D054058", "Coronary Stenosis"),
    ("D003326", "Coronary Thrombosis"),
    ("D016893", "Coronary Restenosis"),
]

ENDOTHELIAL_SEEDS = [
    ("D004715", "Endothelial Cells"),
    ("D000070678", "Vascular Endothelial Cells"),
    ("D000068820", "Lymphatic Endothelial Cells"),
    ("D000068821", "Microvascular Endothelial Cells"),
    ("D000071254", "Venous Endothelial Cells"),
    ("D000070679", "Arterial Endothelial Cells"),
    ("D000071261", "Coronary Endothelial Cells"),
    ("D004717", "Endothelium, Vascular"),
    ("D004718", "Endothelium, Lymphatic"),
    ("D000068879", "Endothelial Dysfunction"),
]

HEART_SEEDS = [
    ("D006321", "Heart"),
    ("D004698", "Endocardium"),
    ("D005315", "Fetal Heart"),
    ("D006329", "Heart Atria"),
    ("D006330", "Heart Conduction System"),
    ("D006331", "Heart Septum"),
    ("D006333", "Heart Valves"),
    ("D006334", "Heart Ventricles"),
    ("D009205", "Myocardium"),
    ("D010493", "Pericardium"),
]

HEART_DISEASE_SEEDS = [
    ("D006331", "Heart Diseases"),
    ("D001145", "Arrhythmias, Cardiac"),
    ("D006301", "Carcinoid Heart Disease"),
    ("D018377", "Cardiac Conduction System Disease"),
    ("D006329", "Cardiac Output, High"),
    ("D006330", "Cardiac Output, Low"),
    ("D002319", "Cardiac Tamponade"),
    ("D006333", "Cardiomegaly"),
    ("D009203", "Cardiomyopathies"),
    ("D000068877", "Cardiotoxicity"),
    ("D004696", "Endocarditis"),
    ("D006324", "Heart Aneurysm"),
    ("D006323", "Heart Arrest"),
    ("D006402", "Heart Defects, Congenital"),
    ("D006333", "Heart Failure"),
    ("D006330", "Heart Neoplasms"),
    ("D006329", "Heart Rupture"),
    ("D006484", "Heart Valve Diseases"),
    ("D009203", "Myocardial Ischemia"),
    ("D015217", "Myocardial Stunning"),
    ("D010494", "Pericardial Effusion"),
    ("D010496", "Pericarditis"),
    ("D011005", "Pneumopericardium"),
    ("D000068879", "Post-Cardiac Arrest Syndrome"),
    ("D011339", "Postpericardiotomy Syndrome"),
    ("D011662", "Pulmonary Heart Disease"),
    ("D012213", "Rheumatic Heart Disease"),
    ("D014658", "Ventricular Dysfunction"),
    ("D014662", "Ventricular Outflow Obstruction"),
]


# =========================
# HELPERS
# =========================
def _safe_norm_mesh(db_id: str) -> str:
    """Normalize or safely format MeSH ID."""
    try:
        curie = norm_id("mesh", db_id)
        return curie or f"mesh:{db_id}"
    except Exception:
        return f"mesh:{db_id}"


def _get_mesh_child_terms(
    mesh_term: Tuple[str, str],
    depth: Optional[int] = None,
    *,
    client: Neo4jClient
) -> Set[Tuple[str, str, str]]:
    """Return (mesh_id, mesh_name, parent_name) for all descendants."""
    db_id, parent_name = mesh_term
    meshid_norm = _safe_norm_mesh(db_id)
    if not meshid_norm:
        return set()

    depth_clause = f"*1..{depth}" if depth and depth > 0 else "*1.."
    query = f"""
        MATCH (c:BioEntity)-[:isa|partof{depth_clause}]->(:BioEntity {{id: $mesh_id}})
        RETURN DISTINCT c.id AS id, c.name AS name
    """

    try:
        rows = client.query_tx(query, mesh_id=meshid_norm)
    except Exception as e:
        print(f"⚠️ Error expanding {db_id}: {e}")
        return set()

    out = set()
    for r in rows:
        cid = r.get("id") if isinstance(r, dict) else r[0]
        cname = r.get("name") if isinstance(r, dict) else (r[1] if len(r) > 1 else "")
        if not cid or cid.replace("mesh:", "") in BROAD_STOPLIST:
            continue
        out.add((cid.replace("mesh:", ""), cname or "", parent_name))
    return out


def _expand_family(
    seeds: Iterable[Tuple[str, str]],
    depth: Optional[int],
    origin: str,
    client: Neo4jClient
) -> Set[Tuple[str, str, str, str]]:
    """Expand a family of MeSH seeds and label origin + child_of."""
    results = set()
    for mid, name in seeds:
        results.add((mid, name, origin, ""))  # Root term
        children = _get_mesh_child_terms((mid, name), depth=depth, client=client)
        for cid, cname, parent in children:
            results.add((cid, cname, origin, parent))
    return results


def _get_names_for_mesh_ids(mesh_ids: List[str], *, client: Neo4jClient) -> Dict[str, str]:
    """Fetch human-readable MeSH names for given MeSH IDs."""
    if not mesh_ids:
        return {}
    normalized = [_safe_norm_mesh(mid) for mid in mesh_ids]
    query = """
        UNWIND $ids AS mid
        MATCH (m:BioEntity {id: mid})
        RETURN m.id AS id, m.name AS name
    """
    try:
        rows = client.query_tx(query, ids=normalized)
    except Exception as e:
        print(f"⚠️ Error fetching MeSH names: {e}")
        return {}

    name_map = {}
    for r in rows:
        if isinstance(r, dict):
            mesh_id = r.get("id", "").replace("mesh:", "")
            mesh_name = r.get("name", "")
        else:
            mesh_id = r[0].replace("mesh:", "")
            mesh_name = r[1] if len(r) > 1 else ""
        if mesh_id:
            name_map[mesh_id] = mesh_name
    return name_map


def main():
    t0 = time.time()
    client = Neo4jClient()

    # --- Extract from PubMed ---
    print(f"📄 Extracting MeSH from PubMed {PAPER_PMID}")
    paper_map = get_mesh_ids_for_pmids([PAPER_PMID], client=client)
    raw_terms = paper_map.get(PAPER_PMID, []) or []
    filtered_terms = [mid for mid in raw_terms if mid not in BROAD_STOPLIST]

    expanded_paper = set()
    for mid in filtered_terms:
        expanded_paper.add((mid, "", "Paper", ""))  # names filled later

    # --- Expand Families ---
    print("🫀 Expanding CAD, Endothelial, Heart, Heart Disease families...")
    expanded_cad = _expand_family(CAD_SEEDS, CHILD_DEPTH, "CAD", client)
    expanded_endo = _expand_family(ENDOTHELIAL_SEEDS, CHILD_DEPTH, "Endothelial", client)
    expanded_heart = _expand_family(HEART_SEEDS, CHILD_DEPTH, "Heart", client)
    expanded_hd = _expand_family(HEART_DISEASE_SEEDS, CHILD_DEPTH, "Heart Disease", client)

    # --- Combine ---
    combined = expanded_paper | expanded_cad | expanded_endo | expanded_heart | expanded_hd

    # --- Fetch missing names ---
    all_ids = [r[0] for r in combined if not r[1]]
    if all_ids:
        name_map = _get_names_for_mesh_ids(all_ids, client=client)
        updated = set()
        for mid, name, origin, parent in combined:
            if not name:
                name = name_map.get(mid, "")
            updated.add((mid, name, origin, parent))
        combined = updated

    # --- Save ---
    df = pd.DataFrame(sorted(combined), columns=["mesh_id", "mesh_name", "origin", "child_of"])
    df.drop_duplicates(subset=["mesh_id", "origin"], inplace=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Total unique MeSH terms: {len(df)}")
    print(f"💾 Saved merged MeSH catalog → {OUTPUT_FILE}")
    print(f"⏱ Done in {(time.time() - t0)/60:.2f} minutes.")


if __name__ == "__main__":
    main()
