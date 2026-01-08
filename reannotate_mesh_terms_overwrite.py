import argparse
import pandas as pd
import re

from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client import get_mesh_ids_for_pmids


def load_valid_mesh_ids(reference_csv: str) -> set[str]:
    ref = pd.read_csv(reference_csv, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
    if "mesh_id" not in ref.columns:
        raise ValueError(f"Reference CSV must contain 'mesh_id'. Columns={ref.columns.tolist()}")
    s = (
        ref["mesh_id"]
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.replace(r"[^A-Za-z0-9]", "", regex=True)
        .str.upper()
    )
    return set(s[s.str.startswith("D", na=False)].tolist())


def parse_pmids_cell(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, float) and pd.isna(x):
        return []
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return []
    return [tok for tok in s.replace(" ", "").split(";") if tok.isdigit()]


def batch_iter(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def build_mesh_id_to_name_for_ids(client: Neo4jClient, mesh_ids: list[str]) -> dict[str, str]:
    curies = [f"mesh:{mid}" for mid in mesh_ids]
    q = """
    UNWIND $ids AS mid
    MATCH (b:BioEntity {id: mid})
    RETURN b.id AS mesh_id, b.name AS mesh_name
    """
    rows = client.query_tx(q, ids=curies)
    m = {}
    for r in rows:
        if isinstance(r, dict):
            mid = r.get("mesh_id", "")
            name = r.get("mesh_name", "")
        else:
            mid = r[0] if len(r) > 0 else ""
            name = r[1] if len(r) > 1 else ""
        if mid and name:
            m[mid.replace("mesh:", "").upper()] = str(name)
    return m


def annotate_from_pmids(df: pd.DataFrame, pmid_col: str, pmid_to_mesh: dict, mesh_id_to_name: dict, valid_ids: set[str]) -> pd.Series:
    def f(cell):
        pmids = parse_pmids_cell(cell)
        mesh_ids = set()
        for p in pmids:
            mesh_ids.update(pmid_to_mesh.get(p, []) or [])
        kept = []
        for mid in sorted(mesh_ids):
            mid_u = str(mid).upper().strip()
            if mid_u in valid_ids:
                name = mesh_id_to_name.get(mid_u)
                if name:
                    kept.append(f"{name} ({mid_u})")
        return ", ".join(kept)
    return df[pmid_col].apply(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument(
        "--mode",
        choices=["1hop", "2hop"],
        required=True,
        help="1hop overwrites 'Annotated MeSH terms' using 'pmids'. 2hop overwrites hop1/hop2 columns using pmids_hop1/pmids_hop2.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    valid_ids = load_valid_mesh_ids(args.reference)
    print(f"Loaded {len(valid_ids):,} valid MeSH IDs from reference")

    if args.mode == "1hop":
        pmid_cols = ["pmids"]
        out_cols = ["Annotated MeSH terms"]
    else:
        pmid_cols = ["pmids_hop1", "pmids_hop2"]
        out_cols = ["Annotated MeSH terms hop1", "Annotated MeSH terms hop2"]

    for c in pmid_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # collect all PMIDs
    all_pmids = set()
    for c in pmid_cols:
        for v in df[c].tolist():
            all_pmids.update(parse_pmids_cell(v))
    all_pmids = sorted(all_pmids, key=lambda x: int(x))
    print(f"Unique PMIDs found: {len(all_pmids):,}")

    # fetch PMID -> MeSH IDs
    client = Neo4jClient()
    pmid_to_mesh = {}
    for batch in batch_iter(all_pmids, args.batch_size):
        pmid_to_mesh.update(get_mesh_ids_for_pmids(batch, client=client))

    # fetch names only for observed descriptor IDs
    seen_mesh = set()
    for mids in pmid_to_mesh.values():
        for mid in (mids or []):
            mid_u = str(mid).upper().strip()
            if re.fullmatch(r"D\d{5,10}", mid_u):
                seen_mesh.add(mid_u)
    seen_mesh = sorted(seen_mesh)
    print(f"Unique MeSH descriptor IDs observed from PMIDs: {len(seen_mesh):,}")

    mesh_id_to_name = build_mesh_id_to_name_for_ids(client, seen_mesh)

    # overwrite target columns
    for pmid_col, out_col in zip(pmid_cols, out_cols):
        df[out_col] = annotate_from_pmids(df, pmid_col, pmid_to_mesh, mesh_id_to_name, valid_ids)
        print(f"Overwrote column: {out_col}")

    df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()