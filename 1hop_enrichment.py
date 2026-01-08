import argparse
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from indra.databases import hgnc_client
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client import get_mesh_ids_for_pmids

# Silence noisy logs
logging.getLogger("indra_cogex").setLevel(logging.ERROR)
logging.getLogger("indra_cogex.client.queries").setLevel(logging.ERROR)

GWAS_GENES = {
    "ARHGEF26", "BCAR1", "BMP1", "CALCRL", "CCM2", "CDKN1A", "CDKN2B", "CFDP1", "COL4A1", "COL4A2", "EDN1", "EXOC3L2",
    "FBN2", "FGD6", "FLT1", "FURIN", "GDPD5", "GGT5", "GOSR2", "IBTK", "JCAD", "LAMB2", "LOX", "MORF4L1", "N4BP2L2",
    "NOS3", "PALLD", "PECAM1", "PGF", "PLPP3", "PRDM16", "PREX1", "PRKAR1A", "SCUBE1", "SERPINH1", "SH3PXD2A", "SLK",
    "SMAD3", "SPRY4", "SVIL", "SWAP70", "TFPI", "TLNRD1", "TSPAN14", "ZEB2",
}

INDRA_URL_FMT = "https://db.indra.bio/statements/from_hash/{h}?format=html"
MESH_PAIR_RE = re.compile(r"([^,]+?)\((D\d{5,10})\)")


def is_nonempty(x) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and pd.isna(x):
        return False
    s = str(x).strip()
    return s != "" and s.lower() not in {"nan", "none"}


def parse_pmids_cell(x) -> list[str]:
    if not is_nonempty(x):
        return []
    pmids = []
    for tok in str(x).replace(" ", "").split(";"):
        if tok.isdigit():
            pmids.append(tok)
    return pmids


def load_valid_mesh_ids(reference_csv: str) -> set[str]:
    ref = pd.read_csv(reference_csv, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
    if "mesh_id" not in ref.columns:
        raise ValueError("Reference file must contain a 'mesh_id' column.")
    ref["mesh_id"] = (
        ref["mesh_id"]
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.replace(r"[^A-Za-z0-9]", "", regex=True)
        .str.upper()
    )
    valid = set(ref.loc[ref["mesh_id"].str.startswith("D", na=False), "mesh_id"].tolist())
    return valid


def build_mesh_id_to_name_map(client: Neo4jClient) -> dict[str, str]:
    query = "MATCH (b:BioEntity) WHERE b.id STARTS WITH 'mesh:' RETURN b.id AS mesh_id, b.name AS mesh_name"
    results = client.query_tx(query)
    mapping = {}
    for rec in results:
        if isinstance(rec, dict):
            mid, name = rec.get("mesh_id"), rec.get("mesh_name")
        else:
            mid, name = rec[0], rec[1]
        if mid and name:
            norm = str(mid).replace("mesh:", "").upper()
            mapping[norm] = str(name)
    return mapping


def annotate_mesh_terms(df: pd.DataFrame, pmids_col: str, client: Neo4jClient, mesh_batch_size: int, valid_mesh_ids: set[str]) -> pd.Series:
    # collect all unique pmids
    all_pmids = set()
    for v in df[pmids_col].tolist():
        all_pmids.update(parse_pmids_cell(v))
    all_pmids = sorted(all_pmids, key=lambda x: int(x))

    pmid_to_mesh = {}
    for i in range(0, len(all_pmids), mesh_batch_size):
        batch = all_pmids[i:i + mesh_batch_size]
        pmid_to_mesh.update(get_mesh_ids_for_pmids(batch, client=client))

    mesh_id_to_name = build_mesh_id_to_name_map(client)

    def annotate_row(v):
        pmids = parse_pmids_cell(v)
        mesh_ids = set()
        for pmid in pmids:
            mesh_ids.update(pmid_to_mesh.get(pmid, []) or [])
        # normalize + filter to reference list (D only)
        kept = []
        for mid in sorted(mesh_ids):
            mid_u = str(mid).upper().strip()
            if mid_u in valid_mesh_ids:
                name = mesh_id_to_name.get(mid_u)
                if name:
                    kept.append(f"{name} ({mid_u})")
        return ", ".join(kept)

    return df[pmids_col].apply(annotate_row)


def looks_like_structured_evidence(text: str) -> bool:
    if not is_nonempty(text):
        return False
    s = str(text)
    return bool(re.search(r"(^|\n)\s*1\)\s+", s))


def compute_directionality(stmt_type: str, logfc) -> str:
    try:
        lfc = float(logfc)
    except Exception:
        return ""
    if stmt_type == "IncreaseAmount" and lfc < 0:
        return "Yes"
    if stmt_type == "DecreaseAmount" and lfc > 0:
        return "Yes"
    return "No"


def safe_float(x, default=None):
    try:
        v = float(x)
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default


def choose_stmt_hash(rows, target_belief=None, target_evcnt=None):
    """
    rows: list[dict] with keys stmt_hash, belief, evidence_count
    preference: evidence_count match (if provided) then closest belief (if provided),
                else max evidence_count then max belief.
    """
    if not rows:
        return None

    if target_evcnt is not None:
        ev_matches = [r for r in rows if r.get("evidence_count") == target_evcnt]
        if ev_matches:
            if target_belief is not None:
                ev_matches.sort(key=lambda r: abs((r.get("belief") or 0.0) - target_belief))
            else:
                ev_matches.sort(key=lambda r: (r.get("evidence_count") or 0, r.get("belief") or 0.0), reverse=True)
            return ev_matches[0].get("stmt_hash")

    if target_belief is not None:
        rows_sorted = sorted(rows, key=lambda r: abs((r.get("belief") or 0.0) - target_belief))
        return rows_sorted[0].get("stmt_hash")

    rows_sorted = sorted(rows, key=lambda r: (r.get("evidence_count") or 0, r.get("belief") or 0.0), reverse=True)
    return rows_sorted[0].get("stmt_hash")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--mesh-reference", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--mesh-batch-size", type=int, default=200)
    ap.add_argument("--hash-workers", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv, low_memory=False)

    # --- validate columns ---
    for c in ["source", "target"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # stmt_type naming varies in 1-hop files
    if "stmt_type" in df.columns:
        stmt_col = "stmt_type"
    elif "stmt_type_1" in df.columns:
        stmt_col = "stmt_type_1"
    else:
        raise ValueError("Missing stmt type column (expected stmt_type or stmt_type_1).")

    # pmids column naming varies
    if "pmids" in df.columns:
        pmids_col = "pmids"
    elif "pmids_hop1" in df.columns:
        pmids_col = "pmids_hop1"
    else:
        raise ValueError("Missing pmids column (expected pmids or pmids_hop1).")

    if "logfoldchange" not in df.columns:
        raise ValueError("Missing logfoldchange column (needed for directionality).")

    # --- 1) MeSH annotate + reference filter ---
    client = Neo4jClient()
    valid_mesh_ids = load_valid_mesh_ids(args.mesh_reference)
    annotated = annotate_mesh_terms(df, pmids_col=pmids_col, client=client, mesh_batch_size=args.mesh_batch_size, valid_mesh_ids=valid_mesh_ids)

    # insert after pmids col
    out_mesh_col = "Annotated MeSH terms"
    if out_mesh_col in df.columns:
        df.drop(columns=[out_mesh_col], inplace=True)
    insert_at = df.columns.get_loc(pmids_col) + 1
    df.insert(insert_at, out_mesh_col, annotated)

    # --- 2) directionality ---
    df["directionality"] = df.apply(lambda r: compute_directionality(r.get(stmt_col, ""), r.get("logfoldchange")), axis=1)

    # --- 3) GWAS_genes_in_path (1-hop: only source/target) ---
    def gwas_in_path_row(r):
        hits = []
        s = str(r.get("source", "")).strip()
        t = str(r.get("target", "")).strip()
        if s in GWAS_GENES:
            hits.append(s)
        if t in GWAS_GENES:
            hits.append(t)
        return ", ".join(sorted(set(hits)))

    df["GWAS_genes_in_path"] = df.apply(gwas_in_path_row, axis=1)

    # --- 4) stmt_hash + indra_url ---
    if "stmt_hash" not in df.columns:
        df["stmt_hash"] = ""
    if "indra_url" not in df.columns:
        df["indra_url"] = ""

    # Optional belief/evidence_count columns (if present, used to pick best stmt_hash)
    belief_col = "belief" if "belief" in df.columns else ("belief_1" if "belief_1" in df.columns else None)
    evcnt_col = "evidence_count" if "evidence_count" in df.columns else ("evidence_1" if "evidence_1" in df.columns else None)

    QUERY = """
    MATCH (source:BioEntity {id: $source_id})-[r:indra_rel {stmt_type: $stmt_type}]->(target:BioEntity {id: $target_id})
    RETURN r.stmt_hash AS stmt_hash, r.belief AS belief, r.evidence_count AS evidence_count
    """

    def hgnc_id_for(symbol):
        if not is_nonempty(symbol):
            return None
        hid = hgnc_client.get_current_hgnc_id(str(symbol).strip())
        # handle ambiguous mappings (list)
        if isinstance(hid, (list, tuple, set)):
            hid = next(iter(hid), None)
        return hid

    def fetch_hash_for_idx(i):
        r = df.loc[i]
        src = str(r["source"]).strip()
        tgt = str(r["target"]).strip()
        stmt_type = str(r[stmt_col]).strip()

        hid1 = hgnc_id_for(src)
        hid2 = hgnc_id_for(tgt)
        if not hid1 or not hid2 or not stmt_type:
            return i, "", ""

        source_id = f"hgnc:{hid1}"
        target_id = f"hgnc:{hid2}"

        rows = client.query_tx(QUERY, source_id=source_id, target_id=target_id, stmt_type=stmt_type) or []
        norm = []
        for rec in rows:
            if isinstance(rec, dict):
                norm.append({
                    "stmt_hash": rec.get("stmt_hash"),
                    "belief": safe_float(rec.get("belief"), default=None),
                    "evidence_count": int(rec.get("evidence_count")) if is_nonempty(rec.get("evidence_count")) else None,
                })
            else:
                norm.append({
                    "stmt_hash": rec[0],
                    "belief": safe_float(rec[1], default=None),
                    "evidence_count": int(rec[2]) if rec[2] is not None else None,
                })

        target_belief = safe_float(r.get(belief_col), default=None) if belief_col else None
        target_evcnt = None
        if evcnt_col and is_nonempty(r.get(evcnt_col)):
            try:
                target_evcnt = int(float(r.get(evcnt_col)))
            except Exception:
                target_evcnt = None

        h = choose_stmt_hash(norm, target_belief=target_belief, target_evcnt=target_evcnt)
        if h is None:
            return i, "", ""
        h = int(h)
        return i, str(h), INDRA_URL_FMT.format(h=h)

    pending = [i for i in df.index if (not is_nonempty(df.at[i, "stmt_hash"]) or not is_nonempty(df.at[i, "indra_url"]))]

    if pending:
        with ThreadPoolExecutor(max_workers=args.hash_workers) as ex:
            futs = [ex.submit(fetch_hash_for_idx, i) for i in pending]
            for fut in as_completed(futs):
                i, h, url = fut.result()
                if h:
                    df.at[i, "stmt_hash"] = h
                    df.at[i, "indra_url"] = url

    # --- reorder: directionality BEFORE stmt_hash + indra_url ---
    cols = df.columns.tolist()
    for c in ["directionality", "stmt_hash", "indra_url"]:
        if c in cols:
            cols.remove(c)

    # Keep existing order, append these 3 at the end in required order
    cols.extend(["directionality", "stmt_hash", "indra_url"])
    df = df[cols]

    df.to_csv(args.out_csv, index=False)
    print(f"Saved -> {args.out_csv} (rows={len(df):,})")


if __name__ == "__main__":
    main()