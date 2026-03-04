"""Superseded legacy script for 2-hop one-stop pipeline.

End-to-end: 2hop extraction -> endothelial intermediate filter ->
evidence/pmids -> mesh -> gwas/directionality -> stmt hash + html urls.

Refactored into src/indra_perturbseq/pipelines/.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local

import pandas as pd

from indra_cogex.analysis.source_targets_explanation import get_valid_gene_ids
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client.queries import get_statements
from indra_cogex.client import get_mesh_ids_for_pmids
from indra.databases import hgnc_client
from indra.databases.hgnc_client import get_current_hgnc_id, get_hgnc_name

logger = logging.getLogger(__name__)

logging.getLogger("indra_cogex").setLevel(logging.WARNING)
logging.getLogger("indra_cogex.client.queries").setLevel(logging.WARNING)

GWAS_GENES = {
    "BCAR1", "BMP1", "CALCRL", "CCM2", "CDKN1A", "CDKN2B", "CFDP1", "COL4A1", "COL4A2",
    "EXOC3L2", "FBN2", "FGD6", "FLT1", "FURIN", "GDPD5", "GGT5", "GOSR2", "IBTK", "LAMB2",
    "LOX", "MORF4L1", "N4BP2L2", "NOS3", "PALLD", "PECAM1", "PGF", "PLPP3", "PREX1", "PRKAR1A",
    "SCUBE1", "SERPINH1", "SH3PXD2A", "SLK", "SMAD3", "SPRY4", "SVIL", "SWAP70", "TFPI",
    "TLNRD1", "TSPAN14", "ZEB2",
}

BATCH_2HOP_QUERY = """
UNWIND $target_list AS target_id
MATCH (a:BioEntity {id: $source})-[r1:indra_rel]->(m:BioEntity)-[r2:indra_rel]->(b:BioEntity {id: target_id})
WHERE r2.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
RETURN a.id, m.id, b.id, target_id,
       r1.stmt_type, r2.stmt_type,
       r1.belief, r2.belief,
       r1.evidence_count, r2.evidence_count
"""

EVIDENCE_JSON_QUERY = """
MATCH (source:BioEntity {id: $source_id})-[r:indra_rel {stmt_type: $stmt_type}]->(target:BioEntity {id: $target_id})
WITH r.stmt_hash AS stmt_hash
MATCH (e:Evidence {stmt_hash: stmt_hash})
RETURN e.evidence
"""

_thread_state = local()


def get_thread_client():
    if not hasattr(_thread_state, "client") or _thread_state.client is None:
        _thread_state.client = Neo4jClient()
    return _thread_state.client


def format_evidence_text(text: str) -> str:
    """Convert '1. ...' -> '1) ...' and put blank lines between items."""
    if not isinstance(text, str):
        return text
    if text.startswith("Evidence from:") or text.startswith("No evidence found"):
        return text
    pattern = r"(^|\n|; )(\d+\.\s)"
    parts = []
    last_idx = 0
    for match in re.finditer(pattern, text):
        start = match.start(2)
        if start > last_idx:
            parts.append(text[last_idx:start].strip())
        last_idx = start
    parts.append(text[last_idx:].strip())
    return "\n\n".join([re.sub(r"^(\d+)\.\s", r"\1) ", p) for p in parts])


def normalize_gene_symbol(symbol: str) -> str:
    """Normalize to current HGNC symbol when possible."""
    if symbol is None or (isinstance(symbol, float) and pd.isna(symbol)):
        return symbol

    if isinstance(symbol, (list, tuple, set)):
        symbol = "; ".join([str(x) for x in symbol if x is not None])

    symbol = str(symbol).strip()
    if symbol == "":
        return symbol

    hid = hgnc_client.get_current_hgnc_id(symbol)

    if isinstance(hid, (list, tuple, set)):
        names = []
        for one in hid:
            try:
                n = hgnc_client.get_hgnc_name(one)
            except Exception:
                n = None
            names.append(n or str(one))
        return "; ".join([n for n in names if n])

    if hid:
        try:
            return hgnc_client.get_hgnc_name(hid) or symbol
        except Exception:
            return symbol

    return symbol


def load_endothelial_gene_set(path: str) -> set[str]:
    df = pd.read_csv(path)
    if "gene" not in df.columns:
        raise ValueError(f"Endothelial list must have a 'gene' column. Columns: {df.columns.tolist()}")
    genes = set(df["gene"].astype(str).str.strip())
    genes.discard("")
    return genes


def pick_sig_column(df: pd.DataFrame, prefer_fdr: bool) -> str:
    fdr_candidates = ["pvals_adj", "padj", "qval", "fdr", "p_adj"]
    p_candidates = ["pvals", "pval", "p_value", "p_val"]
    fdr_col = next((c for c in fdr_candidates if c in df.columns), None)
    p_col = next((c for c in p_candidates if c in df.columns), None)
    if prefer_fdr and fdr_col:
        return fdr_col
    if p_col:
        return p_col
    if fdr_col:
        return fdr_col
    raise ValueError(f"DEG file missing p-value columns. Columns: {df.columns.tolist()}")


def run_2hop_for_gene(
    gene: str,
    deg_dir: str,
    p_threshold: float,
    prefer_fdr: bool,
    batch_size: int,
    allowed_intermediates: set[str],
) -> list[dict]:
    gene = gene.strip()
    deg_path = os.path.join(deg_dir, f"{gene}_vs_control.csv")
    if not os.path.exists(deg_path):
        logger.warning("SKIP: missing DEG file for %s: %s", gene, deg_path)
        return []

    hgnc_id = get_current_hgnc_id(gene.upper())
    if not hgnc_id:
        logger.warning("SKIP: no HGNC id for %s", gene)
        return []
    source_id = f"hgnc:{hgnc_id}"

    df = pd.read_csv(deg_path, low_memory=False)
    if "names" not in df.columns:
        raise ValueError(f"{deg_path} missing 'names'. Columns: {df.columns.tolist()}")

    sig_col = pick_sig_column(df, prefer_fdr=prefer_fdr)
    df[sig_col] = pd.to_numeric(df[sig_col], errors="coerce")
    df = df[df[sig_col] < p_threshold].copy()
    if df.empty:
        return []

    df["names"] = df["names"].astype(str)
    if "logfoldchanges" in df.columns:
        df["logfoldchanges"] = pd.to_numeric(df["logfoldchanges"], errors="coerce")
    else:
        df["logfoldchanges"] = pd.NA

    target_symbols = df["names"].dropna().unique().tolist()
    converted = get_valid_gene_ids(target_symbols)
    symbol_to_hgnc = {sym: f"hgnc:{hid}" for sym, hid in zip(target_symbols, converted) if hid}
    if not symbol_to_hgnc:
        return []

    hgnc_to_symbol = {v: k for k, v in symbol_to_hgnc.items()}
    target_ids = list(symbol_to_hgnc.values())
    deg_map = df.set_index("names")[["logfoldchanges", sig_col]].to_dict("index")

    client = Neo4jClient()
    out = []

    for i in range(0, len(target_ids), batch_size):
        batch_targets = target_ids[i:i + batch_size]
        rows = client.query_tx(BATCH_2HOP_QUERY, source=source_id, target_list=batch_targets)

        for r in rows:
            _, intermediate_id, _, target_id, stmt1, stmt2, belief1, belief2, ev1, ev2 = r[:10]

            if isinstance(intermediate_id, str) and intermediate_id.startswith("hgnc:"):
                interm_symbol = get_hgnc_name(intermediate_id.replace("hgnc:", "")) or intermediate_id
            else:
                interm_symbol = intermediate_id

            if str(interm_symbol) not in allowed_intermediates:
                continue

            target_symbol = hgnc_to_symbol.get(target_id, target_id.replace("hgnc:", ""))
            stats = deg_map.get(target_symbol, {})
            out.append({
                "source": gene,
                "intermediate": str(interm_symbol),
                "target": str(target_symbol),
                "stmt_type_1": stmt1,
                "stmt_type_2": stmt2,
                "belief_1": belief1,
                "belief_2": belief2,
                "evidence_1": ev1,
                "evidence_2": ev2,
                "logfoldchange": stats.get("logfoldchanges", None),
                "pval": stats.get(sig_col, None),
                "pval_col_used": sig_col,
            })

    return out


def get_evidence_info(agent1: str, agent2: str, stmt_type: str, client: Neo4jClient):
    """Return (db_sources_text, pmids_list) via Neo4j Evidence nodes."""
    h1 = hgnc_client.get_current_hgnc_id(agent1)
    h2 = hgnc_client.get_current_hgnc_id(agent2)
    if not h1 or not h2:
        return "No evidence found", []

    results = client.query_tx(
        EVIDENCE_JSON_QUERY,
        source_id=f"hgnc:{h1}",
        target_id=f"hgnc:{h2}",
        stmt_type=stmt_type,
    )
    if not results:
        return "No evidence found", []

    sources_seen = OrderedDict()
    pmids_seen = set()

    for (ev_json,) in results:
        try:
            ev = json.loads(ev_json)
        except Exception:
            continue

        pmid = ev.get("pmid")
        if pmid:
            pmids_seen.add(str(pmid))

        source_api = ev.get("source_api", "") or ""
        source_sub_id = (ev.get("annotations", {}) or {}).get("source_sub_id", "") or ""
        key = f"{source_api}:{source_sub_id}" if source_sub_id else source_api
        if key:
            sources_seen[key] = None

    db_info = f"Evidence from: {', '.join(sources_seen.keys())}" if sources_seen else "No evidence found"
    pmids = sorted(pmids_seen, key=lambda x: int(x) if x.isdigit() else x)
    return db_info, pmids


def fetch_evidence_text(agent1: str, agent2: str, stmt_type: str, client: Neo4jClient, limit: int = 20):
    """Fetch statement evidence text via get_statements; fallback to Neo4j source list text."""
    try:
        stmts = get_statements(
            agent=agent1,
            other_agent=agent2,
            agent_role="subject",
            other_role="object",
            rel_types=stmt_type,
            limit=limit,
            evidence_limit=limit,
            client=client,
        )
        if not stmts:
            db_info, _ = get_evidence_info(agent1, agent2, stmt_type, client)
            return db_info

        evidences = []
        counter = 1
        for stmt in stmts:
            for ev in (stmt.evidence or []):
                if ev.text:
                    evidences.append(f"{counter}. {ev.text.strip()}")
                    counter += 1

        if evidences:
            return "\n".join(evidences)

        db_info, _ = get_evidence_info(agent1, agent2, stmt_type, client)
        return db_info

    except Exception as e:
        return f"Error fetching evidence: {e}"


def enrich_with_evidence(
    df: pd.DataFrame,
    out_csv: str,
    max_workers: int,
    checkpoint_every: int,
    resume: bool,
) -> pd.DataFrame:
    df = df.copy()
    for col in ["evidence_text_hop1", "pmids_hop1", "evidence_text_hop2", "pmids_hop2"]:
        if col not in df.columns:
            df[col] = ""

    def row_needs(i: int) -> bool:
        return (
            (not isinstance(df.at[i, "evidence_text_hop1"], str) or df.at[i, "evidence_text_hop1"] == "") or
            (not isinstance(df.at[i, "evidence_text_hop2"], str) or df.at[i, "evidence_text_hop2"] == "")
        )

    pending = [i for i in df.index if (row_needs(i) if resume else True)]
    total_pending = len(pending)
    if total_pending == 0:
        logger.info("Evidence step: nothing to do (all rows already filled).")
        return df

    logger.info(
        "Evidence step: %d/%d rows pending | workers=%d | checkpoint_every=%d",
        total_pending, len(df), max_workers, checkpoint_every,
    )

    start_time = time.time()

    def work(i: int):
        client = get_thread_client()
        row = df.loc[i]

        src = normalize_gene_symbol(row["source"])
        mid = normalize_gene_symbol(row["intermediate"])
        tgt = normalize_gene_symbol(row["target"])

        stmt1 = row["stmt_type_1"]
        stmt2 = row["stmt_type_2"]

        ev1 = fetch_evidence_text(src, mid, stmt1, client)
        _, pmids1 = get_evidence_info(src, mid, stmt1, client)

        ev2 = fetch_evidence_text(mid, tgt, stmt2, client)
        _, pmids2 = get_evidence_info(mid, tgt, stmt2, client)

        return i, format_evidence_text(ev1), "; ".join(pmids1), format_evidence_text(ev2), "; ".join(pmids2)

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(work, i) for i in pending]
        for fut in as_completed(futures):
            i, ev1, pm1, ev2, pm2 = fut.result()
            df.at[i, "evidence_text_hop1"] = ev1
            df.at[i, "pmids_hop1"] = pm1
            df.at[i, "evidence_text_hop2"] = ev2
            df.at[i, "pmids_hop2"] = pm2

            done += 1

            if checkpoint_every and done % checkpoint_every == 0:
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = total_pending - done
                eta_secs = (remaining / rate) if rate > 0 else float("inf")

                df.to_csv(out_csv, index=False)
                logger.info(
                    "[checkpoint:evidence] done=%d/%d | left=%d | "
                    "elapsed=%.1fm | rate=%.2f rows/s | ETA=%.1fm | saved=%s",
                    done, total_pending, remaining,
                    elapsed / 60, rate, eta_secs / 60, out_csv,
                )

    elapsed = time.time() - start_time
    remaining = total_pending - done
    logger.info("Evidence step finished: done=%d/%d | left=%d | elapsed=%.1fm", done, total_pending, remaining, elapsed / 60)
    return df


def extract_unique_pmids(df: pd.DataFrame, cols: list[str]) -> list[str]:
    pmids = set()
    for col in cols:
        for entry in df[col].astype(str):
            for pmid in entry.replace(" ", "").split(";"):
                if pmid.isdigit():
                    pmids.add(pmid)
    return sorted(pmids)


def build_mesh_id_to_name_map(client: Neo4jClient) -> dict[str, str]:
    query = "MATCH (b:BioEntity) WHERE b.id STARTS WITH 'mesh:' RETURN b.id AS mesh_id, b.name AS mesh_name"
    results = client.query_tx(query)
    mapping = {}
    for record in results:
        if isinstance(record, dict):
            mesh_id, mesh_name = record.get("mesh_id"), record.get("mesh_name")
        else:
            mesh_id, mesh_name = record[0], record[1]
        if mesh_id and mesh_name:
            mapping[mesh_id.replace("mesh:", "").upper()] = mesh_name
    return mapping


def is_valid_mesh_id(mid: str) -> bool:
    return bool(re.fullmatch(r"[DC]\d{6,7}", mid))


def annotate_mesh(df: pd.DataFrame, mesh_batch_size: int) -> pd.DataFrame:
    df = df.copy()
    client = Neo4jClient()

    if "Annotated MeSH terms hop1" not in df.columns:
        df["Annotated MeSH terms hop1"] = ""
    if "Annotated MeSH terms hop2" not in df.columns:
        df["Annotated MeSH terms hop2"] = ""

    all_pmids = extract_unique_pmids(df, ["pmids_hop1", "pmids_hop2"])
    if not all_pmids:
        return df

    pmid_to_mesh = {}
    for i in range(0, len(all_pmids), mesh_batch_size):
        batch = all_pmids[i:i + mesh_batch_size]
        pmid_to_mesh.update(get_mesh_ids_for_pmids(batch, client=client))

    mesh_id_to_name = build_mesh_id_to_name_map(client)

    def annotate_pmid_string(pmid_string: str) -> str:
        mesh_terms = set()
        for pmid in str(pmid_string).replace(" ", "").split(";"):
            mesh_terms.update(pmid_to_mesh.get(pmid, []))
        valid_terms = []
        for mid in sorted(mesh_terms):
            if is_valid_mesh_id(mid):
                name = mesh_id_to_name.get(mid.upper())
                if name:
                    valid_terms.append(f"{name} ({mid})")
        return ", ".join(valid_terms)

    df["Annotated MeSH terms hop1"] = df["pmids_hop1"].apply(annotate_pmid_string)
    df["Annotated MeSH terms hop2"] = df["pmids_hop2"].apply(annotate_pmid_string)
    return df


def add_gwas_and_directionality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def gwas_in_path(row):
        hits = []
        for col in ["source", "intermediate", "target"]:
            val = str(row.get(col, "")).strip()
            if val in GWAS_GENES:
                hits.append(val)
        return ", ".join(sorted(set(hits)))

    df["GWAS_genes_in_path"] = df.apply(gwas_in_path, axis=1)

    df["directionality"] = "No"
    lfc = pd.to_numeric(df["logfoldchange"], errors="coerce")
    df.loc[
        ((df["stmt_type_2"] == "IncreaseAmount") & (lfc < 0)) |
        ((df["stmt_type_2"] == "DecreaseAmount") & (lfc > 0)),
        "directionality"
    ] = "Yes"

    return df


def find_matching_statement(stmts, target_belief: float, target_evcnt: int):
    best = None
    best_diff = float("inf")

    for stmt in stmts:
        evcnt = len(stmt.evidence or [])
        if evcnt == target_evcnt:
            diff = abs((stmt.belief or 0.0) - target_belief)
            if diff < best_diff:
                best_diff = diff
                best = stmt
    if best:
        return best

    for stmt in stmts:
        diff = abs((stmt.belief or 0.0) - target_belief)
        if diff < best_diff:
            best_diff = diff
            best = stmt
    return best


def add_stmt_hash_and_urls(
    df: pd.DataFrame,
    out_csv: str,
    max_workers: int,
    checkpoint_every: int,
    resume: bool,
) -> pd.DataFrame:
    df = df.copy()
    for col in ["hop1_hash", "hop1_indra_url", "hop2_hash", "hop2_indra_url"]:
        if col not in df.columns:
            df[col] = ""

    def row_needs(i: int) -> bool:
        return (
            (not isinstance(df.at[i, "hop1_hash"], str) or df.at[i, "hop1_hash"] == "") or
            (not isinstance(df.at[i, "hop2_hash"], str) or df.at[i, "hop2_hash"] == "")
        )

    pending = [i for i in df.index if (row_needs(i) if resume else True)]
    total_pending = len(pending)
    if total_pending == 0:
        logger.info("Hash step: nothing to do (all rows already filled).")
        return df

    logger.info(
        "Hash step: %d/%d rows pending | workers=%d | checkpoint_every=%d",
        total_pending, len(df), max_workers, checkpoint_every,
    )

    start_time = time.time()

    def work(i: int):
        client = get_thread_client()
        row = df.loc[i]

        src = row["source"]
        mid = row["intermediate"]
        tgt = row["target"]

        stmt1 = row["stmt_type_1"]
        stmt2 = row["stmt_type_2"]

        belief1 = float(row.get("belief_1", 0.0) or 0.0)
        belief2 = float(row.get("belief_2", 0.0) or 0.0)
        evcnt1 = int(row.get("evidence_1", 0) or 0)
        evcnt2 = int(row.get("evidence_2", 0) or 0)

        s1 = get_statements(agent=src, other_agent=mid, rel_types=stmt1, evidence_limit=50, client=client)
        m1 = find_matching_statement(s1, belief1, evcnt1)
        if m1:
            h1 = m1.get_hash()
            u1 = f"https://db.indra.bio/statements/from_hash/{h1}?format=html"
        else:
            h1, u1 = "", ""

        s2 = get_statements(agent=mid, other_agent=tgt, rel_types=stmt2, evidence_limit=50, client=client)
        m2 = find_matching_statement(s2, belief2, evcnt2)
        if m2:
            h2 = m2.get_hash()
            u2 = f"https://db.indra.bio/statements/from_hash/{h2}?format=html"
        else:
            h2, u2 = "", ""

        return i, h1, u1, h2, u2

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(work, i) for i in pending]
        for fut in as_completed(futures):
            i, h1, u1, h2, u2 = fut.result()
            df.at[i, "hop1_hash"] = h1
            df.at[i, "hop1_indra_url"] = u1
            df.at[i, "hop2_hash"] = h2
            df.at[i, "hop2_indra_url"] = u2

            done += 1

            if checkpoint_every and done % checkpoint_every == 0:
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = total_pending - done
                eta_secs = (remaining / rate) if rate > 0 else float("inf")

                df.to_csv(out_csv, index=False)
                logger.info(
                    "[checkpoint:hash] done=%d/%d | left=%d | "
                    "elapsed=%.1fm | rate=%.2f rows/s | ETA=%.1fm | saved=%s",
                    done, total_pending, remaining,
                    elapsed / 60, rate, eta_secs / 60, out_csv,
                )

    elapsed = time.time() - start_time
    remaining = total_pending - done
    logger.info("Hash step finished: done=%d/%d | left=%d | elapsed=%.1fm", done, total_pending, remaining, elapsed / 60)
    return df


def main():
    ap = argparse.ArgumentParser(
        description="One-stop 2-hop pipeline: 2hop -> endothelial intermediate filter -> "
                    "evidence/pmids -> mesh -> gwas/directionality -> stmt hash + html urls",
    )

    ap.add_argument(
        "--genes", nargs="+", required=False,
        help="Source genes to run. If omitted, runs all GWAS_GENES.",
    )
    ap.add_argument("--deg-dir", required=True, help="Folder containing DEG CSVs")
    ap.add_argument("--endothelial-list", required=True, help="CSV with column 'gene' listing allowed intermediates")
    ap.add_argument("--out-csv", required=True, help="Output CSV path")

    ap.add_argument("--p-threshold", type=float, default=0.05)
    ap.add_argument("--prefer-fdr", action="store_true")

    ap.add_argument("--twohop-workers", type=int, default=3)
    ap.add_argument("--twohop-batch-size", type=int, default=50)

    ap.add_argument("--evidence-workers", type=int, default=5)
    ap.add_argument("--hash-workers", type=int, default=5)

    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--mesh-batch-size", type=int, default=200)

    args = ap.parse_args()

    if not args.genes:
        args.genes = sorted(GWAS_GENES)
        logger.info("No --genes provided; defaulting to all GWAS genes: %d", len(args.genes))

    t0 = time.time()
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    allowed_intermediates = load_endothelial_gene_set(args.endothelial_list)
    logger.info("Loaded endothelial intermediate whitelist: %d genes", len(allowed_intermediates))

    if args.resume and os.path.exists(args.out_csv):
        df = pd.read_csv(args.out_csv)
        logger.info("Resuming from existing output: %s (rows=%d)", args.out_csv, len(df))
    else:
        rows = []

        def gene_job(g):
            return g, run_2hop_for_gene(
                gene=g,
                deg_dir=args.deg_dir,
                p_threshold=args.p_threshold,
                prefer_fdr=args.prefer_fdr,
                batch_size=args.twohop_batch_size,
                allowed_intermediates=allowed_intermediates,
            )

        with ThreadPoolExecutor(max_workers=args.twohop_workers) as ex:
            futures = [ex.submit(gene_job, g) for g in args.genes]
            for fut in as_completed(futures):
                g, res = fut.result()
                logger.info("2-hop done for %s: kept %d rows (after endothelial intermediate filter)", g, len(res))
                rows.extend(res)

        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning("No 2-hop rows produced after endothelial intermediate filtering. Exiting.")
            return

        df["source"] = df["source"].apply(normalize_gene_symbol)
        df["intermediate"] = df["intermediate"].apply(normalize_gene_symbol)
        df["target"] = df["target"].apply(normalize_gene_symbol)

        df.to_csv(args.out_csv, index=False)
        logger.info("Saved initial 2-hop output: %s (rows=%d)", args.out_csv, len(df))

    logger.info("=== evidence + pmids (formatted) ===")
    df = enrich_with_evidence(
        df=df,
        out_csv=args.out_csv,
        max_workers=args.evidence_workers,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )
    df.to_csv(args.out_csv, index=False)

    logger.info("=== mesh annotation ===")
    df = annotate_mesh(df, mesh_batch_size=args.mesh_batch_size)
    df.to_csv(args.out_csv, index=False)

    logger.info("=== gwas + directionality ===")
    df = add_gwas_and_directionality(df)
    df.to_csv(args.out_csv, index=False)

    logger.info("=== stmt hashes + indra html urls ===")
    df = add_stmt_hash_and_urls(
        df=df,
        out_csv=args.out_csv,
        max_workers=args.hash_workers,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )
    df.to_csv(args.out_csv, index=False)

    logger.info("DONE. Final CSV: %s", args.out_csv)
    logger.info("Rows: %d", len(df))
    logger.info("Total time: %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
