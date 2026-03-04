"""POST-PROCESS ONE-STOP 2-HOP OUTPUT."""
from __future__ import annotations


import argparse
import re
import pandas as pd

import logging


logger = logging.getLogger(__name__)

# Basic utilities
def is_nonempty(x) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and pd.isna(x):
        return False
    s = str(x).strip()
    return s != "" and s.lower() not in {"nan", "none"}


def nonempty_count(series: pd.Series) -> int:
    return series.apply(is_nonempty).sum()


def looks_like_structured_evidence(text: str) -> bool:
    # your formatter typically produces "1) ..." separated by blank lines
    if not is_nonempty(text):
        return False
    s = str(text).strip()
    return bool(re.search(r"(^|\n)\s*1\)\s+", s))


def is_placeholder_evidence(text: str) -> bool:
    if not is_nonempty(text):
        return True
    s = str(text).strip()
    return (
        s.startswith("No evidence found")
        or s.startswith("Evidence from:")
        or s.startswith("Database evidence only")
        or s.startswith("Error")
        or s.startswith("Error fetching evidence")
    )


def is_valid_indra_url(url: str) -> bool:
    if not is_nonempty(url):
        return False
    s = str(url).strip()
    return bool(re.fullmatch(r"https://db\.indra\.bio/statements/from_hash/-?\d+\?format=html", s))


def safe_float(x, default=None):
    try:
        v = float(x)
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


# MeSH reference filtering (matches map_mesh_terms.py behavior)
def load_valid_mesh_ids(reference_file: str) -> set[str]:
    ref_df = pd.read_csv(reference_file, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
    if "mesh_id" not in ref_df.columns:
        raise ValueError(f"Reference MeSH CSV must contain column 'mesh_id'. Columns: {ref_df.columns.tolist()}")

    ref_df["mesh_id"] = (
        ref_df["mesh_id"]
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.replace(r"[^A-Za-z0-9]", "", regex=True)
    )

    # Keep only D* IDs (same as your map_mesh_terms.py)
    return set(ref_df["mesh_id"][ref_df["mesh_id"].str.startswith("D")])


def filter_mesh_terms_to_reference(text: str, valid_ids: set[str]) -> str:
    if not is_nonempty(text):
        return ""

    # Find "Name (D#######)" chunks
    pairs = re.findall(r"[^,]+?\(D\d{5,10}\)", str(text))
    kept = []
    for p in pairs:
        m = re.search(r"\((D\d{5,10})\)", p)
        if m and m.group(1) in valid_ids:
            kept.append(p.strip())
    return ", ".join(kept)


def apply_mesh_reference_filter(df: pd.DataFrame, reference_file: str) -> pd.DataFrame:
    df = df.copy()
    valid_ids = load_valid_mesh_ids(reference_file)
    logger.info("\n### MeSH reference")
    logger.info("- loaded valid MeSH IDs: %d (D*)", len(valid_ids))

    mesh_cols = ["Annotated MeSH terms hop1", "Annotated MeSH terms hop2"]
    for col in mesh_cols:
        if col not in df.columns:
            df[col] = ""

        before_nonempty = nonempty_count(df[col])
        before_chars = df[col].fillna("").astype(str).str.len().sum()

        df[col] = df[col].apply(lambda x: filter_mesh_terms_to_reference(x, valid_ids))

        after_nonempty = nonempty_count(df[col])
        after_chars = df[col].fillna("").astype(str).str.len().sum()

        logger.info("- %s: non-empty %d -> %d | total chars %d -> %d", col, before_nonempty, after_nonempty, before_chars, after_chars)

    any_mesh_after = (
        df["Annotated MeSH terms hop1"].apply(is_nonempty)
        | df["Annotated MeSH terms hop2"].apply(is_nonempty)
    ).sum()
    logger.info("- rows with ANY kept MeSH after filtering: %d/%d (%.1f%%)", any_mesh_after, len(df), any_mesh_after/len(df)*100)
    return df


# Uniqueness selection
def row_quality_score(row: pd.Series) -> float:
    # Strong signals
    url1_ok = is_valid_indra_url(row.get("hop1_indra_url", ""))
    url2_ok = is_valid_indra_url(row.get("hop2_indra_url", ""))
    hash1_ok = is_nonempty(row.get("hop1_hash", ""))
    hash2_ok = is_nonempty(row.get("hop2_hash", ""))

    ev1 = row.get("evidence_text_hop1", "")
    ev2 = row.get("evidence_text_hop2", "")
    ev1_struct = looks_like_structured_evidence(ev1)
    ev2_struct = looks_like_structured_evidence(ev2)
    ev1_bad = is_placeholder_evidence(ev1)
    ev2_bad = is_placeholder_evidence(ev2)

    pm1_ok = is_nonempty(row.get("pmids_hop1", ""))
    pm2_ok = is_nonempty(row.get("pmids_hop2", ""))

    # Completeness over key fields
    required_fields = [
        "source", "intermediate", "target",
        "stmt_type_1", "stmt_type_2",
        "belief_1", "belief_2",
        "evidence_1", "evidence_2",
        "logfoldchange", "pval",
        "GWAS_genes_in_path", "directionality",
        "Annotated MeSH terms hop1", "Annotated MeSH terms hop2",
    ]
    completeness = sum(1 for f in required_fields if is_nonempty(row.get(f, "")))

    # Numeric tie-breakers (kept in score as small weights)
    abs_lfc = abs(safe_float(row.get("logfoldchange", None), default=0.0) or 0.0)
    ev_cnt = safe_int(row.get("evidence_1", 0)) + safe_int(row.get("evidence_2", 0))
    belief_sum = (safe_float(row.get("belief_1", 0.0), default=0.0) or 0.0) + (safe_float(row.get("belief_2", 0.0), default=0.0) or 0.0)

    # Score assembly
    score = 0.0
    score += completeness * 1.0

    # Prefer working URLs/hashes heavily
    score += 10.0 * (url1_ok + url2_ok)
    score += 5.0 * (hash1_ok + hash2_ok)

    # Prefer structured evidence text; penalize placeholders/errors
    score += 6.0 * (ev1_struct + ev2_struct)
    score -= 6.0 * (ev1_bad + ev2_bad)

    # Prefer PMIDs present
    score += 3.0 * (pm1_ok + pm2_ok)

    # Small numeric nudges
    score += 0.01 * ev_cnt
    score += 0.5 * belief_sum
    score += 0.1 * abs_lfc

    return score


def make_unique_by_source_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "source" not in df.columns or "target" not in df.columns:
        raise ValueError("Input must contain columns: source, target")

    # Compute quality score
    logger.info("\n### Uniqueness selection (source,target)")
    df["_quality_score"] = df.apply(row_quality_score, axis=1)

    # Sort so the best row per (source,target) comes first
    # additional stable tie-breakers after score:
    df["_abs_logfc"] = df["logfoldchange"].apply(lambda x: abs(safe_float(x, default=0.0) or 0.0))
    df["_ev_sum"] = df.apply(lambda r: safe_int(r.get("evidence_1", 0)) + safe_int(r.get("evidence_2", 0)), axis=1)
    df["_belief_sum"] = df.apply(lambda r: (safe_float(r.get("belief_1", 0.0), 0.0) or 0.0) + (safe_float(r.get("belief_2", 0.0), 0.0) or 0.0), axis=1)

    df_sorted = df.sort_values(
        by=["_quality_score", "_abs_logfc", "_ev_sum", "_belief_sum"],
        ascending=[False, False, False, False],
        kind="mergesort",  # stable
    )

    before = len(df_sorted)
    unique_df = df_sorted.drop_duplicates(subset=["source", "target"], keep="first").copy()
    after = len(unique_df)

    # Cleanup temp columns
    unique_df.drop(columns=["_quality_score", "_abs_logfc", "_ev_sum", "_belief_sum"], inplace=True, errors="ignore")

    logger.info("- rows before: %d", before)
    logger.info("- unique (source,target) rows after: %d", after)
    logger.info("- reduction: %d (%.1f%%)", before-after, (before-after)/before*100)

    # Some quick quality stats on selected rows
    url_ok = unique_df["hop1_indra_url"].apply(is_valid_indra_url).sum() if "hop1_indra_url" in unique_df.columns else 0
    url_ok2 = unique_df["hop2_indra_url"].apply(is_valid_indra_url).sum() if "hop2_indra_url" in unique_df.columns else 0
    logger.info("- selected rows with valid hop1 URL: %d/%d", url_ok, after)
    logger.info("- selected rows with valid hop2 URL: %d/%d", url_ok2, after)

    return unique_df


# Reporting
def print_dataset_stats(df: pd.DataFrame, title: str):
    logger.info("\n### %s", title)
    logger.info("- rows: %d", len(df))
    logger.info("- columns: %d", len(df.columns))
    logger.info("- column names: %s", df.columns.tolist())

    logger.info("\n- non-empty values per column:")
    for col in df.columns:
        ne = nonempty_count(df[col])
        logger.info("  - %s: %d/%d (%.1f%%)", col, ne, len(df), ne/len(df)*100)

    # Key checks
    key_cols = [
        "evidence_text_hop1", "evidence_text_hop2",
        "pmids_hop1", "pmids_hop2",
        "hop1_indra_url", "hop2_indra_url",
        "hop1_hash", "hop2_hash",
        "Annotated MeSH terms hop1", "Annotated MeSH terms hop2",
    ]
    present = [c for c in key_cols if c in df.columns]
    if present:
        logger.info("\n- key sanity checks:")
        if "evidence_text_hop1" in df.columns:
            good = df["evidence_text_hop1"].apply(looks_like_structured_evidence).sum()
            bad = df["evidence_text_hop1"].apply(is_placeholder_evidence).sum()
            logger.info("  - hop1 evidence structured: %d/%d | placeholders/errors: %d/%d", good, len(df), bad, len(df))
        if "evidence_text_hop2" in df.columns:
            good = df["evidence_text_hop2"].apply(looks_like_structured_evidence).sum()
            bad = df["evidence_text_hop2"].apply(is_placeholder_evidence).sum()
            logger.info("  - hop2 evidence structured: %d/%d | placeholders/errors: %d/%d", good, len(df), bad, len(df))
        if "hop1_indra_url" in df.columns:
            ok = df["hop1_indra_url"].apply(is_valid_indra_url).sum()
            logger.info("  - hop1 URL valid format: %d/%d", ok, len(df))
        if "hop2_indra_url" in df.columns:
            ok = df["hop2_indra_url"].apply(is_valid_indra_url).sum()
            logger.info("  - hop2 URL valid format: %d/%d", ok, len(df))


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--mesh-reference", required=True)
    ap.add_argument("--out-mesh-filtered", required=True)
    ap.add_argument("--out-unique", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv, low_memory=False)
    print_dataset_stats(df, title="ONE-STOP OUTPUT (raw)")

    # 1) Apply reference MeSH filtering (mapping/filtering step)
    df_mesh = apply_mesh_reference_filter(df, reference_file=args.mesh_reference)
    df_mesh.to_csv(args.out_mesh_filtered, index=False)
    logger.info("\nSaved MeSH-reference-filtered CSV -> %s", args.out_mesh_filtered)

    print_dataset_stats(df_mesh, title="AFTER MeSH reference filtering")

    # 2) Enforce (source,target) uniqueness with strong quality selection
    df_unique = make_unique_by_source_target(df_mesh)
    df_unique.to_csv(args.out_unique, index=False)
    logger.info("\nSaved UNIQUE (source,target) CSV -> %s", args.out_unique)

    print_dataset_stats(df_unique, title="FINAL UNIQUE (source,target) output")


if __name__ == "__main__":
    main()
