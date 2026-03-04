import argparse
import re
import pandas as pd

# GWAS genes list
GWAS_GENES = {
    "ARHGEF26", "BCAR1", "BMP1", "CALCRL", "CCM2", "CDKN1A", "CDKN2B", "CFDP1", "COL4A1", "COL4A2", "EDN1", "EXOC3L2",
    "FBN2", "FGD6", "FLT1", "FURIN", "GDPD5", "GGT5", "GOSR2", "IBTK", "JCAD", "LAMB2", "LOX", "MORF4L1", "N4BP2L2",
    "NOS3", "PALLD", "PECAM1", "PGF", "PLPP3", "PRDM16", "PREX1", "PRKAR1A", "SCUBE1", "SERPINH1", "SH3PXD2A", "SLK",
    "SMAD3", "SPRY4", "SVIL", "SWAP70", "TFPI", "TLNRD1", "TSPAN14", "ZEB2",
}

# Canonical columns we care about for scoring/dedup
CANON_COLS = [
    "source", "intermediate", "target",
    "logfoldchange",
    "evidence_text_hop1", "pmids_hop1", "hop1_hash", "hop1_indra_url",
    "evidence_text_hop2", "pmids_hop2", "hop2_hash", "hop2_indra_url",
    "GWAS_genes_in_path",
    "directionality",
    "Annotated MeSH terms hop1", "Annotated MeSH terms hop2",
    "stmt_type_1", "stmt_type_2",
    "belief_1", "belief_2",
    "evidence_1", "evidence_2",
    "pval", "pval_col_used",
]

URL_RE = re.compile(r"^https://db\.indra\.bio/statements/from_hash/-?\d+\?format=html$")


def is_nonempty(x) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and pd.isna(x):
        return False
    s = str(x).strip()
    return s != "" and s.lower() not in {"nan", "none"}


def looks_like_structured_evidence(text: str) -> bool:
    # Your formatted evidence tends to contain "1) ..."
    if not is_nonempty(text):
        return False
    return bool(re.search(r"(^|\n)\s*1\)\s+", str(text)))


def is_placeholder_evidence(text: str) -> bool:
    if not is_nonempty(text):
        return True
    s = str(text).strip()
    return (
        s.startswith("No evidence found")
        or s.startswith("Evidence from:")  # provenance-only
        or s.startswith("Database evidence only")
        or s.startswith("Error")
        or s.startswith("Error fetching evidence")
    )


def is_valid_indra_url(url: str) -> bool:
    if not is_nonempty(url):
        return False
    return bool(URL_RE.fullmatch(str(url).strip()))


def safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default


def print_file_stats(df: pd.DataFrame, title: str):
    print(f"\n### {title}")
    print(f"- rows: {len(df):,}")
    print(f"- cols: {len(df.columns):,}")
    print(f"- columns: {df.columns.tolist()}")
    key = ["evidence_text_hop1", "pmids_hop1", "hop1_indra_url", "evidence_text_hop2", "pmids_hop2", "hop2_indra_url"]
    present = [c for c in key if c in df.columns]
    if present:
        for c in present:
            nn = df[c].apply(is_nonempty).sum()
            print(f"  - {c}: non-empty {nn:,}/{len(df):,} ({nn/len(df)*100:.1f}%)")
    if "GWAS_genes_in_path" in df.columns:
        nn = df["GWAS_genes_in_path"].apply(is_nonempty).sum()
        print(f"  - GWAS_genes_in_path non-empty: {nn:,}/{len(df):,} ({nn/len(df)*100:.1f}%)")


def report_column_mismatches(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str):
    s1 = set(df1.columns)
    s2 = set(df2.columns)
    only1 = sorted(s1 - s2)
    only2 = sorted(s2 - s1)
    print(f"\n### Column comparison")
    print(f"- {name1} only: {only1 if only1 else 'None'}")
    print(f"- {name2} only: {only2 if only2 else 'None'}")


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Add missing canonical columns
    for c in CANON_COLS:
        if c not in df.columns:
            df[c] = ""

    # Ensure numeric column
    df["logfoldchange"] = pd.to_numeric(df["logfoldchange"], errors="coerce")

    # If GWAS_genes_in_path missing/empty, recompute from source/intermediate/target
    def compute_gwas_in_path(row):
        hits = []
        for col in ["source", "intermediate", "target"]:
            v = str(row.get(col, "")).strip()
            if v in GWAS_GENES:
                hits.append(v)
        return ", ".join(sorted(set(hits)))

    missing_or_empty = (~df["GWAS_genes_in_path"].apply(is_nonempty))
    if missing_or_empty.any():
        df.loc[missing_or_empty, "GWAS_genes_in_path"] = df.loc[missing_or_empty].apply(compute_gwas_in_path, axis=1)

    return df


def row_quality_score(row: pd.Series) -> float:
    # Primary: has both working URLs
    url1 = is_valid_indra_url(row.get("hop1_indra_url", ""))
    url2 = is_valid_indra_url(row.get("hop2_indra_url", ""))

    # Evidence (prefer structured, penalize placeholders)
    ev1 = row.get("evidence_text_hop1", "")
    ev2 = row.get("evidence_text_hop2", "")
    ev1_struct = looks_like_structured_evidence(ev1)
    ev2_struct = looks_like_structured_evidence(ev2)
    ev1_bad = is_placeholder_evidence(ev1)
    ev2_bad = is_placeholder_evidence(ev2)

    # PMIDs present
    pm1 = is_nonempty(row.get("pmids_hop1", ""))
    pm2 = is_nonempty(row.get("pmids_hop2", ""))

    # Overall completeness across important fields
    important = [
        "source", "intermediate", "target",
        "stmt_type_1", "stmt_type_2",
        "belief_1", "belief_2",
        "evidence_1", "evidence_2",
        "logfoldchange", "pval",
        "hop1_hash", "hop1_indra_url", "hop2_hash", "hop2_indra_url",
        "Annotated MeSH terms hop1", "Annotated MeSH terms hop2",
    ]
    completeness = sum(1 for c in important if is_nonempty(row.get(c, "")))

    abs_lfc = abs(safe_float(row.get("logfoldchange", 0.0), default=0.0))

    score = 0.0
    score += 100.0 * (url1 and url2)          # strongest preference: BOTH urls exist
    score += 20.0 * (url1 + url2)             # next: at least one url
    score += 8.0 * (ev1_struct + ev2_struct)
    score -= 8.0 * (ev1_bad + ev2_bad)
    score += 4.0 * (pm1 + pm2)
    score += 1.0 * completeness
    score += 0.2 * abs_lfc
    return score


def dedupe_by_source_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "source" not in df.columns or "target" not in df.columns:
        raise ValueError("Need 'source' and 'target' columns for dedupe.")

    df["_score"] = df.apply(row_quality_score, axis=1)
    df["_abs_lfc"] = df["logfoldchange"].apply(lambda x: abs(safe_float(x, default=0.0)))

    # sort best-first, then abs_lfc
    df = df.sort_values(by=["_score", "_abs_lfc"], ascending=[False, False], kind="mergesort")
    before = len(df)
    out = df.drop_duplicates(subset=["source", "target"], keep="first").copy()
    after = len(out)

    out.drop(columns=["_score", "_abs_lfc"], inplace=True, errors="ignore")

    print(f"\n### De-dup (source,target)")
    print(f"- before: {before:,}")
    print(f"- after:  {after:,}")
    print(f"- reduced: {before-after:,} ({(before-after)/before*100:.1f}%)")

    # quick quality stats
    hop2_ok = out["hop2_indra_url"].apply(is_valid_indra_url).sum()
    print(f"- kept rows with valid hop2 url: {hop2_ok:,}/{after:,} ({hop2_ok/after*100:.1f}%)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one-stop-csv", required=True, help="Your one_stop output CSV (new file)")
    ap.add_argument("--gwas-url-enriched-csv", required=True, help="Existing gwas_endothelial_paths_url_enriched.csv")
    ap.add_argument("--out-combined", required=True, help="Combined + deduped output")
    ap.add_argument("--out-gwas-sources", required=True, help="Subset where GWAS genes are sources")
    ap.add_argument("--out-gwas-targets", required=True, help="Subset where GWAS genes are targets")
    args = ap.parse_args()

    # Load, with gwas-url-enriched first (as requested)
    df_a = pd.read_csv(args.gwas_url_enriched_csv, low_memory=False)
    df_b = pd.read_csv(args.one_stop_csv, low_memory=False)

    print_file_stats(df_a, "GWAS URL ENRICHED (input A, first)")
    print_file_stats(df_b, "ONE-STOP OUTPUT (input B, second)")
    report_column_mismatches(df_a, df_b, "A", "B")

    # Standardize schemas (add missing cols, recompute GWAS_genes_in_path if needed)
    df_a = ensure_schema(df_a)
    df_b = ensure_schema(df_b)

    # Merge (A first)
    combined = pd.concat([df_a, df_b], ignore_index=True)
    print_file_stats(combined, "COMBINED (before dedupe)")

    # Dedupe by (source,target) with quality scoring
    combined_unique = dedupe_by_source_target(combined)
    combined_unique.to_csv(args.out_combined, index=False)
    print(f"\nSaved combined deduped -> {args.out_combined}")

    # Split 1: GWAS genes are sources
    gwas_sources = combined_unique[combined_unique["source"].astype(str).isin(GWAS_GENES)].copy()
    gwas_sources.to_csv(args.out_gwas_sources, index=False)
    print(f"Saved GWAS-as-source -> {args.out_gwas_sources} (rows={len(gwas_sources):,})")

    # Split 2: GWAS genes are targets
    gwas_targets = combined_unique[combined_unique["target"].astype(str).isin(GWAS_GENES)].copy()
    gwas_targets.to_csv(args.out_gwas_targets, index=False)
    print(f"Saved GWAS-as-target -> {args.out_gwas_targets} (rows={len(gwas_targets):,})")


if __name__ == "__main__":
    main()