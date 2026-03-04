"""
COMBINE + STRICT DEDUPE + SPLIT (GWAS sources / GWAS targets)
- Uses column ORDER from the FIRST file: gwas_endothelial_paths_url_enriched.csv
- Drops pval_col_used everywhere
- Combines FIRST file + SECOND file, then de-dupes on (source,target)
- STRICT mode (default): keeps only rows where hop2 is "strong":
    * hop2_hash non-empty
    * hop2 evidence is structured OR starts with "Evidence from:"
    * optionally require pmids_hop2 (enable with --require-hop2-pmids)
  If a (source,target) pair has no such row, it is dropped.
- Outputs 3 CSVs:
    1) combined strict-deduped
    2) GWAS genes as sources
    3) GWAS genes as targets

Run example:
python combine_gwas_datsets_strict.py \
  --one-stop-csv "/Users/prashammarfatia/Downloads/2hop_mesh_reference_filtered_unique_source_target.csv" \
  --gwas-url-enriched-csv "/Users/prashammarfatia/Downloads/gwas_endothelial_paths_url_enriched.csv" \
  --out-combined "/Users/prashammarfatia/Downloads/combined_deduped_STRICT.csv" \
  --out-gwas-sources "/Users/prashammarfatia/Downloads/combined_deduped_STRICT_gwas_sources.csv" \
  --out-gwas-targets "/Users/prashammarfatia/Downloads/combined_deduped_STRICT_gwas_targets.csv" \
  --require-hop2-pmids
"""

import argparse
import re
import pandas as pd


# ---- GWAS genes (49) ----
GWAS_GENES = {
    "ARHGEF26",
    "BCAR1",
    "BMP1",
    "CALCRL",
    "CCM2",
    "CDKN1A",
    "CDKN2B",
    "CFDP1",
    "COL4A1",
    "COL4A2",
    "EDN1",
    "EXOC3L2",
    "FBN2",
    "FGD6",
    "FLT1",
    "FURIN",
    "GDPD5",
    "GGT5",
    "GOSR2",
    "IBTK",
    "JCAD",
    "LAMB2",
    "LOX",
    "MORF4L1",
    "N4BP2L2",
    "NOS3",
    "PALLD",
    "PECAM1",
    "PGF",
    "PLPP3",
    "PRDM16",
    "PREX1",
    "PRKAR1A",
    "SCUBE1",
    "SERPINH1",
    "SH3PXD2A",
    "SLK",
    "SMAD3",
    "SPRY4",
    "SVIL",
    "SWAP70",
    "TFPI",
    "TLNRD1",
    "TSPAN14",
    "ZEB2",
}

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


def evidence_ok_strict(text: str) -> bool:
    # acceptable: structured evidence OR explicit DB provenance
    if not is_nonempty(text):
        return False
    s = str(text).strip()
    return looks_like_structured_evidence(s) or s.startswith("Evidence from:")


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
    for c in present:
        nn = df[c].apply(is_nonempty).sum()
        print(f"  - {c}: non-empty {nn:,}/{len(df):,} ({nn/len(df)*100:.1f}%)")


def report_column_mismatches(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str):
    s1 = set(df1.columns)
    s2 = set(df2.columns)
    only1 = sorted(s1 - s2)
    only2 = sorted(s2 - s1)
    print(f"\n### Column comparison")
    print(f"- {name1} only: {only1 if only1 else 'None'}")
    print(f"- {name2} only: {only2 if only2 else 'None'}")


def ensure_schema(df: pd.DataFrame, a_cols: list[str]) -> pd.DataFrame:
    """
    Make sure df has at least all columns in A (first file).
    Any missing columns are created as empty strings.
    Also drop pval_col_used.
    """
    df = df.copy()
    df = df.drop(columns=["pval_col_used"], errors="ignore")

    for c in a_cols:
        if c not in df.columns:
            df[c] = ""

    # Ensure numeric logfoldchange if present
    if "logfoldchange" in df.columns:
        df["logfoldchange"] = pd.to_numeric(df["logfoldchange"], errors="coerce")

    # Ensure GWAS_genes_in_path exists and populated
    if "GWAS_genes_in_path" in df.columns:
        missing_or_empty = (~df["GWAS_genes_in_path"].apply(is_nonempty))
    else:
        df["GWAS_genes_in_path"] = ""
        missing_or_empty = pd.Series([True] * len(df), index=df.index)

    def compute_gwas_in_path(row):
        hits = []
        for col in ["source", "intermediate", "target"]:
            if col in row:
                v = str(row.get(col, "")).strip()
                if v in GWAS_GENES:
                    hits.append(v)
        return ", ".join(sorted(set(hits)))

    if missing_or_empty.any():
        df.loc[missing_or_empty, "GWAS_genes_in_path"] = df.loc[missing_or_empty].apply(compute_gwas_in_path, axis=1)

    return df


def hop2_strict_ok(row: pd.Series, require_pmids: bool) -> bool:
    # hop2_hash must be present
    if not is_nonempty(row.get("hop2_hash", "")):
        return False
    # hop2 evidence must be structured or "Evidence from:"
    if not evidence_ok_strict(row.get("evidence_text_hop2", "")):
        return False
    # optionally require PMIDs
    if require_pmids and (not is_nonempty(row.get("pmids_hop2", ""))):
        return False
    return True


def row_quality_score(row: pd.Series) -> float:
    """
    Score used AFTER strict hop2 filter.
    Still prefers more complete rows & stronger signals.
    """
    url1 = is_valid_indra_url(row.get("hop1_indra_url", ""))
    url2 = is_valid_indra_url(row.get("hop2_indra_url", ""))

    ev1 = row.get("evidence_text_hop1", "")
    ev2 = row.get("evidence_text_hop2", "")
    ev1_ok = evidence_ok_strict(ev1)
    ev2_ok = evidence_ok_strict(ev2)

    pm1 = is_nonempty(row.get("pmids_hop1", ""))
    pm2 = is_nonempty(row.get("pmids_hop2", ""))

    # completeness over A's schema columns (since we will output in that order)
    completeness = 0
    for c in row.index:
        if is_nonempty(row.get(c, "")):
            completeness += 1

    abs_lfc = abs(safe_float(row.get("logfoldchange", 0.0), default=0.0))

    score = 0.0
    score += 20.0 * (url1 + url2)     # both urls preferred
    score += 10.0 * (ev1_ok + ev2_ok)
    score += 4.0 * (pm1 + pm2)
    score += 0.5 * completeness
    score += 1.0 * abs_lfc            # tie-breaker: higher abs logFC
    return score


def dedupe_by_source_target_strict(df: pd.DataFrame, require_hop2_pmids: bool) -> pd.DataFrame:
    if "source" not in df.columns or "target" not in df.columns:
        raise ValueError("Need 'source' and 'target' columns for dedupe.")

    df = df.copy()
    before_all = len(df)

    # STRICT hop2 eligibility filter FIRST
    df["_hop2_ok"] = df.apply(lambda r: hop2_strict_ok(r, require_pmids=require_hop2_pmids), axis=1)
    df = df[df["_hop2_ok"]].copy()
    after_filter = len(df)

    print(f"\n### Strict hop2 filter")
    print(f"- before: {before_all:,}")
    print(f"- after strict-hop2: {after_filter:,} ({after_filter/before_all*100:.1f}%)")
    if after_filter == 0:
        raise RuntimeError("No rows left after strict hop2 filtering. Relax criteria (or disable --require-hop2-pmids).")

    # Score and dedupe
    df["_score"] = df.apply(row_quality_score, axis=1)
    df["_abs_lfc"] = df["logfoldchange"].apply(lambda x: abs(safe_float(x, default=0.0)))

    df = df.sort_values(by=["_score", "_abs_lfc"], ascending=[False, False], kind="mergesort")

    before = len(df)
    out = df.drop_duplicates(subset=["source", "target"], keep="first").copy()
    after = len(out)

    out.drop(columns=["_hop2_ok", "_score", "_abs_lfc"], inplace=True, errors="ignore")

    print(f"\n### De-dup (source,target) [STRICT]")
    print(f"- candidates before dedupe: {before:,}")
    print(f"- unique pairs after:       {after:,}")
    print(f"- reduced:                 {before-after:,} ({(before-after)/before*100:.1f}%)")

    hop2_url_ok = out["hop2_indra_url"].apply(is_valid_indra_url).sum() if "hop2_indra_url" in out.columns else 0
    pm2_ok = out["pmids_hop2"].apply(is_nonempty).sum() if "pmids_hop2" in out.columns else 0
    print(f"- kept rows with valid hop2 url (format): {hop2_url_ok:,}/{after:,} ({hop2_url_ok/after*100:.1f}%)")
    print(f"- kept rows with hop2 pmids:              {pm2_ok:,}/{after:,} ({pm2_ok/after*100:.1f}%)")
    return out


def reorder_like_a(df: pd.DataFrame, a_cols: list[str]) -> pd.DataFrame:
    """
    Output must match column order of file A exactly.
    Also ensures pval_col_used is absent.
    Drops any extra columns not in A.
    """
    df = df.copy()
    df = df.drop(columns=["pval_col_used"], errors="ignore")
    keep = [c for c in a_cols if c in df.columns]
    return df[keep].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one-stop-csv", required=True)
    ap.add_argument("--gwas-url-enriched-csv", required=True)
    ap.add_argument("--out-combined", required=True)
    ap.add_argument("--out-gwas-sources", required=True)
    ap.add_argument("--out-gwas-targets", required=True)
    ap.add_argument("--require-hop2-pmids", action="store_true", help="Also require pmids_hop2 non-empty (strictest).")
    args = ap.parse_args()

    # Load A first (requested)
    df_a = pd.read_csv(args.gwas_url_enriched_csv, low_memory=False)
    df_b = pd.read_csv(args.one_stop_csv, low_memory=False)

    # Drop pval_col_used early (latter part first)
    df_a = df_a.drop(columns=["pval_col_used"], errors="ignore")
    df_b = df_b.drop(columns=["pval_col_used"], errors="ignore")

    a_cols = list(df_a.columns)

    print_file_stats(df_a, "GWAS URL ENRICHED (A, first)")
    print_file_stats(df_b, "ONE-STOP (B, second)")
    report_column_mismatches(df_a, df_b, "A", "B")

    # Ensure both have at least A's schema columns
    df_a = ensure_schema(df_a, a_cols=a_cols)
    df_b = ensure_schema(df_b, a_cols=a_cols)

    # Combine A first
    combined = pd.concat([df_a, df_b], ignore_index=True)
    print_file_stats(combined, "COMBINED (before strict filter/dedupe)")

    # Strict dedupe (drops pairs with no strong hop2)
    combined_unique = dedupe_by_source_target_strict(combined, require_hop2_pmids=args.require_hop2_pmids)

    # Enforce output column order EXACTLY like A
    combined_unique = reorder_like_a(combined_unique, a_cols=a_cols)

    # Save combined
    combined_unique.to_csv(args.out_combined, index=False)
    print(f"\nSaved combined strict deduped -> {args.out_combined} (rows={len(combined_unique):,})")

    # Split 1: GWAS genes are sources
    gwas_sources = combined_unique[combined_unique["source"].astype(str).isin(GWAS_GENES)].copy()
    gwas_sources = reorder_like_a(gwas_sources, a_cols=a_cols)
    gwas_sources.to_csv(args.out_gwas_sources, index=False)
    print(f"Saved GWAS-as-source -> {args.out_gwas_sources} (rows={len(gwas_sources):,})")

    # Split 2: GWAS genes are targets
    gwas_targets = combined_unique[combined_unique["target"].astype(str).isin(GWAS_GENES)].copy()
    gwas_targets = reorder_like_a(gwas_targets, a_cols=a_cols)
    gwas_targets.to_csv(args.out_gwas_targets, index=False)
    print(f"Saved GWAS-as-target -> {args.out_gwas_targets} (rows={len(gwas_targets):,})")


if __name__ == "__main__":
    main()