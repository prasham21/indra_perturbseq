import argparse
import re
import pandas as pd


PAIR_RE = re.compile(r"[^,]+?\(D\d{5,10}\)")
ID_RE = re.compile(r"\((D\d{5,10})\)")


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


def filter_mesh_terms_cell(x, valid_ids: set[str]) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    text = str(x).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return ""

    pairs = PAIR_RE.findall(text)
    kept = []
    for p in pairs:
        m = ID_RE.search(p)
        if m and m.group(1).upper() in valid_ids:
            kept.append(p.strip())
    return ", ".join(kept)


def pick_mesh_columns(df: pd.DataFrame) -> list[str]:
    # Filter all columns that look like your annotation columns
    cols = []
    for c in df.columns:
        lc = str(c).lower()
        if "annotated mesh terms" in lc:
            cols.append(c)
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="New reference list CSV (must contain mesh_id).")
    ap.add_argument("--input", required=True, help="Input 1-hop or 2-hop final CSV (already has annotated MeSH columns).")
    ap.add_argument("--output", required=True, help="Output CSV with filtered annotated MeSH columns.")
    args = ap.parse_args()

    valid_ids = load_valid_mesh_ids(args.reference)
    print(f"Loaded {len(valid_ids):,} valid MeSH IDs from reference")

    df = pd.read_csv(args.input, low_memory=False)
    mesh_cols = pick_mesh_columns(df)

    if not mesh_cols:
        raise ValueError(
            "No columns found containing 'Annotated MeSH terms'. "
            f"Columns seen: {df.columns.tolist()}"
        )

    for col in mesh_cols:
        before_nonempty = (df[col].fillna("").astype(str).str.strip() != "").sum()
        df[col] = df[col].apply(lambda v: filter_mesh_terms_cell(v, valid_ids))
        after_nonempty = (df[col].fillna("").astype(str).str.strip() != "").sum()
        print(f"{col}: non-empty {before_nonempty:,} -> {after_nonempty:,}")

    df.to_csv(args.output, index=False)
    print(f"\nWrote filtered file: {args.output}")


if __name__ == "__main__":
    main()