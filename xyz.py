import argparse
import re
from collections import defaultdict

import pandas as pd
from indra_cogex.client.neo4j_client import Neo4jClient


C14_LINE_RE = re.compile(r"^(?P<name>.+?)\s*\[(?P<tree>C14(?:\.\d+)*?)\]\s*$", re.IGNORECASE)


def parse_c14_names_from_text(text: str) -> list[str]:
    names = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = C14_LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name").strip()
        # drop obvious headers/noise if any sneak in
        if name.lower() in {"details", "qualifiers", "mesh tree structures", "concepts"}:
            continue
        names.append(name)
    # de-dup while preserving order
    seen = set()
    out = []
    for n in names:
        k = n.lower()
        if k not in seen:
            seen.add(k)
            out.append(n)
    return out


def normalize_mesh_id(mid: str) -> str:
    mid = str(mid or "").strip().upper()
    # Your downstream reference filters keep only D* MeSH descriptor IDs
    return mid if re.fullmatch(r"D\d{5,10}", mid) else ""


def map_names_to_mesh_ids(client: Neo4jClient, names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Exact case-insensitive name match in Neo4j:
      input name -> mesh_id, mesh_name
    Returns (mapped_df, unmapped_names)
    """
    if not names:
        return pd.DataFrame(columns=["mesh_id", "mesh_name", "origin", "child_of"]), []

    q = """
    UNWIND $names AS n
    MATCH (b:BioEntity)
    WHERE b.id STARTS WITH 'mesh:'
      AND b.name IS NOT NULL
      AND toLower(b.name) = toLower(n)
    RETURN DISTINCT
      n AS input_name,
      replace(b.id, 'mesh:', '') AS mesh_id,
      b.name AS mesh_name
    """
    rows = client.query_tx(q, names=names)

    mapped = []
    hit_names_lower = set()
    for r in rows:
        if isinstance(r, dict):
            input_name = r.get("input_name", "")
            mesh_id = r.get("mesh_id", "")
            mesh_name = r.get("mesh_name", "")
        else:
            input_name = r[0] if len(r) > 0 else ""
            mesh_id = r[1] if len(r) > 1 else ""
            mesh_name = r[2] if len(r) > 2 else ""

        if input_name:
            hit_names_lower.add(str(input_name).lower())

        mapped.append(
            {
                "mesh_id": mesh_id,
                "mesh_name": mesh_name,
                "origin": "C14_from_pasted_tree",
                "child_of": "",
            }
        )

    unmapped = [n for n in names if n.lower() not in hit_names_lower]

    df = pd.DataFrame(mapped)
    if not df.empty:
        df["mesh_id"] = df["mesh_id"].apply(normalize_mesh_id)
        df = df[df["mesh_id"] != ""].copy()

    return df, unmapped


def fetch_endothelium_keyword_mesh(client: Neo4jClient) -> pd.DataFrame:
    q = """
    MATCH (b:BioEntity)
    WHERE b.id STARTS WITH 'mesh:'
      AND b.name IS NOT NULL
      AND (
        toLower(b.name) CONTAINS 'endothelial'
        OR toLower(b.name) CONTAINS 'endothelium'
      )
    RETURN DISTINCT
      replace(b.id, 'mesh:', '') AS mesh_id,
      b.name AS mesh_name
    """
    rows = client.query_tx(q)

    out = []
    for r in rows:
        if isinstance(r, dict):
            mesh_id = r.get("mesh_id", "")
            mesh_name = r.get("mesh_name", "")
        else:
            mesh_id = r[0] if len(r) > 0 else ""
            mesh_name = r[1] if len(r) > 1 else ""
        out.append(
            {
                "mesh_id": mesh_id,
                "mesh_name": mesh_name,
                "origin": "endothelium_keyword",
                "child_of": "",
            }
        )

    df = pd.DataFrame(out)
    if not df.empty:
        df["mesh_id"] = df["mesh_id"].apply(normalize_mesh_id)
        df = df[df["mesh_id"] != ""].copy()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c14-text-file", required=True, help="Text file containing your pasted C14 tree listing.")
    ap.add_argument("--out-csv", required=True, help="Output reference CSV (mesh_id, mesh_name, origin, child_of).")
    ap.add_argument("--unmapped-out", default="", help="Optional: write unmapped C14 names here.")
    args = ap.parse_args()

    text = open(args.c14_text_file, "r", encoding="utf-8").read()
    c14_names = parse_c14_names_from_text(text)
    print(f"Parsed {len(c14_names):,} unique C14 names from pasted text")

    client = Neo4jClient()

    df_c14, unmapped = map_names_to_mesh_ids(client, c14_names)
    print(f"Mapped {len(df_c14):,} C14 names to MeSH descriptor IDs (D*)")
    print(f"Unmapped C14 names (case-insensitive exact match): {len(unmapped):,}")

    df_endo = fetch_endothelium_keyword_mesh(client)
    print(f"Keyword endothelium/endothelial matches (D*): {len(df_endo):,}")

    df = pd.concat([df_c14, df_endo], ignore_index=True)
    df["mesh_id"] = df["mesh_id"].apply(normalize_mesh_id)
    df = df[df["mesh_id"] != ""].copy()

    # one row per descriptor ID; keep a stable origin label if duplicates exist
    # (if a term matches both constraints, prefer C14 label)
    origin_rank = {"C14_from_pasted_tree": 0, "endothelium_keyword": 1}
    df["_rank"] = df["origin"].map(lambda x: origin_rank.get(x, 99))
    df = df.sort_values(["mesh_id", "_rank"]).drop_duplicates(subset=["mesh_id"], keep="first")
    df = df.drop(columns=["_rank"]).sort_values(["mesh_id"]).reset_index(drop=True)

    df.to_csv(args.out_csv, index=False)
    print(f"\n✅ Wrote {len(df):,} MeSH descriptor IDs to: {args.out_csv}")
    print("Use this file anywhere you currently pass --mesh-reference")

    if args.unmapped_out:
        with open(args.unmapped_out, "w", encoding="utf-8") as f:
            for n in unmapped:
                f.write(n + "\n")
        print(f"📝 Wrote unmapped C14 names to: {args.unmapped_out}")


if __name__ == "__main__":
    main()