#!/usr/bin/env python3
"""
Audit OmniPath graph choices (pyomnipath 1.0.12).
- Pull broad interactions table
- Compare ALL vs CAUSAL-only rows
- Compare STRICT (HGNC) vs LOOSE (UniProt gene name) mapping
- Build 4 graphs and report node/edge counts
- Save a CSV summary ("graph_choice_summary.csv")
"""

import time
import pandas as pd
import networkx as nx
import omnipath as op
from omnipath.interactions import AllInteractions
from indra.databases import uniprot_client, hgnc_client

CACHE = "omnipath_broad.parquet"
SUMMARY_CSV = "graph_choice_summary.csv"


def load_broad():
    try:
        df = pd.read_parquet(CACHE)
        print(f"Loaded cached OmniPath data from {CACHE} ({len(df):,} rows).")
    except FileNotFoundError:
        print("Downloading OmniPath interactions (broad table)…")
        t0 = time.time()
        df = AllInteractions.get()  # v1.0.12 API
        print(f"Downloaded {len(df):,} rows in {time.time()-t0:.2f}s")
        df.to_parquet(CACHE)
        print(f"Saved to {CACHE}")
    # Ensure expected columns exist
    for c in ["source", "target", "is_stimulation", "is_inhibition"]:
        if c not in df.columns:
            df[c] = None
    # Keep only rows with both endpoints
    df = df.dropna(subset=["source", "target"])
    return df


def summarize_df(tag, df):
    n = len(df)
    n_causal = ((df["is_stimulation"] == True) | (df["is_inhibition"] == True)).sum()
    print(f"[{tag}] rows: {n:,} | causal-flagged: {n_causal:,} | non-causal: {n - n_causal:,}")
    return n, n_causal


def build_maps(uids):
    # STRICT: UniProt -> HGNC symbol via HGNC ID
    strict = {}
    for uid in uids:
        hid = uniprot_client.get_hgnc_id(uid)
        if hid:
            sym = hgnc_client.get_hgnc_name(hid)
            if sym:
                strict[uid] = sym
    # LOOSE: UniProt -> gene symbol (UniProt gene name)
    loose = {}
    for uid in uids:
        sym = uniprot_client.get_gene_name(uid)
        if sym:
            loose[uid] = sym
    return strict, loose


def map_df(df, mapping, tag):
    msrc = df["source"].map(mapping)
    mtgt = df["target"].map(mapping)
    hits = msrc.notna().sum() + mtgt.notna().sum()
    total = len(msrc) + len(mtgt)
    print(f"[{tag}] mapping hits: {hits:,}/{total:,} ({(hits/total):.1%})")
    dfm = df.copy()
    dfm["source"] = msrc
    dfm["target"] = mtgt
    before = len(dfm)
    dfm = dfm.dropna(subset=["source", "target"])
    after = len(dfm)
    print(f"[{tag}] kept rows after dropping unmapped: {after:,}/{before:,} ({(after/before):.1%})")
    return dfm


def build_graph(dfm, tag):
    G = nx.from_pandas_edgelist(
        dfm, source="source", target="target",
        edge_attr=["is_stimulation", "is_inhibition"], create_using=nx.DiGraph()
    )
    for _, _, d in G.edges(data=True):
        d["sign"] = 1 if d.get("is_stimulation") else -1 if d.get("is_inhibition") else 0
    print(f"[{tag}] graph => nodes: {G.number_of_nodes():,}, edges: {G.number_of_edges():,}")
    return G


if __name__ == "__main__":
    print("pyomnipath version:", op.__version__)

    df = load_broad()
    summarize_df("RAW broad table", df)

    # Build mappings
    all_uids = set(df["source"]).union(df["target"])
    t0 = time.time()
    strict_map, loose_map = build_maps(all_uids)
    print(f"Built mappings in {time.time()-t0:.2f}s "
          f"(STRICT keys: {len(strict_map):,}, LOOSE keys: {len(loose_map):,})")

    # Apply mappings
    df_strict = map_df(df, strict_map, "STRICT (HGNC)")
    df_loose  = map_df(df, loose_map,  "LOOSE (UniProt gene name)")

    # CAUSAL-only subsets
    causal_strict = df_strict[(df_strict["is_stimulation"] == True) | (df_strict["is_inhibition"] == True)]
    causal_loose  = df_loose[(df_loose["is_stimulation"] == True) | (df_loose["is_inhibition"] == True)]

    # Summaries
    rows = []
    for tag, d in [
        ("ANY + STRICT", df_strict),
        ("ANY + LOOSE",  df_loose),
        ("CAUSAL + STRICT", causal_strict),
        ("CAUSAL + LOOSE",  causal_loose),
    ]:
        n, n_causal = summarize_df(tag, d)
        rows.append({"graph": tag, "rows": n, "causal_rows": n_causal})

    # Build graphs and record sizes
    sizes = []
    for tag, d in [
        ("ANY + STRICT", df_strict),
        ("ANY + LOOSE",  df_loose),
        ("CAUSAL + STRICT", causal_strict),
        ("CAUSAL + LOOSE",  causal_loose),
    ]:
        G = build_graph(d, tag)
        sizes.append({"graph": tag, "nodes": G.number_of_nodes(), "edges": G.number_of_edges()})

    # Save summary CSV
    out = (pd.DataFrame(rows)
             .merge(pd.DataFrame(sizes), on="graph", how="left"))
    out.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved graph summary to {SUMMARY_CSV}")

    print("\nRecommendation:")
    print("- For 1-hop (INDRA parity): use CAUSAL ONLY (+ STRICT mapping).")
    print("- For 2/3-hop: use ANY interactions for the first N-1 hops, but require CAUSAL on the final hop; use STRICT mapping.")
