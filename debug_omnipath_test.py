"""
SMAD3 test: 1-hop and 2-hop coverage using OmniPath (cached).
- Reuses the hybrid UniProt->HGNC mapping (correct + fast)
- Uses shortest-path lengths to get exactly-1-hop and exactly-2-hop sets
- Prints counts and explained descendant lists
- Writes a small CSV report for SMAD3
"""

import os
import time
import pandas as pd
import networkx as nx
import omnipath as op
from indra.databases import uniprot_client, hgnc_client

# Paths
CACHE_FILE = "omnipath_hgnc.parquet"
SMAD3_DE_PATH = "/Users/prashammarfatia/Downloads/de_results_per_gene/SMAD3_vs_control.csv"
OUTPUT_FILE = "omnipath_smad3_1hop_2hop.csv"


def load_significant_descendants(file_path: str, pval_threshold: float = 0.05) -> dict[str, int]:
    """Load descendants from Perturb-seq DE results and filter by significance."""
    df = pd.read_csv(file_path)
    sig = df[df["pvals"] < pval_threshold].copy()
    sig["direction"] = sig["logfoldchanges"].apply(lambda x: 1 if x > 0 else -1)
    print(f"Loaded {len(sig)} significant descendants with p < {pval_threshold}")
    return dict(zip(sig["names"], sig["direction"]))


def build_uniprot_to_hgnc(uniprot_ids: set[str]) -> dict[str, str]:
    """Build UniProt -> HGNC symbol dictionary once using INDRA clients."""
    mapping = {}
    for uid in uniprot_ids:
        hgnc_id = uniprot_client.get_hgnc_id(uid)
        if hgnc_id:
            sym = hgnc_client.get_hgnc_name(hgnc_id)
            if sym:
                mapping[uid] = sym
    return mapping


def fetch_omnipath_graph(force_reload: bool = False) -> nx.DiGraph:
    """
    Fetch OmniPath causal interactions, map UniProt -> HGNC (vectorized),
    cache the mapped DataFrame, and build a signed directed graph.
    """
    if os.path.exists(CACHE_FILE) and not force_reload:
        df = pd.read_parquet(CACHE_FILE)
        print(f"Loading cached OmniPath data from {CACHE_FILE}")
    else:
        t0 = time.time()
        print("Downloading OmniPath interactions...")
        df = op.interactions.OmniPath.get()
        print(f"Download done in {time.time() - t0:.2f} s")

        df = df[["source", "target", "is_stimulation", "is_inhibition"]].dropna()

        # Build mapping once, apply vectorized
        t1 = time.time()
        all_ids = set(df["source"]).union(df["target"])
        up2hgnc = build_uniprot_to_hgnc(all_ids)
        df["source"] = df["source"].map(up2hgnc)
        df["target"] = df["target"].map(up2hgnc)
        df = df.dropna(subset=["source", "target"])
        print(f"Mapping done in {time.time() - t1:.2f} s")

        df.to_parquet(CACHE_FILE)
        print(f"Saved mapped OmniPath data to {CACHE_FILE}")

    # Build graph
    t2 = time.time()
    G = nx.from_pandas_edgelist(
        df,
        source="source",
        target="target",
        edge_attr=["is_stimulation", "is_inhibition"],
        create_using=nx.DiGraph(),
    )
    for _, _, d in G.edges(data=True):
        d["sign"] = 1 if d["is_stimulation"] else -1 if d["is_inhibition"] else 0
    print(
        f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges "
        f"in {time.time() - t2:.2f} s"
    )
    return G


def explained_descendants_at_hop(
    G: nx.DiGraph, source_gene: str, descendants: dict[str, int], hop: int
) -> list[str]:
    """
    Return descendants explained by exactly `hop` steps from source_gene.
    Uses shortest-path lengths: nodes with length == hop.
    """
    if source_gene not in G:
        return []
    lengths = nx.single_source_shortest_path_length(G, source_gene, cutoff=hop)
    return sorted([g for g in descendants.keys() if lengths.get(g) == hop])


if __name__ == "__main__":
    # 1) Load SMAD3 descendants
    descendants = load_significant_descendants(SMAD3_DE_PATH)

    # 2) Build/load OmniPath graph
    G = fetch_omnipath_graph(force_reload=False)

    # 3) Compute 1-hop and 2-hop explained sets
    smad3 = "SMAD3"
    explained_1 = explained_descendants_at_hop(G, smad3, descendants, hop=1)
    explained_2 = explained_descendants_at_hop(G, smad3, descendants, hop=2)

    # 4) Print report
    total = len(descendants)
    print("\n=== SMAD3 coverage ===")
    print(f"Total descendants (p<0.05): {total}")
    print(f"1-hop explained: {len(explained_1)}  ({len(explained_1)/total:.2%})")
    print(f"Explained@1-hop: {explained_1}")
    print(f"2-hop explained: {len(explained_2)}  ({len(explained_2)/total:.2%})")
    print(f"Explained@2-hop: {explained_2}")

    # 5) Save small CSV for SMAD3
    rows = []
    rows.append(
        {
            "perturbation": smad3,
            "hop": 1,
            "total_descendants": total,
            "explained_count": len(explained_1),
            "coverage": len(explained_1) / total if total else 0.0,
            "explained_descendants": ",".join(explained_1),
        }
    )
    rows.append(
        {
            "perturbation": smad3,
            "hop": 2,
            "total_descendants": total,
            "explained_count": len(explained_2),
            "coverage": len(explained_2) / total if total else 0.0,
            "explained_descendants": ",".join(explained_2),
        }
    )
    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved SMAD3 1-hop/2-hop report to {OUTPUT_FILE}")
