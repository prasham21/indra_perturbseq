"""Legacy script: omnipath_client_1hop."""
from __future__ import annotations

import argparse
import os
import time
import pandas as pd
import networkx as nx
import omnipath as op
from indra.databases import uniprot_client, hgnc_client

import logging


logger = logging.getLogger(__name__)

CACHE_FILE = "omnipath_hgnc.parquet"
INPUT_DIR = "de_results_per_gene"
MANIFEST = "target_validation_expanded.csv"
OUTPUT_FILE = "omnipath_1hop_all_perturbations.csv"


def load_significant_descendants(file_path: str, pval_threshold: float = 0.05) -> dict[str, int]:
    """Load descendants from Perturb-seq DE results and filter by significance."""
    df = pd.read_csv(file_path)
    sig = df[df["pvals"] < pval_threshold].copy()
    sig["direction"] = sig["logfoldchanges"].apply(lambda x: 1 if x > 0 else -1)
    return dict(zip(sig["names"], sig["direction"]))


def build_uniprot_to_hgnc(uniprot_ids: set[str]) -> dict[str, str]:
    """Build UniProt -> HGNC symbol dictionary using INDRA clients."""
    mapping = {}
    for uid in uniprot_ids:
        hgnc_id = uniprot_client.get_hgnc_id(uid)
        if hgnc_id:
            hgnc_symbol = hgnc_client.get_hgnc_name(hgnc_id)
            if hgnc_symbol:
                mapping[uid] = hgnc_symbol
    return mapping


def fetch_omnipath_graph(force_reload: bool = False) -> nx.DiGraph:
    """Fetch OmniPath interactions, map UniProt -> HGNC, cache, and build graph."""
    if os.path.exists(CACHE_FILE) and not force_reload:
        df = pd.read_parquet(CACHE_FILE)
    else:
        logger.info("Downloading OmniPath interactions...")
        df = op.interactions.OmniPath.get()
        df = df[["source", "target", "is_stimulation", "is_inhibition"]].dropna()

        all_ids = set(df["source"]).union(df["target"])
        uniprot_to_hgnc = build_uniprot_to_hgnc(all_ids)

        df["source"] = df["source"].map(uniprot_to_hgnc)
        df["target"] = df["target"].map(uniprot_to_hgnc)
        df = df.dropna(subset=["source", "target"])

        df.to_parquet(CACHE_FILE)
        logger.info("Saved mapped OmniPath data to %s", CACHE_FILE)

    G = nx.from_pandas_edgelist(
        df,
        source="source",
        target="target",
        edge_attr=["is_stimulation", "is_inhibition"],
        create_using=nx.DiGraph(),
    )
    for u, v, d in G.edges(data=True):
        d["sign"] = 1 if d["is_stimulation"] else -1 if d["is_inhibition"] else 0
    logger.info("Graph built with %s nodes and %s edges", G.number_of_nodes(), G.number_of_edges())
    return G


def one_hop_coverage(G: nx.DiGraph, perturb: str, descendants: dict[str, int]):
    """Return explained descendants list plus counts for 1-hop coverage."""
    if perturb not in G:
        return [], 0, len(descendants), 0.0

    explained = [d for d in descendants if G.has_edge(perturb, d)]
    coverage = len(explained) / len(descendants) if descendants else 0
    return explained, len(explained), len(descendants) - len(explained), coverage




def main():
    ap = argparse.ArgumentParser()



if __name__ == "__main__":
    main()
