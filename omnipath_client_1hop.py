import os
import time
import pandas as pd
import networkx as nx
import omnipath as op
from indra.databases import uniprot_client, hgnc_client


CACHE_FILE = "omnipath_hgnc.parquet"
INPUT_DIR = "/Users/prashammarfatia/Downloads/de_results_per_gene"
MANIFEST = "/Users/prashammarfatia/Downloads/target_validation_expanded.csv"
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
        print("Downloading OmniPath interactions...")
        df = op.interactions.OmniPath.get()
        df = df[["source", "target", "is_stimulation", "is_inhibition"]].dropna()

        all_ids = set(df["source"]).union(df["target"])
        uniprot_to_hgnc = build_uniprot_to_hgnc(all_ids)

        df["source"] = df["source"].map(uniprot_to_hgnc)
        df["target"] = df["target"].map(uniprot_to_hgnc)
        df = df.dropna(subset=["source", "target"])

        df.to_parquet(CACHE_FILE)
        print(f"Saved mapped OmniPath data to {CACHE_FILE}")

    G = nx.from_pandas_edgelist(
        df,
        source="source",
        target="target",
        edge_attr=["is_stimulation", "is_inhibition"],
        create_using=nx.DiGraph(),
    )
    for u, v, d in G.edges(data=True):
        d["sign"] = 1 if d["is_stimulation"] else -1 if d["is_inhibition"] else 0
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def one_hop_coverage(G: nx.DiGraph, perturb: str, descendants: dict[str, int]):
    """Return explained descendants list plus counts for 1-hop coverage."""
    if perturb not in G:
        return [], 0, len(descendants), 0.0

    explained = [d for d in descendants if G.has_edge(perturb, d)]
    coverage = len(explained) / len(descendants) if descendants else 0
    return explained, len(explained), len(descendants) - len(explained), coverage


if __name__ == "__main__":
    # Build OmniPath graph once
    G = fetch_omnipath_graph(force_reload=False)

    # Load perturbation manifest
    perturb_df = pd.read_csv(MANIFEST)
    perturb_df = perturb_df[perturb_df["Karen_Flag"] == "Use_for_analysis"]
    print(f"Perturbations selected: {len(perturb_df)}")

    all_results = []
    start_time = time.time()

    for idx, row in perturb_df.iterrows():
        perturb_gene = row["Gene"].upper()
        print(f"\n({idx + 1}/{len(perturb_df)}) Processing: {perturb_gene}")

        file_path = os.path.join(INPUT_DIR, f"{perturb_gene}_vs_control.csv")
        if not os.path.exists(file_path):
            print(f" DEG file not found: {file_path}")
            continue

        descendants = load_significant_descendants(file_path)
        explained_list, explained, not_explained, coverage = one_hop_coverage(G, perturb_gene, descendants)

        all_results.append({
            "perturbation": perturb_gene,
            "total_descendants": len(descendants),
            "explained_count": explained,
            "not_explained_count": not_explained,
            "coverage": coverage,
            "explained_descendants": ",".join(explained_list) if explained_list else ""
        })

        # Progress log
        loop_time = time.time() - start_time
        avg_time = loop_time / (idx + 1)
        remaining = avg_time * (len(perturb_df) - idx - 1)
        print(f" Explained: {explained} | Coverage: {coverage:.2%} | "
              f"ETA: {remaining/60:.1f} mins")

    # Save final CSV
    output_df = pd.DataFrame(all_results)
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved all results to {OUTPUT_FILE}")
