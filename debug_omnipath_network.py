import omnipath as op
from omnipath.interactions import AllInteractions
import pandas as pd
import networkx as nx
import time

CACHE_FILE = "omnipath_broad.parquet"


def fetch_omnipath_graph():
    try:
        print("Loading cached OmniPath data from", CACHE_FILE)
        df = pd.read_parquet(CACHE_FILE)
    except FileNotFoundError:
        print("Downloading OmniPath interactions (broad table)…")
        # ✅ This is the old v1.0.12 API
        df = AllInteractions.get()
        print(f"Downloaded {len(df)} interactions")
        df.to_parquet(CACHE_FILE)
        print(f"Saved to {CACHE_FILE}")

    # Make sure the columns are consistent
    df = df.rename(columns={"source": "source", "target": "target"})
    print("Columns available:", list(df.columns))

    # Build directed graph
    start = time.time()
    G = nx.from_pandas_edgelist(df, "source", "target", create_using=nx.DiGraph)
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges "
          f"in {time.time() - start:.2f}s")
    return G


if __name__ == "__main__":
    print("Using pyomnipath version:", op.__version__)
    G = fetch_omnipath_graph()
