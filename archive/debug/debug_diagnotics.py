import pandas as pd


def mesh_percentage(filepath, mesh_cols):
    df = pd.read_csv(filepath)
    df["has_mesh"] = df[mesh_cols].apply(
        lambda row: any(isinstance(v, str) and v.strip() != "" for v in row),
        axis=1
    )
    unique_pairs = df.drop_duplicates(subset=["source", "target"])
    percent = (unique_pairs["has_mesh"].sum() / len(unique_pairs)) * 100
    return round(percent, 2)


# === File paths and columns ===
onehop = "/Users/prashammarfatia/Downloads/1hop_mesh_filtered__.csv"
twohop = "/Users/prashammarfatia/Downloads/2hop_mesh_filtered__.csv"
threehop = "/Users/prashammarfatia/Downloads/3hop_mesh_filtered__.csv"

p1 = mesh_percentage(onehop, ["Annotated MeSH terms"])
p2 = mesh_percentage(twohop, ["Annotated MeSH terms hop1", "Annotated MeSH terms hop2"])
p3 = mesh_percentage(threehop, [
    "Annotated MeSH terms (hop1)", "Annotated MeSH terms (hop2)", "Annotated MeSH terms (hop3)"
])

print(f"1-hop:  {p1}%")
print(f"2-hop:  {p2}%")
print(f"3-hop:  {p3}%")
