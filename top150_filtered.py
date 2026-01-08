import pandas as pd

# === CONFIG ===
INPUT_FILE = "/Users/prashammarfatia/Downloads/1hop_mesh_filtered__.csv"
OUTPUT_FILE = "/Users/prashammarfatia/Downloads/1hop_top150_unique_pairs.csv"

# === STEP 1: Load data ===
df = pd.read_csv(INPUT_FILE)
print(f"✅ Loaded {len(df)} rows from {INPUT_FILE}")

# === STEP 2: Drop duplicates based on unique source-target pair ===
# Keep the row with the highest logfoldchange per pair
df_unique = df.sort_values("logfoldchange", ascending=False).drop_duplicates(
    subset=["source", "target"], keep="first"
)

print(f"✅ After deduplication: {len(df_unique)} unique (source, target) pairs")

# === STEP 3: Sort by logfoldchange and take top 150 ===
df_top150 = df_unique.sort_values("logfoldchange", ascending=False).head(150)

# === STEP 4: Save to output ===
df_top150.to_csv(OUTPUT_FILE, index=False)
print(f"📁 Top 150 unique source-target pairs saved to: {OUTPUT_FILE}")

# === STEP 5: Optional summary printout ===
print("\n📊 === Summary ===")
print(f"Total unique pairs analyzed: {len(df_unique)}")
print(f"Top 150 max logfoldchange range: {df_top150['logfoldchange'].min():.4f} – {df_top150['logfoldchange'].max():.4f}")
