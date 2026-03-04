import pandas as pd
import re

# === CONFIG ===
REFERENCE_FILE = "/Users/prashammarfatia/Downloads/comprehensive_mesh_list_EXPANDED_.csv"
HOP_FILE = "/Users/prashammarfatia/Downloads/indra_2hop_with_mesh_terms.csv"
OUTPUT_FILE = "/Users/prashammarfatia/Downloads/_new_2hop_mesh_filtered.csv"

# === STEP 1: Load reference MeSH list (robust) ===
ref_df = pd.read_csv(REFERENCE_FILE, encoding='utf-8-sig', on_bad_lines='skip')

# Clean mesh_id thoroughly
ref_df["mesh_id"] = (
    ref_df["mesh_id"]
    .astype(str)
    .str.replace(r"\s+", "", regex=True)          # remove whitespace, tabs, newlines
    .str.replace(r"[^A-Za-z0-9]", "", regex=True) # remove non-alphanumeric chars
)

# Keep unique valid MeSH IDs (start with D)
valid_ids = set(ref_df["mesh_id"][ref_df["mesh_id"].str.startswith("D")])
print(f"✅ Loaded {len(valid_ids)} valid MeSH IDs from reference list")

# === STEP 2: Load 2-hop dataset ===
hop_df = pd.read_csv(HOP_FILE)
print(f"✅ Loaded {len(hop_df)} rows from {HOP_FILE}\n")

# === STEP 3: Define helper function for filtering ===
def filter_mesh_terms(text):
    """Keep only valid MeSH terms present in the reference list."""
    if pd.isna(text):
        return ""
    pairs = re.findall(r'[^,]+?\(D\d{5,10}\)', text)
    filtered = []
    for p in pairs:
        match = re.search(r'\((D\d{5,10})\)', p)
        if match and match.group(1) in valid_ids:
            filtered.append(p.strip())
    return ", ".join(filtered)

# === STEP 4: Apply cleaning to BOTH MeSH columns ===
mesh_columns = ["Annotated MeSH terms hop1", "Annotated MeSH terms hop2"]

for target_col in mesh_columns:
    if target_col in hop_df.columns:
        print(f"🧹 Filtering column: {target_col}")
        hop_df[target_col] = hop_df[target_col].apply(filter_mesh_terms)
    else:
        print(f"⚠️ Column '{target_col}' not found — skipping.")

# === STEP 5: Drop unwanted or duplicate columns if any ===
drop_cols = [
    c for c in hop_df.columns
    if any(s in c for s in [".1", ".2", ".3", "(copy)", "n_kept", "n_total", "n_dropped"])
]
if drop_cols:
    hop_df.drop(columns=drop_cols, inplace=True)
    print(f"🗑️ Dropped extra columns: {drop_cols}")
else:
    print("✅ No extra or duplicate columns found to drop.")

# === STEP 6: Save cleaned output ===
hop_df.to_csv(OUTPUT_FILE, index=False)
print(f"📁 Clean filtered 2-hop file saved to: {OUTPUT_FILE}")