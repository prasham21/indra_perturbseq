import pandas as pd

# === Input files ===
old_file = "/Users/prashammarfatia/Downloads/indra_4hop_results_converted_2.csv"
new_file = "/Users/prashammarfatia/Downloads/4hop_intermediate_20251016_213433.csv"
output_file = "/Users/prashammarfatia/Downloads/indra_4hop_merged_final.csv"

# === Load both datasets ===
old_df = pd.read_csv(old_file)
new_df = pd.read_csv(new_file)

# Ensure consistent column order and names
common_cols = [c for c in new_df.columns if c in old_df.columns]
old_df = old_df[common_cols]
new_df = new_df[common_cols]

# Identify rows in old file that contain UniProt intermediates
intermediate_cols = [c for c in old_df.columns if c.startswith("intermediate_")]
mask_uniprot = old_df[intermediate_cols].astype(str).apply(
    lambda x: x.str.contains("uniprot:", case=False, na=False)
).any(axis=1)

# Remove those rows (we'll replace them)
old_clean = old_df[~mask_uniprot].copy()

# Combine old clean rows with new clean rerun data
merged_df = pd.concat([old_clean, new_df], ignore_index=True)

# Drop duplicate source-target pairs
merged_df.drop_duplicates(subset=["source", "target"], inplace=True)

# Save final result
merged_df.to_csv(output_file, index=False)
print(f"✅ Final merged file saved: {output_file}")
print(f"Rows in final dataset: {len(merged_df)}")
print(f"Unique sources: {merged_df['source'].nunique()}")
print(f"Unique targets: {merged_df['target'].nunique()}")

# Optional: sanity check for UniProt remnants
uniprot_rows = merged_df[intermediate_cols].astype(str).apply(
    lambda x: x.str.contains("uniprot:", case=False, na=False)
).any(axis=1)
print(f"Remaining UniProt rows after merge: {uniprot_rows.sum()}")
