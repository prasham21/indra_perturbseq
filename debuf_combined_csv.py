import pandas as pd
import os

# === CONFIG ===
HOP_FILES = {
    "1hop": "/Users/prashammarfatia/Downloads/indra_1hop_with_statements__main.csv",
    "2hop": "/Users/prashammarfatia/Downloads/indra_2hop_with_evidence_statements_main (2).csv",
    "3hop": "/Users/prashammarfatia/Downloads/indra_3hop_with_statements_main.csv",
    "4hop": "/Users/prashammarfatia/Downloads/indra_4hop_with_evidence_and_pmids_.csv",
}
OUTPUT_DIR = "/Users/prashammarfatia/Downloads/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "indra_top100_full_combined_standardized_1.csv")
DUPLICATES_FILE = os.path.join(OUTPUT_DIR, "indra_top100_duplicates_remove.csv")
SELF_PATHS_FILE = os.path.join(OUTPUT_DIR, "indra_self_paths_remove.csv")

# === STANDARDIZED COLUMN ORDER ===
STANDARD_COLUMNS = [
    "source", "target", "hop_number",
    "intermediate_1", "intermediate_2", "intermediate_3",
    "stmt_type_1", "edge1_evidence_text", "edge1_pmids",
    "stmt_type_2", "edge2_evidence_text", "edge2_pmids",
    "stmt_type_3", "edge3_evidence_text", "edge3_pmids",
    "stmt_type_4", "edge4_evidence_text", "edge4_pmids",
    "logfoldchange", "pval",
    "belief_1", "belief_2", "belief_3", "belief_4",
    "evidence_1", "evidence_2", "evidence_3", "evidence_4"
]

# === COLUMN MAPPINGS PER HOP FILE ===
COLUMN_MAPPINGS = {
    "1hop": {
        "stmt_type": "stmt_type_1",
        "evidence_text": "edge1_evidence_text",
        "pmids": "edge1_pmids",
        "belief": "belief_1",
        "evidence_count": "evidence_1",
    },

    "2hop": {
        "intermediate": "intermediate_1",
        "stmt_type_1": "stmt_type_1",
        "stmt_type_2": "stmt_type_2",
        "evidence_text_hop1": "edge1_evidence_text",
        "evidence_text_hop2": "edge2_evidence_text",
        "pmids_hop1": "edge1_pmids",
        "pmids_hop2": "edge2_pmids",
        "belief_1": "belief_1",
        "belief_2": "belief_2",
        "evidence_1": "evidence_1",
        "evidence_2": "evidence_2",
    },

    "3hop": {
        "intermediate_1": "intermediate_1",
        "intermediate_2": "intermediate_2",
        "stmt_type_1": "stmt_type_1",
        "stmt_type_2": "stmt_type_2",
        "stmt_type_3": "stmt_type_3",
        "evidence_text_hop1": "edge1_evidence_text",
        "evidence_text_hop2": "edge2_evidence_text",
        "evidence_text_hop3": "edge3_evidence_text",
        "pmids_hop1": "edge1_pmids",
        "pmids_hop2": "edge2_pmids",
        "pmids_hop3": "edge3_pmids",
        "belief_1": "belief_1",
        "belief_2": "belief_2",
        "belief_3": "belief_3",
        "evidence_1": "evidence_1",
        "evidence_2": "evidence_2",
        "evidence_3": "evidence_3",
    },

    "4hop": {
        "intermediate_1": "intermediate_1",
        "intermediate_2": "intermediate_2",
        "intermediate_3": "intermediate_3",
        "stmt_type_1": "stmt_type_1",
        "edge1_evidence_text": "edge1_evidence_text",
        "edge1_pmids": "edge1_pmids",
        "stmt_type_2": "stmt_type_2",
        "edge2_evidence_text": "edge2_evidence_text",
        "edge2_pmids": "edge2_pmids",
        "stmt_type_3": "stmt_type_3",
        "edge3_evidence_text": "edge3_evidence_text",
        "edge3_pmids": "edge3_pmids",
        "stmt_type_4": "stmt_type_4",
        "edge4_evidence_text": "edge4_evidence_text",
        "edge4_pmids": "edge4_pmids",
        "belief_1": "belief_1",
        "belief_2": "belief_2",
        "belief_3": "belief_3",
        "belief_4": "belief_4",
        "evidence_1": "evidence_1",
        "evidence_2": "evidence_2",
        "evidence_3": "evidence_3",
        "evidence_4": "evidence_4",
    }
}

# === LOAD AND STANDARDIZE ===
dfs = []
for hop, path in HOP_FILES.items():
    if not os.path.exists(path):
        print(f"⚠️ Missing file for {hop}: {path}")
        continue

    df = pd.read_csv(path)

    print(f"\n📁 Processing {hop} file ({len(df)} rows)")

    df["hop_number"] = hop

    # Check required columns
    missing_cols = [c for c in ["source", "target", "logfoldchange", "pval"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{hop} file missing columns: {missing_cols}")

    # Create standardized dataframe
    standardized_df = pd.DataFrame()

    # Always copy these basic columns
    standardized_df["source"] = df["source"]
    standardized_df["target"] = df["target"]
    standardized_df["hop_number"] = df["hop_number"]
    standardized_df["logfoldchange"] = df["logfoldchange"]
    standardized_df["pval"] = df["pval"]

    # Initialize all other columns as empty strings
    for col in STANDARD_COLUMNS:
        if col not in standardized_df.columns:
            standardized_df[col] = ""

    # Apply hop-specific column mapping
    mapping = COLUMN_MAPPINGS.get(hop, {})
    mapped_count = 0
    for original_col, standard_col in mapping.items():
        if original_col in df.columns:
            standardized_df[standard_col] = df[original_col]
            mapped_count += 1
        else:
            print(f"   ⚠️ Expected column not found: {original_col}")

    print(f"   ✓ Mapped {mapped_count}/{len(mapping)} columns")
    dfs.append(standardized_df)

if not dfs:
    raise FileNotFoundError("No valid hop files found!")

combined = pd.concat(dfs, ignore_index=True, sort=False)
print(f"\n✅ Loaded {len(combined)} rows across all hops.")

# === REMOVE SELF-PATHS ===
self_paths = combined[combined["source"] == combined["target"]]
if not self_paths.empty:
    self_paths_output = self_paths[STANDARD_COLUMNS]
    self_paths_output.to_csv(SELF_PATHS_FILE, index=False)
    print(f"\n🚫 Self-paths detected and removed: {len(self_paths)} rows")
    print(f"   Self-path genes: {self_paths['source'].unique().tolist()}")
    print(f"   Saved to: {SELF_PATHS_FILE}")

    # Remove self-paths from combined data
    combined = combined[combined["source"] != combined["target"]]
    print(f"   Remaining rows after removal: {len(combined)}")
else:
    print(f"\n✓ No self-paths detected")

# === CLEAN + RANK ===
combined["logfoldchange"] = pd.to_numeric(combined["logfoldchange"], errors="coerce")
combined["pval"] = pd.to_numeric(combined["pval"], errors="coerce")

combined["abs_logfc"] = combined["logfoldchange"].abs()
hop_priority = {"1hop": 1, "2hop": 2, "3hop": 3, "4hop": 4}
combined["hop_rank"] = combined["hop_number"].map(hop_priority)

# === LOG DUPLICATES BEFORE FILTERING ===
dupes = combined[combined.duplicated(subset=["source", "target"], keep=False)]
if not dupes.empty:
    dupes_output = dupes[STANDARD_COLUMNS]
    dupes_output.to_csv(DUPLICATES_FILE, index=False)
    print(f"\n📋 Duplicates saved to: {DUPLICATES_FILE} ({len(dupes)} rows)")

# === SELECT TOP 100 HIGH |LOGFC| ===
top_fc = (
    combined.sort_values(["abs_logfc", "hop_rank"], ascending=[False, True])
    .drop_duplicates(subset=["source", "target"], keep="first")
    .head(100)
)
print(f"\n📊 Top 100 by |logFC|: {len(top_fc)}")

# === SELECT TOP 100 LOW PVALUE ===
top_pval = (
    combined.sort_values(["pval", "hop_rank"], ascending=[True, True])
    .drop_duplicates(subset=["source", "target"], keep="first")
    .head(100)
)
print(f"📊 Top 100 by p-value: {len(top_pval)}")

# === COMBINE BOTH SETS (UNION, PRIORITIZE EARLIER HOPS) ===
final = (
    pd.concat([top_fc, top_pval], ignore_index=True, sort=False)
    .sort_values(["hop_rank", "pval"], ascending=[True, True])
    .drop_duplicates(subset=["source", "target"], keep="first")
)

print(f"\n📊 Final results before hop reduction:")
print(f"   Total rows: {len(final)}")
hop_counts_before = final["hop_number"].value_counts().to_dict()
print(f"   Hops: {hop_counts_before}")

# === REDUCE 3HOP AND 4HOP IN FINAL RESULTS ===
print(f"\n📉 Reducing 3hop and 4hop rows in final results:")

# Separate by hop type
final_1hop = final[final["hop_number"] == "1hop"]
final_2hop = final[final["hop_number"] == "2hop"]
final_3hop = final[final["hop_number"] == "3hop"]
final_4hop = final[final["hop_number"] == "4hop"]

# Reduce 3hop: keep 75% (remove 25%)
if len(final_3hop) > 0:
    keep_count_3hop = int(len(final_3hop) * 0.75)
    final_3hop_reduced = final_3hop.head(keep_count_3hop)  # Keep top rows by current sort order
    print(
        f"   3hop: {len(final_3hop)} → {len(final_3hop_reduced)} (removed {len(final_3hop) - len(final_3hop_reduced)})")
else:
    final_3hop_reduced = final_3hop
    print(f"   3hop: 0 rows (nothing to reduce)")

# Reduce 4hop: keep 50% (remove 50%)
if len(final_4hop) > 0:
    keep_count_4hop = int(len(final_4hop) * 0.50)
    final_4hop_reduced = final_4hop.head(keep_count_4hop)  # Keep top rows by current sort order
    print(
        f"   4hop: {len(final_4hop)} → {len(final_4hop_reduced)} (removed {len(final_4hop) - len(final_4hop_reduced)})")
else:
    final_4hop_reduced = final_4hop
    print(f"   4hop: 0 rows (nothing to reduce)")

# Recombine final results
final = pd.concat([final_1hop, final_2hop, final_3hop_reduced, final_4hop_reduced], ignore_index=True)

# === SAVE FINAL OUTPUT WITH STANDARDIZED COLUMNS ===
final_output = final[STANDARD_COLUMNS]
final_output.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Saved standardized output: {OUTPUT_FILE}")
print(f"   Rows: {len(final_output)}")
hop_counts_after = final['hop_number'].value_counts().to_dict()
print(f"   Hops represented: {hop_counts_after}")

# === OPTIONAL THRESHOLDS ===
# Recalculate thresholds based on reduced final set
final_fc_rows = final[final.index.isin(top_fc.index)]
final_pval_rows = final[final.index.isin(top_pval.index)]

if not final_fc_rows.empty:
    fc_thresh = final_fc_rows["abs_logfc"].min()
    print(f"   |logFC| threshold ≥ {fc_thresh:.3f}")
if not final_pval_rows.empty:
    pval_thresh = final_pval_rows["pval"].max()
    print(f"   p-value threshold ≤ {pval_thresh:.3e}")

print("\n🎯 Done — standardized format with reduced 3hop and 4hop representation.")