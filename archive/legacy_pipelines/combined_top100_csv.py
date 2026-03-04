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
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "indra_top100_full_combined_standardized.csv")
DUPLICATES_FILE = os.path.join(OUTPUT_DIR, "indra_top100_duplicates_removed.csv")

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

# === LOAD AND STANDARDIZE ===
dfs = []
for hop, path in HOP_FILES.items():
    if not os.path.exists(path):
        print(f"⚠️ Missing file for {hop}: {path}")
        continue

    df = pd.read_csv(path)
    df["hop_number"] = hop

    # Check required columns
    missing_cols = [c for c in ["source", "target", "logfoldchange", "pval"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{hop} file missing columns: {missing_cols}")

    # Create a new dataframe with standardized columns
    standardized_df = pd.DataFrame()

    # Copy basic columns
    standardized_df["source"] = df["source"]
    standardized_df["target"] = df["target"]
    standardized_df["hop_number"] = df["hop_number"]
    standardized_df["logfoldchange"] = df["logfoldchange"]
    standardized_df["pval"] = df["pval"]

    # Initialize all other columns as empty
    for col in STANDARD_COLUMNS:
        if col not in standardized_df.columns:
            standardized_df[col] = ""

    # Map existing columns to standardized format
    # You'll need to adjust these mappings based on your actual column names
    column_mapping = {
        # Intermediates
        "intermediate_1": "intermediate_1",
        "intermediate_2": "intermediate_2",
        "intermediate_3": "intermediate_3",

        # Statement types
        "stmt_type_1": "stmt_type_1",
        "stmt_type_2": "stmt_type_2",
        "stmt_type_3": "stmt_type_3",
        "stmt_type_4": "stmt_type_4",

        # Evidence text
        "edge1_evidence_text": "edge1_evidence_text",
        "edge2_evidence_text": "edge2_evidence_text",
        "edge3_evidence_text": "edge3_evidence_text",
        "edge4_evidence_text": "edge4_evidence_text",

        # PMIDs
        "edge1_pmids": "edge1_pmids",
        "edge2_pmids": "edge2_pmids",
        "edge3_pmids": "edge3_pmids",
        "edge4_pmids": "edge4_pmids",

        # Beliefs
        "belief_1": "belief_1",
        "belief_2": "belief_2",
        "belief_3": "belief_3",
        "belief_4": "belief_4",

        # Evidence counts
        "evidence_1": "evidence_1",
        "evidence_2": "evidence_2",
        "evidence_3": "evidence_3",
        "evidence_4": "evidence_4",
    }

    # Copy data from original columns if they exist
    for orig_col, std_col in column_mapping.items():
        if orig_col in df.columns:
            standardized_df[std_col] = df[orig_col]

    dfs.append(standardized_df)

if not dfs:
    raise FileNotFoundError("No valid hop files found!")

combined = pd.concat(dfs, ignore_index=True, sort=False)
print(f"✅ Loaded {len(combined)} rows across all hops.")

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
    print(f"📋 Duplicates saved to: {DUPLICATES_FILE} ({len(dupes)} rows)")

# === SELECT TOP 100 HIGH |LOGFC| ===
top_fc = (
    combined.sort_values(["abs_logfc", "hop_rank"], ascending=[False, True])
    .drop_duplicates(subset=["source", "target"], keep="first")
    .head(100)
)
print(f"📊 Top 100 by |logFC|: {len(top_fc)}")

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

# === SAVE FINAL OUTPUT WITH STANDARDIZED COLUMNS ===
final_output = final[STANDARD_COLUMNS]
final_output.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved standardized output: {OUTPUT_FILE}")
print(f"   Rows: {len(final_output)}")
print(f"   Hops represented: {final['hop_number'].value_counts().to_dict()}")

# === OPTIONAL THRESHOLDS ===
if not top_fc.empty:
    fc_thresh = top_fc["abs_logfc"].min()
    print(f"   |logFC| threshold ≥ {fc_thresh:.3f}")
if not top_pval.empty:
    pval_thresh = top_pval["pval"].max()
    print(f"   p-value threshold ≤ {pval_thresh:.3e}")

print("🎯 Done — standardized format with all columns, blanks where data doesn't exist.")