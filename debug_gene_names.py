import pandas as pd
import re
from indra.databases import hgnc_client

# ========== CONFIG ==========
INPUT_FILE = "/Users/prashammarfatia/Downloads/indra_1hop_with_readable_statements.csv"
OUTPUT_FILE = "/Users/prashammarfatia/Downloads/indra_1hop_gene_names_cleaned.csv"

# ========== FIX + VALIDATE FUNCTION ==========
def fix_and_normalize(symbol):
    """
    Attempt to clean, normalize, and validate gene symbols.
    Examples:
      'TOP 1.00' ➝ 'TOP1' ➝ 'TOP1'
      'LMS 6.00' ➝ 'LMS6' ➝ 'LMS6'
    """
    if not isinstance(symbol, str) or not symbol.strip():
        return None

    raw = symbol.strip()

    # Try original symbol first
    hgnc_id = hgnc_client.get_current_hgnc_id(raw)
    if hgnc_id:
        return hgnc_client.get_hgnc_name(hgnc_id)

    # Try fixing: remove spaces and decimal suffixes like " 1.00"
    fixed = re.sub(r'[\s\-]*(\d+)(?:\.00)?$', r'\1', raw)  # "TOP 1.00" -> "TOP1", "LMS 6.00" -> "LMS6"
    hgnc_id = hgnc_client.get_current_hgnc_id(fixed)
    if hgnc_id:
        return hgnc_client.get_hgnc_name(hgnc_id)

    # Still not valid
    return None

# ========== LOAD + CLEAN ==========
df = pd.read_csv(INPUT_FILE)

df["source_clean"] = df["source"].apply(fix_and_normalize)
df["target_clean"] = df["target"].apply(fix_and_normalize)

# Drop unresolved gene rows
df_cleaned = df.dropna(subset=["source_clean", "target_clean"]).copy()

# Replace original columns
df_cleaned["source"] = df_cleaned["source_clean"]
df_cleaned["target"] = df_cleaned["target_clean"]
df_cleaned.drop(columns=["source_clean", "target_clean"], inplace=True)

# ========== SAVE ==========
df_cleaned.to_csv(OUTPUT_FILE, index=False)
print(f"✔️ Cleaned file saved to: {OUTPUT_FILE}")
print(f"✔️ Rows retained: {len(df_cleaned)}")
