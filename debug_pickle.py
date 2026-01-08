import pandas as pd
import re

# === Paths ===
csv_input_path = "/Users/prashammarfatia/Downloads/indra_3hop_with_statements_.csv"
excel_output_path = "/Users/prashammarfatia/Downloads/indra_3hop_with_statements__.xlsx"

# === Load CSV safely ===
df = pd.read_csv(csv_input_path, dtype=str, keep_default_na=False)

# === Function to remove illegal Excel characters ===
def sanitize_for_excel(value):
    if isinstance(value, str):
        # Remove illegal ASCII control characters except \n and \t
        return re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", value)
    return value

# === Apply sanitization to all cells ===
df_cleaned = df.applymap(sanitize_for_excel)

# === Export to Excel safely ===
df_cleaned.to_excel(excel_output_path, index=False, engine='openpyxl')
print(f"✅ Clean Excel saved to:\n{excel_output_path}")
