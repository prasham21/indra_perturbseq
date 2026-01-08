import os
import pandas as pd
from indra.databases import uniprot_client, hgnc_client

# === CONFIG ===
INPUT_FILE = "/Users/prashammarfatia/Downloads/indra_4hop_results_human_final.csv"
OUTPUT_FILE = "/Users/prashammarfatia/Downloads/indra_4hop_results_converted_2.csv"


# === FUNCTION ===
def convert_to_hgnc_symbol(identifier: str) -> str:
    """Convert UniProt or UniProt.chain IDs to HGNC symbols if possible."""
    if not isinstance(identifier, str):
        return identifier
    # Normalize variants
    identifier = identifier.strip()
    identifier = identifier.replace("uniprot.chain:", "uniprot:")
    identifier = identifier.replace("hgnc:uniprot:", "uniprot:")

    # If not UniProt, return as is
    if not identifier.startswith("uniprot:"):
        return identifier

    uid = identifier.split("uniprot:")[-1]
    hgnc_id = uniprot_client.get_hgnc_id(uid)
    if hgnc_id:
        hgnc_symbol = hgnc_client.get_hgnc_name(hgnc_id)
        if hgnc_symbol:
            return hgnc_symbol
    return identifier  # fallback if not found


def main():
    print("=== UniProt → HGNC Symbol Conversion ===")
    print(f"Loading input: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    intermediate_cols = [col for col in df.columns if col.startswith("intermediate_")]
    print(f"Converting columns: {intermediate_cols}")

    total_before = 0
    total_converted = 0
    not_found = set()

    for col in intermediate_cols:
        def safe_convert(x):
            nonlocal total_before, total_converted
            if isinstance(x, str) and "uniprot" in x:
                total_before += 1
                new = convert_to_hgnc_symbol(x)
                if new != x:
                    total_converted += 1
                else:
                    not_found.add(x)
                return new
            return x

        df[col] = df[col].apply(safe_convert)

    print(f"Total UniProt-like IDs found: {total_before}")
    print(f"Successfully converted: {total_converted}")
    print(f"Unconverted (likely non-human proteins or missing): {len(not_found)}")

    # Save new file
    df.to_csv(OUTPUT_FILE, index=False)
    print(f" Saved converted file to: {OUTPUT_FILE}")

    # Optional: save missing IDs list
    if not_found:
        with open(OUTPUT_FILE.replace(".csv", "_unmapped.txt"), "w") as f:
            f.write("\n".join(sorted(not_found)))
        print(f"Unmapped UniProt IDs saved to: {OUTPUT_FILE.replace('.csv', '_unmapped.txt')}")


if __name__ == "__main__":
    main()
