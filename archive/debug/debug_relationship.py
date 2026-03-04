import pandas as pd

# ==============================
# CONFIGURATION
# ==============================
INPUT_CSV = "/Users/prashammarfatia/Downloads/indra_3hop_with_statements_fixed.csv"  # Script A output
OUTPUT_CSV = "/Users/prashammarfatia/Downloads/indra_3hop_with_statements_cleaned_2.csv"  # Final clean version


def clean_identifier(agent_str):
    """
    Clean up identifier strings for better readability in CSV output.
    Converts 'hgnc:uniprot:P12345' -> 'UNIPROT:P12345'
    """
    if not agent_str or pd.isna(agent_str):
        return agent_str

    agent_str = str(agent_str).strip()

    # Handle known prefix patterns
    if agent_str.startswith("hgnc:uniprot.chain:"):
        return "UNIPROT:" + agent_str.replace("hgnc:uniprot.chain:", "")
    elif agent_str.startswith("hgnc:uniprot:"):
        return "UNIPROT:" + agent_str.replace("hgnc:uniprot:", "")
    elif agent_str.startswith("hgnc:mesh:"):
        return "MESH:" + agent_str.replace("hgnc:mesh:", "")
    elif agent_str.startswith("hgnc:chebi:"):
        return "CHEBI:" + agent_str.replace("hgnc:chebi:", "")
    elif agent_str.startswith("hgnc:fplx:"):
        return "FPLX:" + agent_str.replace("hgnc:fplx:", "")
    elif agent_str.startswith("hgnc:"):
        return "HGNC:" + agent_str.replace("hgnc:", "")
    else:
        # Leave unchanged for gene symbols or other formats
        return agent_str


def main():
    print("Loading Script A CSV...")
    df = pd.read_csv(INPUT_CSV)

    print(f"Original data shape: {df.shape}")

    # Show examples of what will be changed
    print("\nExample transformations:")
    print("Before cleanup:")
    print(f"  intermediate_1 sample: {df['intermediate_1'].iloc[0]}")
    print(f"  intermediate_2 sample: {df['intermediate_2'].iloc[0]}")

    # Clean up the intermediate columns
    df['intermediate_1'] = df['intermediate_1'].apply(clean_identifier)
    df['intermediate_2'] = df['intermediate_2'].apply(clean_identifier)

    print("\nAfter cleanup:")
    print(f"  intermediate_1 sample: {df['intermediate_1'].iloc[0]}")
    print(f"  intermediate_2 sample: {df['intermediate_2'].iloc[0]}")

    # Optional: Also improve evidence text formatting (replace "; " with proper line breaks)
    print("\nOptional: Also fixing evidence text formatting...")

    def format_evidence_text(text):
        """Fix evidence text formatting with proper numbering."""
        if not isinstance(text, str) or "; " not in text:
            return text

        # Split by "; " to get individual evidence pieces
        pieces = text.split("; ")
        formatted_pieces = []

        for i, piece in enumerate(pieces, 1):
            # Remove old numbering (e.g., "2. text" becomes "text")
            if piece.strip() and piece[0].isdigit() and ". " in piece:
                piece = piece.split(". ", 1)[1]  # Remove "2. " part

            # Add new numbering with parentheses
            formatted_pieces.append(f"{i}) {piece.strip()}")

        return "\n\n".join(formatted_pieces)

    for col in ['edge1_statements', 'edge2_statements', 'edge3_statements']:
        if col in df.columns:
            df[col] = df[col].apply(format_evidence_text)

    # Save the cleaned version
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nCleaned CSV saved to: {OUTPUT_CSV}")
    print("✅ Processing complete!")
    print(f"   - Cleaned intermediate_1 and intermediate_2 columns")
    print(f"   - Improved evidence text formatting")
    print(f"   - All original evidence data preserved")


if __name__ == "__main__":
    main()