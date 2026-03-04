import pandas as pd

# Define file paths
files = {
    "1hop": "/Users/prashammarfatia/Downloads/indra_1hop_with_statements__main.csv",
    "2hop": "/Users/prashammarfatia/Downloads/indra_2hop_with_evidence_statements_main (2).csv",
    "3hop": "/Users/prashammarfatia/Downloads/indra_3hop_with_statements_main.csv",
    "4hop": "/Users/prashammarfatia/Downloads/indra_4hop_with_evidence_and_pmids_.csv",
}

# Check for self-paths in each file
print("=" * 60)
print("SELF-PATH ANALYSIS")
print("=" * 60)

for hop_type, filepath in files.items():
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)

        # Check if 'source' and 'target' columns exist
        if 'source' not in df.columns or 'target' not in df.columns:
            print(f"\n{hop_type}:")
            print(f"  ⚠️  Columns 'source' and/or 'target' not found")
            print(f"  Available columns: {list(df.columns)}")
            continue

        # Find self-paths (source == target)
        self_paths = df[df['source'] == df['target']]
        num_self_paths = len(self_paths)
        total_rows = len(df)

        # Display results
        print(f"\n{hop_type}:")
        print(f"  Total rows: {total_rows}")
        print(f"  Self-paths found: {num_self_paths}")

        if num_self_paths > 0:
            print(f"  Percentage: {(num_self_paths / total_rows) * 100:.2f}%")
            print(f"  Example self-paths:")
            # Show first 5 unique self-path genes
            unique_self_genes = self_paths['source'].unique()[:5]
            for gene in unique_self_genes:
                print(f"    - {gene}")
        else:
            print(f"  ✓ No self-paths found")

    except FileNotFoundError:
        print(f"\n{hop_type}:")
        print(f"  ❌ File not found: {filepath}")
    except Exception as e:
        print(f"\n{hop_type}:")
        print(f"  ❌ Error: {str(e)}")

print("\n" + "=" * 60)