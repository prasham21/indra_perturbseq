import pandas as pd
import os


def separate_self_targeting_3hop():
    """
    Separate 3-hop dataset into self-targeting and non-self-targeting files
    """
    # File path for 3-hop data
    input_file = "/Users/prashammarfatia/Downloads/indra_2hop_with_evidence_statements__range_replaced.csv"

    print("=" * 50)
    print("SEPARATING SELF-TARGETING GENES FROM 3-HOP DATA")
    print("=" * 50)

    print(f"Processing {input_file}...")

    # Check if file exists
    if not os.path.exists(input_file):
        print(f" File not found: {input_file}")
        return

    try:
        # Read the input file
        df = pd.read_csv(input_file)
        print(f" File loaded successfully")
        print(f" Total rows: {len(df):,}")

        # Check column names
        print(f" Columns: {list(df.columns)}")

        # Separate self-targeting vs non-self-targeting
        self_targeting = df[df['source'] == df['target']].copy()
        non_self_targeting = df[df['source'] != df['target']].copy()

        print(f"\n Results:")
        print(f"   Self-targeting rows: {len(self_targeting):,}")
        print(f"   Non-self-targeting rows: {len(non_self_targeting):,}")

        # Calculate percentage
        self_percentage = (len(self_targeting) / len(df)) * 100 if len(df) > 0 else 0
        print(f"   Self-targeting percentage: {self_percentage:.2f}%")

        # Generate output file paths
        input_dir = os.path.dirname(input_file)
        base_name = os.path.splitext(os.path.basename(input_file))[0]

        main_output = os.path.join(input_dir, f"{base_name}_main.csv")
        self_output = os.path.join(input_dir, f"{base_name}_self_targeting.csv")

        # Save the files
        print(f"\n Saving files...")
        non_self_targeting.to_csv(main_output, index=False)
        self_targeting.to_csv(self_output, index=False)

        print(f" Main file saved: {main_output}")
        print(f" Self-targeting file saved: {self_output}")

        # Show some examples if self-targeting genes exist
        if len(self_targeting) > 0:
            print(f"\n Sample self-targeting genes:")
            sample_genes = self_targeting['source'].value_counts().head(5)
            for gene, count in sample_genes.items():
                print(f"   {gene}: {count} pathway(s)")
        else:
            print(f"\n  No self-targeting genes found in this dataset")

        print(f"\n Separation completed successfully!")

    except Exception as e:
        print(f" Error processing file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    separate_self_targeting_3hop()