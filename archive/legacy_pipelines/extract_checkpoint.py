import pickle
import pandas as pd
import os
from datetime import datetime

# Load checkpoint file
checkpoint_file = "3hop_optimized_checkpoint.pkl"

if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'rb') as f:
        data = pickle.load(f)

    results = data['results']
    processed_genes = data['processed_genes']
    timestamp = data['timestamp']

    # Print summary
    print("=" * 60)
    print("3-HOP ANALYSIS CHECKPOINT SUMMARY")
    print("=" * 60)
    print(f"Checkpoint saved: {datetime.fromtimestamp(timestamp)}")
    print(f"Genes completed: {len(processed_genes)} out of 358")
    print(f"Progress: {len(processed_genes) / 358 * 100:.1f}%")
    print(f"Total 3-hop pathways found: {len(results)}")
    print(f"Average pathways per gene: {len(results) / len(processed_genes):.1f}")

    # Show completed genes
    print(f"\nCompleted genes:")
    gene_list = sorted(list(processed_genes))
    for i in range(0, len(gene_list), 10):
        print(", ".join(gene_list[i:i + 10]))

    # Convert to DataFrame and save
    if results:
        df = pd.DataFrame(results)
        output_file = "partial_3hop_results.csv"
        df.to_csv(output_file, index=False)

        print(f"\n" + "=" * 60)
        print("RESULTS EXPORTED")
        print("=" * 60)
        print(f"CSV saved to: {output_file}")
        print(f"CSV shape: {df.shape} (rows × columns)")

        # Show column info
        print(f"\nColumns: {list(df.columns)}")

        # Show sample results
        print(f"\nSample 3-hop pathways:")
        print("-" * 80)
        for i, row in df.head(5).iterrows():
            print(f"{row['source']} → {row['intermediate_1']} → {row['intermediate_2']} → {row['target']}")
            print(f"  Types: {row['stmt_type_1']} → {row['stmt_type_2']} → {row['stmt_type_3']}")
            print(f"  Beliefs: {row['belief_1']:.2f} → {row['belief_2']:.2f} → {row['belief_3']:.2f}")
            print()

        # Show some statistics
        print("PATHWAY STATISTICS:")
        print(f"Unique source genes: {df['source'].nunique()}")
        print(f"Unique target genes: {df['target'].nunique()}")
        print(f"Unique intermediate_1 genes: {df['intermediate_1'].nunique()}")
        print(f"Unique intermediate_2 genes: {df['intermediate_2'].nunique()}")

        # Belief score distribution
        avg_belief_1 = df['belief_1'].mean()
        avg_belief_2 = df['belief_2'].mean()
        avg_belief_3 = df['belief_3'].mean()
        print(f"Average belief scores: {avg_belief_1:.3f} → {avg_belief_2:.3f} → {avg_belief_3:.3f}")

    else:
        print("No results found in checkpoint")

else:
    print(f"Checkpoint file '{checkpoint_file}' not found")
    print("Make sure you're in the correct directory")