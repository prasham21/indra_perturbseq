import pandas as pd
import os

tv = pd.read_csv("/Users/prashammarfatia/Downloads/target_validation_expanded.csv")
de_results_dir = "/Users/prashammarfatia/Downloads/de_results_per_gene"

# Filter to the 358 valid perturbations
valid_genes = tv[tv["Karen_Flag"] == "Use_for_analysis"]["Gene"].tolist()

descendant_stats = []
print(f"Analyzing {len(valid_genes)} perturbations using CSVs...\n")

for i, gene in enumerate(valid_genes):
    csv_path = os.path.join(de_results_dir, f"{gene}_vs_control.csv")
    try:
        df = pd.read_csv(csv_path)
        sig_descendants = df[df["pvals"] < 0.05]
        descendant_stats.append({
            "Gene": gene,
            "Num_Significant_DEGs": len(sig_descendants)
        })
        print(f"{i+1:3d}. {gene:<10} → {len(sig_descendants):3d} significant descendants")
    except Exception as e:
        print(f"{i+1:3d}. {gene:<10} → ❌ ERROR: {e}")
        descendant_stats.append({
            "Gene": gene,
            "Num_Significant_DEGs": -1
        })

# Save to CSV
desc_df = pd.DataFrame(descendant_stats)
desc_df = desc_df.sort_values("Num_Significant_DEGs", ascending=False)
desc_df.to_csv("descendant_counts.csv", index=False)

print("\nTop 10 perturbations by number of significant descendants:")
print(desc_df.head(10).to_string(index=False))
