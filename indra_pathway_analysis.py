import pandas as pd
import scanpy as sc
import numpy as np
import time
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

from indra.sources import indra_db_rest
from indra.statements import IncreaseAmount, DecreaseAmount, Activation, Inhibition
from tenacity import retry, stop_after_attempt, wait_exponential


def is_unusual_gene_name(name: str) -> bool:
    """Check for problematic gene names"""
    return name != name.upper() or '-TSS' in name or name.startswith('ENSG')


def clean_gene_name(name: str) -> str:
    """Clean gene names to standard format"""
    if pd.isna(name):
        return None
    name = str(name).upper().strip()
    if '-TSS' in name:
        name = name.split('-TSS')[0]
    return name


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
def safe_get_statements(**kwargs):
    return indra_db_rest.get_statements(**kwargs)


class DirectPathwayAnalyzer:
    def __init__(self, target_validation_path: str, adata_backup_path: str):
        self.target_validation_df = pd.read_csv(target_validation_path)
        self.adata_de = sc.read_h5ad(adata_backup_path)

        # Get successful targets
        self.successful_targets = self.target_validation_df[
            self.target_validation_df['Karen_Flag'] == 'Use_for_analysis'
            ]['Gene'].tolist()

        print(f"Loaded {len(self.successful_targets)} successful knockdown targets")
        print(f"AnnData shape: {self.adata_de.shape}")

    def get_affected_genes(self, target_gene: str,
                           significance_threshold: float = 0.05,
                           max_genes: int = None) -> List[Dict]:
        """Get significantly affected genes with flexible thresholds"""
        try:
            target_gene = clean_gene_name(target_gene) or target_gene.upper()
            de_results = sc.get.rank_genes_groups_df(self.adata_de, group=target_gene)

            # Remove self-targeting
            downstream_genes = de_results[
                de_results['names'].str.upper() != target_gene
                ]

            # Try multiple significance thresholds
            thresholds = [0.05, 0.1, 0.2]
            significant_genes = pd.DataFrame()

            for thresh in thresholds:
                significant_genes = downstream_genes[
                    downstream_genes['pvals_adj'] < thresh
                    ]
                if len(significant_genes) >= 10:  # Need reasonable number of genes
                    print(f"  Using p < {thresh} threshold ({len(significant_genes)} genes)")
                    break

            if significant_genes.empty:
                print(f"  No significant genes found for {target_gene}")
                return []

            # Sort by significance and limit if specified
            significant_genes = significant_genes.sort_values('pvals_adj')
            if max_genes is not None:
                significant_genes = significant_genes.head(max_genes)

            affected_genes = []
            for _, row in significant_genes.iterrows():
                clean_name = clean_gene_name(row['names'])
                if clean_name and not is_unusual_gene_name(clean_name):
                    affected_genes.append({
                        'gene': clean_name,
                        'logfc': row['logfoldchanges'],
                        'pvalue': row['pvals_adj'],
                        'effect_direction': 'increase' if row['logfoldchanges'] > 0 else 'decrease'
                    })

            print(f"  Found {len(affected_genes)} clean affected genes")
            return affected_genes

        except Exception as e:
            print(f"Error getting affected genes for {target_gene}: {e}")
            return []

    def query_direct_pathways_optimized(self, source_gene: str, target_gene: str) -> List[Dict]:
        """Optimized direct pathway queries - single query without statement type filtering"""
        source_gene = clean_gene_name(source_gene) or source_gene.upper()
        target_gene = clean_gene_name(target_gene) or target_gene.upper()

        if is_unusual_gene_name(target_gene):
            print(f"⚠️ Unusual gene name: {target_gene}")
            return []

        try:
            # Single query - let INDRA return all statement types
            ip = safe_get_statements(
                subject=source_gene,
                object=target_gene,
                ev_limit=10  # Reasonable evidence limit
            )
            time.sleep(1)  # Rate limiting

            pathways = []
            for stmt in ip.statements:
                # Filter for relevant statement types and convert to final edge types
                final_edge_type = None

                if isinstance(stmt, IncreaseAmount):
                    final_edge_type = 'increaseamount'
                elif isinstance(stmt, DecreaseAmount):
                    final_edge_type = 'decreaseamount'
                elif isinstance(stmt, Activation):
                    final_edge_type = 'increaseamount'  # Activation -> increase
                elif isinstance(stmt, Inhibition):
                    final_edge_type = 'decreaseamount'  # Inhibition -> decrease
                else:
                    continue  # Skip other statement types

                pathways.append({
                    'source': source_gene,
                    'target': target_gene,
                    'relationship_type': stmt.__class__.__name__,
                    'evidence_count': len(stmt.evidence),
                    'pathway_length': 1,
                    'pathway_string': f"{source_gene} -> {target_gene}",
                    'final_edge_type': final_edge_type,
                    'confidence': 'direct_literature'
                })

            return pathways

        except Exception as e:
            print(f"Error querying INDRA for {source_gene} -> {target_gene}: {e}")
            return []

    def analyze_single_perturbation(self, target_gene: str, max_genes: int = 50) -> pd.DataFrame:
        """Analyze direct pathways only for a single perturbation"""
        target_gene = clean_gene_name(target_gene) or target_gene.upper()
        print(f"\n🔍 Analyzing DIRECT pathways for {target_gene}")

        affected_genes = self.get_affected_genes(target_gene, max_genes=max_genes)
        if not affected_genes:
            return pd.DataFrame()

        all_pathways = []
        successful_queries = 0

        print(f"Querying direct pathways for {len(affected_genes)} affected genes...")

        for i, affected_gene_info in enumerate(affected_genes):
            affected_gene = affected_gene_info['gene']
            print(f"  ({i + 1}/{len(affected_genes)}) {target_gene} → {affected_gene}")

            # ONLY direct pathways - no multi-hop
            direct_pathways = self.query_direct_pathways_optimized(target_gene, affected_gene)

            # Add experimental information to each pathway
            for pathway in direct_pathways:
                pathway.update({
                    'perturbation_gene': target_gene,
                    'affected_gene': affected_gene,
                    'observed_effect': affected_gene_info['effect_direction'],
                    'observed_logfc': affected_gene_info['logfc'],
                    'observed_pvalue': affected_gene_info['pvalue'],

                    # Add consistency check between literature and experiment
                    'literature_experiment_consistent': self._check_consistency(
                        pathway['final_edge_type'],
                        affected_gene_info['effect_direction']
                    )
                })

            all_pathways.extend(direct_pathways)

            if direct_pathways:
                successful_queries += 1
                print(f"    ✓ Found {len(direct_pathways)} pathways")
            else:
                print(f"    ✗ No direct pathways found")

        print(
            f"  ✅ Completed {target_gene}: {successful_queries}/{len(affected_genes)} genes with pathways, {len(all_pathways)} total pathways")
        return pd.DataFrame(all_pathways)

    def _check_consistency(self, literature_effect: str, observed_effect: str) -> bool:
        """Check if literature prediction matches experimental observation"""
        # For knockdown experiments:
        # If literature says TP53 increases target, and we see target decrease when TP53 is knocked down -> CONSISTENT
        # If literature says TP53 decreases target, and we see target increase when TP53 is knocked down -> CONSISTENT

        if literature_effect == 'increaseamount' and observed_effect == 'decrease':
            return True  # TP53 normally increases target, knockdown causes decrease
        elif literature_effect == 'decreaseamount' and observed_effect == 'increase':
            return True  # TP53 normally decreases target, knockdown causes increase
        else:
            return False  # Inconsistent with knockdown expectation

    def analyze_all_perturbations(self, max_perturbations: Optional[int] = None,
                                  max_genes_per_perturbation: int = 50) -> pd.DataFrame:
        """Analyze all perturbations using direct pathways only"""
        from tqdm import tqdm
        results = []
        targets = self.successful_targets[:max_perturbations] if max_perturbations else self.successful_targets

        print(f"\n{'=' * 60}")
        print(f"STARTING DIRECT PATHWAY ANALYSIS")
        print(f"{'=' * 60}")
        print(f"Perturbations to analyze: {len(targets)}")
        print(f"Max genes per perturbation: {max_genes_per_perturbation if max_genes_per_perturbation else 'ALL'}")

        # Handle None case for estimation
        if max_genes_per_perturbation is not None:
            estimated_queries = len(targets) * max_genes_per_perturbation
            estimated_time = estimated_queries * 2 / 60
            print(f"Expected total queries: ~{estimated_queries}")
            print(f"Estimated time: {estimated_time:.1f} minutes")
        else:
            print(f"Expected total queries: ~{len(targets)} perturbations × ALL affected genes")
            print(f"Estimated time: 10-20 minutes (will analyze all affected genes)")

        for i, gene in enumerate(tqdm(targets, desc="Analyzing perturbations")):
            try:
                print(f"\n--- Perturbation {i + 1}/{len(targets)}: {gene} ---")
                df = self.analyze_single_perturbation(gene, max_genes=max_genes_per_perturbation)

                if not df.empty:
                    results.append(df)
                    print(f"  ✅ {gene}: Added {len(df)} pathways")
                else:
                    print(f"  ⚠️ {gene}: No pathways found")

            except Exception as e:
                print(f"❌ Error on {gene}: {e}")
                continue

        if results:
            final_df = pd.concat(results, ignore_index=True)
            print(f"\n🎉 Analysis complete! Total pathways discovered: {len(final_df)}")
            return final_df
        else:
            print("⚠️ No results generated")
            return pd.DataFrame()

    def generate_analysis_summary(self, results_df: pd.DataFrame) -> None:
        """Generate comprehensive analysis summary with consistency analysis"""
        if results_df.empty:
            print("No results to summarize")
            return

        print(f"\n{'=' * 60}")
        print("DIRECT PATHWAY ANALYSIS SUMMARY")
        print(f"{'=' * 60}")

        print(f"Total direct pathways discovered: {len(results_df)}")
        print(f"Unique perturbations analyzed: {results_df['perturbation_gene'].nunique()}")
        print(f"Unique affected genes: {results_df['affected_gene'].nunique()}")

        # Success rate by perturbation
        success_by_perturbation = results_df.groupby('perturbation_gene').size().sort_values(ascending=False)
        print(f"\nTop perturbations by pathway count:")
        for gene, count in success_by_perturbation.head(10).items():
            print(f"  {gene}: {count} pathways")

        if 'final_edge_type' in results_df.columns:
            print(f"\nLiterature relationship types:")
            for edge_type, count in results_df['final_edge_type'].value_counts().items():
                print(f"  {edge_type}: {count}")

        if 'literature_experiment_consistent' in results_df.columns:
            consistent_count = results_df['literature_experiment_consistent'].sum()
            total_count = len(results_df)
            consistency_rate = consistent_count / total_count * 100
            print(f"\nLiterature-Experiment Consistency:")
            print(f"  Consistent pathways: {consistent_count}/{total_count} ({consistency_rate:.1f}%)")
            print(
                f"  Inconsistent pathways: {total_count - consistent_count}/{total_count} ({100 - consistency_rate:.1f}%)")

        if 'evidence_count' in results_df.columns:
            print(f"\nEvidence strength distribution:")
            evidence_stats = results_df['evidence_count'].describe()
            print(f"  Mean evidence per pathway: {evidence_stats['mean']:.1f}")
            print(f"  Median evidence per pathway: {evidence_stats['50%']:.1f}")
            print(f"  Max evidence per pathway: {int(evidence_stats['max'])}")


def main():
    """Main function for TP53 complete direct pathway analysis"""
    target_validation_path = "/Users/prashammarfatia/Downloads/karen_target_validation.csv"
    adata_backup_path = "/Users/prashammarfatia/Downloads/adata_de_backup.h5ad"
    output_path = "/Users/prashammarfatia/Downloads/tp53_all_direct_pathways_results.csv"

    try:
        analyzer = DirectPathwayAnalyzer(target_validation_path, adata_backup_path)

        # Focus specifically on TP53 - the strongest perturbation
        analyzer.successful_targets = ['TP53']
        print(f"\n🎯 COMPLETE TP53 DIRECT PATHWAY ANALYSIS")
        print(f"Analyzing ALL affected genes for TP53...")

        results = analyzer.analyze_all_perturbations(
            max_perturbations=1,  # Just TP53
            max_genes_per_perturbation=None  # ALL 443 affected genes (no limit)
        )

        if not results.empty:
            results.to_csv(output_path, index=False)
            print(f"\n✅ TP53 Complete Results saved to: {output_path}")
            analyzer.generate_analysis_summary(results)

            # TP53-specific analysis
            print(f"\n{'=' * 60}")
            print(f"TP53 COMPREHENSIVE PATHWAY COVERAGE ANALYSIS")
            print(f"{'=' * 60}")

            total_affected_genes = len(analyzer.get_affected_genes('TP53', max_genes=None))
            pathways_found = len(results)
            genes_with_pathways = results['affected_gene'].nunique()
            coverage_rate = (genes_with_pathways / total_affected_genes) * 100 if total_affected_genes > 0 else 0

            print(f"📊 TP53 Coverage Statistics:")
            print(f"  Total TP53 affected genes: {total_affected_genes}")
            print(f"  Genes with direct pathways: {genes_with_pathways}")
            print(f"  Coverage rate: {coverage_rate:.1f}%")
            print(f"  Total direct pathways found: {pathways_found}")
            print(
                f"  Avg pathways per covered gene: {pathways_found / genes_with_pathways:.1f}" if genes_with_pathways > 0 else "  Avg pathways per covered gene: 0")

            # Show top TP53 targets by pathway count
            if not results.empty:
                pathway_counts = results['affected_gene'].value_counts()
                print(f"\n🎯 Top TP53 targets by pathway count:")
                for gene, count in pathway_counts.head(10).items():
                    print(f"  TP53 → {gene}: {count} pathways")

                # Show sample high-confidence pathways
                high_evidence = results[results['evidence_count'] >= 3].sort_values('evidence_count', ascending=False)
                if not high_evidence.empty:
                    print(f"\n⭐ High-evidence TP53 pathways (≥3 pieces of evidence):")
                    for _, row in high_evidence.head(10).iterrows():
                        print(f"  {row['pathway_string']}: {row['evidence_count']} evidence, {row['final_edge_type']}")

                # Consistency analysis for TP53
                consistent_pathways = results[results['literature_experiment_consistent'] == True]
                if not consistent_pathways.empty:
                    consistency_rate = len(consistent_pathways) / len(results) * 100
                    print(f"\n✅ TP53 Consistency Analysis:")
                    print(f"  Consistent pathways: {len(consistent_pathways)}/{len(results)} ({consistency_rate:.1f}%)")
                    print(f"  Sample consistent pathways:")
                    for _, row in consistent_pathways.head(5).iterrows():
                        direction = "↑" if row['final_edge_type'] == 'increaseamount' else "↓"
                        obs_direction = "↓" if row['observed_effect'] == 'decrease' else "↑"
                        print(
                            f"    TP53 {direction} {row['affected_gene']} | Observed: {obs_direction} (p={row['observed_pvalue']:.2e})")

            # Show sample results
            print(f"\nSample TP53 pathways found:")
            sample_cols = ['perturbation_gene', 'affected_gene', 'pathway_string',
                           'final_edge_type', 'evidence_count', 'literature_experiment_consistent']
            if all(col in results.columns for col in sample_cols):
                print(results[sample_cols].head(15).to_string(index=False))
        else:
            print("⚠️ No TP53 pathways discovered.")
            # Debug information
            affected_genes = analyzer.get_affected_genes('TP53', max_genes=None)
            print(f"TP53 has {len(affected_genes)} affected genes but no literature pathways found")
            if affected_genes:
                print("Sample affected genes:")
                for gene_info in affected_genes[:10]:
                    print(f"  {gene_info['gene']}: logFC={gene_info['logfc']:.2f}, p={gene_info['pvalue']:.2e}")

    except Exception as e:
        print(f"TP53 analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()