"""2-hop pathway coverage summary statistics and visualizations."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

plt.style.use('default')
sns.set_palette("husl")


def normalize_columns(df, col_mapping):
    """Normalize column names according to a mapping of standard_name -> variants."""
    for standard_name, variants in col_mapping.items():
        for variant in variants:
            if variant in df.columns and standard_name not in df.columns:
                df[standard_name] = df[variant]
                logger.info("Normalized %s -> %s", variant, standard_name)
    return df


def load_deg_files(perturb_df, deg_folder, pathway_col_mapping):
    """Load all DEG files and build descendant records."""
    deg_col_mapping = {
        'names': ['names', 'gene_names', 'gene', 'target'],
        'pvals': ['pvals', 'pval', 'p_val', 'p_value'],
        'logfoldchanges': ['logfoldchanges', 'logfoldchange', 'log_fold_change', 'logFC']
    }
    use_fdr_col = None

    sample_gene = None
    sample_path = None
    for _, row in perturb_df.iterrows():
        test_gene = row['Gene']
        test_path = os.path.join(deg_folder, f"{test_gene}_vs_control.csv")
        if os.path.exists(test_path):
            sample_gene = test_gene
            sample_path = test_path
            break

    if sample_path is None:
        logger.error("No DEG files found")
        sys.exit(1)

    logger.info("Using %s as sample file for column detection", sample_gene)

    try:
        sample_df = pd.read_csv(sample_path)
        logger.info("Sample DEG file columns: %s", list(sample_df.columns))
        fdr_columns = ['pvals_adj', 'qval', 'padj', 'fdr', 'p_adj']
        for fdr_col in fdr_columns:
            if fdr_col in sample_df.columns:
                use_fdr_col = fdr_col
                logger.info("Found FDR column: %s", fdr_col)
                break
        if use_fdr_col:
            deg_col_mapping['fdr'] = [use_fdr_col]
    except Exception as e:
        logger.warning("Could not read sample DEG file: %s", e)

    all_descendants_cache = {}
    all_descendant_records = []

    for _, row in perturb_df.iterrows():
        gene = row['Gene']
        deg_path = os.path.join(deg_folder, f"{gene}_vs_control.csv")
        try:
            deg_df = pd.read_csv(deg_path)
            deg_df = normalize_columns(deg_df, deg_col_mapping)
            if use_fdr_col and use_fdr_col in deg_df.columns:
                deg_df['fdr'] = deg_df[use_fdr_col]

            significant_df = deg_df[deg_df['pvals'] < 0.05].copy()
            significant_df = significant_df.dropna(subset=['names', 'pvals', 'logfoldchanges'])
            all_descendants_cache[gene] = significant_df

            for _, desc_row in significant_df.iterrows():
                pval_to_use = desc_row.get('fdr', desc_row['pvals'])
                if pd.isna(pval_to_use):
                    pval_to_use = desc_row['pvals']
                all_descendant_records.append({
                    'source': gene,
                    'target': desc_row['names'],
                    'logfoldchange': desc_row['logfoldchanges'],
                    'pval': pval_to_use,
                    'abs_logfc': abs(desc_row['logfoldchanges']),
                    'one_minus_pval': 1 - pval_to_use,
                    'has_pathway': False
                })
            logger.info("%s: %d significant descendants", gene, len(significant_df))
        except FileNotFoundError:
            logger.info("No DEG file found for %s", gene)
            all_descendants_cache[gene] = pd.DataFrame()
        except Exception as e:
            logger.warning("Error loading %s: %s", gene, e)
            all_descendants_cache[gene] = pd.DataFrame()

    return all_descendants_cache, all_descendant_records, use_fdr_col


def main():
    parser = argparse.ArgumentParser(description="2-hop summary statistics")
    parser.add_argument("--pathway-csv", required=True,
                        help="Path to indra_2hop_all_perturbations.csv")
    parser.add_argument("--validation-csv", required=True,
                        help="Path to target_validation_expanded.csv")
    parser.add_argument("--deg-folder", required=True,
                        help="Path to de_results_per_gene folder")
    parser.add_argument("--output-dir", default=".", help="Output directory for plots/CSVs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Loading and validating data")

    with open(args.pathway_csv, 'r') as f:
        first_lines = [f.readline().strip() for _ in range(3)]
        for i, line in enumerate(first_lines):
            logger.info("Line %d: %s...", i, line[:100])

    try:
        df = pd.read_csv(args.pathway_csv, skiprows=1)
        if 'source' not in df.columns:
            df = pd.read_csv(args.pathway_csv)
    except Exception:
        df = pd.read_csv(args.pathway_csv)

    logger.info("Loaded %s pathway results", f"{len(df):,}")

    pathway_col_mapping = {
        'logfoldchange': ['logfoldchange', 'logfoldchanges', 'log_fold_change'],
        'pval': ['pval', 'pvals', 'p_val', 'p_value'],
        'belief_1': ['belief_1', 'belief1'],
        'belief_2': ['belief_2', 'belief2']
    }
    df = normalize_columns(df, pathway_col_mapping)

    perturb_df = pd.read_csv(args.validation_csv)
    perturb_df = perturb_df[perturb_df['analysis_flag'] == "Use_for_analysis"]
    if perturb_df.empty:
        raise ValueError("No perturbations with analysis_flag == 'Use_for_analysis'")

    all_descendants_cache, all_descendant_records, use_fdr_col = load_deg_files(
        perturb_df, args.deg_folder, pathway_col_mapping
    )

    all_descendants_df = pd.DataFrame(all_descendant_records)
    if len(all_descendants_df) == 0:
        logger.error("No descendant data loaded")
        sys.exit(1)

    logger.info("Total descendant records: %s", f"{len(all_descendants_df):,}")

    assert {'source', 'target'}.issubset(df.columns), "Pathway CSV missing source/target columns"
    assert {'belief_1', 'belief_2'}.issubset(df.columns), "Pathway CSV missing belief columns"

    pathway_pairs_df = df[['source', 'target']].dropna().drop_duplicates()
    pathway_pairs_index = pd.MultiIndex.from_frame(pathway_pairs_df)
    all_pairs_index = pd.MultiIndex.from_frame(all_descendants_df[['source', 'target']])
    all_descendants_df['has_pathway'] = all_pairs_index.isin(pathway_pairs_index)

    unique_targets_with_pathways = df['target'].nunique()
    total_unique_descendants = all_descendants_df['target'].nunique()
    overall_coverage = (unique_targets_with_pathways / total_unique_descendants) * 100

    total_explained_pairs = all_descendants_df['has_pathway'].sum()
    total_descendant_pairs = len(all_descendants_df)
    pair_coverage = (total_explained_pairs / total_descendant_pairs) * 100

    logger.info("Unique targets with pathways: %s", f"{unique_targets_with_pathways:,}")
    logger.info("Total unique descendants: %s", f"{total_unique_descendants:,}")
    logger.info("Overall coverage: %.2f%%", overall_coverage)
    logger.info("Pair-wise coverage: %.2f%%", pair_coverage)

    perturbation_coverage = []
    perturbation_data = []

    for gene in all_descendants_cache.keys():
        gene_descendants = all_descendants_df[all_descendants_df['source'] == gene]
        if len(gene_descendants) > 0:
            explained_count = gene_descendants['has_pathway'].sum()
            total_count = len(gene_descendants)
            coverage_pct = (explained_count / total_count) * 100
            perturbation_coverage.append(coverage_pct)
            perturbation_data.append({
                'gene': gene, 'total_descendants': total_count,
                'explained_descendants': explained_count, 'percent_explained': coverage_pct
            })

    perturb_analysis_df = pd.DataFrame(perturbation_data)

    tp53_coverage = 0
    tp53_row = perturb_analysis_df[perturb_analysis_df['gene'] == 'TP53']
    if len(tp53_row) > 0:
        tp53_coverage = tp53_row['percent_explained'].iloc[0]

    logger.info("TP53 coverage: %.1f%% (outlier)", tp53_coverage)
    logger.info("Mean coverage: %.1f%%", np.mean(perturbation_coverage))

    os.makedirs(args.output_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.hist(perturbation_coverage, bins=100, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
    plt.axvline(tp53_coverage, color='red', linestyle='--', linewidth=2,
                label=f'TP53: {tp53_coverage:.1f}% (outlier)')
    plt.axvline(np.mean(perturbation_coverage), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(perturbation_coverage):.1f}%')
    plt.xlabel('Percent of Descendants Explained by 2-Hop Pathways (%)', fontsize=12)
    plt.ylabel('Number of Perturbations', fontsize=12)
    plt.title('Distribution of Pathway Coverage Across Perturbations\n(100 bins, corrected denominator)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'corrected_per_perturbation_coverage_100bins.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    coverage_no_tp53 = perturb_analysis_df.loc[
        perturb_analysis_df['gene'] != 'TP53', 'percent_explained'
    ].tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.hist(perturbation_coverage, bins=100, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
    ax1.axvline(tp53_coverage, color='red', linestyle='--', linewidth=2, label=f'TP53: {tp53_coverage:.1f}%')
    ax1.axvline(np.mean(perturbation_coverage), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(perturbation_coverage):.1f}%')
    ax1.set_xlabel('Percent of Descendants Explained (%)')
    ax1.set_ylabel('Number of Perturbations')
    ax1.set_title('Coverage Distribution - TP53 INCLUDED\n(100 bins)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(coverage_no_tp53, bins=100, alpha=0.7, color='lightcoral', edgecolor='black', linewidth=0.5)
    ax2.axvline(np.mean(coverage_no_tp53), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(coverage_no_tp53):.1f}%')
    ax2.axvline(np.median(coverage_no_tp53), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(coverage_no_tp53):.1f}%')
    ax2.set_xlabel('Percent of Descendants Explained (%)')
    ax2.set_ylabel('Number of Perturbations')
    ax2.set_title('Coverage Distribution - TP53 EXCLUDED\n(Statistical Summary)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'tp53_included_vs_excluded_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    logger.info("TP53 impact: Including - Mean: %.1f%%, Excluding - Mean: %.1f%%",
                np.mean(perturbation_coverage), np.mean(coverage_no_tp53))

    clean_descendants = (all_descendants_df
                         .query("source != 'TP53'")
                         .dropna(subset=['abs_logfc', 'one_minus_pval'])
                         .copy())
    logger.info("Clean descendants for analysis: %s records", f"{len(clean_descendants):,}")

    n_bins = 10
    fold_change_bins = pd.qcut(clean_descendants['abs_logfc'], q=n_bins, duplicates='drop')

    bin_coverage_data = []
    for i, (bin_name, group) in enumerate(clean_descendants.groupby(fold_change_bins)):
        total_in_bin = len(group)
        explained_in_bin = group['has_pathway'].sum()
        coverage_pct = (explained_in_bin / total_in_bin) * 100 if total_in_bin > 0 else 0
        bin_coverage_data.append({
            'bin_label': f"{bin_name.left:.2f}-{bin_name.right:.2f}",
            'bin_center': (bin_name.left + bin_name.right) / 2,
            'total_descendants': total_in_bin,
            'explained_descendants': explained_in_bin,
            'coverage_percent': coverage_pct
        })

    sig_bins = pd.qcut(clean_descendants['one_minus_pval'], q=n_bins, duplicates='drop')
    sig_coverage_data = []
    for i, (bin_name, group) in enumerate(clean_descendants.groupby(sig_bins)):
        total_in_bin = len(group)
        explained_in_bin = group['has_pathway'].sum()
        coverage_pct = (explained_in_bin / total_in_bin) * 100 if total_in_bin > 0 else 0
        sig_coverage_data.append({
            'bin_label': f"{bin_name.left:.3f}-{bin_name.right:.3f}",
            'bin_center': (bin_name.left + bin_name.right) / 2,
            'total_descendants': total_in_bin,
            'explained_descendants': explained_in_bin,
            'coverage_percent': coverage_pct
        })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    fc_data = pd.DataFrame(bin_coverage_data)
    bars1 = ax1.bar(range(len(fc_data)), fc_data['coverage_percent'], alpha=0.7, color='lightblue')
    ax1.set_xlabel('Absolute Log Fold Change Bins (Quantiles)')
    ax1.set_ylabel('Percent Explained (%)')
    ax1.set_title('Coverage vs Experimental Effect Size\n(Binned by |Log Fold Change|)')
    ax1.set_xticks(range(len(fc_data)))
    ax1.set_xticklabels([f"Q{i + 1}" for i in range(len(fc_data))], rotation=45)
    ax1.grid(True, alpha=0.3)
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    sig_data = pd.DataFrame(sig_coverage_data)
    bars2 = ax2.bar(range(len(sig_data)), sig_data['coverage_percent'], alpha=0.7, color='lightgreen')
    ax2.set_xlabel('Statistical Significance Bins (1 - P Value Quantiles)')
    ax2.set_ylabel('Percent Explained (%)')
    ax2.set_title('Coverage vs Statistical Significance\n(Binned by 1 - P Value)')
    ax2.set_xticks(range(len(sig_data)))
    ax2.set_xticklabels([f"Q{i + 1}" for i in range(len(sig_data))], rotation=45)
    ax2.grid(True, alpha=0.3)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'coverage_vs_effect_size_corrected.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.hist(clean_descendants['logfoldchange'], bins=100, alpha=0.7, color='lightsteelblue',
             edgecolor='black', linewidth=0.5)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    plt.axvline(clean_descendants['logfoldchange'].mean(), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {clean_descendants["logfoldchange"].mean():.3f}')
    plt.xlabel('Log Fold Change')
    plt.ylabel('Number of Targets')
    plt.title(f'Fold Change Distribution - All Descendants (TP53 excluded)\nn={len(clean_descendants):,} targets')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'fold_change_all_descendants.png'), dpi=300, bbox_inches='tight')
    plt.show()

    tiniest_fdr_threshold = clean_descendants['pval'].quantile(0.05)
    tiniest_fdrs = clean_descendants[clean_descendants['pval'] <= tiniest_fdr_threshold]

    logger.info("Tiniest FDRs: %s targets (p <= %.2e), %.1f%% with pathways",
                f"{len(tiniest_fdrs):,}", tiniest_fdr_threshold,
                tiniest_fdrs['has_pathway'].mean() * 100)

    plt.figure(figsize=(12, 8))
    plt.hist(tiniest_fdrs['logfoldchange'], bins=100, alpha=0.7, color='salmon', edgecolor='black', linewidth=0.5)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    plt.axvline(tiniest_fdrs['logfoldchange'].mean(), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {tiniest_fdrs["logfoldchange"].mean():.3f}')
    plt.xlabel('Log Fold Change', fontsize=12)
    plt.ylabel('Number of Targets', fontsize=12)
    plt.title(
        f'Fold Change Distribution for Tiniest FDRs\n(Bottom 5% p-values from ALL descendants, n={len(tiniest_fdrs):,})',
        fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'fold_change_tiniest_fdrs_corrected.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    explained_descendants = all_descendants_df[all_descendants_df['has_pathway']].copy()
    explained_no_tp53 = explained_descendants[explained_descendants['source'] != 'TP53'].copy()

    pathway_belief_data = df[['source', 'target', 'belief_1', 'belief_2']].drop_duplicates(
        subset=['source', 'target'])
    explained_with_beliefs = explained_no_tp53.merge(pathway_belief_data, on=['source', 'target'], how='left')
    explained_with_beliefs['mean_belief'] = (
        explained_with_beliefs['belief_1'] + explained_with_beliefs['belief_2']
    ) / 2
    explained_with_beliefs = explained_with_beliefs.dropna(subset=['mean_belief'])

    logger.info("Correlation analysis using %s explained targets (TP53 excluded)",
                f"{len(explained_with_beliefs):,}")

    correlation_coef, p_value = pearsonr(
        explained_with_beliefs['abs_logfc'], explained_with_beliefs['mean_belief']
    )
    logger.info("Pearson r: %.3f, p-value: %.2e", correlation_coef, p_value)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plot_sample = (explained_with_beliefs.sample(n=5000, random_state=42)
                   if len(explained_with_beliefs) > 5000 else explained_with_beliefs)
    plt.scatter(plot_sample['abs_logfc'], plot_sample['mean_belief'], alpha=0.6, s=20)
    z = np.polyfit(explained_with_beliefs['abs_logfc'], explained_with_beliefs['mean_belief'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(explained_with_beliefs['abs_logfc'].min(),
                          explained_with_beliefs['abs_logfc'].max(), 100)
    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
    plt.xlabel('Absolute Log Fold Change')
    plt.ylabel('Mean Belief Score')
    plt.title(f'Mean Belief vs Effect\n(r = {correlation_coef:.3f}, no TP53)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.scatter(plot_sample['abs_logfc'], plot_sample['belief_1'], alpha=0.6, s=15,
                label='First hop', color='blue')
    plt.scatter(plot_sample['abs_logfc'], plot_sample['belief_2'], alpha=0.6, s=15,
                label='Second hop', color='red')
    plt.xlabel('Absolute Log Fold Change')
    plt.ylabel('Individual Belief Scores')
    plt.title('Each Hop Separately')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.hist(explained_with_beliefs['belief_1'], bins=30, alpha=0.7, label='First hop', color='blue')
    plt.hist(explained_with_beliefs['belief_2'], bins=30, alpha=0.7, label='Second hop', color='red')
    plt.xlabel('Belief Score')
    plt.ylabel('Frequency')
    plt.title('Belief Score Distribution\nby Hop')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'correlation_analysis_corrected.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    sig_label = '1 - FDR' if use_fdr_col else '1 - p-value'
    x_sig = explained_with_beliefs['one_minus_pval']
    y_mb = explained_with_beliefs['mean_belief']

    if x_sig.nunique() > 1 and y_mb.nunique() > 1:
        corr_sig, p_sig = pearsonr(x_sig, y_mb)
    else:
        corr_sig, p_sig = np.nan, np.nan

    logger.info("Correlation vs (%s): r = %.3f, p = %.2e", sig_label, corr_sig, p_sig)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_sig, y_mb, alpha=0.6, s=20)
    if x_sig.nunique() > 1:
        z = np.polyfit(x_sig, y_mb, 1)
        pfit = np.poly1d(z)
        xs = np.linspace(x_sig.min(), x_sig.max(), 100)
        plt.plot(xs, pfit(xs), "r--", alpha=0.8, linewidth=2)
    plt.xlabel(sig_label)
    plt.ylabel('Mean Belief Score')
    plt.title(f'Mean Belief vs {sig_label}\n(r = {corr_sig:.3f}, no TP53)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'correlation_meanbelief_vs_one_minus_p.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    orig_pairs = len(df.drop_duplicates(['source', 'target']))
    high_conf_mask = (df['belief_1'] >= 0.7) & (df['belief_2'] >= 0.7)
    high_conf_pathway_pairs = set(zip(df.loc[high_conf_mask, 'source'], df.loc[high_conf_mask, 'target']))

    if high_conf_pathway_pairs:
        hc_pairs_index = pd.MultiIndex.from_tuples(list(high_conf_pathway_pairs))
        all_descendants_df['has_high_conf_pathway'] = all_pairs_index.isin(hc_pairs_index)
    else:
        logger.warning("No high confidence pathways found (belief >= 0.7 for both hops)")
        all_descendants_df['has_high_conf_pathway'] = False

    hc_perturbation_coverage = []
    for gene in all_descendants_cache.keys():
        gene_descendants = all_descendants_df[all_descendants_df['source'] == gene]
        if len(gene_descendants) > 0:
            hc_explained = gene_descendants['has_high_conf_pathway'].sum()
            total_count = len(gene_descendants)
            hc_perturbation_coverage.append((hc_explained / total_count) * 100)

    tp53_hc_coverage = 0
    tp53_descendants = all_descendants_df[all_descendants_df['source'] == 'TP53']
    if len(tp53_descendants) > 0:
        tp53_hc_coverage = (tp53_descendants['has_high_conf_pathway'].sum() / len(tp53_descendants)) * 100

    logger.info("High confidence mean coverage: %.1f%%", np.mean(hc_perturbation_coverage))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    ax1.hist(perturbation_coverage, bins=100, alpha=0.7, color='skyblue')
    ax1.axvline(tp53_coverage, color='red', linestyle='--', alpha=0.8, label=f'TP53: {tp53_coverage:.1f}%')
    ax1.set_xlabel('Percent Explained (%)')
    ax1.set_ylabel('Number of Perturbations')
    ax1.set_title('Original Coverage\n(All pathways, TP53 included)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(hc_perturbation_coverage, bins=100, alpha=0.7, color='orange')
    ax2.axvline(tp53_hc_coverage, color='red', linestyle='--', alpha=0.8, label=f'TP53: {tp53_hc_coverage:.1f}%')
    ax2.set_xlabel('Percent Explained (%)')
    ax2.set_ylabel('Number of Perturbations')
    ax2.set_title('High Confidence Coverage\n(Belief >= 0.7, TP53 included)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    perturb_analysis_with_coverage = pd.DataFrame({
        'gene': [d['gene'] for d in perturbation_data],
        'coverage': perturbation_coverage,
        'hc_coverage': hc_perturbation_coverage
    })

    coverage_no_tp53_df = perturb_analysis_with_coverage.loc[
        perturb_analysis_with_coverage['gene'] != 'TP53', 'coverage'
    ].tolist()
    hc_coverage_no_tp53 = perturb_analysis_with_coverage.loc[
        perturb_analysis_with_coverage['gene'] != 'TP53', 'hc_coverage'
    ].tolist()

    ax3.hist(coverage_no_tp53_df, bins=100, alpha=0.7, color='lightblue')
    ax3.set_xlabel('Percent Explained (%)')
    ax3.set_ylabel('Number of Perturbations')
    ax3.set_title('Original Coverage\n(All pathways, TP53 excluded)')
    ax3.grid(True, alpha=0.3)

    ax4.hist(hc_coverage_no_tp53, bins=100, alpha=0.7, color='lightsalmon')
    ax4.set_xlabel('Percent Explained (%)')
    ax4.set_ylabel('Number of Perturbations')
    ax4.set_title('High Confidence Coverage\n(Belief >= 0.7, TP53 excluded)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'belief_filtering_comparison_corrected.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    logger.info("Original coverage (no TP53) - Mean: %.1f%%, Median: %.1f%%",
                np.mean(coverage_no_tp53_df), np.median(coverage_no_tp53_df))
    logger.info("High conf coverage (no TP53) - Mean: %.1f%%, Median: %.1f%%",
                np.mean(hc_coverage_no_tp53), np.median(hc_coverage_no_tp53))

    logger.info("Effect size correlation: r = %.3f (p = %.2e)", correlation_coef, p_value)

    perturb_analysis_df.to_csv(os.path.join(args.output_dir,
                               'final_corrected_per_perturbation_analysis.csv'), index=False)
    all_descendants_df.to_csv(os.path.join(args.output_dir,
                              'master_descendants_with_pathway_status.csv'), index=False)
    pd.DataFrame(bin_coverage_data).to_csv(os.path.join(args.output_dir,
                                           'coverage_by_fold_change_bins.csv'), index=False)
    pd.DataFrame(sig_coverage_data).to_csv(os.path.join(args.output_dir,
                                           'coverage_by_significance_bins.csv'), index=False)
    tiniest_fdrs.to_csv(os.path.join(args.output_dir, 'tiniest_fdr_targets.csv'), index=False)

    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
