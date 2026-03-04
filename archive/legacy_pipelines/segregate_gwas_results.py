"""Legacy script: segregate_gwas_results."""
from __future__ import annotations

import argparse
import pandas as pd

import logging


logger = logging.getLogger(__name__)
# Load the GWAS CSV
def get_genes_from_set(row, gene_set):
    genes_in_path = []
    if row['source'] in gene_set:
        genes_in_path.append(row['source'])
    if row['intermediate'] in gene_set:
        genes_in_path.append(row['intermediate'])
    if row['target'] in gene_set:
        genes_in_path.append(row['target'])
    return ', '.join(sorted(set(genes_in_path))) if genes_in_path else ''

# Create masks for each gene set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gwas-endothelial-paths-url-enriched-1", default="gwas_endothelial_paths_url_enriched (1).csv", help="Path: gwas_endothelial_paths_url_enriched (1).csv")
    ap.add_argument("--gwas-8gene-set", default="gwas_8gene_set_.csv", help="Path: gwas_8gene_set_.csv")
    ap.add_argument("--gwas-17gene-set", default="gwas_17gene_set_.csv", help="Path: gwas_17gene_set_.csv")
    ap.add_argument("--gwas-residual", default="gwas_residual_.csv", help="Path: gwas_residual_.csv")
    args = ap.parse_args()

    df = pd.read_csv("gwas_endothelial_paths_url_enriched (1).csv")

    logger.info("="*80)
    logger.info("SEGREGATING GWAS CSV INTO 3 PARTS")
    logger.info("="*80)
    logger.info("Original GWAS CSV shape: %s\n", df.shape)

    # Define gene sets
    genes_8 = {
        'PRDM16', 'PLPP3', 'NOS3', 'JCAD', 'FLT1', 'EDN1', 'PECAM1', 'ARHGEF26'
    }

    genes_17 = {
        'CALCRL', 'CCM2', 'CDKN1A', 'EXOC3L2', 'GDPD5', 'GGT5', 'IBTK', 'N4BP2L2',
        'PREX1', 'PRKAR1A', 'SCUBE1', 'SLK', 'SPRY4', 'SVIL', 'TFPI', 'TLNRD1', 'TSPAN14'
    }

    genes_420 = {
        'AAK1', 'ABCG5', 'ABCG8', 'ABHD2', 'AC003986.6', 'AC105384.1', 'ACTA2', 'ACTN4',
        'ACTRT2', 'ACVR2A', 'ACVRL1', 'ADAM19', 'ADAMTS3', 'ADAMTS7', 'AEBP1', 'AFAP1L2',
        'AFF4', 'AGAP5', 'AHI1', 'AIDA', 'AKAP12', 'AL592148.3', 'ANGPTL4', 'ANKRD13B',
        'ANTXR2', 'ANXA11', 'AP000318.2', 'AP002989.1', 'APOA1', 'APOA5', 'APOB', 'APOE',
        'APOM', 'ARAP3', 'ARHGAP21', 'ARHGAP26', 'ARHGAP42', 'ARHGEF12', 'ARHGEF26',
        'ARNT', 'ARNTL', 'ARVCF', 'ATF6B', 'ATP1B1', 'ATP2B1', 'ATXN7L2', 'AXL', 'B4GALT5',
        'BACH1', 'BAG6', 'BASP1', 'BCAR1', 'BCAS3', 'BCKDHA', 'BMP1', 'BMPR1B', 'C1QTNF1',
        'C4A', 'CABIN1', 'CALCRL', 'CAND1', 'CARF', 'CAV1', 'CBS', 'CBX5', 'CCDC3',
        'CCDC97', 'CCM2', 'CD36', 'CDC123', 'CDH13', 'CDK8', 'CDKN1A', 'CDKN2A', 'CDKN2B',
        'CDKN2BAS', 'CEL', 'CELSR2', 'CETP', 'CFDP1', 'CHRNB4', 'CLDN5', 'COBLL1',
        'COL4A1', 'COL4A2', 'COL6A3', 'COPRS', 'CORO6', 'CRYAB', 'CSF1', 'CTD-3253I12.1',
        'CTSH', 'CXCL12', 'CYP17A1', 'CYP46A1', 'DAAM2', 'DAB2IP', 'DCUN1D3', 'DENND4C',
        'DHX36', 'DHX38', 'DMAC2', 'DOCK5', 'DOCK9', 'DPYD', 'DST', 'EDN1', 'EDNRA',
        'EFCAB13', 'EGFLAM', 'EHBP1L1', 'EIF2B2', 'EPAS1', 'ESYT3', 'EXOC3L2', 'EZR',
        'F10', 'FAM114A1', 'FAM117B', 'FAM177B', 'FBN2', 'FCHO1', 'FER', 'FES', 'FGD5',
        'FGD6', 'FGF5', 'FHL3', 'FHL5', 'FIGN', 'FLOT1', 'FLT1', 'FN1', 'FNDC3B', 'FOCAD',
        'FOXC1', 'FURIN', 'GAS8', 'GATA6', 'GATAD2A', 'GDPD5', 'GEM', 'GFOD1', 'GGCX',
        'GGT5', 'GIGYF1', 'GIGYF2', 'GNA12', 'GNAS', 'GOSR2', 'GRK4', 'GUCY1A1', 'GUCY1A3',
        'HDAC9', 'HDGFL1', 'HEY2', 'HHAT', 'HHIPL1', 'HIPK2', 'HIVEP2', 'HLA-C',
        'HLA-DQB1', 'HMHB1', 'HNF1A', 'HOMER3', 'HSD17B1', 'HSD17B12', 'HTRA1', 'HTT',
        'IBTK', 'ICA1L', 'IGFBP7', 'IL6R', 'IL6ST', 'ILK', 'INPP5B', 'IRS1', 'ITGA1',
        'ITGB3', 'ITIH4', 'JCAD', 'JUN', 'KANK1', 'KCNE2', 'KCNK5', 'KCTD8', 'KIAA0040',
        'KLF2', 'KPTN', 'LAMA4', 'LAMB1', 'LAMB2', 'LDLR', 'LIMS2', 'LINC00189',
        'LINC00310', 'LINC01312', 'LIPA', 'LIPC', 'LMAN1', 'LMOD1', 'LOX', 'LOXL4', 'LPA',
        'LPIN3', 'LPL', 'LRP1', 'LRRC10B', 'LSM2', 'MAD2L1', 'MAGI3', 'MAN2A2', 'MAP1S',
        'MAP3K1', 'MAP3K3', 'MAP3K7CL', 'MAP9', 'MAT2A', 'MC4R', 'MCAM', 'MCF2L', 'MECOM',
        'MED1', 'MESD', 'MFGE8', 'MGP', 'MIA3', 'MLH3', 'MORF4L1', 'MRAS', 'MRPS6',
        'MRVI1', 'MSH5', 'MTAP', 'MTUS1', 'MYH11', 'MYL2', 'MYLK', 'MYO9B', 'N4BP2L2',
        'NBEAL1', 'NCOA6', 'NEK8', 'NEK9', 'NF2', 'NFIB', 'NGF', 'NIPBL', 'NISCH', 'NLRC4',
        'NME9', 'NOB1', 'NOS3', 'NOTCH1', 'NR2F2', 'NR3C1', 'NRP1', 'NT5C2', 'NUPR1',
        'OPRL1', 'PAFAH1B1', 'PALLD', 'PARP12', 'PCNX3', 'PCSK9', 'PDE1A', 'PDE1C',
        'PDE3A', 'PDE5A', 'PDGFD', 'PDGFRA', 'PECAM1', 'PGF', 'PHACTR1', 'PHB', 'PHETA1',
        'PHLPP2', 'PID1', 'PLCE1', 'PLCG1', 'PLCG2', 'PLG', 'PLPP3', 'PLTP', 'PMAIP1',
        'PNPLA3', 'POLK', 'PPAP2B', 'PPARD', 'PPP1R12A', 'PRDM16', 'PREX1', 'PRIM2',
        'PRKAR1A', 'PRKCE', 'PRL', 'PROCR', 'PRRT1', 'PSMA4', 'PSMA5', 'PSORS1C1', 'PSRC1',
        'R3HCC1L', 'R3HDM1', 'RAC1', 'RASGEF1B', 'RCOR3', 'RDX', 'RELA', 'REST', 'RGS19',
        'RHOB', 'RIIAD1', 'RP1-257A7.4', 'RP1-257A7.5', 'RP11-298D21.1', 'RP11-298D21.3',
        'RP11-543N12.1', 'RP11-588K22.2', 'RP11-752L20.5', 'RP11-755F10.1', 'RRBP1',
        'RUNX1', 'SARS', 'SCAMP1-AS1', 'SCARB1', 'SCUBE1', 'SDCCAG3', 'SEMA5A', 'SEPT11',
        'SERPINA1', 'SERPINH1', 'SH3PXD2A', 'SHROOM3', 'SKI', 'SKIV2L', 'SLC18A1',
        'SLC22A1', 'SLC22A3', 'SLC22A4', 'SLC22A5', 'SLC2A12', 'SLC5A3', 'SLK', 'SMAD1',
        'SMAD3', 'SMAD7', 'SMG6', 'SMTN', 'SNF8', 'SORT1', 'SPC24', 'SPRY4', 'SREBF1',
        'ST3GAL4', 'ST5', 'STAG1', 'STARD13', 'STAT3', 'STX4', 'SUMO1', 'SUMO2', 'SVIL',
        'SWAP70', 'TAF1A', 'TARID', 'TBC1D7', 'TBX2', 'TBX20', 'TBX3', 'TCF21', 'TCF7L2',
        'TENT5A', 'TFAP2B', 'TFPI', 'TGFB1', 'THOC5', 'TIE1', 'TIMP3', 'TIPARP', 'TLNRD1',
        'TMEM133', 'TNFAIP8', 'TNKS', 'TNS1', 'TRIB1', 'TSPAN11', 'TSPAN14', 'TTC32',
        'TWIST1', 'TWISTNB', 'TXNRD3', 'UBC', 'UBE2H', 'UFL1', 'UMPS', 'UNC119B', 'USP34',
        'VAMP5', 'VEGFA', 'VWF', 'WASF1', 'WASF2', 'WIPI1', 'WT1', 'WWOX', 'WWP2',
        'ZBTB38', 'ZC3HC1', 'ZEB2', 'ZFHX3', 'ZFPM2', 'ZNF100', 'ZNF335', 'ZNF43',
        'ZNF462', 'ZNF532', 'ZNF589', 'ZNF652', 'ZNF831'
    }

    logger.info("Gene set sizes:")
    logger.info("  8-gene set: %s genes", len(genes_8))
    logger.info("  17-gene set: %s genes", len(genes_17))
    logger.info("  420-gene set: %s genes\n", len(genes_420))

    # Function to get genes in path from a specific set
    mask_8genes = (
        df['source'].isin(genes_8) |
        df['intermediate'].isin(genes_8) |
        df['target'].isin(genes_8)
    )

    mask_17genes = (
        df['source'].isin(genes_17) |
        df['intermediate'].isin(genes_17) |
        df['target'].isin(genes_17)
    )

    logger.info("="*80)
    logger.info("FILTERING STATISTICS")
    logger.info("="*80)
    logger.info("Rows with 8-gene set: %d", mask_8genes.sum())
    logger.info("Rows with 17-gene set: %d", mask_17genes.sum())
    logger.info("Rows with BOTH 8-gene and 17-gene sets: %d", (mask_8genes & mask_17genes).sum())
    logger.info("Rows with EITHER 8-gene or 17-gene sets: %d", (mask_8genes | mask_17genes).sum())
    logger.info("Residual rows (neither): %d\n", (~(mask_8genes | mask_17genes)).sum())

    # CREATE 8-GENE CSV
    logger.info("="*80)
    logger.info("CREATING 8-GENE CSV")
    logger.info("="*80)
    df_8genes = df[mask_8genes].copy()
    df_8genes['Genes_in_path_8gene_set'] = df_8genes.apply(lambda row: get_genes_from_set(row, genes_8), axis=1)
    df_8genes['Genes_in_path_420gene_set'] = df_8genes.apply(lambda row: get_genes_from_set(row, genes_420), axis=1)

    # Count unique genes found from 8-gene set
    genes_8_found = set()
    for col in ['source', 'intermediate', 'target']:
        genes_8_found.update(df_8genes[col][df_8genes[col].isin(genes_8)].unique())

    logger.info("Rows in 8-gene CSV: %d", len(df_8genes))
    logger.info("Unique genes from 8-gene set found: %s out of %s", len(genes_8_found), len(genes_8))
    logger.info("  Genes found: %s", sorted(genes_8_found))
    missing_8 = genes_8 - genes_8_found
    if missing_8:
        logger.info("  Genes NOT found: %s", sorted(missing_8))

    output_8genes = "gwas_8gene_set_.csv"
    df_8genes.to_csv(output_8genes, index=False)
    logger.info("\n 8-GENE CSV SAVED: %s", output_8genes)
    logger.info("  Shape: %s\n", df_8genes.shape)

    # CREATE 17-GENE CSV
    logger.info("="*80)
    logger.info("CREATING 17-GENE CSV")
    logger.info("="*80)
    df_17genes = df[mask_17genes].copy()
    df_17genes['Genes_in_path_17gene_set'] = df_17genes.apply(lambda row: get_genes_from_set(row, genes_17), axis=1)
    df_17genes['Genes_in_path_420gene_set'] = df_17genes.apply(lambda row: get_genes_from_set(row, genes_420), axis=1)

    # Count unique genes found from 17-gene set
    genes_17_found = set()
    for col in ['source', 'intermediate', 'target']:
        genes_17_found.update(df_17genes[col][df_17genes[col].isin(genes_17)].unique())

    logger.info("Rows in 17-gene CSV: %d", len(df_17genes))
    logger.info("Unique genes from 17-gene set found: %s out of %s", len(genes_17_found), len(genes_17))
    logger.info("  Genes found: %s", sorted(genes_17_found))
    missing_17 = genes_17 - genes_17_found
    if missing_17:
        logger.info("  Genes NOT found: %s", sorted(missing_17))

    output_17genes = "gwas_17gene_set_.csv"
    df_17genes.to_csv(output_17genes, index=False)
    logger.info("\n 17-GENE CSV SAVED: %s", output_17genes)
    logger.info("  Shape: %s\n", df_17genes.shape)

    # CREATE RESIDUAL CSV
    logger.info("="*80)
    logger.info("CREATING RESIDUAL CSV")
    logger.info("="*80)
    df_residual = df[~(mask_8genes | mask_17genes)].copy()
    df_residual['Genes_in_path_420gene_set'] = df_residual.apply(lambda row: get_genes_from_set(row, genes_420), axis=1)

    logger.info("Rows in residual CSV: %d", len(df_residual))

    # Check if any residual rows have 420-gene set genes
    residual_with_420 = (df_residual['Genes_in_path_420gene_set'] != '').sum()
    logger.info("Residual rows with 420-gene set genes: %d", residual_with_420)

    output_residual = "gwas_residual_.csv"
    df_residual.to_csv(output_residual, index=False)
    logger.info("\n RESIDUAL CSV SAVED: %s", output_residual)
    logger.info("  Shape: %s\n", df_residual.shape)

    # FINAL SUMMARY
    logger.info("="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    logger.info("Original GWAS CSV: %d rows", len(df))
    logger.info(" 8-gene CSV: %d rows", len(df_8genes))
    logger.info("  File: %s", output_8genes)
    logger.info("  Genes covered: %s/%s", len(genes_8_found), len(genes_8))
    logger.info(" 17-gene CSV: %d rows", len(df_17genes))
    logger.info("  File: %s", output_17genes)
    logger.info("  Genes covered: %s/%s", len(genes_17_found), len(genes_17))
    logger.info(" Residual CSV: %d rows", len(df_residual))
    logger.info("  File: %s", output_residual)
    logger.info("Verification: %s (8-only) + %s (17-only) + %s (overlap) + %s (residual) = %s rows", len(df_8genes) - (mask_8genes & mask_17genes).sum(), len(df_17genes) - (mask_8genes & mask_17genes).sum(), (mask_8genes & mask_17genes).sum(), len(df_residual), len(df_8genes) - (mask_8genes & mask_17genes).sum() + len(df_17genes) - (mask_8genes & mask_17genes).sum() + (mask_8genes & mask_17genes).sum() + len(df_residual))
    logger.info("Note: Overlap rows appear in BOTH 8-gene and 17-gene CSVs")
    logger.info("="*80)



if __name__ == "__main__":
    main()
