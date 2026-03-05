"""GWAS gene-set constants.
This module provides preprocessing utilities and command-line data preparation workflows.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
GWAS_GENES: set[str] = {
    "ARHGEF26", "BCAR1", "BMP1", "CALCRL", "CCM2", "CDKN1A", "CDKN2B",
    "CFDP1", "COL4A1", "COL4A2", "EDN1", "EXOC3L2", "FBN2", "FGD6",
    "FLT1", "FURIN", "GDPD5", "GGT5", "GOSR2", "IBTK", "JCAD", "LAMB2",
    "LOX", "MORF4L1", "N4BP2L2", "NOS3", "PALLD", "PECAM1", "PGF",
    "PLPP3", "PRDM16", "PREX1", "PRKAR1A", "SCUBE1", "SERPINH1",
    "SH3PXD2A", "SLK", "SMAD3", "SPRY4", "SVIL", "SWAP70", "TFPI",
    "TLNRD1", "TSPAN14", "ZEB2",
}

GENES_8: set[str] = {
    "PRDM16", "PLPP3", "NOS3", "JCAD", "FLT1", "EDN1", "PECAM1",
    "ARHGEF26",
}

GENES_17: set[str] = {
    "CALCRL", "CCM2", "CDKN1A", "EXOC3L2", "GDPD5", "GGT5", "IBTK",
    "N4BP2L2", "PREX1", "PRKAR1A", "SCUBE1", "SLK", "SPRY4", "SVIL",
    "TFPI", "TLNRD1", "TSPAN14",
}

GENES_420: set[str] = {
    "AAK1", "ABCG5", "ABCG8", "ABHD2", "AC003986.6", "AC105384.1",
    "ACTA2", "ACTN4", "ACTRT2", "ACVR2A", "ACVRL1", "ADAM19", "ADAMTS3",
    "ADAMTS7", "AEBP1", "AFAP1L2", "AFF4", "AGAP5", "AHI1", "AIDA",
    "AKAP12", "AL592148.3", "ANGPTL4", "ANKRD13B", "ANTXR2", "ANXA11",
    "AP000318.2", "AP002989.1", "APOA1", "APOA5", "APOB", "APOE", "APOM",
    "ARAP3", "ARHGAP21", "ARHGAP26", "ARHGAP42", "ARHGEF12", "ARHGEF26",
    "ARNT", "ARNTL", "ARVCF", "ATF6B", "ATP1B1", "ATP2B1", "ATXN7L2",
    "AXL", "B4GALT5", "BACH1", "BAG6", "BASP1", "BCAR1", "BCAS3",
    "BCKDHA", "BMP1", "BMPR1B", "C1QTNF1", "C4A", "CABIN1", "CALCRL",
    "CAND1", "CARF", "CAV1", "CBS", "CBX5", "CCDC3", "CCDC97", "CCM2",
    "CD36", "CDC123", "CDH13", "CDK8", "CDKN1A", "CDKN2A", "CDKN2B",
    "CDKN2BAS", "CEL", "CELSR2", "CETP", "CFDP1", "CHRNB4", "CLDN5",
    "COBLL1", "COL4A1", "COL4A2", "COL6A3", "COPRS", "CORO6", "CRYAB",
    "CSF1", "CTD-3253I12.1", "CTSH", "CXCL12", "CYP17A1", "CYP46A1",
    "DAAM2", "DAB2IP", "DCUN1D3", "DENND4C", "DHX36", "DHX38", "DMAC2",
    "DOCK5", "DOCK9", "DPYD", "DST", "EDN1", "EDNRA", "EFCAB13",
    "EGFLAM", "EHBP1L1", "EIF2B2", "EPAS1", "ESYT3", "EXOC3L2", "EZR",
    "F10", "FAM114A1", "FAM117B", "FAM177B", "FBN2", "FCHO1", "FER",
    "FES", "FGD5", "FGD6", "FGF5", "FHL3", "FHL5", "FIGN", "FLOT1",
    "FLT1", "FN1", "FNDC3B", "FOCAD", "FOXC1", "FURIN", "GAS8", "GATA6",
    "GATAD2A", "GDPD5", "GEM", "GFOD1", "GGCX", "GGT5", "GIGYF1",
    "GIGYF2", "GNA12", "GNAS", "GOSR2", "GRK4", "GUCY1A1", "GUCY1A3",
    "HDAC9", "HDGFL1", "HEY2", "HHAT", "HHIPL1", "HIPK2", "HIVEP2",
    "HLA-C", "HLA-DQB1", "HMHB1", "HNF1A", "HOMER3", "HSD17B1",
    "HSD17B12", "HTRA1", "HTT", "IBTK", "ICA1L", "IGFBP7", "IL6R",
    "IL6ST", "ILK", "INPP5B", "IRS1", "ITGA1", "ITGB3", "ITIH4", "JCAD",
    "JUN", "KANK1", "KCNE2", "KCNK5", "KCTD8", "KIAA0040", "KLF2",
    "KPTN", "LAMA4", "LAMB1", "LAMB2", "LDLR", "LIMS2", "LINC00189",
    "LINC00310", "LINC01312", "LIPA", "LIPC", "LMAN1", "LMOD1", "LOX",
    "LOXL4", "LPA", "LPIN3", "LPL", "LRP1", "LRRC10B", "LSM2",
    "MAD2L1", "MAGI3", "MAN2A2", "MAP1S", "MAP3K1", "MAP3K3", "MAP3K7CL",
    "MAP9", "MAT2A", "MC4R", "MCAM", "MCF2L", "MECOM", "MED1", "MESD",
    "MFGE8", "MGP", "MIA3", "MLH3", "MORF4L1", "MRAS", "MRPS6", "MRVI1",
    "MSH5", "MTAP", "MTUS1", "MYH11", "MYL2", "MYLK", "MYO9B",
    "N4BP2L2", "NBEAL1", "NCOA6", "NEK8", "NEK9", "NF2", "NFIB", "NGF",
    "NIPBL", "NISCH", "NLRC4", "NME9", "NOB1", "NOS3", "NOTCH1", "NR2F2",
    "NR3C1", "NRP1", "NT5C2", "NUPR1", "OPRL1", "PAFAH1B1", "PALLD",
    "PARP12", "PCNX3", "PCSK9", "PDE1A", "PDE1C", "PDE3A", "PDE5A",
    "PDGFD", "PDGFRA", "PECAM1", "PGF", "PHACTR1", "PHB", "PHETA1",
    "PHLPP2", "PID1", "PLCE1", "PLCG1", "PLCG2", "PLG", "PLPP3", "PLTP",
    "PMAIP1", "PNPLA3", "POLK", "PPAP2B", "PPARD", "PPP1R12A", "PRDM16",
    "PREX1", "PRIM2", "PRKAR1A", "PRKCE", "PRL", "PROCR", "PRRT1",
    "PSMA4", "PSMA5", "PSORS1C1", "PSRC1", "R3HCC1L", "R3HDM1", "RAC1",
    "RASGEF1B", "RCOR3", "RDX", "RELA", "REST", "RGS19", "RHOB",
    "RIIAD1", "RP1-257A7.4", "RP1-257A7.5", "RP11-298D21.1",
    "RP11-298D21.3", "RP11-543N12.1", "RP11-588K22.2", "RP11-752L20.5",
    "RP11-755F10.1", "RRBP1", "RUNX1", "SARS", "SCAMP1-AS1", "SCARB1",
    "SCUBE1", "SDCCAG3", "SEMA5A", "SEPT11", "SERPINA1", "SERPINH1",
    "SH3PXD2A", "SHROOM3", "SKI", "SKIV2L", "SLC18A1", "SLC22A1",
    "SLC22A3", "SLC22A4", "SLC22A5", "SLC2A12", "SLC5A3", "SLK", "SMAD1",
    "SMAD3", "SMAD7", "SMG6", "SMTN", "SNF8", "SORT1", "SPC24", "SPRY4",
    "SREBF1", "ST3GAL4", "ST5", "STAG1", "STARD13", "STAT3", "STX4",
    "SUMO1", "SUMO2", "SVIL", "SWAP70", "TAF1A", "TARID", "TBC1D7",
    "TBX2", "TBX20", "TBX3", "TCF21", "TCF7L2", "TENT5A", "TFAP2B",
    "TFPI", "TGFB1", "THOC5", "TIE1", "TIMP3", "TIPARP", "TLNRD1",
    "TMEM133", "TNFAIP8", "TNKS", "TNS1", "TRIB1", "TSPAN11", "TSPAN14",
    "TTC32", "TWIST1", "TWISTNB", "TXNRD3", "UBC", "UBE2H", "UFL1",
    "UMPS", "UNC119B", "USP34", "VAMP5", "VEGFA", "VWF", "WASF1",
    "WASF2", "WIPI1", "WT1", "WWOX", "WWP2", "ZBTB38", "ZC3HC1", "ZEB2",
    "ZFHX3", "ZFPM2", "ZNF100", "ZNF335", "ZNF43", "ZNF462", "ZNF532",
    "ZNF589", "ZNF652", "ZNF831",
}

URL_RE = re.compile(
    r"^https://db\.indra\.bio/statements/from_hash/-?\d+\?format=html$",
)


__all__ = ["GWAS_GENES", "GENES_8", "GENES_17", "GENES_420", "URL_RE"]
