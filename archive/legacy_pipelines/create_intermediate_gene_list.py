"""Legacy script: create intermediate endothelial gene list with manual additions."""
from __future__ import annotations

import argparse

import pandas as pd

from debug_cell_count import present_gene_names

import logging


logger = logging.getLogger(__name__)

MANUAL_GENES = [
    "BCAR1", "BMP1", "CALCRL", "CCM2", "CDKN1A", "CDKN2B", "CFDP1", "COL4A1", "COL4A2",
    "EXOC3L2", "FBN2", "FGD6", "FLT1", "FURIN", "GDPD5", "GGT5", "GOSR2", "IBTK", "LAMB2",
    "LOX", "MORF4L1", "N4BP2L2", "NOS3", "PALLD", "PECAM1", "PGF", "PLPP3", "PREX1", "PRKAR1A",
    "SCUBE1", "SERPINH1", "SH3PXD2A", "SLK", "SMAD3", "SPRY4", "SVIL", "SWAP70", "TFPI",
    "TLNRD1", "TSPAN14", "ZEB2",
]


def main():
    ap = argparse.ArgumentParser(description="Merge detected + manual endothelial gene lists.")
    ap.add_argument("--output", required=True, help="Output CSV path for the combined gene list.")
    args = ap.parse_args()

    present_genes = list(present_gene_names)

    combined_genes = sorted(set(present_genes + MANUAL_GENES))
    logger.info("Total unique endothelial-present genes after adding manual list: %d", len(combined_genes))

    pd.Series(combined_genes, name="gene").to_csv(args.output, index=False)
    logger.info("Saved combined gene list to: %s", args.output)

    logger.info("Example genes: %s", combined_genes[:20])

    present_set = set(present_genes)
    manual_set = set(MANUAL_GENES)

    overlap_genes = sorted(present_set.intersection(manual_set))
    missing_genes = sorted(manual_set - present_set)

    logger.info("Total manual/GWAS genes: %d", len(manual_set))
    logger.info("Present in endothelial list: %d", len(overlap_genes))
    logger.info("Missing from endothelial list: %d", len(missing_genes))
    logger.info("Overlapping genes: %s", overlap_genes)
    logger.info("Missing genes: %s", missing_genes)


if __name__ == "__main__":
    main()
