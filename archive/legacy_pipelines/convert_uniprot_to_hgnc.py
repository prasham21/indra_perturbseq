"""Legacy script: convert_uniprot_to_hgnc."""
from __future__ import annotations

import argparse

import os
import pandas as pd
from indra.databases import uniprot_client, hgnc_client

import logging


logger = logging.getLogger(__name__)


def convert_to_hgnc_symbol(identifier: str) -> str:
    """Convert UniProt or UniProt.chain IDs to HGNC symbols if possible."""
    if not isinstance(identifier, str):
        return identifier
    identifier = identifier.strip()
    identifier = identifier.replace("uniprot.chain:", "uniprot:")
    identifier = identifier.replace("hgnc:uniprot:", "uniprot:")

    if not identifier.startswith("uniprot:"):
        return identifier

    uid = identifier.split("uniprot:")[-1]
    hgnc_id = uniprot_client.get_hgnc_id(uid)
    if hgnc_id:
        hgnc_symbol = hgnc_client.get_hgnc_name(hgnc_id)
        if hgnc_symbol:
            return hgnc_symbol
    return identifier  # fallback if not found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", default="indra_4hop_results_human_final.csv", help="Path for args.input_file.")
    ap.add_argument("--output-file", default="indra_4hop_results_converted_2.csv", help="Path for args.output_file.")
    args = ap.parse_args()

    logger.info("Loading input: %s", args.input_file)
    df = pd.read_csv(args.input_file)

    intermediate_cols = [col for col in df.columns if col.startswith("intermediate_")]
    logger.info("Converting columns: %s", intermediate_cols)

    total_before = 0
    total_converted = 0
    not_found = set()

    for col in intermediate_cols:
        def safe_convert(x):
            nonlocal total_before, total_converted
            if isinstance(x, str) and "uniprot" in x:
                total_before += 1
                new = convert_to_hgnc_symbol(x)
                if new != x:
                    total_converted += 1
                else:
                    not_found.add(x)
                return new
            return x

        df[col] = df[col].apply(safe_convert)

    logger.info("Total UniProt-like IDs found: %s", total_before)
    logger.info("Successfully converted: %s", total_converted)
    logger.info("Unconverted (likely non-human proteins or missing): %s", len(not_found))

    df.to_csv(args.output_file, index=False)
    logger.info("Saved converted file to: %s", args.output_file)

    if not_found:
        with open(args.output_file.replace(".csv", "_unmapped.txt"), "w") as f:
            f.write("\n".join(sorted(not_found)))
        logger.info("Unmapped UniProt IDs saved to: %s", args.output_file.replace('.csv', '_unmapped.txt'))


if __name__ == "__main__":
    main()
