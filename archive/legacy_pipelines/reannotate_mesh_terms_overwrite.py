"""Legacy script: reannotate mesh terms overwrite."""
from __future__ import annotations

import argparse
import re

import pandas as pd
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client import get_mesh_ids_for_pmids

import logging

logger = logging.getLogger(__name__)


def load_valid_mesh_ids(reference_csv: str) -> set[str]:
    ref = pd.read_csv(reference_csv, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
    if "mesh_id" not in ref.columns:
        raise ValueError(f"Reference CSV must contain 'mesh_id'. Columns={ref.columns.tolist()}")
    s = (
        ref["mesh_id"]
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.replace(r"[^A-Za-z0-9]", "", regex=True)
        .str.upper()
    )
    return set(s[s.str.startswith("D", na=False)].tolist())


def parse_pmids_cell(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, float) and pd.isna(x):
        return []
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return []
    return [tok for tok in s.replace(" ", "").split(";") if tok.isdigit()]


def batch_iter(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def build_mesh_id_to_name_for_ids(client: Neo4jClient, mesh_ids: list[str]) -> dict[str, str]:
    curies = [f"mesh:{mid}" for mid in mesh_ids]
    q = """
    UNWIND $ids AS mid
    MATCH (b:BioEntity {id: mid})
    RETURN b.id AS mesh_id, b.name AS mesh_name


def main():
    pass


if __name__ == "__main__":
    main()
