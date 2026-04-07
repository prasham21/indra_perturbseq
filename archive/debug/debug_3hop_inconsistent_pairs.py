#!/usr/bin/env python3
"""3-hop pathway extraction for inconsistent source-target pairs with sign consistency.

- Input: CSV with source, target, logfoldchange, pval
- Uses direct graph traversal (graph.successors), NOT shortest_simple_paths
- Only keeps sign-consistent pathways:
  - IncreaseAmount => logfoldchange < 0
  - DecreaseAmount => logfoldchange > 0
- At most one pathway per (source, target) pair
- Checkpoints every 100 rows; resumes from checkpoint if present.
  Checkpoint stored under outputs/ (gitignored).
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

# Ensure src is on path when run from project root
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from indra_perturbseq.gene_lists import load_gene_set
from indra_perturbseq.graph import is_hgnc_node, load_graph
from indra_perturbseq.hgnc import normalize_hgnc_symbol
from indra_perturbseq.statements import best_statement, indra_html_url

logger = logging.getLogger(__name__)

INCDEC = frozenset({"IncreaseAmount", "DecreaseAmount"})
CHECKPOINT_EVERY = 100


def is_sign_consistent(stmt_type: str | None, logfc: float | None) -> bool:
    """For knockdown: IncreaseAmount => target decreases (logfc<0); DecreaseAmount => target increases (logfc>0)."""
    if stmt_type not in INCDEC:
        return False
    if logfc is None:
        return False
    if stmt_type == "IncreaseAmount":
        return logfc < 0.0
    return logfc > 0.0


def load_pairs_csv(csv_path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    """Load (source, target) -> (logfoldchange, pval). First row wins on duplicates."""
    df = pd.read_csv(csv_path, low_memory=False)
    for col in ("source", "target", "logfoldchange", "pval"):
        if col not in df.columns:
            raise ValueError(f"CSV missing column '{col}'. Has: {df.columns.tolist()}")

    pairs: dict[tuple[str, str], tuple[float, float]] = {}
    for _, row in df.iterrows():
        src = normalize_hgnc_symbol(str(row["source"]).strip())
        tgt = normalize_hgnc_symbol(str(row["target"]).strip())
        if not src or not tgt or src == tgt:
            continue
        key = (src, tgt)
        if key in pairs:
            continue
        try:
            lfc = float(row["logfoldchange"])
        except (TypeError, ValueError):
            lfc = float("nan")
        try:
            pval = float(row["pval"])
        except (TypeError, ValueError):
            pval = float("nan")
        pairs[key] = (lfc, pval)
    return pairs


def run_3hop_sign_consistent(
    graph,
    src: str,
    target_to_deg: dict[str, tuple[float, float]],
    allowed_intermediates: set[str],
) -> list[dict]:
    """Find at most one sign-consistent 3-hop path per target. Direct traversal."""
    if src not in graph or not is_hgnc_node(graph, src):
        return []

    found_targets: set[str] = set()
    rows: list[dict] = []

    for mid1 in graph.successors(src):
        if not is_hgnc_node(graph, mid1) or mid1 not in allowed_intermediates or mid1 == src:
            continue
        s1 = best_statement(graph.get_edge_data(src, mid1), require_incdec=False)
        if not s1:
            continue

        for mid2 in graph.successors(mid1):
            if not is_hgnc_node(graph, mid2) or mid2 not in allowed_intermediates:
                continue
            if mid2 in (src, mid1):
                continue
            s2 = best_statement(graph.get_edge_data(mid1, mid2), require_incdec=False)
            if not s2:
                continue

            for tgt in graph.successors(mid2):
                if tgt in found_targets:
                    continue
                if not is_hgnc_node(graph, tgt) or tgt not in target_to_deg:
                    continue
                if tgt in (src, mid1, mid2):
                    continue

                s3 = best_statement(graph.get_edge_data(mid2, tgt), require_incdec=True)
                if not s3:
                    continue

                stmt_type_3 = s3.get("stmt_type")
                lfc, pval = target_to_deg[tgt]
                if not is_sign_consistent(stmt_type_3, lfc if lfc == lfc else None):
                    continue

                found_targets.add(tgt)
                h1, h2, h3 = s1.get("stmt_hash"), s2.get("stmt_hash"), s3.get("stmt_hash")
                rows.append({
                    "source": src,
                    "intermediate_1": mid1,
                    "intermediate_2": mid2,
                    "target": tgt,
                    "stmt_type_1": s1.get("stmt_type"),
                    "stmt_type_2": s2.get("stmt_type"),
                    "stmt_type_3": stmt_type_3,
                    "belief_1": s1.get("belief"),
                    "belief_2": s2.get("belief"),
                    "belief_3": s3.get("belief"),
                    "evidence_1": s1.get("evidence_count"),
                    "evidence_2": s2.get("evidence_count"),
                    "evidence_3": s3.get("evidence_count"),
                    "logfoldchange": lfc,
                    "pval": pval,
                    "hop1_hash": h1,
                    "hop2_hash": h2,
                    "hop3_hash": h3,
                    "hop1_indra_url": indra_html_url(h1),
                    "hop2_indra_url": indra_html_url(h2),
                    "hop3_indra_url": indra_html_url(h3),
                })
                break

    return rows


def load_checkpoint(checkpoint_path: Path) -> tuple[set[str], list[dict]]:
    """Return (processed_sources, rows). Empty if no checkpoint."""
    if not checkpoint_path.exists():
        return set(), []
    try:
        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)
        processed = set(data.get("processed_sources", []))
        rows = list(data.get("rows", []))
        logger.info("Resumed from checkpoint: %d sources done, %d rows", len(processed), len(rows))
        return processed, rows
    except Exception as e:
        logger.warning("Failed to load checkpoint %s: %s. Starting fresh.", checkpoint_path, e)
        return set(), []


def save_checkpoint(checkpoint_path: Path, processed_sources: set[str], rows: list[dict]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "wb") as f:
        pickle.dump({"processed_sources": list(processed_sources), "rows": rows}, f)
    logger.info("Checkpoint saved: %d sources, %d rows -> %s", len(processed_sources), len(rows), checkpoint_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(
        description="3-hop sign-consistent pathways for inconsistent pairs (direct graph traversal).",
    )
    ap.add_argument(
        "--pairs-csv",
        required=True,
        help="CSV with columns: source, target, logfoldchange, pval",
    )
    ap.add_argument("--graph-pkl", required=True, help="INDRA network export pickle")
    ap.add_argument(
        "--endothelial-list",
        required=True,
        help="CSV with 'gene' column for allowed intermediates",
    )
    ap.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV path",
    )
    ap.add_argument(
        "--checkpoint-file",
        default=None,
        help="Checkpoint path (default: outputs/.checkpoints/run_3hop_inconsistent_pairs_checkpoint.pkl)",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save checkpoint every N rows (default: 100)",
    )
    args = ap.parse_args()

    checkpoint_path = Path(
        args.checkpoint_file
        or str(_ROOT / "outputs" / ".checkpoints" / "run_3hop_inconsistent_pairs_checkpoint.pkl")
    )

    pairs_path = Path(args.pairs_csv)
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs CSV not found: {pairs_path}")

    logger.info("Loading pairs from %s", pairs_path)
    pairs = load_pairs_csv(pairs_path)
    logger.info("Loaded %d unique (source, target) pairs", len(pairs))

    by_source: dict[str, dict[str, tuple[float, float]]] = {}
    for (src, tgt), (lfc, pval) in pairs.items():
        by_source.setdefault(src, {})[tgt] = (lfc, pval)

    logger.info("Sources with targets: %d", len(by_source))

    processed_sources, all_rows = load_checkpoint(checkpoint_path)
    last_checkpoint_rows = len(all_rows)

    # Deterministic order for resume
    source_order = sorted(by_source.keys())
    remaining = [s for s in source_order if s not in processed_sources]
    logger.info("Remaining sources to process: %d", len(remaining))

    if not remaining:
        logger.info("All sources already processed (from checkpoint). Writing final output.")
    else:
        logger.info("Loading graph: %s", args.graph_pkl)
        graph, _ = load_graph(args.graph_pkl)
        logger.info("Loading intermediate gene set: %s", args.endothelial_list)
        allowed_intermediates = load_gene_set(args.endothelial_list)
        logger.info("Allowed intermediates: %d", len(allowed_intermediates))

        for i, src in enumerate(remaining):
            target_to_deg = by_source[src]
            rows = run_3hop_sign_consistent(graph, src, target_to_deg, allowed_intermediates)
            all_rows.extend(rows)
            processed_sources.add(src)

            if len(all_rows) >= last_checkpoint_rows + args.checkpoint_every:
                save_checkpoint(checkpoint_path, processed_sources, all_rows)
                last_checkpoint_rows = len(all_rows)

            if (i + 1) % 50 == 0:
                logger.info("Processed %d/%d remaining sources, %d total rows", i + 1, len(remaining), len(all_rows))

    df = pd.DataFrame(all_rows)
    if df.empty:
        logger.warning("No sign-consistent 3-hop paths found. Exiting.")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Removed checkpoint (no results).")
        return

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), out_path)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Removed checkpoint (run complete).")


if __name__ == "__main__":
    main()