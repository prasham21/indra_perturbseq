"""One-shot cleanup script for legacy pipeline files.
This module removes stale formatting artifacts and normalizes legacy script text.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002B50"
    "\U0000200D"
    "\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)

AI_COMMENT_PATTERNS = [
    re.compile(r"^(\s*)#\s*={3,}\s*$"),
    re.compile(r"^(\s*)#\s*-{3,}\s*$"),
    re.compile(r"^(\s*)#\s*~{3,}\s*$"),
    re.compile(r"^(\s*)#\s*\d+\.\s+[A-Z]"),
    re.compile(r"^(\s*)#\s*Step\s+\d+", re.IGNORECASE),
    re.compile(r"^(\s*)#\s*---+\s*$"),
    re.compile(r'^(\s*)# ---+\s*Global'),
    re.compile(r"^(\s*)#\s*Edit these if"),
]

SECTION_HEADER_PATTERN = re.compile(
    r"^(\s*)#\s*={3,}.*$\n"
    r"^(\s*)#\s*\d+\.\s+.*$\n"
    r"^(\s*)#\s*={3,}.*$",
    re.MULTILINE,
)


def remove_emojis(text: str) -> str:
    return EMOJI_PATTERN.sub("", text)


def is_ai_section_comment(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("#"):
        return False
    for pat in AI_COMMENT_PATTERNS:
        if pat.match(line):
            return True
    if re.match(r"^\s*#\s*-+\s*$", stripped):
        return True
    return False


def convert_print_to_logger(line: str) -> str:
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]

    if not stripped.startswith("print("):
        return line

    inside = stripped[6:]
    if inside.endswith(")"):
        inside = inside[:-1]
    elif inside.endswith(")\n"):
        inside = inside[:-2]

    inside = inside.strip()

    if inside.startswith("f\"") or inside.startswith("f'"):
        inside_content = inside[2:-1]
        parts = re.split(r"\{([^}]+)\}", inside_content)
        if len(parts) == 1:
            return f'{indent}logger.info("{inside_content}")\n'
        fmt = ""
        args = []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                fmt += part.replace("%", "%%")
            else:
                expr = part.strip()
                if ":" in expr:
                    var, spec = expr.rsplit(":", 1)
                    fmt += f"%{spec}"
                    args.append(var.strip())
                else:
                    fmt += "%s"
                    args.append(expr)
        args_str = ", ".join(args)
        return f'{indent}logger.info("{fmt}", {args_str})\n'

    return f"{indent}logger.info({inside})\n"


def has_future_annotations(lines: list[str]) -> bool:
    for line in lines:
        if "from __future__ import annotations" in line:
            return True
    return False


def has_logging_import(lines: list[str]) -> bool:
    for line in lines:
        if "import logging" in line:
            return True
    return False


def has_logger_setup(lines: list[str]) -> bool:
    for line in lines:
        if "logger = logging.getLogger" in line:
            return True
        if "log = logging.getLogger" in line:
            return True
    return False


def has_module_docstring(lines: list[str]) -> bool:
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "from __future__ import annotations":
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            return True
        return False
    return False


def get_filename_docstring(filename: str) -> str:
    name = Path(filename).stem
    return f'"""Legacy pipeline: {name}."""\n'


def clean_docstring(lines: list[str]) -> list[str]:
    """Trim overly long AI-style docstrings to a brief version."""
    result = []
    i = 0
    in_docstring = False
    docstring_start = -1
    docstring_lines = []
    found_code = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not found_code:
            if stripped and not stripped.startswith("#") and stripped != "from __future__ import annotations":
                found_code = True

        if found_code and not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
            quote = stripped[:3]
            if stripped.count(quote) >= 2 and len(stripped) > 6:
                result.append(line)
                i += 1
                continue
            in_docstring = True
            docstring_start = i
            docstring_lines = [line]
            i += 1
            continue

        if in_docstring:
            docstring_lines.append(line)
            if '"""' in stripped or "'''" in stripped:
                in_docstring = False
                if len(docstring_lines) > 8:
                    first_line = docstring_lines[0].strip().replace('"""', "").replace("'''", "").strip()
                    if not first_line:
                        for dl in docstring_lines[1:]:
                            candidate = dl.strip().replace('"""', "").replace("'''", "").strip()
                            if candidate and not candidate.startswith("-") and not candidate.startswith("*"):
                                first_line = candidate
                                break
                    if first_line:
                        if not first_line.endswith("."):
                            first_line += "."
                        result.append(f'"""{first_line}"""\n')
                    else:
                        for dl in docstring_lines:
                            result.append(dl)
                else:
                    for dl in docstring_lines:
                        result.append(dl)
                found_code = False
                i += 1
                continue
            i += 1
            continue

        result.append(line)
        i += 1

    return result


def process_file(filepath: str) -> bool:
    path = Path(filepath)
    if not path.exists():
        print(f"  SKIP (not found): {filepath}")
        return False

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    if not lines:
        print(f"  SKIP (empty): {filepath}")
        return False

    has_print = any(
        line.lstrip().startswith("print(") for line in lines
    )

    new_lines = []
    for line in lines:
        line = remove_emojis(line)

        if is_ai_section_comment(line):
            stripped = line.strip()
            if re.match(r"^#\s*={3,}$", stripped) or re.match(r"^#\s*-{3,}$", stripped) or re.match(r"^#\s*~{3,}$", stripped):
                continue
            if re.match(r"^#\s*\d+\.\s+[A-Z]", stripped):
                continue
            content = re.sub(r"^#\s*-+\s*$", "", stripped)
            if not content:
                continue

        if line.lstrip().startswith("print("):
            line = convert_print_to_logger(line)

        new_lines = new_lines or []
        new_lines.append(line)

    if not new_lines:
        new_lines = lines

    new_lines = clean_docstring(new_lines)

    needs_future = not has_future_annotations(new_lines)
    needs_logging_import = not has_logging_import(new_lines) and has_print
    needs_logger = not has_logger_setup(new_lines) and has_print

    insert_idx = 0
    inserts = []

    if needs_future:
        for idx, line in enumerate(new_lines):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_doc = True
                if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                    insert_idx = idx + 1
                    break
                for j in range(idx + 1, len(new_lines)):
                    if '"""' in new_lines[j] or "'''" in new_lines[j]:
                        insert_idx = j + 1
                        break
                break
            if not stripped or stripped.startswith("#"):
                continue
            insert_idx = idx
            break
        inserts.append("from __future__ import annotations\n")
        inserts.append("\n")

    if not has_module_docstring(new_lines) and not needs_future:
        docstring = get_filename_docstring(filepath)
        for idx, line in enumerate(new_lines):
            stripped = line.strip()
            if stripped == "from __future__ import annotations":
                insert_idx = idx
                break
            if not stripped or stripped.startswith("#"):
                continue
            insert_idx = idx
            break
        inserts.insert(0, docstring)
    elif not has_module_docstring(new_lines) and needs_future:
        docstring = get_filename_docstring(filepath)
        inserts.insert(0, docstring)

    if inserts:
        for i, ins in enumerate(inserts):
            new_lines.insert(insert_idx + i, ins)

    if needs_logging_import:
        import_block_end = 0
        for idx, line in enumerate(new_lines):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                import_block_end = idx + 1
            elif stripped and import_block_end > 0 and not stripped.startswith("#"):
                break
        new_lines.insert(import_block_end, "\n")
        new_lines.insert(import_block_end + 1, "import logging\n")
        import_block_end += 2

    if needs_logger:
        import_block_end = 0
        for idx, line in enumerate(new_lines):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                import_block_end = idx + 1
            elif stripped and import_block_end > 0 and not stripped.startswith("#"):
                break

        insert_pos = import_block_end
        if insert_pos < len(new_lines) and new_lines[insert_pos].strip() == "":
            insert_pos += 1
        new_lines.insert(insert_pos, "\nlogger = logging.getLogger(__name__)\n")

    new_text = "".join(new_lines)

    if new_text.rstrip() and not new_text.endswith("\n"):
        new_text += "\n"

    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        print(f"  CLEANED: {path.name}")
        return True
    else:
        print(f"  OK (no changes): {path.name}")
        return False


FILES = [
    "run_1hop_network_export.py",
    "2hop_one_stop_network_export.py",
    "3hop_one_stop_network_export.py",
    "false_positive_paths_eval.py",
    "tpr_calculation.py",
    "fpr_calculations.py",
    "compute_3hop_tp_fp_tpr_fpr.py",
    "recompute_tpr_fpr_with_outliers.py",
    "run_outliers_endothelial_universe.py",
    "calulate_ROC_network_export.py",
    "create_boxplots.py",
    "plot_tpr_fpr_overlaid.py",
    "overlaid_line_plots_outlier_comparison.py",
    "postprocess_one_stop.py",
    "indra_multihop.py",
    "permuted_network_eval.py",
    "create_ROC_curves.py",
    "create_ROC_curves_2hop.py",
    "missing_1.8k.py",
    "fc_pval_landscape_by_hop.py",
    "bulk_rna_deg.py",
    "sc_rna_deg_validation.py",
    "calculate_tp_fp_fn_bulk_rna.py",
    "bulk_vs_sc_rna_coparison.py",
    "mesh_coverage_analysis.py",
    "inspect_target_genes.py",
    "descendants_vs_non-descendants.py",
    "reannotate_mesh_terms_overwrite.py",
    "xyz.py",
    "deduplicate_gwas_dataset.py",
    "combine_gwas_datsets.py",
    "segregate_gwas_results.py",
    "cell_count_visualization.py",
    "create_html_asembler.py",
    "create_interrmediate_gene_list.py",
    "2hop_dataset_and _GWAS_filtering.py",
    "find_2hop_overlap_with_endothelial_genes.py",
    "map_mesh_terms.py",
    "exapnd_mesh_terms_list.py",
    "extract_mesh_terms.py",
    "top150_filtered.py",
    "compare_original_vs_permuted.py",
    "create_permuted_dataset.py",
    "mesh_annotation_1hop.py",
    "mesh_annotation_2hop.py",
    "mesh_annotation_3hop.py",
    "comapare_with_permuted_data.py",
    "combined_top100_csv.py",
    "4hop_evidence_pmid_extraction.py",
    "indra_4hop_analysis.py",
    "convert_uniprot_to_hgnc.py",
    "op_dataset_creation.py",
    "overlay_network_visualization.py",
    "omnipath_dataset_3hop.py",
    "network_visualization.py",
    "direct_pathway_anlaysis.py",
    "omnipath_client_1hop.py",
    "separate_self_taget_genes.py",
    "indra_2hop_cleanup.py",
    "extract_checkpoint.py",
    "pathway_analysis.py",
    "indra_pathway_analysis.py",
]


def main():
    base = Path(__file__).parent / "archive" / "legacy_pipelines"
    changed = 0
    skipped = 0
    for name in FILES:
        filepath = base / name
        if process_file(str(filepath)):
            changed += 1
        else:
            skipped += 1
    print(f"\nDone. Changed: {changed}, Unchanged/Skipped: {skipped}")


if __name__ == "__main__":
    main()
