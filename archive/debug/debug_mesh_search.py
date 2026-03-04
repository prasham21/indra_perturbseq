import argparse
import re
import pandas as pd


def pct(x, total):
    return 0.0 if total == 0 else 100.0 * x / total


def summarize_numeric(df: pd.DataFrame, col: str):
    s = pd.to_numeric(df[col], errors="coerce")
    n = len(s)
    nn = s.notna().sum()
    print(f"\n== {col} ==")
    print(f"- numeric: {nn:,}/{n:,} ({pct(nn,n):.1f}%)")
    if nn == 0:
        return
    q = s.quantile([0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]).to_dict()
    print(f"- min/median/max: {q[0.0]:.0f} / {q[0.5]:.0f} / {q[1.0]:.0f}")
    print(f"- p95/p99: {q[0.95]:.0f} / {q[0.99]:.0f}")
    neg = (s < 0).sum()
    zeros = (s == 0).sum()
    print(f"- negatives: {neg:,}")
    print(f"- zeros: {zeros:,} ({pct(zeros,n):.1f}%)")


def detect_weird_gene_strings(genes: pd.Series):
    g = genes.astype(str)

    weird = {
        "has_nbsp": g.str.contains("\xa0", regex=False).sum(),
        "has_space": g.str.contains(r"\s", regex=True).sum(),
        "has_dot_number": g.str.contains(r"\d+\.\d+$", regex=True).sum(),
        "has_non_alnum": g.str.contains(r"[^A-Za-z0-9_-]", regex=True).sum(),
        "empty": (g.str.strip() == "").sum(),
    }
    print("\n### Gene string hygiene checks")
    for k, v in weird.items():
        print(f"- {k}: {v:,}")

    # show examples
    def show_examples(mask, title, k=20):
        ex = g[mask].head(k).tolist()
        if ex:
            print(f"\n- sample genes ({title}):")
            for x in ex:
                print(f"  - {repr(x)}")

    show_examples(g.str.contains("\xa0", regex=False), "contains NBSP (\\xa0)")
    show_examples(g.str.contains(r"\s", regex=True), "contains whitespace")
    show_examples(g.str.contains(r"\d+\.\d+$", regex=True), "ends with a decimal number")
    show_examples(g.str.contains(r"[^A-Za-z0-9_-]", regex=True), "contains non [A-Za-z0-9_-]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-csv", required=True, help="Path to gene_characterization CSV")
    ap.add_argument("--top", type=int, default=25, help="How many rows to display in top/bottom tables")
    args = ap.parse_args()

    df = pd.read_csv(args.gene_csv, low_memory=False)

    required = ["gene", "literature_count", "database_count", "characterization_count", "evidence_mentions_count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nColumns present: {df.columns.tolist()}")

    print("\n### File overview")
    print(f"- rows: {len(df):,}")
    print(f"- cols: {len(df.columns):,}")
    print(f"- columns: {df.columns.tolist()}")

    # gene uniqueness
    df["gene"] = df["gene"].astype(str)
    dup = df["gene"].duplicated().sum()
    print(f"- duplicate gene rows: {dup:,}")

    detect_weird_gene_strings(df["gene"])

    # numeric summaries
    for col in ["literature_count", "database_count", "characterization_count", "evidence_mentions_count"]:
        summarize_numeric(df, col)

    # integrity checks
    lit = pd.to_numeric(df["literature_count"], errors="coerce").fillna(0).astype(int)
    db = pd.to_numeric(df["database_count"], errors="coerce").fillna(0).astype(int)
    char = pd.to_numeric(df["characterization_count"], errors="coerce").fillna(0).astype(int)
    evm = pd.to_numeric(df["evidence_mentions_count"], errors="coerce").fillna(0).astype(int)

    print("\n### Integrity checks")
    bad_sum = (char != (lit + db)).sum()
    print(f"- characterization_count == literature_count + database_count: "
          f"{len(df)-bad_sum:,}/{len(df):,} ok; {bad_sum:,} bad")

    # Evidence mentions should be >= unique PMIDs in most cases; but can be 0 if no evidence nodes were found
    lt_evm = (evm < lit).sum()
    print(f"- evidence_mentions_count < literature_count: {lt_evm:,} rows")
    if lt_evm:
        ex = df.loc[evm < lit, ["gene", "literature_count", "evidence_mentions_count", "database_count", "characterization_count"]].head(20)
        print("  sample rows where evidence_mentions_count < literature_count (investigate):")
        print(ex.to_string(index=False))

    # Sanity: evidence mentions should be >= characterization_count? Not guaranteed, because characterization counts unique sources.
    # But evm should typically be >= char; show how often it isn't.
    lt_char = (evm < char).sum()
    print(f"- evidence_mentions_count < characterization_count: {lt_char:,} rows")
    if lt_char:
        ex = df.loc[evm < char, ["gene", "characterization_count", "evidence_mentions_count", "literature_count", "database_count"]].head(20)
        print("  sample rows where evidence_mentions_count < characterization_count (possible if DB-only sources counted but evidence JSON missing):")
        print(ex.to_string(index=False))

    # Zeros breakdown
    zeros = df[(char == 0) & (evm == 0)]
    print("\n### Zero cases")
    print(f"- genes with characterization_count=0 AND evidence_mentions_count=0: {len(zeros):,}")
    if len(zeros):
        print(zeros[["gene", "literature_count", "database_count", "characterization_count", "evidence_mentions_count"]].head(50).to_string(index=False))

    # Top/bottom tables
    out = df.copy()
    out["_char"] = char
    out["_evm"] = evm

    print(f"\n### Top {args.top} by characterization_count")
    print(
        out.sort_values(["_char", "_evm"], ascending=[False, False])[[
            "gene", "literature_count", "database_count", "characterization_count", "evidence_mentions_count"
        ]].head(args.top).to_string(index=False)
    )

    print(f"\n### Top {args.top} by evidence_mentions_count")
    print(
        out.sort_values(["_evm", "_char"], ascending=[False, False])[[
            "gene", "evidence_mentions_count", "literature_count", "database_count", "characterization_count"
        ]].head(args.top).to_string(index=False)
    )

    print(f"\n### Bottom {args.top} (lowest characterization_count, then evidence_mentions_count)")
    print(
        out.sort_values(["_char", "_evm"], ascending=[True, True])[[
            "gene", "literature_count", "database_count", "characterization_count", "evidence_mentions_count"
        ]].head(args.top).to_string(index=False)
    )

    # Relationship between unique sources and evidence mentions
    print("\n### Relationship checks (coarse)")
    ratio = pd.Series([None] * len(out))
    with pd.option_context("mode.use_inf_as_na", True):
        ratio = (out["evidence_mentions_count"] / out["characterization_count"].replace(0, pd.NA)).astype("Float64")
    ratio_nn = ratio.dropna()
    if len(ratio_nn):
        print(f"- evidence_mentions_count / characterization_count: "
              f"median={ratio_nn.median():.2f}, p95={ratio_nn.quantile(0.95):.2f}, max={ratio_nn.max():.2f}")
    else:
        print("- evidence_mentions_count / characterization_count: no non-null ratios (all characterization_count=0?)")


if __name__ == "__main__":
    main()