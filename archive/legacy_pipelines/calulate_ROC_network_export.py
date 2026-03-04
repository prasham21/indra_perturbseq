
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from indra.databases import hgnc_client

# ===== PATHS =====
TV_PATH = "/Users/prashammarfatia/Downloads/target_validation_expanded.csv"
DE_DIR  = "/Users/prashammarfatia/Downloads/de_results_per_gene/"

POS_1HOP_PATHS = "/Users/prashammarfatia/Downloads/indra_1hop_NETWORK_EXPORT_main.csv"
POS_2HOP_PATHS = "/Users/prashammarfatia/Downloads/2hop_network_export_main.csv"

NEG_1HOP_PATHS = "/Users/prashammarfatia/Downloads/fp_1hop_paths_p005.csv"
NEG_2HOP_PATHS = "/Users/prashammarfatia/Downloads/fp_2hop_paths_p005.csv"

OUT_CSV = "/Users/prashammarfatia/Downloads/roc_le2hop_overall_and_per_source_by_threshold_HGNC_norm.csv"
OUT_ROC_OVERALL = "/Users/prashammarfatia/Downloads/roc_le2hop_overall_HGNC_norm_ZOOM.png"
OUT_ROC_PERSRC  = "/Users/prashammarfatia/Downloads/roc_le2hop_per_source_avg_HGNC_norm_ZOOM.png"

# ===== SETTINGS =====
TV_SOURCE_COL = "Gene"
TV_FLAG_COL = "Karen_Flag"
TV_FLAG_KEEP = "Use_for_analysis"

THRESHOLDS = np.array([0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001], dtype=float)

plt.rcParams.update({
    "figure.figsize": (6, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "savefig.dpi": 300,
})

# -----------------------------
# HGNC normalization
# -----------------------------
_norm_cache = {}
def normalize_hgnc_symbol(symbol: str):
    if symbol is None:
        return None
    s = str(symbol).strip()
    if not s:
        return None
    if s in _norm_cache:
        return _norm_cache[s]

    hid = hgnc_client.get_current_hgnc_id(s.upper())
    if not hid:
        out = s.upper()
        _norm_cache[s] = out
        return out

    if isinstance(hid, (list, tuple, set)):
        hid = sorted(list(hid))[0] if hid else None
        if not hid:
            out = s.upper()
            _norm_cache[s] = out
            return out

    name = hgnc_client.get_hgnc_name(hid) if hid else None
    out = name or s.upper()
    _norm_cache[s] = out
    return out

# -----------------------------
# Helpers
# -----------------------------
def pick_pcol(df):
    for c in ["pvals", "pval", "p_value", "p_val", "pvals_adj", "padj", "qval", "fdr", "p_adj"]:
        if c in df.columns:
            return c
    raise ValueError(f"No p-value column found. Columns: {df.columns.tolist()}")

def load_sources():
    tv = pd.read_csv(TV_PATH, low_memory=False)
    if TV_FLAG_COL in tv.columns:
        tv = tv[tv[TV_FLAG_COL] == TV_FLAG_KEEP].copy()
    srcs = [str(x).strip() for x in tv[TV_SOURCE_COL].dropna().tolist() if str(x).strip()]
    out = []
    seen = set()
    for s in srcs:
        sn = normalize_hgnc_symbol(s)
        if sn and sn not in seen:
            out.append(sn)
            seen.add(sn)
    return out

def load_explained_pairs(path):
    df = pd.read_csv(path, low_memory=False)
    if not {"source","target"}.issubset(df.columns):
        raise ValueError(f"{path} missing source/target. Has: {df.columns.tolist()}")
    df["source"] = df["source"].astype(str).str.strip().map(normalize_hgnc_symbol)
    df["target"] = df["target"].astype(str).str.strip().map(normalize_hgnc_symbol)
    df = df.dropna(subset=["source","target"])
    df = df[df["source"] != df["target"]].copy()
    return set(zip(df["source"], df["target"]))

def auc_trapezoid(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]; y = y[mask]
    if len(x) < 2:
        return np.nan
    order = np.argsort(x)
    return np.trapezoid(y[order], x[order])

def plot_roc_zoom(x, y, title, outpath):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]; y = y[mask]
    order = np.argsort(x)
    x = x[order]; y = y[order]
    auc = auc_trapezoid(x, y)

    dx = x.max() - x.min()
    dy = y.max() - y.min()
    pad_x = max(0.005, 0.15*dx if dx > 0 else 0.01)
    pad_y = max(0.01, 0.15*dy if dy > 0 else 0.02)

    xmin = max(0, x.min() - pad_x)
    xmax = min(1, x.max() + pad_x)
    ymin = max(0, y.min() - pad_y)
    ymax = min(1, y.max() + pad_y)

    # Chance line restricted to the zoom window so it's visible
    chance_x = np.array([xmin, xmax])
    chance_y = chance_x

    plt.figure()
    plt.plot(x, y, marker="o", markersize=6, label=f"AUC = {auc:.3f}")
    plt.plot(chance_x, chance_y, "r--", linewidth=1.2, label="Chance")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    print("Wrote ROC plot:", outpath)
    return auc

# -----------------------------
# Load explained pairs (<=2 hops union)
# -----------------------------
print("Loading explained pairs from 4 files (pos/neg, 1hop/2hop)...")
pos1 = load_explained_pairs(POS_1HOP_PATHS)
pos2 = load_explained_pairs(POS_2HOP_PATHS)
neg1 = load_explained_pairs(NEG_1HOP_PATHS)
neg2 = load_explained_pairs(NEG_2HOP_PATHS)

explained_pairs = pos1 | pos2 | neg1 | neg2
print("Explained pairs (<=2 hops union):", len(explained_pairs))

expl_by_src = {}
for s, t in explained_pairs:
    expl_by_src.setdefault(s, set()).add(t)

sources = load_sources()
print("Sources (Karen-approved, normalized):", len(sources))

# -----------------------------
# Cache control-file p-values once per source (normalized)
# -----------------------------
print("Caching control-file p-values once per source (HGNC-normalized)...")
cache = {}
missing = 0

for src in sources:
    fpath = os.path.join(DE_DIR, f"{src}_vs_control.csv")
    if not os.path.exists(fpath):
        missing += 1
        continue

    d = pd.read_csv(fpath, low_memory=False)
    if "names" not in d.columns:
        missing += 1
        continue

    pcol = pick_pcol(d)
    d[pcol] = pd.to_numeric(d[pcol], errors="coerce")

    pmap = {}
    for _, r in d.iterrows():
        t_raw = r["names"]
        p = r[pcol]
        if pd.isna(p):
            continue
        t = normalize_hgnc_symbol(t_raw)
        if not t or t == src:
            continue
        p = float(p)
        if (t not in pmap) or (p < pmap[t]):
            pmap[t] = p

    if not pmap:
        missing += 1
        continue

    pvals_all = np.fromiter(pmap.values(), dtype=float)
    cache[src] = (pvals_all, pmap)

print(f"Cached sources: {len(cache)} | missing/unusable: {missing}")

# -----------------------------
# Compute both methods per threshold
# -----------------------------
rows = []
overall_tprs = []
overall_fprs = []
persrc_tprs = []
persrc_fprs = []

for thr in THRESHOLDS:
    TP_all = FP_all = POS_all = NEG_all = 0
    tpr_list = []
    fpr_list = []

    for src, (pvals_all, pmap) in cache.items():
        pos_size = int(np.sum(pvals_all <= thr))
        neg_size = int(np.sum(pvals_all > thr))

        explained = expl_by_src.get(src, set())

        tp = fp = 0
        for t in explained:
            p = pmap.get(t)
            if p is None:
                continue
            if p <= thr:
                tp += 1
            else:
                fp += 1

        TP_all += tp
        FP_all += fp
        POS_all += pos_size
        NEG_all += neg_size

        tpr = (tp / pos_size) if pos_size else np.nan
        fpr = (fp / neg_size) if neg_size else np.nan
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    tpr_overall = (TP_all / POS_all) if POS_all else np.nan
    fpr_overall = (FP_all / NEG_all) if NEG_all else np.nan

    tpr_avg = float(np.nanmean(tpr_list)) if len(tpr_list) else np.nan
    fpr_avg = float(np.nanmean(fpr_list)) if len(fpr_list) else np.nan

    overall_tprs.append(tpr_overall)
    overall_fprs.append(fpr_overall)
    persrc_tprs.append(tpr_avg)
    persrc_fprs.append(fpr_avg)

    rows.append({
        "threshold": float(thr),
        "TP_total": int(TP_all),
        "FP_total": int(FP_all),
        "positives_total": int(POS_all),
        "negatives_total": int(NEG_all),
        "TPR_overall": float(tpr_overall) if not np.isnan(tpr_overall) else np.nan,
        "FPR_overall": float(fpr_overall) if not np.isnan(fpr_overall) else np.nan,
        "TPR_per_source_avg": float(tpr_avg) if not np.isnan(tpr_avg) else np.nan,
        "FPR_per_source_avg": float(fpr_avg) if not np.isnan(fpr_avg) else np.nan,
    })

    print(
        f"thr {thr:g} | "
        f"Overall: TPR={tpr_overall:.4f}, FPR={fpr_overall:.4f} | "
        f"Per-source avg: TPR={tpr_avg:.4f}, FPR={fpr_avg:.4f}"
    )

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print("Wrote per-threshold table:", OUT_CSV)

auc_overall = plot_roc_zoom(
    overall_fprs, overall_tprs,
    title="ROC curve from p-value thresholds (≤2 hops)\nOverall rate across all pairs",
    outpath=OUT_ROC_OVERALL,
)

auc_persrc = plot_roc_zoom(
    persrc_fprs, persrc_tprs,
    title="ROC curve from p-value thresholds (≤2 hops)\nAverage rate across perturbations",
    outpath=OUT_ROC_PERSRC,
)

print("\nAUC summary:")
print(f"  Overall rate AUC: {auc_overall:.3f}")
print(f"  Per-source average AUC: {auc_persrc:.3f}")
