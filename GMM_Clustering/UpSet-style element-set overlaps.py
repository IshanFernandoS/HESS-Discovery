# ============================ Fig. 3d — Ranking robustness / sensitivity ============================
# Assumes these exist from earlier cells:
#   OUT (Path), dfX, plot_feats_unique, pool9_ranked, selected, TARGET_N_METALS
# Also uses weights & zscore from earlier; safe fallbacks added without changing behavior.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path

# ---- Guards & paths ----
_required = ["OUT", "dfX", "plot_feats_unique", "pool9_ranked", "selected", "TARGET_N_METALS"]
_missing = [v for v in _required if v not in globals()]
if _missing:
    raise RuntimeError(f"Missing variables for Fig. 3d: {_missing}. Run the GMM + bands + ranking cells first.")

FIG_DIR = OUT / "fig3"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- Weights (use prior if set; otherwise identical fallbacks) ----
W_CUM  = globals().get("W_CUM",  0.33)
W_POST = globals().get("W_POST", 0.33)
W_COV  = globals().get("W_COV",  0.34)

# ---- z-score helper: identical policy to earlier blocks ----
try:
    zscore  # defined upstream
except NameError:
    def zscore(series: pd.Series) -> pd.Series:
        x = pd.to_numeric(series, errors="coerce").astype(float)
        mu = np.nanmean(x); sd = np.nanstd(x)
        if not np.isfinite(sd) or sd < 1e-12:
            return (x - mu)  # all zeros when constant
        return (x - mu) / sd

# ----- Helper: re-rank with arbitrary band hyperparameters -----
def rank_with_params(n_bins: int, favor_q: float, roll_smooth: int) -> pd.DataFrame:
    # Build posterior–feature bands
    bands = []
    for (_, feat_col) in plot_feats_unique:
        x = pd.to_numeric(dfX[feat_col], errors="coerce")
        p = dfX["P_single_cal"].astype(float)
        ok = ~(x.isna() | p.isna())
        x, p = x[ok], p[ok]

        q = pd.qcut(pd.Series(x).astype(float), q=n_bins, duplicates="drop")
        tmp = pd.DataFrame({"x": x.values, "p": p.values, "q": q})
        g = tmp.groupby("q", observed=True)
        bin_mid = g["x"].mean()
        bin_p   = g["p"].mean()

        if roll_smooth and roll_smooth > 1:
            bin_p = bin_p.rolling(roll_smooth, center=True, min_periods=1).mean()

        k_top  = max(1, int(np.ceil(len(bin_p) * favor_q)))
        top_ix = bin_p.sort_values(ascending=False).index[:k_top]
        is_fav = bin_p.index.isin(top_ix)

        bands.append(pd.DataFrame({
            "feature":   feat_col,
            "bin_left":  [iv.left  for iv in bin_mid.index],
            "bin_right": [iv.right for iv in bin_mid.index],
            "bin_pmean": bin_p.values,
            "is_fav":    is_fav.astype(bool),
        }))

    bands_df = pd.concat(bands, ignore_index=True)

    # Fast lookup: feature value → bin
    lookup = {}
    for f, grp in bands_df.groupby("feature"):
        intervals = pd.IntervalIndex.from_arrays(
            grp["bin_left"].values, grp["bin_right"].values, closed="right"
        )
        lookup[f] = {
            "intervals": intervals,
            "is_fav": grp["is_fav"].to_numpy(),
            "pmean":  grp["bin_pmean"].to_numpy(),
        }

    # Score each composition
    rows = []
    for _, r in dfX.iterrows():
        pmeans_in_bin, hits, used = [], 0, 0
        for (_, feat_col) in plot_feats_unique:
            val = r.get(feat_col, np.nan)
            if pd.isna(val):
                continue
            L = lookup[feat_col]
            bidx = L["intervals"].get_indexer([val])[0]
            if bidx == -1:
                continue
            used += 1
            hits += 1 if L["is_fav"][bidx] else 0
            pmeans_in_bin.append(L["pmean"][bidx])

        rows.append({
            "Composition":   r["Composition"],
            "n_metals":      float(r["n_metals"]),
            "P_single_cal":  float(r["P_single_cal"]),
            "n_feat_used":   int(used),
            "cum_pmean":     float(np.nansum(pmeans_in_bin)) if used else np.nan,
            "bin_pmean_avg": float(np.nanmean(pmeans_in_bin)) if used else np.nan,
            "hits_count":    int(hits),
        })

    scores = pd.DataFrame(rows)
    pool   = scores[scores["n_metals"] == float(TARGET_N_METALS)].copy()

    # Coverage + composite Score (same weights as main text)
    n_total_feats = max(1, len(plot_feats_unique))
    pool["coverage"]       = pool["n_feat_used"] / float(n_total_feats)
    pool["cum_pmean_z"]    = zscore(pool["cum_pmean"])
    pool["P_single_cal_z"] = zscore(pool["P_single_cal"])
    pool["coverage_z"]     = zscore(pool["coverage"])
    pool["Score"] = (
        W_CUM  * pool["cum_pmean_z"] +
        W_POST * pool["P_single_cal_z"] +
        W_COV  * pool["coverage_z"]
    )

    ranked = pool.sort_values(
        by=["Score", "cum_pmean", "bin_pmean_avg"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    ranked["Rank"] = np.arange(1, len(ranked) + 1)
    return ranked

# ----- Hyperparameter grid -----
BINS_LIST  = [15, 20, 25]
FAVOR_LIST = [0.25, 0.30, 0.35]
ROLL_LIST  = [1, 3, 5]
grid = [(nb, fq, rs) for nb, fq, rs in product(BINS_LIST, FAVOR_LIST, ROLL_LIST)]

# Baseline top-30 (from the run you already executed above)
baseline_top30 = pool9_ranked["Composition"].head(30).tolist()
sel_set = set(selected)

# Compute ranks for every setting
rank_maps = {}   # key -> {comp: rank}
top10_sets = {}  # key -> set of comps in top-10
keys = []
for (nb, fq, rs) in grid:
    key = f"B{nb}_Q{int(100*fq):02d}_R{rs}"
    keys.append(key)
    ranked = rank_with_params(nb, fq, rs)
    rank_maps[key] = dict(zip(ranked["Composition"], ranked["Rank"]))
    top10_sets[key] = set(ranked["Composition"].head(10))

# ============ Panel 1: fraction-of-settings staying top-10 (baseline top-30 only) ============
frac = []
is_sel = []
for comp in baseline_top30:
    count = sum(comp in top10_sets[k] for k in keys)
    frac.append(count / len(keys))
    is_sel.append(comp in sel_set)

robust_df = pd.DataFrame({
    "Composition": baseline_top30,
    "top10_fraction": frac,
    "is_selected": is_sel,
})

def short_comp(s: str) -> str:
    s = str(s)
    for k in [")C5",")C6",")C7",")C8",")C9",")C10",")C11"]:
        s = s.replace(k, "")
    return s

# Plot bars
N = len(robust_df)
x = np.arange(N)
fig_w = max(10, 0.35 * N)
fig_h = 4.0
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

# Shade selected
for i, sel in enumerate(robust_df["is_selected"]):
    if sel:
        ax.axvspan(i - 0.5, i + 0.5, color="#2ca02c", alpha=0.12, lw=0)

bars = ax.bar(x, robust_df["top10_fraction"].values)
# Emphasize selected with edge
for i, sel in enumerate(robust_df["is_selected"]):
    if sel:
        bars[i].set_edgecolor("k")
        bars[i].set_linewidth(1.2)

ax.set_ylim(0, 1.0)
ax.set_ylabel("Fraction of settings where candidate is top-10")
ax.set_xticks(x)
ax.set_xticklabels([short_comp(s) for s in robust_df["Composition"]], rotation=45, ha="right", fontsize=8)
ax.set_title("Fig. 3d — Ranking robustness across band hyperparameters")
ax.axhline(0.5, color="k", lw=0.7, ls="--", alpha=0.6)

plt.tight_layout()
out_png1 = FIG_DIR / "fig3d_top10_fraction_bar.png"
plt.savefig(out_png1, dpi=300, bbox_inches="tight")
plt.close(fig)

# Save the table too (for SI)
robust_df.to_csv(FIG_DIR / "fig3d_top10_fraction_bar_values.csv", index=False)
print("Saved Fig. 3d (top-10 stability bars) →", out_png1.resolve())

# ============ Panel 2: rank vs setting (selected + near ties) ============
# Choose: your selections + next best non-selected to fill ~10–12 lines
near_ties = [c for c in baseline_top30 if c not in sel_set][:6]  # first six non-selected in baseline top-30
focus = [c for c in selected if c in baseline_top30] + near_ties
focus = list(dict.fromkeys(focus))  # unique, preserve order

# Collect ranks across settings
rank_matrix = []
for comp in focus:
    ranks = [rank_maps[k].get(comp, np.nan) for k in keys]
    rank_matrix.append(ranks)

rank_df = pd.DataFrame(rank_matrix, index=focus, columns=keys)

# Plot lines: lower is better (rank 1 at top), so invert y
fig_h = 4.6
fig_w = max(10, 0.42 * len(keys))
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

xx = np.arange(len(keys))
for comp in focus:
    yy = rank_df.loc[comp].values.astype(float)
    mask = np.isfinite(yy)
    if comp in sel_set:
        ax.plot(xx[mask], yy[mask], marker="o", lw=2.2, label=short_comp(comp))
    else:
        ax.plot(xx[mask], yy[mask], marker="o", lw=1.0, alpha=0.7, label=short_comp(comp))

ax.set_xlim(-0.5, len(keys) - 0.5)
ax.set_ylim(rank_df.min().min() - 1, rank_df.max().max() + 1)
ax.invert_yaxis()  # rank 1 at top
ax.set_xlabel("Hyperparameter setting")
ax.set_ylabel("Rank (lower is better)")
ax.set_title("Fig. 3d — Rank vs. setting (selections highlighted)")
ax.set_xticks(xx)
ax.set_xticklabels(keys, rotation=60, ha="right", fontsize=7)

# Legend outside if many lines
ax.legend(title="Compositions", bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
plt.tight_layout()
out_png2 = FIG_DIR / "fig3d_rank_vs_setting_selected.png"
plt.savefig(out_png2, dpi=300, bbox_inches="tight")
plt.close(fig)

# Save ranks table
rank_df.to_csv(FIG_DIR / "fig3d_rank_vs_setting_selected_values.csv")
print("Saved Fig. 3d (rank vs setting lines) →", out_png2.resolve())