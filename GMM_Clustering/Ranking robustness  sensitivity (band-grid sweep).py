# ============================ Fig. 3d — Ranking robustness / sensitivity ============================
# Expects from earlier cells: OUT (Path), dfX, plot_feats_unique, pool9_ranked, selected, TARGET_N_METALS
# Uses W_CUM, W_POST, W_COV and the same z-score policy as your HEC9 logic.
# Note: Behavior and outputs are unchanged; comments/variable names are tidied for readability.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path

# ---- Output dir ----
fig3 = OUT / "fig3"
fig3.mkdir(parents=True, exist_ok=True)

# ---- Safe z-score (vectorized); returns zeros when std ~ 0 (same policy as earlier) ----
def _zscore(arr_like):
    a = np.asarray(arr_like, dtype=float)
    m = np.nanmean(a)
    s = np.nanstd(a)
    if not np.isfinite(s) or s < 1e-15:
        return (a - m)  # all zeros if constant
    return (a - m) / s

# ----- Re-rank with arbitrary band hyperparameters (n_bins, favor_q, roll_smooth) -----
def rank_with_params(n_bins, favor_q, roll_smooth):
    # Build posterior–feature bands from dfX for the features used in clustering
    bands_tables = []
    for _, feat_col in plot_feats_unique:
        x = pd.to_numeric(dfX[feat_col], errors="coerce")
        p = dfX["P_single_cal"].astype(float)
        ok = ~(x.isna() | p.isna())
        x, p = x[ok], p[ok]

        # Quantile bins; drop duplicates if data are clumpy
        q = pd.qcut(pd.Series(x).astype(float), q=n_bins, duplicates="drop")
        tmp = pd.DataFrame({"x": x.values, "p": p.values, "q": q})
        g = tmp.groupby("q", observed=True)
        bin_mid = g["x"].mean()
        bin_p   = g["p"].mean()

        # Optional centered smoothing over adjacent bins
        if roll_smooth and roll_smooth > 1:
            bin_p = bin_p.rolling(roll_smooth, center=True, min_periods=1).mean()

        # Favor the top-q fraction of bins by mean P(single)
        k_top   = max(1, int(np.ceil(len(bin_p) * favor_q)))
        top_idx = bin_p.sort_values(ascending=False).index[:k_top]
        is_fav  = bin_p.index.isin(top_idx)

        bands_tables.append(pd.DataFrame({
            "feature":   feat_col,
            "bin_left":  [iv.left  for iv in bin_mid.index],
            "bin_right": [iv.right for iv in bin_mid.index],
            "bin_pmean": bin_p.values,
            "is_fav":    is_fav.astype(bool),
        }))

    bands_df = pd.concat(bands_tables, ignore_index=True)

    # Fast lookup: feature value → bin index → (is_fav, bin_pmean)
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

    # Score each composition (no EFA in this block)
    rows = []
    for _, r in dfX.iterrows():
        pmeans_in_bin, hits, used = [], 0, 0
        for _, feat_col in plot_feats_unique:
            val = r.get(feat_col, np.nan)
            if pd.isna(val):
                continue
            L = lookup.get(feat_col)
            if L is None:
                continue
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

    # Coverage + composite Score (weights as in main text)
    n_total_feats = max(1, len(plot_feats_unique))
    pool["coverage"]       = pool["n_feat_used"] / float(n_total_feats)
    pool["cum_pmean_z"]    = _zscore(pool["cum_pmean"])
    pool["P_single_cal_z"] = _zscore(pool["P_single_cal"])
    pool["coverage_z"]     = _zscore(pool["coverage"])
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

# ----- Hyperparameter grid to sweep -----
BINS_LIST  = [15, 20, 25]
FAVOR_LIST = [0.25, 0.30, 0.35]
ROLL_LIST  = [1, 3, 5]
grid = [(nb, fq, rs) for nb, fq, rs in product(BINS_LIST, FAVOR_LIST, ROLL_LIST)]

# Baseline: top-30 from your previously ranked HEC9 pool
baseline_top30 = pool9_ranked["Composition"].head(30).tolist()
selected_set   = set(selected)

# Compute ranks for each setting
ranks_by_setting           = {}  # key -> {composition: rank}
top10_set_by_setting       = {}  # key -> set of top-10 compositions
ranked_tables_by_setting   = {}  # key -> ranked DataFrame
setting_keys               = []
for nb, fq, rs in grid:
    key = f"B{nb}_Q{int(100*fq):02d}_R{rs}"
    setting_keys.append(key)
    ranked_tbl = rank_with_params(nb, fq, rs)
    ranks_by_setting[key]     = dict(zip(ranked_tbl["Composition"], ranked_tbl["Rank"]))
    top10_set_by_setting[key] = set(ranked_tbl["Composition"].head(10))
    ranked_tables_by_setting[key] = ranked_tbl

# ============ Panel 1: fraction of settings where baseline top-30 remain top-10 ============
top10_frac = []
is_selected_flag = []
for comp in baseline_top30:
    count = sum(comp in top10_set_by_setting[k] for k in setting_keys)
    top10_frac.append(count / len(setting_keys))
    is_selected_flag.append(comp in selected_set)

robust_df = pd.DataFrame({
    "Composition":    baseline_top30,
    "top10_fraction": top10_frac,
    "is_selected":    is_selected_flag,
})

def short_comp(s):
    s = str(s)
    for tag in [")C5",")C6",")C7",")C8",")C9",")C10",")C11"]:
        s = s.replace(tag, "")
    return s

# Bar plot
N = len(robust_df)
x = np.arange(N)
fig_w = max(10, 0.35 * N)
fig_h = 4.0
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

# Shade selections
for i, sel in enumerate(robust_df["is_selected"]):
    if sel:
        ax.axvspan(i - 0.5, i + 0.5, color="#2ca02c", alpha=0.12, lw=0)

bars = ax.bar(x, robust_df["top10_fraction"].values)
# Outline selected
for i, sel in enumerate(robust_df["is_selected"]):
    if sel:
        bars[i].set_edgecolor("k")
        bars[i].set_linewidth(1.2)

ax.set_ylim(0, 1.0)
ax.set_ylabel("Fraction of settings where candidate is top-10")
ax.set_xticks(x)
ax.set_xticklabels([short_comp(s) for s in robust_df["Composition"]],
                   rotation=45, ha="right", fontsize=8)
ax.set_title("Fig. 3d — Ranking robustness across band hyperparameters")
ax.axhline(0.5, color="k", lw=0.7, ls="--", alpha=0.6)

plt.tight_layout()
out_png1 = fig3 / "fig3d_top10_fraction_bar.png"
plt.savefig(out_png1, dpi=300, bbox_inches="tight")
plt.close(fig)

robust_df.to_csv(fig3 / "fig3d_top10_fraction_bar_values.csv", index=False)
print("Saved Fig. 3d (top-10 stability bars) →", out_png1.resolve())

# ============ Panel 2: rank vs setting (your selections + near ties) ============
near_ties = [c for c in baseline_top30 if c not in selected_set][:6]  # first six non-selected
focus_list = [c for c in selected if c in baseline_top30] + near_ties
focus_list = list(dict.fromkeys(focus_list))  # keep order, make unique

# Build rank matrix for focus lines
rank_matrix = []
for comp in focus_list:
    rank_series = [ranks_by_setting[k].get(comp, np.nan) for k in setting_keys]
    rank_matrix.append(rank_series)

rank_df = pd.DataFrame(rank_matrix, index=focus_list, columns=setting_keys)

# Line plot (lower rank is better → invert y)
fig_h = 4.6
fig_w = max(10, 0.42 * len(setting_keys))
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

xx = np.arange(len(setting_keys))
for comp in focus_list:
    yy = rank_df.loc[comp].values.astype(float)
    mask = np.isfinite(yy)
    if comp in selected_set:
        ax.plot(xx[mask], yy[mask], marker="o", lw=2.2, label=short_comp(comp))
    else:
        ax.plot(xx[mask], yy[mask], marker="o", lw=1.0, alpha=0.7, label=short_comp(comp))

ax.set_xlim(-0.5, len(setting_keys) - 0.5)
ax.set_ylim(rank_df.min().min() - 1, rank_df.max().max() + 1)
ax.invert_yaxis()
ax.set_xlabel("Hyperparameter setting")
ax.set_ylabel("Rank (lower is better)")
ax.set_title("Fig. 3d — Rank vs. setting (selections highlighted)")
ax.set_xticks(xx)
ax.set_xticklabels(setting_keys, rotation=60, ha="right", fontsize=7)
ax.legend(title="Compositions", bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)

plt.tight_layout()
out_png2 = fig3 / "fig3d_rank_vs_setting_selected.png"
plt.savefig(out_png2, dpi=300, bbox_inches="tight")
plt.close(fig)

rank_df.to_csv(fig3 / "fig3d_rank_vs_setting_selected_values.csv")
print("Saved Fig. 3d (rank vs setting lines) →", out_png2.resolve())

# ------------------------------ Reproducibility & audit outputs ------------------------------
# (a) Hyperparameter grid map
hp_rows = [{"key": f"B{nb}_Q{int(100*fq):02d}_R{rs}", "n_bins": nb, "favor_q": fq, "roll_smooth": rs}
           for (nb, fq, rs) in grid]
hp_df = pd.DataFrame(hp_rows)
hp_df.to_csv(fig3 / "fig3d_hyperparam_grid.csv", index=False)

# (b) Baseline lists used
pd.DataFrame({"Composition": baseline_top30}).to_csv(fig3 / "fig3d_baseline_top30.csv", index=False)
pd.DataFrame({"Composition": sorted(selected_set)}).to_csv(fig3 / "fig3d_selected_list.csv", index=False)

# (c) Per-setting ranked tables + top-10 (ordered)
for k in setting_keys:
    ranked_tables_by_setting[k].to_csv(fig3 / f"fig3d_ranked_{k}.csv", index=False)
    ranked_tables_by_setting[k].head(10)[["Composition", "Rank", "Score"]].to_csv(
        fig3 / f"fig3d_top10_{k}.csv", index=False
    )

# (d) Full rank matrix (all compositions × all settings)
all_comps = sorted({c for mapping in ranks_by_setting.values() for c in mapping.keys()})
full_rank_mat = pd.DataFrame({k: [ranks_by_setting[k].get(c, np.nan) for c in all_comps]
                              for k in setting_keys},
                             index=all_comps)
full_rank_mat.index.name = "Composition"
full_rank_mat.to_csv(fig3 / "fig3d_rank_matrix_all.csv")

# (e) Long table: every composition, every setting, with hyperparameters attached
long_tables = []
for k in setting_keys:
    df_k = ranked_tables_by_setting[k].copy()
    df_k["setting"] = k
    long_tables.append(df_k)
rank_long = pd.concat(long_tables, ignore_index=True)
rank_long = rank_long.merge(hp_df, left_on="setting", right_on="key", how="left").drop(columns=["key"])
rank_long.to_csv(fig3 / "fig3d_rank_long.csv", index=False)

# (f) Focus list used in Panel 2
pd.DataFrame({"Composition": focus_list,
              "is_selected": [c in selected_set for c in focus_list]}).to_csv(
    fig3 / "fig3d_focus_list.csv", index=False
)