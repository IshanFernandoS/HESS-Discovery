# ===================== Enhanced Fig. 3b — Feature-band coverage heatmap =====================
# Assumes: OUT, dfX, bands_df, band_lookup (or rebuild), pool9_ranked, plot_feats_unique, selected
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from pathlib import Path

# ---- Output dir ----
FIG_DIR = OUT / "fig3"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- Config (kept identical) ----
TOP_ROWS   = 30            # rows = top-N HEC9 by Score
MAX_FEATS  = None          # None = use all descriptors in bands; or cap (e.g., 16)
ANNOTATE_SELECTED = True   # add left stripe to mark your selected set

# ---- Utility ----
def short_comp(s: str) -> str:
    s = str(s)
    for k in [")C5",")C6",")C7",")C8",")C9",")C10",")C11"]:
        s = s.replace(k, "")
    return s

# Columns (features) to display = raw axis names from the bands (keep original order)
all_feats = [raw for (_, raw) in plot_feats_unique]
feats = all_feats[:MAX_FEATS] if MAX_FEATS is not None else all_feats[:]
if len(feats) == 0:
    raise RuntimeError("No features available for the feature-band heatmap.")

# Top-N rows by Score
top_df = pool9_ranked.sort_values("Score", ascending=False).head(min(TOP_ROWS, len(pool9_ranked))).copy()
rowsN, colsN = len(top_df), len(feats)

# Build matrices: M = bin-mean posterior; Fav = favorable-band mask; Used = bin exists
M    = np.full((rowsN, colsN), np.nan, dtype=float)
Fav  = np.zeros((rowsN, colsN), dtype=bool)
Used = np.zeros((rowsN, colsN), dtype=bool)

# Rebuild band_lookup if missing
if "band_lookup" not in globals():
    band_lookup = {}
    for f, grp in bands_df.groupby("feature"):
        intervals = pd.IntervalIndex.from_arrays(grp["bin_left"].values, grp["bin_right"].values, closed="right")
        band_lookup[f] = {
            "intervals": intervals,
            "is_fav": grp["is_favorable"].to_numpy(),
            "pmean": grp["bin_pmean"].to_numpy(),
        }

# Fill matrices
for i, comp in enumerate(top_df["Composition"]):
    r = dfX[dfX["Composition"] == comp].iloc[0]
    for j, f in enumerate(feats):
        if f not in dfX.columns or f not in band_lookup:
            continue
        val = pd.to_numeric(r.get(f), errors="coerce")
        if pd.isna(val):
            continue
        L = band_lookup[f]
        bidx = L["intervals"].get_indexer([val])[0]
        if bidx == -1:
            continue
        M[i, j]    = float(L["pmean"][bidx])
        Fav[i, j]  = bool(L["is_fav"][bidx])
        Used[i, j] = True

# Row summaries for the right-side bars
row_hits     = Fav.sum(axis=1)
row_used     = Used.sum(axis=1)
row_cov      = np.divide(row_used, colsN, out=np.zeros_like(row_used, dtype=float), where=(colsN > 0))
row_p_single = top_df["P_single_cal"].astype(float).to_numpy()
row_score    = top_df["Score"].astype(float).to_numpy()

# Column summaries (top bar)
col_fav_rate  = np.where(Used.sum(axis=0) > 0, Fav.sum(axis=0) / Used.sum(axis=0), 0.0)
col_mean_binP = np.nanmean(M, axis=0)

# Selected mask for left stripe
sel_set  = set(selected) if ANNOTATE_SELECTED else set()
sel_mask = np.array([c in sel_set for c in top_df["Composition"]], dtype=bool)

# ---- Figure layout ----
# Gridspec: top bar (feature summary), main heatmap, left stripe, right bars
wL, wH, wR = (0.25, 10.0, 2.8)
hT, hM     = (1.1, max(2.5, 0.35 * rowsN))
fig = plt.figure(figsize=(wL + wH + wR, hT + hM), dpi=300)
gs  = gridspec.GridSpec(2, 3, width_ratios=[wL, wH, wR], height_ratios=[hT, hM], wspace=0.15, hspace=0.15)

# (Top) feature summary bars
ax_top = fig.add_subplot(gs[0, 1])
Xpos = np.arange(colsN)
ax_top.bar(Xpos, col_fav_rate, alpha=0.85, label="Favorable-band rate")
ax_top.plot(Xpos, col_mean_binP, marker="o", lw=1.0, label="Mean bin-mean P(single)")
ax_top.set_ylim(0, 1.05)
ax_top.set_xticks(Xpos)
ax_top.set_xticklabels([])  # labels appear on the heatmap axis
ax_top.set_ylabel("Feature summary")
ax_top.set_title("Fig. 3b (top): per-feature support across top-30 HEC9")
ax_top.legend(loc="upper right", fontsize=8, frameon=True)

# (Main left) selection stripe — label ONLY selected rows
SELECTED_LABEL_COLOR = "#1b7837"  # deep green
ax_left = fig.add_subplot(gs[1, 0])
ax_left.imshow(
    sel_mask[:, None],
    aspect="auto",
    cmap=mpl.colors.ListedColormap(["#e0e0e0", SELECTED_LABEL_COLOR]),
    vmin=0,
    vmax=1,
)
ax_left.set_yticks(np.arange(rowsN))
ax_left.set_yticklabels([])  # clean; labels drawn manually for selected
ax_left.set_xticks([])
ax_left.set_title("Selected", fontsize=9)
for spine in ["top", "right", "bottom", "left"]:
    ax_left.spines[spine].set_visible(False)

for i, comp in enumerate(top_df["Composition"]):
    if comp in sel_set:
        ax_left.annotate(
            short_comp(comp),
            xy=(0.0, i), xycoords=("axes fraction", "data"),
            xytext=(-6, 0), textcoords="offset points",
            ha="right", va="center",
            fontsize=8, fontweight="bold",
            color=SELECTED_LABEL_COLOR,
            clip_on=False,
        )

# (Main center) heatmap + favorable-band squares
ax = fig.add_subplot(gs[1, 1])
im = ax.imshow(M, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="viridis")
ii, jj = np.where(Fav)
ax.scatter(jj, ii, marker="s", s=30, facecolors="none", edgecolors="w", linewidths=0.9)
ax.set_yticks(np.arange(rowsN))
ax.set_yticklabels([])  # row labels already on left stripe
ax.set_xticks(np.arange(colsN))
ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
ax.set_title("Fig. 3b — Feature-band coverage: bin-mean P(single)   (white squares = favorable bands)")
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.set_label("Bin-mean P(single)")

# (Main right) row summary bars (Score, P(single), coverage)
def _minmax(z):
    z = np.asarray(z, float)
    m, M = np.nanmin(z), np.nanmax(z)
    if not np.isfinite(m) or not np.isfinite(M) or M - m < 1e-12:
        return np.zeros_like(z)
    return (z - m) / (M - m)

ax_right = fig.add_subplot(gs[1, 2])
y = np.arange(rowsN)
bar_h = 0.22
ax_right.barh(y + 0.35, _minmax(row_score), height=bar_h, label="Score", alpha=0.9)
ax_right.barh(y + 0.00, row_p_single,       height=bar_h, label="P(single)", alpha=0.9)
ax_right.barh(y - 0.35, row_cov,            height=bar_h, label="Coverage",  alpha=0.9)
ax_right.set_yticks(y)
ax_right.set_yticklabels([])
ax_right.set_xlim(0, 1.05)
ax_right.invert_yaxis()
ax_right.set_xlabel("Row summaries (normalized)")
ax_right.set_title("Row summaries")
ax_right.legend(loc="lower right", fontsize=7, frameon=True)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig3b_explainable_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---- Save CSVs (kept filenames identical) ----
pd.DataFrame(M, index=top_df["Composition"], columns=feats) \
  .to_csv(FIG_DIR / "fig3b_binmean_matrix.csv", float_format="%.6g")

pd.DataFrame(Fav.astype(int),  index=top_df["Composition"], columns=feats) \
  .to_csv(FIG_DIR / "fig3b_favorable_mask.csv")
pd.DataFrame(Used.astype(int), index=top_df["Composition"], columns=feats) \
  .to_csv(FIG_DIR / "fig3b_used_mask.csv")

row_summary = pd.DataFrame({
    "Composition":   top_df["Composition"].values,
    "Score":         row_score,
    "P_single_cal":  row_p_single,
    "Coverage":      row_cov,
    "FavorableHits": row_hits,
    "FeaturesUsed":  row_used,
    "Selected":      sel_mask.astype(int),
})
row_summary.to_csv(FIG_DIR / "fig3b_row_summary.csv", index=False)

col_summary = pd.DataFrame({
    "Feature":            feats,
    "FavorableBandRate":  col_fav_rate,
    "MeanBinMeanP":       col_mean_binP,
})
col_summary.to_csv(FIG_DIR / "fig3b_column_summary.csv", index=False)

records = []
for i, comp in enumerate(top_df["Composition"]):
    for j, feat in enumerate(feats):
        if Used[i, j]:
            records.append({
                "Composition": comp,
                "Feature":     feat,
                "BinMeanP":    M[i, j],
                "IsFavorable": bool(Fav[i, j]),
            })
pd.DataFrame(records).to_csv(FIG_DIR / "fig3b_long.csv", index=False)

# Also save the exact top-N table you plotted
top_df.to_csv(FIG_DIR / "fig3b_topN_table.csv", index=False)