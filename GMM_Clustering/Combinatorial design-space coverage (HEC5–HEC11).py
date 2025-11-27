# Fig. 1c — Class imbalance/overlap: UpSet-style element-set overlaps (Nature-ready)
# Drop-in: just run. Outputs: fig1/fig1c_upset_elements.png and .pdf

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

# -------------------- Literature lists (as given) --------------------
single_phase_lit = [
    "(CrHfMoNbTaTiVZr)C8",
    "(CrHfNbTaTiVWZr)C8",
    "(CrHfNbTaTi)C5",
    "(CrHfNbTaV)C5",
    "(CrHfTaTiZr)C5",
    "(CrMoNbTaW)C5",
    "(CrMoNbVW)C5",
    "(CrMoTaVW)C5",
    "(CrMoTiVW)C5",
    "(HfMoNbTaTi)C5",
    "(HfMoNbTaTiVWZr)C8",
    "(HfMoNbTaTiZr)C6",
    "(HfMoNbTaV)C5",
    "(HfMoNbTaZr)C5",
    "(HfMoTaTiZr)C5",
    "(HfNbTaTiV)C5",
    "(HfNbTaTiVZr)C6",
    "(HfNbTaTiW)C5",
    "(HfNbTaTiWZr)C6",
    "(HfNbTaTiZr)C5",
    "(HfNbTaVW)C5",
    "(HfTaTiWZr)C5",
    "(MoNbTaTiV)C5",
    "(MoNbTaTiW)C5",
    "(MoNbTaTiZr)C5",
    "(MoNbTaVW)C5",
    "(MoNbTaVZr)C5",
    "(NbTaTiVW)C5",
    "(NbTaTiWZr)C5",
    "(NbTaVWZr)C5",
    "(CrNbTaTiZr)C5",
    "(CrHfNbTaZr)C5",
    "(MoNbTaTiVW)C6",
    "(HfNbTaWZr)C5",
    "(MoNbTaWZr)C5", 
]

multi_phase_lit = [
    "(CrHfMoTiW)C5",
    "(CrHfMoVW)C5",
    "(CrHfTaVW)C5",
    "(CrHfTaWZr)C5",
    "(CrMoTiVZr)C5",
    "(CrMoTiWZr)C5",
    "(HfMoTaWZr)C5",
    "(HfMoTiWZr)C5",
    "(HfMoVWZr)C5",
    "(MoTiVWZr)C5",
]

# -------------------- Tunable display parameters --------------------
TOP_ELEMENTS   = 10   # use the N most frequent elements as "sets"
MIN_SET_SIZE   = 2    # keep intersections with at least this many elements
MAX_SET_SIZE   = 5    # ... and at most this many (to stay interpretable)
TOP_INTERSECT  = 50   # show top-N intersections by total count
DOT_SIZE       = 120  # matrix dot size (points^2)
LINE_WIDTH     = 1.2  # line width connecting dots in a row

# Okabe–Ito colorblind-safe palette
COL_SINGLE = "#0072B2"
COL_MULTI  = "#D55E00"
COL_GRID   = "#DDDDDD"

# -------------------- Helpers --------------------
def parse_elements(comp_str: str):
    m = re.search(r"\(([^)]+)\)", str(comp_str))
    if not m: return []
    return re.findall(r"[A-Z][a-z]?", m.group(1))

def make_df(lst, label):
    return pd.DataFrame({"Composition": lst, "Label": label})

# -------------------- Build dataframe --------------------
df = pd.concat([make_df(single_phase_lit, "Single"),
                make_df(multi_phase_lit,  "Multi")], ignore_index=True)
df["Elements"] = df["Composition"].apply(parse_elements)

# Element frequencies → choose focus sets
all_elems = pd.Series([e for row in df["Elements"] for e in row])
elem_counts = all_elems.value_counts()
focus_elems = list(elem_counts.head(TOP_ELEMENTS).index)

# Boolean matrix for focus elements
bool_rows = []
for i, row in df.iterrows():
    present = set(row["Elements"])
    bool_rows.append({e: (e in present) for e in focus_elems})
B = pd.DataFrame(bool_rows, index=df.index).astype(bool)
B["Label"] = df["Label"].values

# Group by boolean membership vector → counts per label
# Each composition maps to exactly one boolean vector over focus_elems
group_cols = focus_elems
grp = B.groupby(group_cols + ["Label"]).size().unstack("Label").fillna(0).astype(int)
# Attach total and intersection size
grp["Total"] = grp.sum(axis=1)
grp = grp[grp["Total"] > 0]
grp = grp.reset_index()

def size_from_row(r):
    return int(sum(bool(r[e]) for e in focus_elems))
grp["SetSize"] = grp.apply(size_from_row, axis=1)

# Keep interpretable intersections
grp = grp[(grp["SetSize"] >= MIN_SET_SIZE) & (grp["SetSize"] <= MAX_SET_SIZE)].copy()
if "Single" not in grp.columns: grp["Single"] = 0
if "Multi" not in grp.columns:  grp["Multi"]  = 0

# Sort and select top intersections
grp = grp.sort_values("Total", ascending=False).head(TOP_INTERSECT).reset_index(drop=True)

# Build a compact label for each intersection (e.g., "Hf·Nb·Ta·Ti")
def combo_label(r):
    members = [e for e in focus_elems if bool(r[e])]
    return "·".join(members)
grp["Combo"] = grp.apply(combo_label, axis=1)

# -------------------- Plot (custom UpSet: bars + membership matrix) --------------------
OUT = Path("fig1"); OUT.mkdir(parents=True, exist_ok=True)

# Prepare geometry
n_rows = len(grp)
n_cols = len(focus_elems)
y_pos = np.arange(n_rows)  # 0 (top) to n_rows-1 (bottom) – we'll invert axis later

fig = plt.figure(figsize=(8.6, 6.0), dpi=220)
gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.3, 2.2], wspace=0.05)

# Left: horizontal stacked bars (Single + Multi)
ax_bar = fig.add_subplot(gs[0, 0])
single_vals = grp["Single"].values
multi_vals  = grp["Multi"].values

ax_bar.barh(y_pos, multi_vals, color=COL_MULTI, edgecolor="white", linewidth=0.5, label="Multi")
ax_bar.barh(y_pos, single_vals, left=multi_vals, color=COL_SINGLE, edgecolor="white", linewidth=0.5, label="Single")

# Annotate totals at bar ends
totals = grp["Total"].values
for yi, t in zip(y_pos, totals):
    if t > 0:
        ax_bar.text(t + max(0.5, 0.02*totals.max()), yi, f"n={t}", va="center", ha="left", fontsize=8, color="#444444")

ax_bar.set_yticks(y_pos)
# Show compact labels on the left for readability
ax_bar.set_yticklabels(grp["Combo"].values, fontsize=8)
ax_bar.invert_yaxis()  # top = largest
ax_bar.grid(axis="x", linestyle=":", linewidth=0.6, color=COL_GRID)
ax_bar.set_xlabel("Literature count", fontsize=9)
ax_bar.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
ax_bar.set_title("Fig. 1c — Dominant element-set intersections", fontsize=10, pad=8)

# Right: membership matrix (dots + connecting lines)
ax_mat = fig.add_subplot(gs[0, 1])
ax_mat.set_xlim(-0.5, n_cols - 0.5)
ax_mat.set_ylim(-0.5, n_rows - 0.5)
ax_mat.invert_yaxis()
ax_mat.set_xticks(np.arange(n_cols))
ax_mat.set_xticklabels(focus_elems, rotation=45, ha="right", fontsize=8)
ax_mat.set_yticks([])

# Light vertical grid for elements
for xc in range(n_cols):
    ax_mat.axvline(x=xc, ymin=0, ymax=1, color=COL_GRID, lw=0.6, zorder=0)

# Draw lines connecting dots within each row (between min and max included columns)
for i, r in grp.iterrows():
    cols_in = [j for j, e in enumerate(focus_elems) if bool(r[e])]
    if len(cols_in) >= 2:
        # small horizontal segments between successive included columns at row y=i
        segments = [((cols_in[k], i), (cols_in[k+1], i)) for k in range(len(cols_in)-1)]
        lc = LineCollection(segments, colors="black", linewidths=LINE_WIDTH, zorder=2)
        ax_mat.add_collection(lc)

# Draw dots where element is included
for i, r in grp.iterrows():
    for j, e in enumerate(focus_elems):
        if bool(r[e]):
            ax_mat.scatter(j, i, s=DOT_SIZE, c="black", zorder=3, marker="o")
        else:
            # faint hollow small marker for context (optional, comment to hide)
            ax_mat.scatter(j, i, s=25, facecolors="none", edgecolors="#BBBBBB", linewidths=0.6, zorder=1)

# Add a small side color stripe showing fraction Single (visual cue)
# (optional, can be commented)
frac_single = np.where(totals>0, single_vals / totals, np.nan)
for yi, fs in zip(y_pos, frac_single):
    if np.isfinite(fs):
        ax_mat.add_patch(plt.Rectangle((n_cols-0.25, yi-0.4), 0.2, 0.8,
                                       color=COL_SINGLE, alpha=0.1 + 0.7*fs, lw=0))

ax_mat.set_xlim(-0.5, n_cols - 0.3)  # leave space for stripe
ax_mat.set_title("Membership matrix (columns = elements in focus)", fontsize=9, pad=6)

plt.tight_layout()
png = OUT / "fig1c_upset_elements.png"
pdf = OUT / "fig1c_upset_elements.pdf"
plt.savefig(png, dpi=300, bbox_inches="tight")
plt.savefig(pdf, dpi=300, bbox_inches="tight")
plt.close(fig)

# --------- Also print a compact table ---------
top_tbl = grp[["Combo", "SetSize", "Single", "Multi", "Total"]].copy()
top_tbl["FracSingle"] = np.where(top_tbl["Total"]>0, top_tbl["Single"]/top_tbl["Total"], np.nan)
print("\nTop intersections among focus elements:")
print(top_tbl.to_string(index=False))

print(f"\nSaved: {png.resolve()}\n       {pdf.resolve()}")
