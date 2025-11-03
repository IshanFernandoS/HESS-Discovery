# Fig. 1c — Element TRIPLETS (k=3): counts & plot + CSVs

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from itertools import combinations
from collections import defaultdict

# -------------------- Literature lists (as given) --------------------
single_phase_lit  = ['(CrHfMoNbTaTiVZr)C8','(CrHfNbTaTiVWZr)C8','(CrHfNbTaTi)C5','(CrHfNbTaV)C5',
'(CrHfTaTiZr)C5','(CrMoNbTaW)C5','(CrMoNbVW)C5','(CrMoTaVW)C5','(CrMoTiVW)C5',
'(HfLaMoNbTaTiWZr)C8','(HfLaNbTaTiYZr)C7','(HfLaTaTiYZr)C6','(HfMoNbTaTi)C5',
'(HfMoNbTaTiVWZr)C8','(HfMoNbTaTiWZr)C7','(HfMoNbTaTiZr)C6','(HfMoNbTaV)C5','(HfMoNbTaZr)C5',
'(HfMoTaTiZr)C5','(HfNbTaTiV)C5','(HfNbTaTiVWZr)C7','(HfNbTaTiVZr)C6','(HfNbTaTiW)C5',
'(HfNbTaTiWZr)C6','(HfNbTaTiZr)C5','(HfNbTaVW)C5','(HfNbTaVZr)C5','(HfNbTiVZr)C5',
'(HfTaTiVZr)C5','(HfTaTiWZr)C5','(MoNbTaTiV)C5','(MoNbTaTiW)C5','(MoNbTaTiZr)C5',
'(MoNbTaVW)C5','(MoNbTaVZr)C5','(NbTaTiVW)C5','(NbTaTiVZr)C5','(NbTaTiWZr)C5','(NbTaVWZr)C5']
multi_phase_lit   = ['(CrHfMoTiW)C5','(CrHfMoVW)C5','(CrHfMoWZr)C5','(CrHfNbVW)C5','(CrHfNbWZr)C5',
'(CrHfTaVW)C5','(CrHfTaWZr)C5','(CrHfTiVW)C5','(CrHfTiWZr)C5','(CrHfVWZr)C5',
'(CrMoTiVZr)C5','(CrMoTiWZr)C5','(CrNbTiWZr)C5','(CrNbVWZr)C5','(CrTaTiWZr)C5',
'(CrTaVWZr)C5','(CrTiVWZr)C5','(HfMoTaWZr)C5','(HfMoTiWZr)C5','(HfMoVWZr)C5','(MoTiVWZr)C5']

# -------------------- Tunables --------------------
TOP_ELEMENTS   = 10    # columns: most frequent elements
TOP_TRIPLETS   = 50    # how many triplets to display/save (set None to keep all)
DOT_SIZE       = 120
LINE_WIDTH     = 1.2

# Okabe–Ito palette
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

# Element frequencies (to pick focus columns)
all_elems = pd.Series([e for row in df["Elements"] for e in row])
elem_counts = all_elems.value_counts()
focus_elems = list(elem_counts.head(TOP_ELEMENTS).index)

# -------------------- Count TRIPLETS explicitly --------------------
def canon_triplet(a, b, c):
    return tuple(sorted((a, b, c)))

counts = defaultdict(lambda: {"Single": 0, "Multi": 0})
for _, row in df.iterrows():
    elems = list(set(row["Elements"]))  # unique per composition
    lbl = row["Label"]
    for a, b, c in combinations(elems, 3):
        t = canon_triplet(a, b, c)
        counts[t][lbl] += 1

# Convert to DataFrame
rows = []
for (a, b, c), d in counts.items():
    total = d.get("Single", 0) + d.get("Multi", 0)
    rows.append({
        "ElemA": a, "ElemB": b, "ElemC": c,
        "Single": d.get("Single", 0),
        "Multi":  d.get("Multi", 0),
        "Total":  total
    })
trip = pd.DataFrame(rows)
trip = trip[trip["Total"] > 0]

# Keep only triplets fully inside the focus_elems (clean matrix columns)
trip = trip[
    trip["ElemA"].isin(focus_elems) &
    trip["ElemB"].isin(focus_elems) &
    trip["ElemC"].isin(focus_elems)
].copy()

# Build labels and sort (respect focus_elems order)
order = {e:i for i, e in enumerate(focus_elems)}
def combo_label(a, b, c):
    aa, bb, cc = sorted([a, b, c], key=lambda x: order.get(x, 1e9))
    return f"{aa}·{bb}·{cc}"

trip["Combo"] = trip.apply(lambda r: combo_label(r["ElemA"], r["ElemB"], r["ElemC"]), axis=1)
trip["FracSingle"] = np.where(trip["Total"]>0, trip["Single"]/trip["Total"], np.nan)

trip = trip.sort_values("Total", ascending=False).reset_index(drop=True)
if TOP_TRIPLETS:
    trip = trip.head(TOP_TRIPLETS)

# -------------------- Save CSVs --------------------
OUT = Path("fig1"); OUT.mkdir(parents=True, exist_ok=True)
# Main summary for triplets
trip_csv = OUT / "fig1c_upset_triplets_k3_ALL.csv"
trip[["Combo", "Single", "Multi", "Total", "FracSingle", "ElemA", "ElemB", "ElemC"]].to_csv(trip_csv, index=False)

# One-hot membership for columns = focus_elems (0/1 per element)
onehot = trip[["Combo", "Single", "Multi", "Total", "FracSingle"]].copy()
for e in focus_elems:
    onehot[e] = ((trip["ElemA"]==e) | (trip["ElemB"]==e) | (trip["ElemC"]==e)).astype(int)
onehot_csv = OUT / "fig1c_upset_triplets_k3_onehot.csv"
onehot.to_csv(onehot_csv, index=False)

print("Saved:", trip_csv.resolve())
print("Saved:", onehot_csv.resolve())

# -------------------- Plot (bars + membership matrix) --------------------
n_rows = len(trip)
n_cols = len(focus_elems)
y_pos = np.arange(n_rows)

fig_height = max(6.0, 0.28 * n_rows)
fig = plt.figure(figsize=(8.6, fig_height), dpi=220)
gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.3, 2.2], wspace=0.05)

# Left: stacked bars
ax_bar = fig.add_subplot(gs[0, 0])
single_vals = trip["Single"].values
multi_vals  = trip["Multi"].values
totals      = trip["Total"].values

ax_bar.barh(y_pos, multi_vals, color=COL_MULTI, edgecolor="white", linewidth=0.5, label="Multi")
ax_bar.barh(y_pos, single_vals, left=multi_vals, color=COL_SINGLE, edgecolor="white", linewidth=0.5, label="Single")

for yi, t in zip(y_pos, totals):
    if t > 0:
        ax_bar.text(t + max(0.5, 0.02*totals.max()), yi, f"n={t}", va="center", ha="left", fontsize=8, color="#444444")

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(trip["Combo"].values, fontsize=8)
ax_bar.invert_yaxis()
ax_bar.grid(axis="x", linestyle=":", linewidth=0.6, color=COL_GRID)
ax_bar.set_xlabel("Literature count", fontsize=9)
ax_bar.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
ax_bar.set_title("Dominant element TRIPLETS (k=3)", fontsize=10, pad=8)

# Right: membership matrix (three black dots per row, two connectors)
ax_mat = fig.add_subplot(gs[0, 1])
ax_mat.set_xlim(-0.5, n_cols - 0.5)
ax_mat.set_ylim(-0.5, n_rows - 0.5)
ax_mat.invert_yaxis()
ax_mat.set_xticks(np.arange(n_cols))
ax_mat.set_xticklabels(focus_elems, rotation=45, ha="right", fontsize=8)
ax_mat.set_yticks([])

# light vertical grid
for xc in range(n_cols):
    ax_mat.axvline(x=xc, ymin=0, ymax=1, color=COL_GRID, lw=0.6, zorder=0)

# draw three dots + two connectors
for i, r in trip.iterrows():
    # indices of the three elements in focus_elems order
    xs = sorted([focus_elems.index(r["ElemA"]),
                 focus_elems.index(r["ElemB"]),
                 focus_elems.index(r["ElemC"])])
    # connectors
    segs = [((xs[0], i), (xs[1], i)), ((xs[1], i), (xs[2], i))]
    ax_mat.add_collection(LineCollection(segs, colors="black", linewidths=LINE_WIDTH, zorder=2))
    # dots
    ax_mat.scatter(xs, [i, i, i], s=DOT_SIZE, c="black", zorder=3, marker="o")

ax_mat.set_title("Membership matrix (columns = focus elements)", fontsize=9, pad=6)

plt.tight_layout()
png = OUT / "fig1c_upset_triplets_k3.png"
pdf = OUT / "fig1c_upset_triplets_k3.pdf"
plt.savefig(png, dpi=300, bbox_inches="tight")
plt.savefig(pdf, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", png.resolve())
print("Saved:", pdf.resolve())