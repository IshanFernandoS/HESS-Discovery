# Fig. 1b — Literature coverage by cardinality (stacked bars, Nature-style)
# Keeps the same outputs, filenames, colors, and annotations.

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- Literature lists (given) --------------------
single_phase_lit = [
    '(CrHfMoNbTaTiVZr)C8','(CrHfNbTaTiVWZr)C8','(CrHfNbTaTi)C5','(CrHfNbTaV)C5',
    '(CrHfTaTiZr)C5','(CrMoNbTaW)C5','(CrMoNbVW)C5','(CrMoTaVW)C5','(CrMoTiVW)C5',
    '(HfLaMoNbTaTiWZr)C8','(HfLaNbTaTiYZr)C7','(HfLaTaTiYZr)C6','(HfMoNbTaTi)C5',
    '(HfMoNbTaTiVWZr)C8','(HfMoNbTaTiWZr)C7','(HfMoNbTaTiZr)C6','(HfMoNbTaV)C5','(HfMoNbTaZr)C5',
    '(HfMoTaTiZr)C5','(HfNbTaTiV)C5','(HfNbTaTiVWZr)C7','(HfNbTaTiVZr)C6','(HfNbTaTiW)C5',
    '(HfNbTaTiWZr)C6','(HfNbTaTiZr)C5','(HfNbTaVW)C5','(HfNbTaVZr)C5','(HfNbTiVZr)C5',
    '(HfTaTiVZr)C5','(HfTaTiWZr)C5','(MoNbTaTiV)C5','(MoNbTaTiW)C5','(MoNbTaTiZr)C5',
    '(MoNbTaVW)C5','(MoNbTaVZr)C5','(NbTaTiVW)C5','(NbTaTiVZr)C5','(NbTaTiWZr)C5','(NbTaVWZr)C5'
]
multi_phase_lit = [
    '(CrHfMoTiW)C5','(CrHfMoVW)C5','(CrHfMoWZr)C5','(CrHfNbVW)C5','(CrHfNbWZr)C5',
    '(CrHfTaVW)C5','(CrHfTaWZr)C5','(CrHfTiVW)C5','(CrHfTiWZr)C5','(CrHfVWZr)C5',
    '(CrMoTiVZr)C5','(CrMoTiWZr)C5','(CrNbTiWZr)C5','(CrNbVWZr)C5','(CrTaTiWZr)C5',
    '(CrTaVWZr)C5','(CrTiVWZr)C5','(HfMoTaWZr)C5','(HfMoTiWZr)C5','(HfMoVWZr)C5','(MoTiVWZr)C5'
]

K_RANGE = [5, 6, 7, 8, 9]
COLOR_SINGLE = "#0072B2"  # Okabe–Ito
COLOR_MULTI  = "#D55E00"  # Okabe–Ito

# -------------------- Helpers --------------------
def k_from_comp(comp_str: str) -> int:
    """Return the number of metals inside parentheses, e.g. '(HfMoNbTaTi)C5' -> 5."""
    m = re.search(r"\(([^)]+)\)", str(comp_str))
    if not m:
        return np.nan
    elems = re.findall(r"[A-Z][a-z]?", m.group(1))
    return len(elems)

def as_df(lst, label) -> pd.DataFrame:
    return pd.DataFrame({"Composition": lst, "Label": label})

# -------------------- Tally by cardinality --------------------
df = pd.concat(
    [as_df(single_phase_lit, "Single"), as_df(multi_phase_lit, "Multi")],
    ignore_index=True
)
df["k"] = df["Composition"].map(k_from_comp).astype(int)

counts = (
    df.groupby(["k", "Label"]).size()
      .unstack("Label")
      .reindex(index=K_RANGE, columns=["Single", "Multi"])
      .fillna(0).astype(int)
)
counts["Total"]      = counts.sum(axis=1)
counts["FracSingle"] = np.where(counts["Total"] > 0, counts["Single"] / counts["Total"], np.nan)

print("Literature coverage table:\n", counts)

# -------------------- Plot --------------------
OUT = Path("fig1"); OUT.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=220)

x_pos = np.arange(len(K_RANGE))
bar_width = 0.65

bars_multi = ax.bar(
    x_pos, counts["Multi"].values, width=bar_width,
    color=COLOR_MULTI, label="Multi",
    edgecolor="white", linewidth=0.5
)
bars_single = ax.bar(
    x_pos, counts["Single"].values, width=bar_width,
    bottom=counts["Multi"].values,
    color=COLOR_SINGLE, label="Single",
    edgecolor="white", linewidth=0.5
)

ax.set_xticks(x_pos)
ax.set_xticklabels([f"HEC{k}" for k in K_RANGE], fontsize=9)
ax.set_ylabel("Count (literature)", fontsize=9)
ax.set_title("Fig. 1b — Literature coverage by cardinality", fontsize=10, pad=8)
ax.set_ylim(0, max(1, counts["Total"].max()) * 1.25)
ax.grid(axis="y", linestyle=":", linewidth=0.6, color="#dddddd")
ax.tick_params(axis="y", labelsize=8)

def annotate_counts(bar_container, color="white", min_show=2):
    """Write the segment count inside each stacked bar segment (centered)."""
    for rect in bar_container:
        height = rect.get_height()
        if height >= min_show:
            x_txt = rect.get_x() + rect.get_width() / 2
            y_txt = rect.get_y() + rect.get_height() / 2
            ax.text(x_txt, y_txt, f"{int(height)}", ha="center", va="center",
                    fontsize=8, color=color)

annotate_counts(bars_multi)
annotate_counts(bars_single)

# Totals + fraction single on top of each stack
top_offset = max(0.6, 0.05 * counts["Total"].max())
for xi, total, frac in zip(x_pos, counts["Total"].values, counts["FracSingle"].values):
    if total > 0:
        frac_txt = f"S={frac*100:.0f}%"
        ax.text(xi, total + top_offset, f"n={total}\n{frac_txt}",
                ha="center", va="bottom", fontsize=8, color="#444444")
    else:
        ax.text(xi, 0 + top_offset, "n=0",
                ha="center", va="bottom", fontsize=8, color="#666666")

ax.legend(ncol=2, frameon=False, fontsize=8, loc="upper left")

plt.tight_layout()
png_path = OUT / "fig1b_literature_coverage.png"
pdf_path = OUT / "fig1b_literature_coverage.pdf"
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {png_path.resolve()}\n       {pdf_path.resolve()}")