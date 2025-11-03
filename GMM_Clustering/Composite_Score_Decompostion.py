# ===================== Fig. 3c — Composite-score decomposition (grouped bars) =====================
# Inputs expected from prior cells:
#   OUT (Path), pool9_ranked (with Score, cum_pmean_z, P_single_cal_z, coverage_z),
#   W_CUM, W_POST, W_COV, selected (list of compositions)
# Outputs:
#   OUT/fig3/fig3c_score_decomposition.png
#   OUT/fig3/fig3c_score_decomposition_values.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Output dir ----
FIG_DIR = OUT / "fig3"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- pick top-N by Score (unchanged) ----
TOP_ROWS = 30
top_df = pool9_ranked.sort_values("Score", ascending=False).head(min(TOP_ROWS, len(pool9_ranked))).copy()

# Make sure the needed columns exist (created earlier in your script)
needed = ["Composition", "Score", "cum_pmean_z", "P_single_cal_z", "coverage_z"]
missing = [c for c in needed if c not in top_df.columns]
if missing:
    raise RuntimeError(f"Missing columns for Fig. 3c: {missing}")

# Contributions (these three bars sum to Score)
contrib_cum  = W_CUM  * top_df["cum_pmean_z"].astype(float).to_numpy()
contrib_post = W_POST * top_df["P_single_cal_z"].astype(float).to_numpy()
contrib_cov  = W_COV  * top_df["coverage_z"].astype(float).to_numpy()
score_total  = top_df["Score"].astype(float).to_numpy()
labels = top_df["Composition"].tolist()

# Highlight selected compositions with a background stripe
sel_set = set(selected)
is_sel  = np.array([lab in sel_set for lab in labels], dtype=bool)

# ---- plotting (grouped bars) ----
N = len(labels)
x = np.arange(N)
bar_w = 0.26
off1, off2, off3 = -bar_w, 0.0, +bar_w

fig_h = max(3.8, 0.38 * N)    # dynamic height for readability
fig_w = max(10,  0.38 * N)
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

# Subtle alternating band to help read across (green for selected)
for k in range(N):
    if is_sel[k]:
        ax.axvspan(k - 0.5, k + 0.5, color="#2ca02c", alpha=0.10, linewidth=0)
    elif k % 2 == 1:
        ax.axvspan(k - 0.5, k + 0.5, color="#000000", alpha=0.03, linewidth=0)

# Bars: each component of the score
b1 = ax.bar(x + off1, contrib_cum,  width=bar_w, label=r"$w_{\mathrm{cum}}\cdot z(\mathrm{cum\_pmean})$")
b2 = ax.bar(x + off2, contrib_post, width=bar_w, label=r"$w_{\mathrm{post}}\cdot z(P_{\mathrm{single}})$")
b3 = ax.bar(x + off3, contrib_cov,  width=bar_w, label=r"$w_{\mathrm{cov}}\cdot z(\mathrm{coverage})$")

# Overlay the total score as a dot (sum of the three)
ax.plot(x, score_total, marker="o", linestyle="none", markersize=4, color="k", label="Total Score (sum)")

def short_comp(s: str) -> str:
    """Drop the trailing )Ck tag for compact x-axis labels."""
    s = str(s)
    for k in [")C5",")C6",")C7",")C8",")C9",")C10",")C11"]:
        s = s.replace(k, "")
    return s

ax.set_xticks(x)
ax.set_xticklabels([short_comp(s) for s in labels], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Score decomposition (weighted z-scores)")
ax.set_title("Fig. 3c — Composite Score = "
             r"$w_{\mathrm{cum}}\cdot z(\mathrm{cum\_pmean}) + "
             r"w_{\mathrm{post}}\cdot z(P_{\mathrm{single}}) + "
             r"w_{\mathrm{cov}}\cdot z(\mathrm{coverage})$")

# Annotate weights (visual aid only)
txt = (f"weights:  cum={W_CUM:.2f}, post={W_POST:.2f}, cov={W_COV:.2f}\n"
       "dot = sum of three bars (Score)")
ax.text(0.995, 0.98, txt, ha="right", va="top", transform=ax.transAxes, fontsize=8,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3))

# Zero line to show positive/negative contributions
ax.axhline(0, color="k", linewidth=0.6, alpha=0.6)

ax.legend(ncol=2, fontsize=8, frameon=True, loc="upper left")

plt.tight_layout()
out_png = FIG_DIR / "fig3c_score_decomposition.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)

# Export the numbers (for SI)
out_csv = FIG_DIR / "fig3c_score_decomposition_values.csv"
export = top_df[["Composition", "Score", "cum_pmean_z", "P_single_cal_z", "coverage_z"]].copy()
export["contrib_cum"]  = contrib_cum
export["contrib_post"] = contrib_post
export["contrib_cov"]  = contrib_cov
export.to_csv(out_csv, index=False)

print("Saved Fig. 3c →", out_png.resolve())
print("Saved values  →", out_csv.resolve())