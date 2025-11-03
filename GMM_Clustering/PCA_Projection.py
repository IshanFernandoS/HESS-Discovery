# ========================== Fig. 3a — PCA feature-space map (P_single_cal) ==========================
# Saves:
#   - OUT/fig3/fig3a_pca_posterior.png
#   - OUT/fig3/fig3a_pca_posterior.pdf
#   - OUT/fig3/fig3a_pca_embedding_with_meta.csv
#
# Requirements from prior cells:
#   OUT (Path), X_scaled (np.ndarray), dfX (DataFrame with Composition, P_single_cal, n_metals, Exp_Phase),
#   pool9_ranked (DataFrame with "Composition" and "Score"), selected (list of compositions)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

# ---- Guards: verify prerequisites without changing behavior ----
_required = ["OUT", "X_scaled", "dfX", "pool9_ranked", "selected"]
_missing = [v for v in _required if v not in globals()]
if _missing:
    raise RuntimeError(f"Missing variables: {_missing}. Run your GMM & ranking cells first.")

# ---- Output paths (kept identical to your originals) ----
FIG_DIR = OUT / "fig3"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_PNG = FIG_DIR / "fig3a_pca_posterior.png"
FIG_PDF = FIG_DIR / "fig3a_pca_posterior.pdf"
EMBED_CSV = FIG_DIR / "fig3a_pca_embedding_with_meta.csv"

# --- PCA on the SAME standardized space used for GMM ---
# (X_scaled aligns row-for-row with dfX after NaN filtering.)
pca = PCA(n_components=2)  # deterministic given fixed X_scaled; no random_state needed
pca_xy = pca.fit_transform(X_scaled)

embedding_df = pd.DataFrame(pca_xy, columns=["pc1", "pc2"], index=dfX.index)
# Attach metadata for plotting
embedding_df["Composition"]  = dfX["Composition"].values
embedding_df["P_single_cal"] = dfX["P_single_cal"].values
embedding_df["n_metals"]     = dfX["n_metals"].values
embedding_df["is_labeled"]   = dfX["Exp_Phase"].isin([0, 1]).values  # True where literature label exists

# Baseline top-30 by Score (from your composite ranking of HEC9)
top30_set = set(pool9_ranked["Composition"].head(30).tolist())

# Selected compositions (already defined upstream)
selected_set = set(selected)

def _short_comp(s: str) -> str:
    """Drop the trailing )Ck tag for compact labels."""
    s = str(s)
    for tag in [")C5", ")C6", ")C7", ")C8", ")C9", ")C10", ")C11"]:
        s = s.replace(tag, "")
    return s

# --- Plot ---
plt.ioff()
fig, ax = plt.subplots(figsize=(6.5, 5.6), dpi=350)

# 1) Background: ALL unlabeled points as faint gray
background_df = embedding_df[~embedding_df["is_labeled"]]
ax.scatter(
    background_df["pc1"], background_df["pc2"],
    s=8, c="0.85", alpha=0.55, linewidths=0, zorder=1
)

# 2) Foreground: labeled points colored by posterior
foreground_df = embedding_df[embedding_df["is_labeled"]]
sc = ax.scatter(
    foreground_df["pc1"], foreground_df["pc2"],
    c=foreground_df["P_single_cal"], vmin=0, vmax=1, cmap="viridis",
    s=14, alpha=0.95, linewidths=0, zorder=2
)

# 3) Outline: Top-30 by Score — black circle outlines
mask_top30 = embedding_df["Composition"].isin(top30_set)
ax.scatter(
    embedding_df.loc[mask_top30, "pc1"], embedding_df.loc[mask_top30, "pc2"],
    s=70, facecolors="none", edgecolors="k", linewidths=1.0, zorder=3,
    label="Top-30 by Score"
)

# 4) Selected comps: red diamond outlines + labels
mask_selected = embedding_df["Composition"].isin(selected_set)
ax.scatter(
    embedding_df.loc[mask_selected, "pc1"], embedding_df.loc[mask_selected, "pc2"],
    s=80, marker="D", facecolors="none", edgecolors="#d62728", linewidths=1.4, zorder=4,
    label="Selected HEC9"
)
for _, row in embedding_df[mask_selected].iterrows():
    ax.annotate(
        _short_comp(row["Composition"]),
        (row["pc1"], row["pc2"]),
        xytext=(3, 3), textcoords="offset points",
        fontsize=7.5, color="#d62728"
    )

# Colorbar & axes
cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.012)
cbar.set_label("P(single) (calibrated)", fontsize=10)
ax.set_xlabel(f"PCA-1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)", fontsize=10)
ax.set_ylabel(f"PCA-2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)", fontsize=10)
ax.set_title("Fig. 3a — PCA of descriptor space colored by single-phase posterior", fontsize=11)
ax.legend(loc="lower right", frameon=True, fontsize=9)
ax.grid(False)

plt.tight_layout()
plt.savefig(FIG_PNG, dpi=350, bbox_inches="tight")
plt.savefig(FIG_PDF, dpi=350, bbox_inches="tight")
plt.close(fig)

# Save the embedding table (for SI/reproducibility)
embedding_out = embedding_df.copy()
embedding_out["is_top30"] = mask_top30.values
embedding_out["is_selected"] = mask_selected.values
embedding_out.to_csv(EMBED_CSV, index=False)

print("Saved Fig. 3a (PCA) →", FIG_PNG.resolve())