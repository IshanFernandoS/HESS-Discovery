# ===== Figure 1d (Nature-style) =====
# Left: Spearman correlation among objectives (nice names, significance outline)
# Right: Physics-family summary heatmap (Count, Isolation)
# Saves: fig1/fig1d_left.png/.pdf, fig1/fig1d_right.png/.pdf, fig1/fig1d_combined.png/.pdf

import io, re, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import spearmanr

# ---- Load your table ----
# If you're in Colab:
# from google.colab import files
# up = files.upload()
# fname = next(iter(up))
# df = pd.read_excel(io.BytesIO(up[fname])) if fname.endswith((".xlsx",".xls")) else pd.read_csv(io.BytesIO(up[fname]))

# If you already have a dataframe `df`, comment the above and just use df directly.
# Expect columns named either "norm_*" or human-readable names (we map both).

OUT = Path("fig1"); OUT.mkdir(parents=True, exist_ok=True)

# -------------------- Choose and sanitize objective columns --------------------
# 1) Define canonical column keys you'd like to include (any of these may exist in df)
#    Add/remove keys as needed to match your file.
CANDIDATE_KEYS = [
    # normalized short names
    "norm_fg", "norm_rmis", "norm_xmis", "norm_ca", "norm_mdri", "norm_mp",
    "norm_vec", "norm_cfdi", "norm_msi",
    # raw fallbacks (verbatim or close)
    "Formation Gap", "Radius Mismatch", "Electronegativity Mismatch",
    "Carbon Affinity", "Magnetic Disorder Risk Index", "Minimum Carbide Melting Point",
    "Average Valence Count", "Carbide Formation Deviation Index", "Metastable Segregation Index",
]

def find_cols(df, candidates):
    have = []
    lower_map = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in df.columns:
            have.append(k)
        elif k.lower() in lower_map:
            have.append(lower_map[k.lower()])
    return list(dict.fromkeys(have))  # unique, preserve order

feature_cols = find_cols(df, CANDIDATE_KEYS)
if len(feature_cols) < 2:
    raise ValueError(f"No usable objective columns found. Looked for any of: {CANDIDATE_KEYS}")

# 2) Coerce to scalar numerics (drop vector-like entries)
def _to_scalar(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (list, tuple, np.ndarray)):
        arr = np.asarray(v)
        return float(arr.ravel()[0]) if arr.size == 1 else np.nan
    try:
        return float(v)
    except Exception:
        return np.nan

X = df[feature_cols].applymap(_to_scalar).astype(float)
# drop columns that are all NaN or constant
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=True) >= 2]
feature_cols = list(X.columns)
if len(feature_cols) < 2:
    raise ValueError("Too few usable features after sanitization.")

# -------------------- Nice display names (LaTeX-safe) --------------------
# Keep mathtext simple; avoid \text{...}. These render well in Matplotlib.
NICE_NAME = {
    "norm_fg":   r"Formation gap $\Delta H_{\mathrm{carb}}$",
    "norm_rmis": r"Radius mismatch $\delta r$",
    "norm_xmis": r"EN mismatch $\delta\chi$",
    "norm_ca":   r"Carbon affinity (max $H_f^{\mathrm{carb}}$)",
    "norm_mdri": r"MDRI (magnetic)",
    "norm_mp":   r"Min.\ carbide melting $T_m^{\min}$",
    "norm_vec":  r"Avg.\ valence $\langle\mathrm{VEC}\rangle$",
    "norm_cfdi": r"CFDI ($H_f^{\mathrm{carb}}$ spread)",
    "norm_msi":  r"MSI (M–M segregation)",

    "Formation Gap":                       r"Formation gap $\Delta H_{\mathrm{carb}}$",
    "Radius Mismatch":                     r"Radius mismatch $\delta r$",
    "Electronegativity Mismatch":          r"EN mismatch $\delta\chi$",
    "Carbon Affinity":                     r"Carbon affinity (max $H_f^{\mathrm{carb}}$)",
    "Magnetic Disorder Risk Index":        r"MDRI (magnetic)",
    "Minimum Carbide Melting Point":       r"Min.\ carbide melting $T_m^{\min}$",
    "Average Valence Count":               r"Avg.\ valence $\langle\mathrm{VEC}\rangle$",
    "Carbide Formation Deviation Index":   r"CFDI ($H_f^{\mathrm{carb}}$ spread)",
    "Metastable Segregation Index":        r"MSI (M–M segregation)",
}

def pretty(col):
    return NICE_NAME.get(col, col)

# -------------------- Spearman correlation with BH–FDR significance --------------------
def spearman_df(df_num):
    cols = list(df_num.columns)
    n = len(cols)
    R = np.eye(n, dtype=float)
    P = np.ones((n, n), dtype=float)
    for i, a in enumerate(cols):
        xa = pd.to_numeric(df_num[a], errors="coerce")
        for j, b in enumerate(cols[:i+1]):
            xb = pd.to_numeric(df_num[b], errors="coerce")
            s = pd.DataFrame({a: xa, b: xb}).dropna()
            if len(s) >= 3:
                r, p = spearmanr(s[a].values, s[b].values)
                if np.ndim(r) > 0:  # defensive on older SciPy returning 2x2
                    r, p = float(r[0, 1]), float(p[0, 1])
            else:
                r, p = np.nan, 1.0
            R[i, j] = R[j, i] = r
            P[i, j] = P[j, i] = p
    return pd.DataFrame(R, index=cols, columns=cols), pd.DataFrame(P, index=cols, columns=cols)

R, P = spearman_df(X)

# Hierarchical order for tidy blocks
D = 1 - R.fillna(0).clip(-1, 1).abs()  # distance proxy from |rho|
Z = linkage(D, method="average")
order = leaves_list(Z)
R_ord = R.iloc[order, order]
P_ord = P.iloc[order, order]

# BH–FDR mask on lower triangle (q=0.05)
def bh_fdr_mask(Pmat, q=0.05):
    m = Pmat.shape[0]
    tri = np.tril(np.ones_like(Pmat, dtype=bool), k=-1)
    pvals = Pmat.values[tri]
    idx = np.where(tri.ravel())[0]
    k = len(pvals)
    if k == 0:
        return np.zeros_like(Pmat, dtype=bool)
    order = np.argsort(pvals)
    pv_sorted = pvals[order]
    thresh = q * (np.arange(1, k+1)/k)
    passed = pv_sorted <= thresh
    cutoff = np.max(np.where(passed)[0]) if np.any(passed) else -1
    sig_mask_flat = np.zeros(k, dtype=bool)
    if cutoff >= 0:
        sig_mask_flat[order[:cutoff+1]] = True
    mask = np.zeros_like(Pmat, dtype=bool)
    mask.ravel()[idx] = sig_mask_flat
    mask = mask | mask.T  # symmetrize
    return mask

SIG = bh_fdr_mask(P_ord, q=0.05)

# -------------------- LEFT panel: correlation heatmap --------------------
plt.close("all")
fig_left, ax = plt.subplots(figsize=(6.6, 6.0), dpi=240)
mask_upper = np.triu(np.ones_like(R_ord, dtype=bool), k=1)

sns.heatmap(R_ord, mask=mask_upper, ax=ax,
            cmap="coolwarm", vmin=-1, vmax=1, center=0,
            linewidths=0.5, linecolor="#F0F0F0",
            cbar_kws={"label": r"Spearman $\rho$"},
            square=True)

# Outline significant cells (BH–FDR) on lower triangle
yy, xx = np.where(np.tril(SIG, k=-1))
ax.scatter(xx + 0.5, yy + 0.5, s=8, marker="s", facecolors="none", edgecolors="k", linewidths=0.8)

# Re-label ticks with nice names
nice_ticks = [pretty(c) for c in R_ord.columns]
ax.set_xticklabels(nice_ticks, rotation=50, ha="right", fontsize=8)
ax.set_yticklabels(nice_ticks, rotation=0, va="center", fontsize=8)

ax.set_title("Objective correlation (Spearman)", fontsize=11, pad=8)
left_png = OUT / "fig1d_left.png"
left_pdf = OUT / "fig1d_left.pdf"
fig_left.tight_layout()
fig_left.savefig(left_png, dpi=300, bbox_inches="tight")
fig_left.savefig(left_pdf, dpi=300, bbox_inches="tight")
plt.close(fig_left)

# -------------------- RIGHT panel: physics-family summary heatmap --------------------
# Map each feature to a family (use your canonical names)
FAMILY_MAP = {
    "norm_fg":   "Thermodynamic competition\n(gap to strongest carbide)",
    "norm_ca":   "Carbide bonding / M–C affinity",
    "norm_afe":  "Thermodynamic formation\n(avg formation enthalpy)",  # optional if present
    "norm_rmis": "Size mismatch / lattice strain",
    "norm_xmis": "Electronegativity mismatch\n(chemical order tendency)",
    "norm_msi":  "Metal–metal segregation tendency",
    "norm_mdri": "Magnetic coupling / disorder",
    "norm_vec":  "Electronic filling (VEC window)",
    "norm_mp":   "Processing constraint\n(min carbide melting point)",
    "norm_cfdi": "Carbide segregation tendency\n(CFDI spread)",
    # raw fallbacks
    "Formation Gap":                       "Thermodynamic competition\n(gap to strongest carbide)",
    "Carbon Affinity":                     "Carbide bonding / M–C affinity",
    "Radius Mismatch":                     "Size mismatch / lattice strain",
    "Electronegativity Mismatch":          "Electronegativity mismatch\n(chemical order tendency)",
    "Metastable Segregation Index":        "Metal–metal segregation tendency",
    "Magnetic Disorder Risk Index":        "Magnetic coupling / disorder",
    "Average Valence Count":               "Electronic filling (VEC window)",
    "Minimum Carbide Melting Point":       "Processing constraint\n(min carbide melting point)",
    "Carbide Formation Deviation Index":   "Carbide segregation tendency\n(CFDI spread)",
}

def family_of(col):
    return FAMILY_MAP.get(col, FAMILY_MAP.get(col.strip(), "Other / uncategorized"))

fam_series = pd.Series({c: family_of(c) for c in feature_cols})

# Compute metrics per family:
# 1) Count of objectives in the family
# 2) Isolation = 1 - median(|rho| to features outside the family)
families = sorted(fam_series.unique())
counts = []
isol = []

for fam in families:
    cols_in = fam_series.index[fam_series == fam].tolist()
    if len(cols_in) == 0:
        continue
    counts.append(len(cols_in))
    cols_out = [c for c in feature_cols if c not in cols_in]
    if len(cols_out) == 0 or R.loc[cols_in, cols_out].size == 0:
        iso = 1.0
    else:
        med_abs = np.nanmedian(np.abs(R.loc[cols_in, cols_out].values))
        iso = float(np.clip(1.0 - med_abs, 0.0, 1.0))
    isol.append(iso)

fam_df = pd.DataFrame({"Family": families, "Count": counts, "Isolation": isol}).set_index("Family")

# Normalize each column to [0,1] for a clean heatmap
def minmax(x):
    if np.all(~np.isfinite(x)):
        return x
    a, b = np.nanmin(x), np.nanmax(x)
    return (x - a) / (b - a + 1e-12) if b > a else np.zeros_like(x)

norm_df = fam_df.copy()
norm_df["Count"] = minmax(norm_df["Count"].values)
norm_df["Isolation"] = minmax(norm_df["Isolation"].values)

plt.close("all")
fig_right, axr = plt.subplots(figsize=(6.8, 4.8), dpi=240)

sns.heatmap(norm_df[["Count", "Isolation"]],
            cmap="viridis", vmin=0, vmax=1, linewidths=0.6, linecolor="#F0F0F0",
            cbar_kws={"label": "Normalized metric"},
            annot=True,
            fmt="",
            annot_kws={"fontsize": 8, "color": "white"},
            ax=axr)

# Show readable family names on y-axis (already)
axr.set_ylabel("")
axr.set_xlabel("")
axr.set_title("Physics-family summary: breadth and isolation", fontsize=11, pad=8)
axr.tick_params(axis="y", labelsize=8)
axr.tick_params(axis="x", labelsize=9)

right_png = OUT / "fig1d_right.png"
right_pdf = OUT / "fig1d_right.pdf"
fig_right.tight_layout()
fig_right.savefig(right_png, dpi=300, bbox_inches="tight")
fig_right.savefig(right_pdf, dpi=300, bbox_inches="tight")
plt.close(fig_right)

# -------------------- Combined 1×2 panel --------------------
left_img  = plt.imread(left_png)
right_img = plt.imread(right_png)

fig_comb, axc = plt.subplots(1, 2, figsize=(12.2, 6.0), dpi=220)
axc[0].imshow(left_img);  axc[0].axis("off");  axc[0].set_title("a) Objective correlation (Spearman)", fontsize=11, pad=6)
axc[1].imshow(right_img); axc[1].axis("off"); axc[1].set_title("b) Physics-family summary", fontsize=11, pad=6)

comb_png = OUT / "fig1d_combined.png"
comb_pdf = OUT / "fig1d_combined.pdf"
fig_comb.tight_layout()
fig_comb.savefig(comb_png, dpi=300, bbox_inches="tight")
fig_comb.savefig(comb_pdf, dpi=300, bbox_inches="tight")
plt.close(fig_comb)

print("Saved:")
print(" -", left_png.resolve())
print(" -", right_png.resolve())
print(" -", comb_png.resolve())