# GMM + posterior–feature bands + composite ranking (NO EFA) + Diagnostics
# (Outputs unchanged. Removed only unused parts and tightened naming/comments.)

# !pip install -q scikit-learn pandas numpy matplotlib

import io, re, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from google.colab import files
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
)

# =================== USER PARAMETERS ===================
TARGET_N_METALS   = 9        # select HEC9
N_BINS_PER_FEAT   = 20       # quantile bins per feature
FAVOR_TOP_Q       = 0.30     # top-q of bins by bin-mean posterior are "favorable" (for shading/hits)
ROLL_SMOOTH       = 3        # rolling window (bins) to smooth bin means; set 1 to disable
SEED              = 42
# Composite score weights (must sum to 1.0)
W_CUM, W_POST, W_COV = 0.33, 0.33, 0.34
# =======================================================

# ------------ helpers ------------
def canonicalize(comp: str) -> str:
    """Ensure (sorted) '(... )Ck' format, e.g., '(HfMoNbTaTi)C5'."""
    s = str(comp).strip()
    m = re.match(r"^\s*\(([^)]+)\)\s*C?\s*(\d+)\s*$", s)
    if not m:
        return s
    metals = re.findall(r"[A-Z][a-z]?", m.group(1))
    return f"({''.join(sorted(metals))})C{int(m.group(2))}"

def count_metals(comp: str) -> float:
    """Return number of metal species from '(...)Ck'."""
    m = re.match(r"^\(([^)]+)\)C(\d+)$", str(comp))
    if not m:
        return np.nan
    return float(len(re.findall(r"[A-Z][a-z]?", m.group(1))))

def find_col(candidates, columns):
    """Pick the first matching column name (case-insensitive)."""
    lower = {c.lower(): c for c in columns}
    for c in candidates:
        if c in columns:
            return c
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    return None

def safe_qcut(x, q):
    """Quantile binning that tolerates duplicates (drop)."""
    s = pd.Series(x).astype(float)
    return pd.qcut(s, q=q, duplicates="drop")

def zscore(series):
    """Simple z-score with nan-safe stats; constant-variance → zero-centered."""
    x = pd.to_numeric(series, errors="coerce").astype(float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        return (x - mu)
    return (x - mu) / sd

# Map normalized cols → preferred raw axis label (if present)
RAW_NAME_MAP = {
    "norm_fg":"Formation Gap",
    "norm_rmis":"Radius Mismatch",
    "norm_xmis":"Electronegativity Mismatch",
    "norm_ca":"Carbon Affinity",
    "norm_mdri":"Magnetic Disorder Risk Index",
    "norm_mp":"Minimum Carbide Melting Point",
    "norm_vec":"Average Valence Count",
    "norm_cfdi":"Carbide Formation Deviation Index",
    "norm_msi":"Metastable Segregation Index",
    "norm_afe":"Average Formation Enthalpy",
}

# ------------ load ------------
print("Upload your CSV (e.g., Compositions_Descriptros(Sheet1).csv)")
uploaded = files.upload()
uploaded_name = next(iter(uploaded))

if uploaded_name.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(io.BytesIO(uploaded[uploaded_name]))
else:
    df = pd.read_csv(io.BytesIO(uploaded[uploaded_name]))

OUT = Path("gmm_band_rank"); OUT.mkdir(parents=True, exist_ok=True)

# Canonical composition & metal count
comp_col = find_col(["Composition", "composition"], df.columns)
if comp_col is None:
    raise ValueError("No 'Composition' column found.")
df["Composition"] = df[comp_col].astype(str).map(canonicalize)
df["n_metals"] = df["Composition"].apply(count_metals)

# Attach weak labels if missing (1=Single, 0=Multi)
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

lab_map = {canonicalize(x): 1 for x in single_phase_lit}
lab_map.update({canonicalize(x): 0 for x in multi_phase_lit})
if "Exp_Phase" not in df.columns:
    df["Exp_Phase"] = df["Composition"].map(lab_map)

# Feature set for clustering (prefer normalized, exclude EFA-like to avoid circularity)
norm_cols = [c for c in df.columns if str(c).startswith("norm_")]
norm_cols = [c for c in norm_cols if c.lower() not in {"norm_efa", "norm_predicted_efa", "norm_cfdi"}]
if len(norm_cols) >= 3:
    feature_cols = norm_cols
else:
    fallback = [
        "Average Formation Enthalpy","Formation Gap","Radius Mismatch",
        "Electronegativity Mismatch","Carbon Affinity","Magnetic Disorder Risk Index",
        "Minimum Carbide Melting Point","Average Valence Count",
        "Carbide Formation Deviation Index","Metastable Segregation Index",
    ]
    feature_cols = [c for c in fallback if c in df.columns]
if len(feature_cols) < 2:
    raise ValueError(f"Too few descriptors for clustering. Found: {feature_cols}")

# Drop rows with NaNs in clustering features
X = df[feature_cols].copy()
row_ok = ~X.isna().any(axis=1)
X = X[row_ok]
dfX = df.loc[row_ok].copy()

# Standardize + choose GMM by BIC
scaler = StandardScaler().fit(X.values)
X_scaled = scaler.transform(X.values)

best = None
for k in range(2, 12):
    for cov in ["full", "diag"]:
        try:
            gm = GaussianMixture(
                n_components=k, covariance_type=cov,
                tol=1e-4, reg_covar=1e-6, max_iter=500,
                n_init=5, init_params="kmeans", random_state=SEED
            )
            gm.fit(X_scaled)
            bic = gm.bic(X_scaled)
            if (best is None) or (bic < best["bic"]):
                best = {"gmm": gm, "bic": bic, "k": k, "cov": cov}
        except Exception:
            pass
assert best is not None, "GMM fit failed."
gmm = best["gmm"]
print(f"Best GMM by BIC: k={best['k']}, cov={best['cov']}, BIC={best['bic']:.1f}")

# Responsibilities and component → class mapping using labels (weak supervision)
resp = gmm.predict_proba(X_scaled)
is_labeled = dfX["Exp_Phase"].isin([0, 1]).to_numpy()
y_lab = dfX.loc[is_labeled, "Exp_Phase"].astype(int).to_numpy()
R_lab = resp[is_labeled]

# --- Diagnostics: component purity & sharpness ---
comp_id = resp.argmax(axis=1)
dfX["gmm_comp"] = comp_id

comp_diag_rows = []
if is_labeled.sum() >= 1 and len(np.unique(y_lab)) == 2:
    for j in range(resp.shape[1]):
        S = R_lab[y_lab == 1, j].sum()
        M = R_lab[y_lab == 0, j].sum()
        tot = S + M
        purity = max(S, M) / tot if tot > 0 else np.nan
        winner = "Single" if S >= M else "Multi"
        ratio = (S / M) if M > 0 else (np.inf if S > 0 else np.nan)
        comp_diag_rows.append({
            "comp": j, "winner": winner, "mass_S": float(S), "mass_M": float(M),
            "purity": float(purity) if purity == purity else None,  # JSON-safe
            "mass_ratio": float(ratio) if np.isfinite(ratio) else None
        })
    diag_df = pd.DataFrame(comp_diag_rows).sort_values("purity", ascending=False)
    print("\nComponent mapping diagnostics:")
    print(diag_df.to_string(index=False))
else:
    diag_df = pd.DataFrame(columns=["comp", "winner", "mass_S", "mass_M", "purity", "mass_ratio"])
    print("\nComponent mapping diagnostics: not computed (insufficient labels)")

# Sharpness of responsibilities (1 = very sharp, ~1/k = diffuse)
sharpness = float(resp.max(axis=1).mean())
print(f"\nMean max responsibility (sharpness): {sharpness:.3f}")

# Save diagnostics
(OUT := OUT)  # keep OUT name stable
(OUT / "eval").mkdir(exist_ok=True, parents=True)
try:
    diag_df.to_csv(OUT / "eval" / "component_mapping_diagnostics.csv", index=False)
except Exception:
    pass

# ----------------- Map components to phase (weak supervision or fallback) -----------------
if is_labeled.sum() >= 1 and len(np.unique(y_lab)) == 2:
    single_mass = R_lab[y_lab == 1].sum(axis=0)
    multi_mass  = R_lab[y_lab == 0].sum(axis=0)
    comp_is_single = (single_mass >= multi_mass)
else:
    # Fallback: split GMM means unsupervised via PC1 median
    means = gmm.means_
    pc1 = PCA(n_components=1, random_state=SEED).fit_transform(means).ravel()
    comp_is_single = pc1 >= np.median(pc1)

# Global posterior and optional calibration
P_single_raw = resp[:, comp_is_single].sum(axis=1)
P_single_cal = P_single_raw.copy()
if is_labeled.sum() >= 25 and len(np.unique(y_lab)) == 2:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(P_single_raw[is_labeled], y_lab)
    P_single_cal = iso.transform(P_single_raw)

dfX["P_single_cal"] = P_single_cal.astype(float)
dfX["n_metals"] = dfX["n_metals"].astype(float)
dfX["comp_phase_label"]   = np.where(comp_is_single[dfX["gmm_comp"]], "Single-like", "Multi-like")
dfX["sample_phase_label"] = np.where(dfX["P_single_cal"] >= 0.5, "Single-like", "Multi-like")

# ================== EVALUATE VS LITERATURE (CONFUSION MATRIX) ==================
if is_labeled.sum() >= 1 and len(np.unique(y_lab)) == 2:
    y_true  = y_lab
    y_score = P_single_cal[is_labeled]
    y_pred  = (y_score >= 0.5).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_score)
    except Exception:
        roc = np.nan
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # rows=true (Multi, Single), cols=pred

    print("\n=== Cluster vs Literature (labeled subset) ===")
    print(f"Labeled compositions: {int(is_labeled.sum())}")
    print(f"Accuracy:           {acc:.3f}")
    print(f"Balanced Accuracy:  {bacc:.3f}")
    print(f"Precision (Single): {prec:.3f}")
    print(f"Recall (Single):    {rec:.3f}")
    print(f"F1 (Single):        {f1:.3f}")
    print(f"ROC-AUC:            {roc:.3f}")
    print("Confusion matrix [rows=true 0/1, cols=pred 0/1]:\n", cm)
    print("\nDetailed report:\n",
          classification_report(y_true, y_pred, target_names=["Multi", "Single"], zero_division=0))

    # save metrics + sharpness
    with open(OUT / "eval" / "metrics.json", "w") as f:
        json.dump({
            "n_labeled": int(is_labeled.sum()),
            "accuracy": float(acc),
            "balanced_accuracy": float(bacc),
            "precision_single": float(prec),
            "recall_single": float(rec),
            "f1_single": float(f1),
            "roc_auc": float(roc) if not np.isnan(roc) else None,
            "mean_max_responsibility": sharpness
        }, f, indent=2)

    # confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Multi", "Single"])
    fig, ax = plt.subplots(figsize=(3.2, 3.0), dpi=220)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("GMM clustering vs literature (labeled subset)")
    plt.tight_layout()
    plt.savefig(OUT / "eval" / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
else:
    print("\nNo (or insufficient) labeled data found for evaluation; skipping confusion matrix.")
    with open(OUT / "eval" / "metrics.json", "w") as f:
        json.dump({"n_labeled": int(is_labeled.sum()),
                   "mean_max_responsibility": sharpness}, f, indent=2)

# ---------------- Posterior-vs-feature bin curves ----------------
def resolve_raw_feature(col):
    """Prefer raw name for plotting if we used a normalized variant."""
    if col.startswith("norm_"):
        raw_guess = RAW_NAME_MAP.get(col, None)
        if raw_guess and raw_guess in dfX.columns:
            return raw_guess
    return col if col in dfX.columns else None

plot_feats = []
for c in feature_cols:
    raw = resolve_raw_feature(c)
    if raw is not None:
        plot_feats.append((c, raw))  # (cluster_feature, plot_feature)

# Deduplicate by plotted feature axis (avoid dup lines)
seen_plot = set()
plot_feats_unique = []
for c_feat, p_feat in plot_feats:
    if p_feat not in seen_plot:
        plot_feats_unique.append((c_feat, p_feat))
        seen_plot.add(p_feat)

band_tables = []  # per-feature bin stats & favorable flags
for (_c, feat_col) in plot_feats_unique:
    x = pd.to_numeric(dfX[feat_col], errors="coerce")
    p = dfX["P_single_cal"].astype(float)
    ok = ~(x.isna() | p.isna())
    x, p = x[ok], p[ok]

    qbins = safe_qcut(x, q=N_BINS_PER_FEAT)
    tmp = pd.DataFrame({"x": x.values, "p": p.values, "q": qbins})
    g = tmp.groupby("q", observed=True)
    bin_mid = g["x"].mean()
    bin_p   = g["p"].mean()
    if ROLL_SMOOTH and ROLL_SMOOTH > 1:
        bin_p = bin_p.rolling(ROLL_SMOOTH, center=True, min_periods=1).mean()

    # Favorable bins: top-q by bin-mean posterior
    k_top = max(1, int(np.ceil(len(bin_p) * FAVOR_TOP_Q)))
    top_idx = bin_p.sort_values(ascending=False).index[:k_top]
    is_fav = bin_p.index.isin(top_idx)

    df_band = pd.DataFrame({
        "feature": feat_col,
        "bin_left":  [iv.left  for iv in bin_mid.index],
        "bin_right": [iv.right for iv in bin_mid.index],
        "bin_mid": bin_mid.values,
        "bin_pmean": bin_p.values,
        "is_favorable": is_fav.astype(bool)
    })
    band_tables.append(df_band)

    # Plot (bin means) + shade favorable bands
    fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=220)
    ax.plot(df_band["bin_mid"], df_band["bin_pmean"], marker="o", lw=1)
    for _, r in df_band[df_band["is_favorable"]].iterrows():
        ax.axvspan(r["bin_left"], r["bin_right"], color="tab:green", alpha=0.12)
    ax.set_xlabel(feat_col, fontsize=9)
    ax.set_ylabel("P(single) (bin mean)", fontsize=9)
    ax.set_title(f"P(single) vs {feat_col}", fontsize=10)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT / f"posterior_vs_{feat_col.replace(' ', '_')}.png", dpi=240)
    plt.close(fig)

bands_df = pd.concat(band_tables, ignore_index=True)
bands_df.to_csv(OUT / "feature_bands.csv", index=False)

# --------------- Score compositions (composite ranking, NO EFA) ---------------
# Build fast lookup: feature value → bin (via IntervalIndex)
band_lookup = {}
for f, grp in bands_df.groupby("feature"):
    intervals = pd.IntervalIndex.from_arrays(grp["bin_left"].values, grp["bin_right"].values, closed="right")
    band_lookup[f] = {
        "intervals": intervals,
        "is_fav": grp["is_favorable"].to_numpy(),
        "pmean": grp["bin_pmean"].to_numpy()
    }

rows = []
for _, r in dfX.iterrows():
    hits = 0
    pmeans_in_bin = []
    per_feat_hits = {}
    used = 0
    for (_c, feat_col) in plot_feats_unique:
        val = r.get(feat_col, np.nan)
        if pd.isna(val):
            per_feat_hits[feat_col] = np.nan
            continue
        L = band_lookup[feat_col]
        bidx = L["intervals"].get_indexer([val])[0]
        if bidx == -1:
            per_feat_hits[feat_col] = 0
            continue
        hit = 1 if L["is_fav"][bidx] else 0
        per_feat_hits[feat_col] = hit
        hits += hit
        pmeans_in_bin.append(L["pmean"][bidx])
        used += 1

    cum_pmean     = float(np.nansum(pmeans_in_bin)) if used else np.nan
    bin_pmean_avg = float(np.nanmean(pmeans_in_bin)) if used else np.nan

    rows.append({
        "Composition": r["Composition"],
        "n_metals": float(r["n_metals"]),
        "P_single_cal": float(r["P_single_cal"]),
        "n_feat_used": int(used),
        "cum_pmean": cum_pmean,
        "bin_pmean_avg": bin_pmean_avg,
        "hits_count": int(hits),
        **{f"hit_{k}": v for k, v in per_feat_hits.items()}
    })

score_df = pd.DataFrame(rows)

# Restrict to HEC9 pool
pool9 = score_df[score_df["n_metals"] == float(TARGET_N_METALS)].copy()

# Coverage (fraction of features used for each composition)
n_total_feats = max(1, len(plot_feats_unique))
pool9["coverage"] = pool9["n_feat_used"] / float(n_total_feats)

# Composite score (z-scored components)
pool9["cum_pmean_z"]    = zscore(pool9["cum_pmean"])
pool9["P_single_cal_z"] = zscore(pool9["P_single_cal"])
pool9["coverage_z"]     = zscore(pool9["coverage"])

pool9["Score"] = (
    W_CUM  * pool9["cum_pmean_z"] +
    W_POST * pool9["P_single_cal_z"] +
    W_COV  * pool9["coverage_z"]
)

# Final ranking: Score, then cum_pmean, then avg bin P
pool9_ranked = pool9.sort_values(
    by=["Score", "cum_pmean", "bin_pmean_avg"],
    ascending=[False, False, False]
).reset_index(drop=True)
pool9_ranked["Rank"] = np.arange(1, len(pool9_ranked) + 1)

# Save results
rank_csv = OUT / "HEC9_rank_composite_noEFA.csv"
keep_cols = ["Rank","Composition","Score","P_single_cal","coverage","cum_pmean",
             "bin_pmean_avg","hits_count","n_feat_used"]
pool9_ranked[keep_cols].to_csv(rank_csv, index=False)
print(f"\nSaved composite ranking → {rank_csv}")

# Show top-20 (text fallback outside notebooks)
try:
    from IPython.display import display
    display(pool9_ranked[keep_cols].head(20))
except Exception:
    print(pool9_ranked[keep_cols].head(20).to_string(index=False))

# =============== Report cluster/component for selected HEC9s ===============
selected_raw = [
    "(HfMoNbReTaTiVWZr)C9",
    "(HfMoNbTaTiVWYZr)C9",
    "(HfMoNbScTaTiVWZr)C9",
]
selected = [canonicalize(s) for s in selected_raw]

cols_to_show = [
    "Composition", "gmm_comp", "comp_phase_label", "sample_phase_label",
    "P_single_cal", "n_metals"
]

sel_info = dfX[dfX["Composition"].isin(selected)][cols_to_show].copy()
sel_info = sel_info.sort_values("Composition").reset_index(drop=True)

print("\n=== Selected HEC9 cluster/component assignments ===")
if len(sel_info):
    print(sel_info.to_string(index=False))
else:
    print("None of the selected compositions were present after filtering/feature NA removal.")

(OUT / "eval").mkdir(exist_ok=True, parents=True)
sel_path = OUT / "eval" / "selected_HEC9_assignments.csv"
sel_info.to_csv(sel_path, index=False)
print(f"Saved selected assignments → {sel_path}")

missing = [s for s in selected if s not in set(sel_info["Composition"])]
if missing:
    print("WARNING: The following selected compositions were not found in the clustering pool:")
    for m in missing:
        print(" -", m, "(possible NA in features or not in input file)")

# OPTIONAL: overlay selected compositions on each feature plot (markers at their x and bin-mean y)
# (Same three as above; re-list kept for clarity of this block.)
selected_raw = [
    "(HfMoNbReTaTiVWZr)C9",
    "(HfMoNbTaTiVWYZr)C9",
    "(HfMoNbScTaTiVWZr)C9",
]
selected = [canonicalize(s) for s in selected_raw]
sel_df = dfX[dfX["Composition"].isin(selected)].copy()

for (_c, feat_col) in plot_feats_unique:
    df_band = bands_df[bands_df["feature"] == feat_col].copy()
    fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=220)
    ax.plot(df_band["bin_mid"], df_band["bin_pmean"], marker="o", lw=1)
    for _, rr in df_band[df_band["is_favorable"]].iterrows():
        ax.axvspan(rr["bin_left"], rr["bin_right"], color="tab:green", alpha=0.12)

    # overlay selected as points at (x_sel, corresponding bin-mean y)
    L = band_lookup[feat_col]
    for _, srow in sel_df.iterrows():
        xv = srow.get(feat_col, np.nan)
        if pd.isna(xv):
            continue
        bidx = L["intervals"].get_indexer([xv])[0]
        if bidx == -1:
            continue
        yv = float(L["pmean"][bidx])
        ax.plot([xv], [yv], marker="D", ms=5, mec="k", mfc="none")
        ax.annotate(srow["Composition"].replace(")C9", ""), (xv, yv),
                    xytext=(3, 3), textcoords="offset points", fontsize=7)

    ax.set_xlabel(feat_col, fontsize=9)
    ax.set_ylabel("P(single) (bin mean)", fontsize=9)
    ax.set_title(f"P(single) vs {feat_col}", fontsize=10)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT / f"posterior_vs_{feat_col.replace(' ', '_')}_with_selected.png", dpi=240)
    plt.close(fig)


print("Saved outputs in:", OUT.resolve())
