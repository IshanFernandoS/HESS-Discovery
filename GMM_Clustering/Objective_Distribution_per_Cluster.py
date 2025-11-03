# ===================== Per-cluster objective distributions (NO REFIT) =====================
# Outputs:
# - gmm_band_rank/cluster_XX_objective_dists.(pdf|png) for each component
# - gmm_band_rank/cluster_objective_stats.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ---- prerequisites from your session (do not refit here) ----
required_names = ["gmm", "resp", "dfX", "X_scaled", "comp_is_single"]
missing_names = [v for v in required_names if v not in globals()]
if missing_names:
    raise RuntimeError(f"Missing variables: {missing_names}. Run your GMM cell first.")

OUT = Path("gmm_band_rank")
OUT.mkdir(parents=True, exist_ok=True)

n_components = gmm.n_components
comp_weights = gmm.weights_
# approximate soft size per component (kept as in original for CSV)
comp_sizes = (comp_weights * len(X_scaled)).astype(int)

# ---- Objectives to visualize (prefer existing OBJECTIVES from session) ----
try:
    OBJECTIVES  # if already set, keep it
except NameError:
    obj_candidates = [
        "Formation Gap", "Radius Mismatch", "Electronegativity Mismatch", "Carbon Affinity",
        "Magnetic Disorder Risk Index", "Minimum Carbide Melting Point", "Average Valence Count",
        "Carbide Formation Deviation Index", "Metastable Segregation Index", "Average Formation Enthalpy",
    ]
    OBJECTIVES = [c for c in obj_candidates if c in dfX.columns]
if len(OBJECTIVES) < 2:
    raise ValueError(f"Need ≥2 objectives present in dfX; found {OBJECTIVES}")

# ---- Data arrays in original units ----
X_obj = dfX[OBJECTIVES].apply(pd.to_numeric, errors="coerce").to_numpy(float)  # [N, J]
N, J = X_obj.shape

# ---- Global stats (z-deltas & shared x-limits) ----
mu_glob = np.nanmean(X_obj, axis=0)
sd_glob = np.nanstd(X_obj, axis=0) + 1e-12  # avoid divide-by-zero

# Per-objective bin edges (shared across clusters) using robust percentiles
bin_edges = []
for j in range(J):
    xj = X_obj[:, j]
    xj = xj[np.isfinite(xj)]
    if xj.size < 10:
        lo, hi = np.nanmin(xj), np.nanmax(xj)
    else:
        lo, hi = np.percentile(xj, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = (0.0, 1.0)
    bin_edges.append(np.linspace(lo, hi, 40))  # keep 40 bins as-is

# ---- Weighted quantile helper (for whiskers/labels) ----
def weighted_quantile(values, quantiles, sample_weight=None):
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    values = values[mask]
    if sample_weight is None:
        sample_weight = np.ones_like(values)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)[mask]
    if values.size == 0 or sample_weight.sum() <= 0:
        return np.full_like(np.atleast_1d(quantiles), np.nan, dtype=float)
    sorter = np.argsort(values)
    values, sample_weight = values[sorter], sample_weight[sorter]
    cdf = np.cumsum(sample_weight) / np.sum(sample_weight)
    return np.interp(np.atleast_1d(quantiles), cdf, values)

# ---- Collect stats for CSV ----
rows = []

# ===== For each component, produce a multi-panel figure =====
for k in range(n_components):
    w = resp[:, k].astype(float)    # responsibilities (soft membership)
    w_sum = float(np.nansum(w))
    phase_color = "#E69F00" if comp_is_single[k] else "#0072B2"  # orange=Single-like, blue=Multi-like
    phase_label = "Single-like" if comp_is_single[k] else "Multi-like"

    # Grid layout: up to 4 columns, auto rows
    ncols = min(4, J)
    nrows = int(np.ceil(J / ncols))
    fig_w = 7.2
    fig_h = max(2.0, 1.8 * nrows)

    # Consistent publication-ish defaults (kept identical)
    mpl.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 600,
        "font.size": 8.5, "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    ax_iter = iter(axes.ravel())

    for j, obj in enumerate(OBJECTIVES):
        ax = next(ax_iter)
        xj = X_obj[:, j]
        mask_glob = np.isfinite(xj)
        mask_k = mask_glob & np.isfinite(w) & (w > 0)

        # Global histogram (gray, normalized)
        hg, eg = np.histogram(xj[mask_glob], bins=bin_edges[j], density=True)
        cx = 0.5 * (eg[:-1] + eg[1:])
        ax.fill_between(cx, hg, step="mid", color="0.8", alpha=0.8, label="Global", zorder=1)

        # Cluster histogram (soft-weighted, normalized)
        if mask_k.sum() >= 5 and w_sum > 1e-8:
            hk, _ = np.histogram(xj[mask_k], bins=bin_edges[j], weights=w[mask_k], density=True)
            ax.step(cx, hk, where="mid", color=phase_color, lw=1.5, label=f"Comp {k}", zorder=3)
            ax.fill_between(cx, hk, step="mid", color=phase_color, alpha=0.20, zorder=2)

            # Weighted mean / quantiles for component
            mu_kj = float(np.nansum(xj[mask_k] * w[mask_k]) / np.nansum(w[mask_k]))
            q25, q50, q75 = weighted_quantile(xj[mask_k], [0.25, 0.50, 0.75], sample_weight=w[mask_k])

            # z-delta (component vs global)
            zd = (mu_kj - mu_glob[j]) / sd_glob[j]
            ax.axvline(mu_kj, color=phase_color, lw=1.2, linestyle="--")
            ax.text(
                0.98, 0.95, f"μ_k={mu_kj:.2g}\nΔz={zd:+.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color=phase_color,
                bbox=dict(fc="white", ec=phase_color, lw=0.6, alpha=0.9)
            )

            # Save stats row (kept column names)
            rows.append({
                "component": k,
                "phase": phase_label,
                "objective": obj,
                "global_mean": float(mu_glob[j]),
                "cluster_mean": float(mu_kj),
                "z_delta": float(zd),
                "q25": float(q25), "q50": float(q50), "q75": float(q75),
                "approx_size": int(comp_sizes[k]),
            })
        else:
            ax.text(
                0.5, 0.5, "insufficient mass",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="0.4"
            )

        # Global mean line and cosmetics
        ax.axvline(mu_glob[j], color="0.4", lw=0.8, linestyle=":")
        ax.set_title(obj, fontsize=7)
        ax.set_yticks([])
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # Turn off any unused subplots (if J not multiple of ncols)
    for ax in ax_iter:
        ax.axis("off")

    fig.suptitle(f"Sub-Cluster {k}", y=0.995)
    # Legend intentionally left off (same as original)

    plt.tight_layout()
    pdf_path = OUT / f"cluster_{k:02d}_objective_dists.pdf"
    png_path = OUT / f"cluster_{k:02d}_objective_dists.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=600)
    plt.show()
    print("Saved:", pdf_path, "and", png_path)

# ---- Save stats table for the supplement ----
stats_df = pd.DataFrame(rows)
stats_path = OUT / "cluster_objective_stats.csv"
stats_df.to_csv(stats_path, index=False)
print("Saved:", stats_path)