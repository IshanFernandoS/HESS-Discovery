from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from validations.common import (
    DEFAULT_DESCRIPTOR_PATH,
    DEFAULT_OUTPUT_DIR,
    descriptor_feature_columns,
    ensure_dir,
    fit_gmm_posterior,
    load_descriptor_table,
)


FEATURE_LABELS = {
    "norm_afe": "Average formation enthalpy",
    "norm_fg": "Formation gap",
    "norm_rmis": "Radius mismatch",
    "norm_xmis": "Electronegativity mismatch",
    "norm_ca": "Carbon affinity",
    "norm_mp": "Minimum carbide melting point",
    "norm_vec": "Average valence count",
    "norm_cfdi": "Carbide formation deviation index",
    "norm_msi": "Metastable segregation index",
}

# Direction convention on the normalized GMM axes used in the frozen code.
# VEC is omitted because it is not treated as one-sided in the SI diagnostics.
SUPPORT_DIRECTIONS = {
    "norm_afe": -1,
    "norm_fg": -1,
    "norm_rmis": -1,
    "norm_xmis": -1,
    "norm_ca": +1,
    "norm_mp": +1,
    "norm_cfdi": -1,
    "norm_msi": +1,
}


def latex_escape(text: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def driver_text(feature: str, z_shift: float, latex: bool = False) -> str:
    label = FEATURE_LABELS.get(feature, feature)
    if latex:
        arrow = r"\(\uparrow\)" if z_shift >= 0 else r"\(\downarrow\)"
        return f"{arrow} {latex_escape(label)}"
    arrow = "up" if z_shift >= 0 else "down"
    return f"{arrow} {label}"


def build_component_table(result) -> pd.DataFrame:
    comp_ids = result.df["gmm_comp"].to_numpy(dtype=int)
    p_single = pd.to_numeric(result.df["P_single"], errors="coerce").to_numpy(dtype=float)
    rows = []
    for comp in range(int(result.gmm.n_components)):
        mask = comp_ids == comp
        if not mask.any():
            mean_z = np.full(len(result.feature_cols), np.nan)
        else:
            mean_z = result.x_scaled[mask].mean(axis=0)

        top_idx = np.argsort(np.abs(np.nan_to_num(mean_z, nan=0.0)))[::-1][:3]
        top_features = [result.feature_cols[i] for i in top_idx]
        top_shifts = [float(mean_z[i]) for i in top_idx]

        support_flags = []
        for feature, z_shift in zip(result.feature_cols, mean_z):
            if feature not in SUPPORT_DIRECTIONS or not np.isfinite(z_shift):
                continue
            if abs(z_shift) < 1e-12:
                continue
            support_flags.append(np.sign(z_shift) == SUPPORT_DIRECTIONS[feature])
        support_fraction = float(np.mean(support_flags)) if support_flags else np.nan

        rows.append(
            {
                "component": int(comp),
                "cluster": f"C{comp}",
                "n_c": int(mask.sum()),
                "phase_mapping": "single-like" if result.component_is_single[comp] else "multi-like",
                "mean_P_single": float(np.nanmean(p_single[mask])) if mask.any() else np.nan,
                "descriptor_support_fraction": support_fraction,
                "top_driver_1": driver_text(top_features[0], top_shifts[0]),
                "top_driver_1_z": top_shifts[0],
                "top_driver_2": driver_text(top_features[1], top_shifts[1]),
                "top_driver_2_z": top_shifts[1],
                "top_driver_3": driver_text(top_features[2], top_shifts[2]),
                "top_driver_3_z": top_shifts[2],
                "top_standardized_drivers": "; ".join(
                    driver_text(feature, z_shift) for feature, z_shift in zip(top_features, top_shifts)
                ),
                "top_standardized_drivers_latex": "; ".join(
                    driver_text(feature, z_shift, latex=True)
                    for feature, z_shift in zip(top_features, top_shifts)
                ),
            }
        )
    return pd.DataFrame(rows)


def write_latex_table(summary: pd.DataFrame, provenance: dict, path: Path) -> None:
    ensure_dir(path.parent)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{\textbf{GMM component diagnostics for the frozen 25-component descriptor-space model.}",
        r"The table was regenerated from the current GMM code and therefore contains components C0--C24,",
        r"consistent with the BIC-selected \(K=25\) full-covariance model reported in Section S4.",
        r"\(n_c\) is the number of complete descriptor rows assigned to each maximum-responsibility component.",
        r"The phase mapping is obtained from the literature-labelled responsibility mass.",
        r"\(\bar{P}_{\mathrm{single}}\) is the mean calibrated single-like posterior of rows assigned to the component.",
        r"The descriptor-support fraction is a diagnostic computed on the normalized descriptor axes using the",
        r"one-sided direction convention stated in Section S4; VEC is excluded from this fraction.",
        r"Top drivers are the three descriptors with the largest absolute standardized component-mean shifts.",
        r"Arrows indicate whether the component mean is above or below the standardized full-data mean and should",
        r"not be interpreted as independent phase rules.}",
        r"\label{tab:s4_gmm_component_diagnostics}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{@{}lrlccp{0.43\textwidth}@{}}",
        r"\toprule",
        r"Component & \(n_c\) & Phase mapping & \(\bar{P}_{\mathrm{single}}\) & Support frac. & Top standardized drivers \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "{} & {:d} & {} & {:.3f} & {:.2f} & {} \\\\".format(
                latex_escape(row["cluster"]),
                int(row["n_c"]),
                latex_escape(row["phase_mapping"]),
                float(row["mean_P_single"]),
                float(row["descriptor_support_fraction"]),
                row["top_standardized_drivers_latex"],
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")

    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate SI Table S4 from the frozen GMM descriptor workflow."
    )
    parser.add_argument("--descriptor-path", default=str(DEFAULT_DESCRIPTOR_PATH))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR / "gmm_cluster_diagnostics"))
    parser.add_argument("--latex-out", default=str(DEFAULT_OUTPUT_DIR / "latex_tables" / "table_s4_gmm_cluster_diagnostics.tex"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    descriptor_path = Path(args.descriptor_path)
    df = load_descriptor_table(descriptor_path)
    feature_cols = descriptor_feature_columns(df)
    result = fit_gmm_posterior(df, feature_cols, seed=args.seed)
    if int(result.best["k"]) != int(result.gmm.n_components):
        raise RuntimeError("Stored best k does not match fitted GMM n_components.")

    summary = build_component_table(result)
    if len(summary) != int(result.best["k"]):
        raise RuntimeError(f"Expected {result.best['k']} components, generated {len(summary)} rows.")

    outdir = ensure_dir(args.outdir)
    csv_path = outdir / "table_s4_gmm_cluster_diagnostics.csv"
    summary.to_csv(csv_path, index=False)

    provenance = {
        "descriptor_path": str(descriptor_path),
        "feature_columns": feature_cols,
        "gmm": result.best,
        "seed": int(args.seed),
        "n_complete_descriptor_rows": int(len(result.df)),
        "n_components_in_table": int(len(summary)),
        "cluster_range": [str(summary["cluster"].iloc[0]), str(summary["cluster"].iloc[-1])],
        "support_direction_convention": SUPPORT_DIRECTIONS,
        "legacy_composite_score_used": False,
    }
    write_latex_table(summary, provenance, Path(args.latex_out))

    print(f"GMM: k={result.best['k']}, covariance={result.best['covariance_type']}, BIC={result.best['bic']:.3f}")
    print(f"Feature columns ({len(feature_cols)}): {', '.join(feature_cols)}")
    print(f"Table rows: {len(summary)} ({summary['cluster'].iloc[0]}--{summary['cluster'].iloc[-1]})")
    print(f"CSV: {csv_path.resolve()}")
    print(f"LaTeX: {Path(args.latex_out).resolve()}")


if __name__ == "__main__":
    main()
