from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from validations.common import (
    DEFAULT_DESCRIPTOR_PATH,
    DEFAULT_OUTPUT_DIR,
    canonicalize,
    descriptor_feature_columns,
    ensure_dir,
    fit_gmm_posterior,
    load_descriptor_table,
    zscore,
)


SELECTED_CANDIDATES = [
    "(HfMoNbReTaTiVWZr)C9",
    "(HfMoNbScTaTiVWZr)C9",
    "(HfMoNbTaTiVWYZr)C9",
    "(HfMoNbReScTaTiVWZr)C10",
    "(HfMoNbScTaTiVWYZr)C10",
    "(HfMoNbReTaTiVWYZr)C10",
    "(HfMoNbReScTaTiVWYZr)C11",
]

RAW_NAME_MAP = {
    "norm_fg": "Formation Gap",
    "norm_rmis": "Radius Mismatch",
    "norm_xmis": "Electronegativity Mismatch",
    "norm_ca": "Carbon Affinity",
    "norm_mdri": "Magnetic Disorder Risk Index",
    "norm_mp": "Minimum Carbide Melting Point",
    "norm_vec": "Average Valence Count",
    "norm_cfdi": "Carbide Formation Deviation Index",
    "norm_msi": "Metastable Segregation Index",
    "norm_afe": "Average Formation Enthalpy",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision_single": "Precision",
    "recall_single": "Recall",
    "f1_single": "F1 score",
}

ABLATION_LABELS = {
    "baseline": "All descriptors",
    "drop_mismatch": "Without mismatch",
    "drop_bonding_carbon_affinity": "Without bonding/C affinity",
    "drop_refractory_floor": "Without refractory floor",
    "drop_segregation_msi": "Without segregation/MSI",
    "drop_vec": "Without VEC",
}


def count_metals(comp: str) -> int:
    match = re.match(r"^\(([^)]+)\)C(\d+)$", canonicalize(comp))
    if not match:
        return 0
    return len(re.findall(r"[A-Z][a-z]?", match.group(1)))


def latex_comp(comp: str) -> str:
    comp = canonicalize(comp)
    match = re.match(r"^\(([^)]+)\)C(\d+)$", comp)
    if not match:
        return str(comp)
    metals, carbon_count = match.groups()
    return rf"\((\mathrm{{{metals}}})\mathrm{{C}}_{{{carbon_count}}}\)"


def fmt(value: float, decimals: int = 3) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{decimals}f}"


def resolve_raw_feature(df: pd.DataFrame, col: str) -> str | None:
    if col.startswith("norm_"):
        raw = RAW_NAME_MAP.get(col)
        if raw in df.columns:
            return raw
    return col if col in df.columns else None


def safe_qcut(x: pd.Series, q: int) -> pd.Series:
    return pd.qcut(pd.Series(x).astype(float), q=q, duplicates="drop")


def compute_raw_band_scores(
    df_x: pd.DataFrame,
    feature_cols: list[str],
    target_n_metals: int,
    n_bins_per_feature: int = 20,
    favor_top_q: float = 0.30,
    roll_smooth: int = 3,
    weights: tuple[float, float, float] = (0.33, 0.33, 0.34),
) -> pd.DataFrame:
    plot_feats = []
    for col in feature_cols:
        raw = resolve_raw_feature(df_x, col)
        if raw is not None:
            plot_feats.append((col, raw))

    seen = set()
    plot_feats_unique = []
    for col, raw in plot_feats:
        if raw not in seen:
            plot_feats_unique.append((col, raw))
            seen.add(raw)

    bands = []
    for _, feat_col in plot_feats_unique:
        x = pd.to_numeric(df_x[feat_col], errors="coerce")
        p = pd.to_numeric(df_x["P_single"], errors="coerce")
        ok = ~(x.isna() | p.isna())
        qbins = safe_qcut(x[ok], q=n_bins_per_feature)
        tmp = pd.DataFrame({"x": x[ok].to_numpy(), "p": p[ok].to_numpy(), "q": qbins})
        grouped = tmp.groupby("q", observed=True)
        bin_mid = grouped["x"].mean()
        bin_p = grouped["p"].mean()
        if roll_smooth and roll_smooth > 1:
            bin_p = bin_p.rolling(roll_smooth, center=True, min_periods=1).mean()
        n_top = max(1, int(math.ceil(len(bin_p) * favor_top_q)))
        top_idx = set(bin_p.sort_values(ascending=False).index[:n_top])
        bands.append(
            pd.DataFrame(
                {
                    "feature": feat_col,
                    "bin_left": [iv.left for iv in bin_mid.index],
                    "bin_right": [iv.right for iv in bin_mid.index],
                    "bin_pmean": bin_p.to_numpy(dtype=float),
                    "is_favorable": [iv in top_idx for iv in bin_mid.index],
                }
            )
        )

    bands_df = pd.concat(bands, ignore_index=True)
    lookup = {}
    for feat, group in bands_df.groupby("feature"):
        lookup[feat] = {
            "intervals": pd.IntervalIndex.from_arrays(
                group["bin_left"].to_numpy(dtype=float),
                group["bin_right"].to_numpy(dtype=float),
                closed="right",
            ),
            "pmean": group["bin_pmean"].to_numpy(dtype=float),
            "is_favorable": group["is_favorable"].to_numpy(dtype=bool),
        }

    rows = []
    for _, row in df_x.iterrows():
        pmeans = []
        hits = 0
        used = 0
        for _, feat_col in plot_feats_unique:
            value = row.get(feat_col, np.nan)
            if pd.isna(value):
                continue
            item = lookup[feat_col]
            idx = item["intervals"].get_indexer([float(value)])[0]
            if idx == -1:
                continue
            used += 1
            hits += int(item["is_favorable"][idx])
            pmeans.append(float(item["pmean"][idx]))
        rows.append(
            {
                "Composition": row["Composition"],
                "n_metals": float(row["n_metals"]),
                "P_single": float(row["P_single"]),
                "n_feat_used": int(used),
                "coverage": used / float(max(1, len(plot_feats_unique))),
                "cum_pmean": float(np.nansum(pmeans)) if pmeans else np.nan,
                "bin_pmean_avg": float(np.nanmean(pmeans)) if pmeans else np.nan,
                "hits_count": int(hits),
            }
        )
    scores = pd.DataFrame(rows)
    pool = scores[scores["n_metals"] == float(target_n_metals)].copy()
    if pool.empty:
        return pool
    w_cum, w_post, w_cov = weights
    pool["cum_pmean_z"] = zscore(pool["cum_pmean"])
    pool["P_single_z"] = zscore(pool["P_single"])
    pool["coverage_z"] = zscore(pool["coverage"])
    pool["Score"] = w_cum * pool["cum_pmean_z"] + w_post * pool["P_single_z"] + w_cov * pool["coverage_z"]
    pool = pool.sort_values(
        by=["Score", "cum_pmean", "bin_pmean_avg"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    pool["Rank_within_order"] = np.arange(1, len(pool) + 1)
    return pool


def build_candidate_risk_table(validation_dir: Path) -> pd.DataFrame:
    df = load_descriptor_table(DEFAULT_DESCRIPTOR_PATH)
    feature_cols = descriptor_feature_columns(df)
    result = fit_gmm_posterior(df, feature_cols, seed=42)
    ranks = []
    for n_metals in [9, 10, 11]:
        ranks.append(compute_raw_band_scores(result.df, feature_cols, target_n_metals=n_metals))
    rank_df = pd.concat(ranks, ignore_index=True)

    selected = [canonicalize(c) for c in SELECTED_CANDIDATES]
    desc_cols = [
        "Composition",
        "Radius Mismatch",
        "Electronegativity Mismatch",
        "Minimum Carbide Melting Point",
    ]
    desc = result.df[desc_cols].copy()
    out = rank_df[rank_df["Composition"].isin(selected)].merge(desc, on="Composition", how="left")
    out["candidate_order"] = out["Composition"].map(lambda c: selected.index(c))
    out = out.sort_values("candidate_order").drop(columns=["candidate_order"]).reset_index(drop=True)
    ensure_dir(validation_dir / "candidate_validation")
    out.to_csv(validation_dir / "candidate_validation" / "selected_candidate_risk_diagnostics.csv", index=False)
    return out


def write_table_file(text: str, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def make_tables(validation_dir: Path) -> str:
    parts: list[str] = []

    boot = pd.read_csv(validation_dir / "gmm_literature_bootstrap" / "bootstrap_metrics.csv")
    boot = boot.set_index("metric").loc[list(METRIC_LABELS)].reset_index()
    parts.append(r"""\begin{table}[htbp]
\centering
\small
\caption{\textbf{Bootstrap uncertainty for literature-label classification metrics.}
Metrics were computed on the curated literature single-/multiphase label set using the single-phase label as the positive class. Confidence intervals are 95\% intervals from 10,000 bootstrap resamples. The calibrated basin recovers the literature labels with 88.9\% accuracy and high single-phase recall/F1, supporting its use as a literature-consistent solid-solution basin. The interval widths reflect the limited number of multiphase labels and should be interpreted as small-sample uncertainty rather than a population-level error bound.}
\label{tab:bootstrap_literature_metrics}
\begin{tabular}{lccc}
\toprule
Metric & Observed & 95\% CI lower & 95\% CI upper \\
\midrule""")
    for _, row in boot.iterrows():
        parts.append(
            f"{METRIC_LABELS[row['metric']]} & {fmt(row['point'])} & {fmt(row['ci_low'])} & {fmt(row['ci_high'])} \\\\"
        )
    parts.append(r"""\bottomrule
\end{tabular}
\end{table}""")

    ab = pd.read_csv(validation_dir / "feature_ablation" / "ablation_metrics.csv")
    ab = ab.set_index("variant").loc[list(ABLATION_LABELS)].reset_index()
    parts.append(r"""
\begin{table}[htbp]
\centering
\small
\caption{\textbf{Descriptor-family ablation metrics.}
The corrected GMM basin analysis was repeated after removing one descriptor family at a time. No descriptor-family removal collapses the literature-label classification, indicating that basin separation is distributed across multiple physically motivated descriptors. Removal of the segregation/MSI family gives the largest reduction in this run, whereas removing the refractory-floor descriptor slightly increases F1 on the small labelled set. These changes should be interpreted as descriptor-sensitivity diagnostics, not as evidence that any retained physical descriptor is unnecessary.}
\label{tab:descriptor_family_ablation}
\begin{tabular}{lccccc}
\toprule
Descriptor set & Accuracy & Balanced accuracy & Precision & Recall & F1 score \\
\midrule""")
    for _, row in ab.iterrows():
        parts.append(
            f"{ABLATION_LABELS[row['variant']]} & {fmt(row['accuracy'])} & {fmt(row['balanced_accuracy'])} & {fmt(row['precision_single'])} & {fmt(row['recall_single'])} & {fmt(row['f1_single'])} \\\\"
        )
    parts.append(r"""\bottomrule
\end{tabular}
\end{table}""")

    null = pd.read_csv(validation_dir / "label_permutation_null" / "permutation_summary.csv")
    null = null.set_index("metric").loc[list(METRIC_LABELS)].reset_index()
    parts.append(r"""
\begin{table}[htbp]
\centering
\small
\caption{\textbf{Label-shuffling null test for literature-label structure.}
Observed-label metrics are compared with the mean metric obtained after randomly shuffling the single-/multiphase labels while using the same uncalibrated GMM label-mapping workflow. The empirical one-sided \(p\)-value is the fraction of shuffled-label runs with metric value greater than or equal to the observed value. Balanced accuracy, precision and F1 exceed the shuffled-label expectation, supporting that the descriptor basin captures non-random structure in the literature labels. Recall alone is not significant under this null because the labelled set is dominated by single-phase reports.}
\label{tab:label_shuffling_null}
\begin{tabular}{lccc}
\toprule
Metric & Observed & Shuffled-label mean & Empirical \(p\)-value \\
\midrule""")
    for _, row in null.iterrows():
        parts.append(
            f"{METRIC_LABELS[row['metric']]} & {fmt(row['real'])} & {fmt(row['null_mean'])} & {fmt(row['empirical_p_value_greater_equal'])} \\\\"
        )
    parts.append(r"""\bottomrule
\end{tabular}
\end{table}""")

    app = pd.read_csv(
        validation_dir / "applicability_domain" / "selected_applicability_literature_labelled_reference.csv"
    )
    order = [canonicalize(c) for c in SELECTED_CANDIDATES]
    app = app.set_index("Composition").loc[order].reset_index()
    parts.append(r"""
\begin{table*}[htbp]
\centering
\small
\caption{\textbf{Applicability-domain metrics for selected experimental-candidate compositions.}
Nearest-neighbour distances were computed relative only to the curated literature-labelled HEC5/HEC6 compositions in the standardized descriptor space used for the corrected GMM analysis. This is stricter than using the full generated HEC5/HEC6 descriptor manifold. All seven selected higher-order compositions lie outside the literature-labelled reference domain by the 95th-percentile criterion, showing that the planned experimental set probes extrapolative chemistry relative to sparse labelled reports. The nearest-reference label is provided for traceability only and should not be interpreted as a deterministic phase assignment for the candidate.}
\label{tab:selected_candidate_applicability}
\begin{tabular}{llccccp{0.16\textwidth}}
\toprule
Candidate & Nearest literature-labelled composition & Label & NN distance & NN percentile & Mahalanobis percentile & Domain call \\
\midrule""")
    for _, row in app.iterrows():
        call = "Outside labelled domain" if str(row["domain_call"]) == "extrapolation_like" else "Within labelled domain"
        parts.append(
            f"{latex_comp(row['Composition'])} & {latex_comp(row['nearest_labelled_reference'])} & {row['nearest_reference_label']} & {fmt(row['nearest_neighbor_distance'])} & {fmt(row['nn_percentile_vs_labelled_ref'], 1)} & {fmt(row['mahalanobis_percentile_vs_labelled_ref'], 1)} & {call} \\\\"
        )
    parts.append(r"""\bottomrule
\end{tabular}
\end{table*}""")

    risk = build_candidate_risk_table(validation_dir)
    interpretations = {
        canonicalize("(HfMoNbReTaTiVWZr)C9"): "HEC9 baseline containing Re; high posterior but less favourable rank than the Sc/Y HEC9 candidates in the raw-band ranking.",
        canonicalize("(HfMoNbScTaTiVWZr)C9"): "HEC9 Sc-containing candidate with strong rank and moderate mismatch relative to the selected set.",
        canonicalize("(HfMoNbTaTiVWYZr)C9"): "HEC9 Y-containing candidate; high posterior, but the largest mismatch descriptors among the HEC9 selected candidates.",
        canonicalize("(HfMoNbReScTaTiVWZr)C10"): "HEC10 Re+Sc chemistry; favourable order-specific rank but lower refractory floor than Re/Y-containing alternatives.",
        canonicalize("(HfMoNbScTaTiVWYZr)C10"): "HEC10 Sc+Y chemistry; strong order-specific rank and posterior, but the largest mismatch among the selected HEC10 candidates.",
        canonicalize("(HfMoNbReTaTiVWYZr)C10"): "HEC10 Re+Y chemistry; reduced size mismatch relative to Sc+Y, but lower order-specific score and outside the labelled applicability domain.",
        canonicalize("(HfMoNbReScTaTiVWYZr)C11"): "HEC11 Re+Sc+Y chemistry; top-ranked within the small HEC11 pool, but highest order and outside the labelled applicability domain.",
    }
    parts.append(r"""
\begin{table*}[htbp]
\centering
\small
\caption{\textbf{Descriptor-level risk diagnostics for selected experimental-candidate compositions.}
Composite ranks were calculated within the same cation order using the corrected GMM posterior-feature-band ranking, excluding EFA from the descriptor set; the ranking pools contain 220 HEC9, 66 HEC10 and 12 HEC11 compositions. The selected candidates retain high calibrated single-phase posterior values, but their mismatch, refractory-floor and applicability-domain diagnostics separate the more conservative candidates from chemically more stressed Sc/Y/Re-containing higher-order compositions. These values are screening diagnostics and do not replace direct experimental phase analysis.}
\label{tab:selected_candidate_risk_diagnostics}
\begin{tabular}{lcccccccp{0.26\textwidth}}
\toprule
Candidate & Order & Rank & Score & \(P_{\mathrm{single}}\) & \(\delta_r\) & \(\delta_\chi\) & \(T_m^{\min}\) (K) & Descriptor-level interpretation \\
\midrule""")
    for _, row in risk.iterrows():
        comp = canonicalize(row["Composition"])
        parts.append(
            f"{latex_comp(comp)} & HEC{int(row['n_metals'])} & {int(row['Rank_within_order'])} & {fmt(row['Score'])} & {fmt(row['P_single'])} & {fmt(row['Radius Mismatch'], 4)} & {fmt(row['Electronegativity Mismatch'], 4)} & {fmt(row['Minimum Carbide Melting Point'], 0)} & {interpretations[comp]} \\\\"
        )
    parts.append(r"""\bottomrule
\end{tabular}
\end{table*}""")

    hec8 = pd.read_csv(validation_dir / "external_hec8_benchmark" / "hec8_bootstrap_metrics.csv")
    hec8_labels = {"mae": "MAE", "rmse": "RMSE", "r2": r"\(R^2\)"}
    parts.append(r"""
\begin{table}[htbp]
\centering
\small
\caption{\textbf{External HEC8 benchmark for GNN EFA predictions.}
The manuscript-consistent GNN checkpoint was evaluated on 190 external HEC8 compositions with reference EFA values. Confidence intervals are 95\% bootstrap intervals from 10,000 resamples. The negative \(R^2\) and sizeable MAE show that this checkpoint is not quantitatively reliable across the full chemically broad HEC8 space. Therefore, GNN EFA values for HEC9--HEC11 candidates should be treated as screening-level estimates with uncertainty, not standalone proof of phase stability.}
\label{tab:gnn_hec8_external_benchmark}
\begin{tabular}{lccc}
\toprule
Metric & Observed & 95\% CI lower & 95\% CI upper \\
\midrule""")
    for _, row in hec8.iterrows():
        metric = row["metric"]
        decimals = 2 if metric != "r2" else 2
        parts.append(
            f"{hec8_labels[metric]} & {fmt(row['point'], decimals)} & {fmt(row['ci_low'], decimals)} & {fmt(row['ci_high'], decimals)} \\\\"
        )
    parts.append(r"""\bottomrule
\end{tabular}
\end{table}""")

    mc = pd.read_csv(validation_dir / "gnn_refined_s10" / "selected_candidate_mc_dropout_predictions.csv")
    mc = mc.set_index("Composition").loc[order].reset_index()
    parts.append(r"""
\begin{table*}[htbp]
\centering
\small
\caption{\textbf{Candidate-level MC-dropout uncertainty for selected GNN EFA predictions.}
Eval-mode predictions are shown with seeded Monte Carlo dropout means, standard deviations and 5th--95th percentile intervals from 500 stochastic forward passes using the manuscript-consistent checkpoint and feature-generation path. The Re-containing HEC9 candidate is closest to the conventional EFA = 45 threshold by the MC mean, whereas the Y-containing HEC9, Y-containing HEC10 candidates and the HEC11 candidate are mostly below the threshold within the MC interval. These results qualify candidate confidence and should be interpreted together with the GMM basin, applicability-domain and descriptor-risk diagnostics.}
\label{tab:mc_dropout_selected_candidates}
\begin{tabular}{lccccc}
\toprule
Candidate & Eval-mode EFA & MC mean EFA & MC SD & MC 5th percentile & MC 95th percentile \\
\midrule""")
    for _, row in mc.iterrows():
        parts.append(
            f"{latex_comp(row['Composition'])} & {fmt(row['Predicted_EFA_eval_mode'], 1)} & {fmt(row['MC_mean'], 1)} & {fmt(row['MC_std'], 1)} & {fmt(row['MC_q05'], 1)} & {fmt(row['MC_q95'], 1)} \\\\"
        )
    parts.append(r"""\bottomrule
\end{tabular}
\end{table*}""")

    return "\n".join(parts) + "\n"


def main() -> None:
    validation_dir = Path(DEFAULT_OUTPUT_DIR)
    out_path = validation_dir / "latex_tables" / "revised_si_validation_tables.tex"
    text = make_tables(validation_dir)
    write_table_file(text, out_path)
    print(f"Saved revised SI validation tables to {out_path.resolve()}")


if __name__ == "__main__":
    main()
