from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from validations.common import (
    DEFAULT_SELECTED_CANDIDATES,
    apply_equation_selection,
    canonicalize,
    ensure_dir,
)
from validations.generate_refined_figure_s10 import (
    build_matminer_element_feature_lookup,
    graph_from_composition_matminer,
    load_model,
    predict_single_graphs,
)


COMPLETE_REALIZATIONS = {
    canonicalize("(HfMoNbReTaTiVWZr)C9"),
    canonicalize("(HfMoNbScTaTiVWZr)C9"),
    canonicalize("(HfMoNbReScTaTiVWZr)C10"),
}

INCOMPLETE_REALIZATIONS = {
    canonicalize("(HfMoNbTaTiVWYZr)C9"),
    canonicalize("(HfMoNbScTaTiVWYZr)C10"),
    canonicalize("(HfMoNbReTaTiVWYZr)C10"),
}

CENSORED_REALIZATIONS = {
    canonicalize("(HfMoNbReScTaTiVWYZr)C11"),
}


def parse_grid(value: str) -> list[float]:
    grid = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not grid or any(item <= 0 or item > 1 for item in grid):
        raise argparse.ArgumentTypeError("Grid values must lie in (0, 1].")
    return grid


def parse_orders(value: str) -> list[int]:
    orders = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not orders or any(item <= 0 for item in orders):
        raise argparse.ArgumentTypeError("Orders must be positive integers.")
    return orders


def outcome_label(composition: str) -> str:
    comp = canonicalize(composition)
    if comp in COMPLETE_REALIZATIONS:
        return "complete"
    if comp in INCOMPLETE_REALIZATIONS:
        return "incomplete"
    if comp in CENSORED_REALIZATIONS:
        return "processing-censored"
    return "not experimentally assessed"


def selection_frequencies(
    basin_scores: pd.DataFrame,
    efa: pd.DataFrame,
    rho_grid: list[float],
    eta_grid: list[float],
    candidate_order: list[str],
) -> pd.Series:
    details = []
    for rho in rho_grid:
        for eta in eta_grid:
            selected = apply_equation_selection(basin_scores, efa, rho=rho, eta=eta)
            part = selected[["Composition", "Eq5_selected"]].copy()
            details.append(part)
    detail = pd.concat(details, ignore_index=True)
    return (
        detail.groupby("Composition")["Eq5_selected"]
        .mean()
        .reindex(candidate_order)
        .fillna(0.0)
        .astype(float)
    )


def separation_margin(freq: pd.Series, complete: set[str], incomplete: set[str]) -> float:
    complete_values = [float(freq[c]) for c in complete if c in freq.index]
    incomplete_values = [float(freq[c]) for c in incomplete if c in freq.index]
    if not complete_values or not incomplete_values:
        return float("nan")
    return float(min(complete_values) - max(incomplete_values))


def summarize_draw_frequencies(
    draw_freq: pd.DataFrame,
    deterministic_freq: pd.Series,
    selected_candidates: list[str],
) -> pd.DataFrame:
    rows = []
    for comp in selected_candidates:
        values = draw_freq.loc[:, comp].to_numpy(dtype=float)
        rows.append(
            {
                "Composition": comp,
                "n_metals": int(comp.rsplit("C", 1)[1]),
                "experimental_outcome": outcome_label(comp),
                "deterministic_frequency": float(deterministic_freq[comp]),
                "dropout_median_frequency": float(np.quantile(values, 0.50)),
                "dropout_q05_frequency": float(np.quantile(values, 0.05)),
                "dropout_q95_frequency": float(np.quantile(values, 0.95)),
                "dropout_mean_frequency": float(np.mean(values)),
                "n_draws": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}\\%"


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            " & ".join(
                [
                    str(row["Composition"]),
                    str(row["experimental_outcome"]),
                    format_percent(row["deterministic_frequency"]),
                    format_percent(row["dropout_median_frequency"]),
                    f"{format_percent(row['dropout_q05_frequency'])}--{format_percent(row['dropout_q95_frequency'])}",
                ]
            )
            + r" \\"
        )
    table = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\scriptsize",
            r"\caption{\textbf{Post hoc dropout-perturbation sensitivity of Eq. (5) selection frequency.} Deterministic frequencies use the frozen evaluation-mode GNN predictions and the reported 25-point \((\rho,\eta)\) grid. Dropout summaries are empirical sensitivity intervals from stochastic forward-pass perturbations of the same checkpoint; they are not confidence intervals for synthesis success.}",
            r"\label{tab:dropout_selection_frequency_sensitivity}",
            r"\begin{tabular}{lllll}",
            r"\toprule",
            r"Composition & Outcome & Deterministic & Dropout median & Dropout 5--95\% \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text(table, encoding="utf-8")


def mc_dropout_graphs(
    model: torch.nn.Module,
    graphs: list,
    device: torch.device,
    n_draws: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_graphs = [graph.to(device) for graph in graphs]
    samples = np.empty((int(n_draws), len(device_graphs)), dtype=float)
    model.train()
    with torch.no_grad():
        for draw_idx in range(int(n_draws)):
            for graph_idx, graph in enumerate(device_graphs):
                samples[draw_idx, graph_idx] = float(
                    model(graph).view(-1).detach().cpu().numpy()[0]
                )
    model.eval()
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post hoc sensitivity of Eq. (5) selection frequency to MC-dropout "
            "perturbations of the manuscript-consistent EFA-GNN predictions."
        )
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--checkpoint", default="checkpoints/best_gnn_model_4_2_32_88.0.pth")
    parser.add_argument(
        "--basin-scores",
        default="validation_outputs/equation_1_5_selection/equation_2_basin_scores.csv",
    )
    parser.add_argument(
        "--deterministic-efa",
        default="validation_outputs/equation_1_5_selection/equation_4_gnn_efa_predictions.csv",
    )
    parser.add_argument(
        "--outdir",
        default="validation_outputs/selection_frequency_dropout_sensitivity",
    )
    parser.add_argument("--rho-grid", type=parse_grid, default=parse_grid("0.10,0.20,0.30,0.40,0.50"))
    parser.add_argument("--eta-grid", type=parse_grid, default=parse_grid("0.05,0.10,0.20,0.30,0.50"))
    parser.add_argument("--n-draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--orders", type=parse_orders, default=parse_orders("9,10,11"))
    args = parser.parse_args()

    repo = Path(args.repo_root)
    outdir = ensure_dir(args.outdir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    orders = list(args.orders)

    basin_scores = pd.read_csv(args.basin_scores)
    basin_scores["Composition"] = basin_scores["Composition"].astype(str).map(canonicalize)
    basin_scores = basin_scores[basin_scores["n_metals"].isin([float(x) for x in orders])].copy()
    basin_scores = basin_scores.sort_values(["n_metals", "Composition"]).reset_index(drop=True)
    candidate_order = basin_scores["Composition"].tolist()

    deterministic_efa = pd.read_csv(args.deterministic_efa)
    deterministic_efa["Composition"] = deterministic_efa["Composition"].astype(str).map(canonicalize)
    deterministic_freq = selection_frequencies(
        basin_scores,
        deterministic_efa,
        rho_grid=args.rho_grid,
        eta_grid=args.eta_grid,
        candidate_order=candidate_order,
    )

    element_features, feature_labels = build_matminer_element_feature_lookup(repo)
    graphs = [graph_from_composition_matminer(comp, element_features) for comp in candidate_order]
    model = load_model(Path(args.checkpoint), device, in_channels=len(feature_labels))
    eval_pred = predict_single_graphs(model, graphs, device=device)

    merged_eval = (
        deterministic_efa[["Composition", "Predicted_EFA"]]
        .drop_duplicates("Composition")
        .set_index("Composition")
        .reindex(candidate_order)["Predicted_EFA"]
        .to_numpy(dtype=float)
    )
    max_eval_abs_diff = float(np.nanmax(np.abs(eval_pred - merged_eval)))
    if max_eval_abs_diff > 1e-4:
        raise ValueError(
            "Eval-mode predictions do not reproduce the deterministic Eq. (4) table; "
            f"maximum absolute difference is {max_eval_abs_diff:.6g}."
        )

    samples = mc_dropout_graphs(
        model,
        graphs,
        device=device,
        n_draws=args.n_draws,
        seed=args.seed,
    )

    draw_freq_rows = []
    selected_candidates = [canonicalize(c) for c in DEFAULT_SELECTED_CANDIDATES]
    for draw_idx in range(samples.shape[0]):
        efa_draw = pd.DataFrame(
            {"Composition": candidate_order, "Predicted_EFA": samples[draw_idx, :]}
        )
        freq = selection_frequencies(
            basin_scores,
            efa_draw,
            rho_grid=args.rho_grid,
            eta_grid=args.eta_grid,
            candidate_order=candidate_order,
        )
        row = {"draw": int(draw_idx)}
        row.update({comp: float(freq[comp]) for comp in candidate_order})
        draw_freq_rows.append(row)
    draw_freq = pd.DataFrame(draw_freq_rows)

    draw_efa = pd.DataFrame(samples, columns=candidate_order)
    draw_efa.insert(0, "draw", np.arange(samples.shape[0], dtype=int))
    draw_efa.to_csv(outdir / "mc_dropout_efa_draws_wide.csv", index=False)
    draw_freq.to_csv(outdir / "dropout_selection_frequency_draws_wide.csv", index=False)

    summary = summarize_draw_frequencies(draw_freq, deterministic_freq, selected_candidates)
    summary.to_csv(outdir / "experimental_candidate_dropout_frequency_summary.csv", index=False)
    write_latex_table(summary, outdir / "table_dropout_selection_frequency_sensitivity.tex")

    complete = COMPLETE_REALIZATIONS
    incomplete = INCOMPLETE_REALIZATIONS
    hec9_complete = {c for c in complete if c.endswith("C9")}
    hec9_incomplete = {c for c in incomplete if c.endswith("C9")}
    hec10_complete = {c for c in complete if c.endswith("C10")}
    hec10_incomplete = {c for c in incomplete if c.endswith("C10")}

    margins = []
    for _, row in draw_freq.iterrows():
        freq = row.drop(labels=["draw"]).astype(float)
        margins.append(
            {
                "draw": int(row["draw"]),
                "D_all": separation_margin(freq, complete, incomplete),
                "D_HEC9": separation_margin(freq, hec9_complete, hec9_incomplete),
                "D_HEC10": separation_margin(freq, hec10_complete, hec10_incomplete),
            }
        )
    margins_df = pd.DataFrame(margins)
    margins_df.to_csv(outdir / "dropout_separation_margins.csv", index=False)

    separation_summary = []
    for name, column in [
        ("all_characterized_HEC9_HEC10", "D_all"),
        ("HEC9_only", "D_HEC9"),
        ("HEC10_only", "D_HEC10"),
    ]:
        values = margins_df[column].to_numpy(dtype=float)
        ok = np.isfinite(values)
        separation_summary.append(
            {
                "comparison": name,
                "n_draws": int(ok.sum()),
                "fraction_D_gt_0": float(np.mean(values[ok] > 0.0)) if ok.any() else float("nan"),
                "fraction_D_eq_0": float(np.mean(values[ok] == 0.0)) if ok.any() else float("nan"),
                "fraction_D_lt_0": float(np.mean(values[ok] < 0.0)) if ok.any() else float("nan"),
                "median_D": float(np.quantile(values[ok], 0.50)) if ok.any() else float("nan"),
                "q05_D": float(np.quantile(values[ok], 0.05)) if ok.any() else float("nan"),
                "q95_D": float(np.quantile(values[ok], 0.95)) if ok.any() else float("nan"),
            }
        )
    separation_summary_df = pd.DataFrame(separation_summary)
    separation_summary_df.to_csv(outdir / "dropout_separation_summary.csv", index=False)

    with (outdir / "provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "analysis": "post hoc dropout perturbation sensitivity of Eq. (5) selection frequency",
                "repo_root": str(repo.resolve()),
                "checkpoint": str(Path(args.checkpoint)),
                "basin_scores": str(Path(args.basin_scores)),
                "deterministic_efa": str(Path(args.deterministic_efa)),
                "feature_generation": "build_matminer_element_feature_lookup plus graph_from_composition_matminer",
                "n_node_features": int(len(feature_labels)),
                "n_candidates": int(len(candidate_order)),
                "orders": orders,
                "rho_grid": args.rho_grid,
                "eta_grid": args.eta_grid,
                "n_draws": int(args.n_draws),
                "seed": int(args.seed),
                "device": str(device),
                "max_eval_abs_diff_vs_deterministic_table": max_eval_abs_diff,
                "interpretation": (
                    "Dropout masks are sampled during ordinary stochastic forward passes; "
                    "the result is a perturbation sensitivity analysis, not a calibrated "
                    "confidence interval for synthesis success."
                ),
            },
            handle,
            indent=2,
        )

    print(f"Saved outputs to {outdir.resolve()}")
    print(summary.to_string(index=False))
    print()
    print(separation_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
