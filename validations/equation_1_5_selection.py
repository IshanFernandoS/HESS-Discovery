from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from validations.common import (
    DEFAULT_DESCRIPTOR_PATH,
    DEFAULT_SELECTED_CANDIDATES,
    apply_equation_selection,
    canonicalize,
    compute_equation_basin_scores,
    descriptor_feature_columns,
    fit_gmm_posterior,
    load_descriptor_table,
)
from validations.generate_refined_figure_s10 import (
    build_matminer_element_feature_lookup,
    graph_from_composition_matminer,
    load_model,
    predict_single_graphs,
)


def parse_grid(value: str) -> list[float]:
    grid = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not grid or any(item <= 0 or item > 1 for item in grid):
        raise argparse.ArgumentTypeError("Grid values must lie in (0, 1].")
    return grid


def get_efa_predictions(
    compositions: list[str],
    deephec_dir: Path,
    checkpoint: Path,
    device: torch.device,
    efa_csv: Path | None,
) -> tuple[pd.DataFrame, dict]:
    if efa_csv is not None:
        table = pd.read_csv(efa_csv)
        if "Composition" not in table.columns:
            raise ValueError("The EFA CSV must contain a Composition column.")
        source_column = next(
            (
                column
                for column in ["Predicted_EFA", "Predicted_EFA_eval_mode", "MC_mean"]
                if column in table.columns
            ),
            None,
        )
        if source_column is None:
            raise ValueError(
                "The EFA CSV must contain Predicted_EFA, Predicted_EFA_eval_mode, or MC_mean."
            )
        efa = table[["Composition", source_column]].copy()
        efa["Composition"] = efa["Composition"].astype(str).map(canonicalize)
        efa = efa.rename(columns={source_column: "Predicted_EFA"})
        provenance = {"source": "csv", "path": str(efa_csv), "column": source_column}
        return efa, provenance

    element_features, feature_labels = build_matminer_element_feature_lookup(deephec_dir)
    model = load_model(checkpoint, device, in_channels=len(feature_labels))
    graphs = [graph_from_composition_matminer(comp, element_features) for comp in compositions]
    predictions = predict_single_graphs(model, graphs, device)
    efa = pd.DataFrame({"Composition": compositions, "Predicted_EFA": predictions})
    provenance = {
        "source": "model",
        "checkpoint": str(checkpoint),
        "architecture": "GCNEncoder(hidden_dim=4, num_layers=2, heads=32, dropout=0.2)",
        "feature_generation": "manuscript-consistent matminer/pymatgen path",
        "n_node_features": len(feature_labels),
        "device": str(device),
    }
    return efa, provenance


def stability_analysis(
    basin_scores: pd.DataFrame,
    efa: pd.DataFrame,
    rho_grid: list[float],
    eta_grid: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = []
    details = []
    for rho in rho_grid:
        for eta in eta_grid:
            selected = apply_equation_selection(basin_scores, efa, rho=rho, eta=eta)
            runs.append(
                {
                    "rho": rho,
                    "eta": eta,
                    "n_Eq3": int(selected["Eq3_selected"].sum()),
                    "n_Eq5": int(selected["Eq5_selected"].sum()),
                }
            )
            part = selected[
                [
                    "Composition",
                    "n_metals",
                    "Basin_score",
                    "Basin_rank",
                    "Predicted_EFA",
                    "EFA_rank_within_Eq3",
                    "Eq3_selected",
                    "Eq5_selected",
                ]
            ].copy()
            part["rho"] = rho
            part["eta"] = eta
            details.append(part)
    detail = pd.concat(details, ignore_index=True)
    frequency = (
        detail.groupby(["Composition", "n_metals"], as_index=False)
        .agg(
            Eq3_frequency=("Eq3_selected", "mean"),
            Eq5_frequency=("Eq5_selected", "mean"),
            Basin_score=("Basin_score", "first"),
            Predicted_EFA=("Predicted_EFA", "first"),
            median_Basin_rank=("Basin_rank", "median"),
        )
        .sort_values(["n_metals", "Eq5_frequency", "Basin_score"], ascending=[True, False, False])
    )
    return pd.DataFrame(runs), frequency


def candidate_audit(selection: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    wanted = {canonicalize(comp) for comp in DEFAULT_SELECTED_CANDIDATES}
    columns = [
        "Composition",
        "n_metals",
        "P_single",
        "Basin_score",
        "Basin_rank",
        "Eq3_selected",
        "Predicted_EFA",
        "EFA_rank_within_Eq3",
        "Eq5_selected",
        "rho",
        "eta",
        "Eq3_count",
        "Eq5_count",
    ]
    audit = selection[selection["Composition"].isin(wanted)][columns].copy()
    audit = audit.merge(
        stability[["Composition", "Eq3_frequency", "Eq5_frequency"]],
        on="Composition",
        how="left",
        validate="one_to_one",
    )
    return audit.sort_values(["n_metals", "Composition"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the manuscript's Equations 1--5 without the legacy composite score."
    )
    parser.add_argument("--descriptor-path", default=str(DEFAULT_DESCRIPTOR_PATH))
    parser.add_argument(
        "--deephec-dir",
        default=".",
        help="Repository root containing data/ and checkpoints/.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--efa-csv",
        default=None,
        help="Optional complete EFA prediction table; otherwise the manuscript checkpoint is run.",
    )
    parser.add_argument("--outdir", default="validation_outputs/equation_1_5_selection")
    parser.add_argument("--rho", type=float, default=0.30)
    parser.add_argument("--eta", type=float, default=0.10)
    parser.add_argument("--rho-grid", type=parse_grid, default=parse_grid("0.10,0.20,0.30,0.40,0.50"))
    parser.add_argument("--eta-grid", type=parse_grid, default=parse_grid("0.05,0.10,0.20,0.30,0.50"))
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--roll-smooth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.rho <= 1 or not 0 < args.eta <= 1:
        raise ValueError("rho and eta must lie in (0, 1].")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    descriptor_path = Path(args.descriptor_path)
    deephec = Path(args.deephec_dir)
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else deephec / "checkpoints" / "best_gnn_model_4_2_32_88.0.pth"
    )
    if args.efa_csv is None and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    descriptors = load_descriptor_table(descriptor_path)
    feature_cols = descriptor_feature_columns(descriptors)
    gmm = fit_gmm_posterior(descriptors, feature_cols, seed=args.seed)
    basin_scores, bands = compute_equation_basin_scores(
        gmm.df,
        feature_cols,
        n_bins_per_feature=args.n_bins,
        roll_smooth=args.roll_smooth,
    )
    candidate_scores = basin_scores[basin_scores["n_metals"].isin([9.0, 10.0, 11.0])].copy()
    compositions = candidate_scores["Composition"].tolist()
    efa, efa_provenance = get_efa_predictions(
        compositions,
        deephec_dir=deephec,
        checkpoint=checkpoint,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        efa_csv=Path(args.efa_csv) if args.efa_csv else None,
    )
    if len(set(compositions).difference(set(efa["Composition"]))) > 0:
        missing = sorted(set(compositions).difference(set(efa["Composition"])))
        raise ValueError(f"Missing EFA predictions for {len(missing)} candidates; first: {missing[:5]}")

    selection = apply_equation_selection(
        candidate_scores,
        efa,
        rho=args.rho,
        eta=args.eta,
    )
    grid_summary, stability = stability_analysis(
        candidate_scores,
        efa,
        rho_grid=args.rho_grid,
        eta_grid=args.eta_grid,
    )
    audit = candidate_audit(selection, stability)

    bands.to_csv(outdir / "equation_1_descriptor_evidence_bands.csv", index=False)
    candidate_scores.to_csv(outdir / "equation_2_basin_scores.csv", index=False)
    efa.to_csv(outdir / "equation_4_gnn_efa_predictions.csv", index=False)
    selection.to_csv(outdir / "equations_1_5_full_selection.csv", index=False)
    grid_summary.to_csv(outdir / "rho_eta_grid_summary.csv", index=False)
    stability.to_csv(outdir / "rho_eta_selection_stability.csv", index=False)
    audit.to_csv(outdir / "experimental_candidate_audit.csv", index=False)
    with (outdir / "provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "descriptor_path": str(descriptor_path),
                "feature_columns": feature_cols,
                "gmm": gmm.best,
                "gmm_seed": args.seed,
                "equation_1": "s_j(c) = smoothed quantile-bin mean of calibrated GMM P_single",
                "equation_2": "B(c) = arithmetic mean of available s_j(c)",
                "equation_3_rho": args.rho,
                "equation_5_eta": args.eta,
                "selection_scope": "Equations 3 and 5 applied independently within HEC9, HEC10, HEC11",
                "rho_grid": args.rho_grid,
                "eta_grid": args.eta_grid,
                "efa": efa_provenance,
                "legacy_composite_score_used": False,
            },
            handle,
            indent=2,
        )

    print(f"GMM: k={gmm.best['k']}, covariance={gmm.best['covariance_type']}")
    print(f"Default selection: rho={args.rho:.2f}, eta={args.eta:.2f}")
    print("\nExperimental candidate audit:")
    print(audit.to_string(index=False))
    print(f"\nOutputs: {outdir.resolve()}")


if __name__ == "__main__":
    main()
