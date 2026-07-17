from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)


def canonicalize(comp: str) -> str:
    match = re.match(r"^\s*\(([^)]+)\)\s*C?\s*(\d+)\s*$", str(comp).strip())
    if not match:
        return str(comp).strip()
    metals = re.findall(r"[A-Z][a-z]?", match.group(1))
    return f"({''.join(sorted(metals))})C{int(match.group(2))}"


def find_column(df: pd.DataFrame, *needles: str) -> str:
    lowered = {col: str(col).lower() for col in df.columns}
    for col, low in lowered.items():
        if all(needle.lower() in low for needle in needles):
            return col
    raise KeyError(f"No column matching {needles}; available columns: {list(df.columns)}")


def metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float | int]:
    true_label = y_true >= threshold
    pred_label = y_pred >= threshold
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(true_label, pred_label, labels=[False, True]).ravel()]
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_label,
        pred_label,
        average="binary",
        zero_division=0,
    )
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "accuracy": float(accuracy_score(true_label, pred_label)),
        "balanced_accuracy": float(balanced_accuracy_score(true_label, pred_label)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "bias": float(np.mean(y_pred - y_true)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def build_source_table(gnn_path: Path, rf_benchmark_path: Path) -> pd.DataFrame:
    gnn = pd.read_csv(gnn_path)
    required_gnn = {"composition", "efa_actual", "efa_pred"}
    missing = required_gnn.difference(gnn.columns)
    if missing:
        raise ValueError(f"GNN table is missing columns: {sorted(missing)}")
    gnn = gnn[["composition", "efa_actual", "efa_pred"]].copy()
    gnn["composition_key"] = gnn["composition"].map(canonicalize)
    if gnn["composition_key"].duplicated().any():
        dupes = gnn.loc[gnn["composition_key"].duplicated(), "composition"].tolist()
        raise ValueError(f"GNN table contains duplicated compositions: {dupes}")

    rf = pd.read_excel(rf_benchmark_path)
    comp_col = find_column(rf, "composition")
    rf_col = find_column(rf, "random", "forest")
    true_col = find_column(rf, "true")
    rf = rf[[comp_col, rf_col, true_col]].copy()
    rf.columns = ["composition_rf_source", "rf_pred_efa", "true_efa_rf_source"]
    rf["composition_key"] = rf["composition_rf_source"].map(canonicalize)
    rf = rf.drop_duplicates("composition_key", keep="first")

    merged = gnn.merge(
        rf[["composition_key", "rf_pred_efa", "true_efa_rf_source"]],
        on="composition_key",
        how="left",
        validate="one_to_one",
    )
    if merged["rf_pred_efa"].isna().any():
        missing_rf = merged.loc[merged["rf_pred_efa"].isna(), "composition"].tolist()
        raise ValueError(f"Missing RF predictions for: {missing_rf}")

    merged["true_efa"] = pd.to_numeric(merged["efa_actual"], errors="raise")
    merged["gnn_pred_efa"] = pd.to_numeric(merged["efa_pred"], errors="raise")
    merged["rf_pred_efa"] = pd.to_numeric(merged["rf_pred_efa"], errors="raise")
    merged["true_efa_rf_source"] = pd.to_numeric(merged["true_efa_rf_source"], errors="raise")
    max_true_delta = float(np.max(np.abs(merged["true_efa"] - merged["true_efa_rf_source"])))
    if max_true_delta > 0.01:
        raise ValueError(f"True-EFA mismatch between GNN and RF sources exceeds tolerance: {max_true_delta}")

    out = merged[["composition", "true_efa", "gnn_pred_efa", "rf_pred_efa"]].copy()
    if len(out) != 22:
        raise ValueError(f"Expected 22 unique HEC6 compositions, found {len(out)}")
    return out


def configure_matplotlib() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = "EB Garamond" if "EB Garamond" in available_fonts else "DejaVu Serif"
    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.serif": [font_family],
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "font.size": 15.0,
            "axes.labelsize": 18.0,
            "axes.titlesize": 16.5,
            "xtick.labelsize": 15.0,
            "ytick.labelsize": 15.0,
            "legend.fontsize": 15.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot_figure(df: pd.DataFrame, metric_rows: dict[str, dict], output_prefix: Path) -> None:
    configure_matplotlib()
    true = df["true_efa"].to_numpy(dtype=float)
    gnn = df["gnn_pred_efa"].to_numpy(dtype=float)
    rf = df["rf_pred_efa"].to_numpy(dtype=float)

    color_gnn = "#C9A227"
    color_rf = "#4C72B0"
    diag_color = "#8A8A8A"
    grid_color = "#D9D9D9"
    text_color = "#222222"

    fig = plt.figure(figsize=(14.2, 7.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.38, 1.05], wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    lims_min = min(true.min(), gnn.min(), rf.min()) - 2.0
    lims_max = max(true.max(), gnn.max(), rf.max()) + 2.0

    ax1.scatter(true, gnn, s=78, color=color_gnn, alpha=0.92, label="GNN")
    ax1.scatter(true, rf, s=78, color=color_rf, alpha=0.88, label="Random Forest")
    ax1.plot([lims_min, lims_max], [lims_min, lims_max], "--", lw=1.7, color=diag_color)
    ax1.set_xlim(lims_min, lims_max)
    ax1.set_ylim(lims_min, lims_max)
    ax1.set_xlabel("True EFA")
    ax1.set_ylabel("Predicted EFA")
    ax1.grid(True, color=grid_color, linewidth=0.95, linestyle="--", alpha=0.8)
    ax1.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.98), handletextpad=0.5)
    ax1.text(
        1.0,
        1.025,
        f"$n$ = {len(df)}",
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        color="#444444",
        fontsize=14.8,
        clip_on=False,
    )
    ax1.text(-0.13, 1.045, "(a)", transform=ax1.transAxes, fontsize=20.0, ha="left", va="bottom")

    metric_order = [
        ("MAE", "mae"),
        ("RMSE", "rmse"),
        (r"$R^2$", "r2"),
        ("Accuracy\n(EFA $\\geq$ 45)", "accuracy"),
    ]
    sg = gs[0, 1].subgridspec(2, 2, wspace=0.50, hspace=0.66)
    axes = [fig.add_subplot(sg[i, j]) for i in range(2) for j in range(2)]
    for ax, (title, key) in zip(axes, metric_order):
        vals = np.array([metric_rows["GNN"][key], metric_rows["RF"][key]], dtype=float)
        ax.bar([0, 1], vals, width=0.50, color=[color_gnn, color_rf], edgecolor="none")
        ax.set_xticks([0, 1], ["GNN", "RF"])
        ax.set_title(title, pad=8)
        if key in {"r2", "accuracy"}:
            ymin = min(0.0, vals.min() - 0.06)
            ymax = max(1.0, vals.max() + 0.08)
        else:
            ymin = 0.0
            ymax = vals.max() * 1.33
        ax.set_ylim(ymin, ymax)
        ax.grid(axis="y", color=grid_color, linewidth=0.95, linestyle="--", alpha=0.8)
        for i, value in enumerate(vals):
            ax.text(
                i,
                value + (ymax - ymin) * 0.055,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=14.4,
                color=text_color,
            )
    axes[0].text(-0.82, 1.20, "(b)", transform=axes[0].transAxes, fontsize=20.0, ha="left", va="bottom")

    fig.subplots_adjust(left=0.078, right=0.985, bottom=0.12, top=0.90)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=1200, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate Fig. S7 using 22 unique HEC6 compositions.")
    parser.add_argument("--gnn", default="validation_outputs/gnn_hec6_unique/HEC6_predictions_unique.csv")
    parser.add_argument("--rf-benchmark", default=r"C:\Users\Ishan\Downloads\Figure_S7_Benchmark.xlsx")
    parser.add_argument("--output-prefix", default="validation_outputs/figures/figure_S7_gnn_rf_benchmark_unique")
    parser.add_argument("--source-csv", default="validation_outputs/figure_sources/figure_S7_gnn_rf_benchmark_unique_source.csv")
    parser.add_argument("--metrics-json", default="validation_outputs/figure_sources/figure_S7_gnn_rf_benchmark_unique_metrics.json")
    parser.add_argument("--metrics-csv", default="validation_outputs/figure_sources/figure_S7_gnn_rf_benchmark_unique_metrics.csv")
    parser.add_argument("--threshold", type=float, default=45.0)
    args = parser.parse_args()

    source = build_source_table(Path(args.gnn), Path(args.rf_benchmark))
    metric_rows = {
        "GNN": metrics(source["true_efa"].to_numpy(float), source["gnn_pred_efa"].to_numpy(float), args.threshold),
        "RF": metrics(source["true_efa"].to_numpy(float), source["rf_pred_efa"].to_numpy(float), args.threshold),
    }

    source_path = Path(args.source_csv)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(source_path, index=False)

    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump({"threshold": args.threshold, **metric_rows}, handle, indent=2)

    pd.DataFrame(
        [{"model": name, **values} for name, values in metric_rows.items()]
    ).to_csv(args.metrics_csv, index=False)
    plot_figure(source, metric_rows, Path(args.output_prefix))

    print(source_path.resolve())
    print(Path(args.metrics_json).resolve())
    print(Path(args.output_prefix).with_suffix(".png").resolve())
    print(pd.DataFrame([{"model": name, **values} for name, values in metric_rows.items()]).to_string(index=False))


if __name__ == "__main__":
    main()
