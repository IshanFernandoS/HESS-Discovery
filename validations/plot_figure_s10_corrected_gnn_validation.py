from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


COLORS = {
    "blue": "#2F6B9A",
    "orange": "#D9822B",
    "red": "#B54A4A",
    "purple": "#725A9C",
    "gray": "#666666",
    "light_gray": "#E6E6E6",
    "dark": "#222222",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.07,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["dark"],
    )


def load_metric_row(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return df.iloc[0].to_dict()


def draw_parity(
    ax: plt.Axes,
    actual: np.ndarray,
    pred: np.ndarray,
    x_label: str,
    y_label: str,
    metrics: dict[str, float],
    threshold: float = 45.0,
    limits: tuple[float, float] | None = None,
) -> None:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    true_cls = actual > threshold
    pred_cls = pred > threshold
    correct = true_cls == pred_cls

    if limits is None:
        lo = float(min(actual.min(), pred.min()))
        hi = float(max(actual.max(), pred.max()))
        pad = 0.07 * (hi - lo)
        limits = (lo - pad, hi + pad)

    ax.scatter(
        actual[correct],
        pred[correct],
        s=18,
        color=COLORS["blue"],
        alpha=0.72,
        linewidths=0,
        label="Correct phase class",
    )
    if np.any(~correct):
        ax.scatter(
            actual[~correct],
            pred[~correct],
            s=24,
            facecolor="white",
            edgecolor=COLORS["red"],
            linewidth=0.9,
            alpha=0.95,
            label="Phase-class error",
        )

    ax.plot(limits, limits, color=COLORS["dark"], linewidth=0.85)
    ax.axvline(threshold, color=COLORS["gray"], linestyle="--", linewidth=0.8)
    ax.axhline(threshold, color=COLORS["gray"], linestyle="--", linewidth=0.8)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(color=COLORS["light_gray"], linewidth=0.55)

    text = (
        f"n = {int(metrics['n'])}\n"
        f"$R^2$ = {metrics['r2']:.3f}"
        f" [{metrics['r2_low']:.3f}, {metrics['r2_high']:.3f}]\n"
        f"MAE = {metrics['mae']:.2f}"
        f" [{metrics['mae_low']:.2f}, {metrics['mae_high']:.2f}]"
    )
    ax.text(
        0.045,
        0.955,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.9,
        color=COLORS["dark"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.88},
    )


def draw_classification_ci(ax: plt.Axes, hec5: dict[str, float], hec6: dict[str, float]) -> None:
    rows = [
        ("Accuracy", "acc", "Acc"),
        ("Precision", "precision", "Precision"),
        ("Recall", "recall", "Recall"),
        ("F1 score", "f1", "F1"),
    ]
    y = np.arange(len(rows))[::-1]
    offset = 0.11

    hec5_points = np.array([hec5[f"{key}"] for _, key, _ in rows], dtype=float)
    hec5_low = np.array([hec5[f"{key}_low"] for _, key, _ in rows], dtype=float)
    hec5_high = np.array([hec5[f"{key}_high"] for _, key, _ in rows], dtype=float)
    hec6_points = np.array([hec6[f"{key}"] for _, key, _ in rows], dtype=float)
    hec6_low = np.array([hec6[f"{key}_low"] for _, key, _ in rows], dtype=float)
    hec6_high = np.array([hec6[f"{key}_high"] for _, key, _ in rows], dtype=float)

    ax.errorbar(
        hec5_points,
        y + offset,
        xerr=np.vstack([hec5_points - hec5_low, hec5_high - hec5_points]),
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["blue"],
        elinewidth=1.1,
        capsize=3,
        markersize=4.2,
        label="HEC5 holdout",
    )
    ax.errorbar(
        hec6_points,
        y - offset,
        xerr=np.vstack([hec6_points - hec6_low, hec6_high - hec6_points]),
        fmt="s",
        color=COLORS["orange"],
        ecolor=COLORS["orange"],
        elinewidth=1.1,
        capsize=3,
        markersize=4.0,
        label="HEC6 DFT",
    )

    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _, _ in rows])
    ax.set_xlim(0.34, 1.04)
    ax.set_xlabel("Metric value (EFA > 45)")
    ax.grid(axis="x", color=COLORS["light_gray"], linewidth=0.6)
    ax.legend(frameon=False, loc="upper left", handlelength=1.2, handletextpad=0.45)


def save_all(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate corrected Figure S10 from the manuscript-consistent GNN checkpoint outputs."
    )
    parser.add_argument(
        "--deephec-dir",
        default=r"C:\Users\Ishan\OneDrive - Queen Mary, University of London\deepHEC-opt",
    )
    parser.add_argument(
        "--outdir",
        default="validation_outputs/figures",
    )
    parser.add_argument("--stem", default="figure_S10_gnn_validation_corrected")
    args = parser.parse_args()

    setup_style()
    deephec = Path(args.deephec_dir)
    outdir = Path(args.outdir)

    hec5_pred = pd.read_csv(deephec / "fig2" / "fig2a_hec5_predictions.csv")
    hec5_metrics_raw = load_metric_row(deephec / "fig2" / "fig2a_hec5_metrics.csv")
    hec6_pred = pd.read_csv(deephec / "fig2" / "HEC6_predictions.csv")
    hec6_metrics_raw = load_metric_row(deephec / "fig2" / "HEC6_metrics_with_ci.csv")

    hec5_metrics = {
        "n": hec5_metrics_raw["N_test"],
        "r2": hec5_metrics_raw["R2"],
        "r2_low": hec5_metrics_raw["R2_CI_low"],
        "r2_high": hec5_metrics_raw["R2_CI_high"],
        "mae": hec5_metrics_raw["MAE"],
        "mae_low": hec5_metrics_raw["MAE_CI_low"],
        "mae_high": hec5_metrics_raw["MAE_CI_high"],
        "acc": hec5_metrics_raw["Accuracy"],
        "acc_low": hec5_metrics_raw["Acc_CI_low"],
        "acc_high": hec5_metrics_raw["Acc_CI_high"],
        "precision": hec5_metrics_raw["Precision_Single"],
        "precision_low": hec5_metrics_raw["Precision_CI_low"],
        "precision_high": hec5_metrics_raw["Precision_CI_high"],
        "recall": hec5_metrics_raw["Recall_Single"],
        "recall_low": hec5_metrics_raw["Recall_CI_low"],
        "recall_high": hec5_metrics_raw["Recall_CI_high"],
        "f1": hec5_metrics_raw["F1_Single"],
        "f1_low": hec5_metrics_raw["F1_CI_low"],
        "f1_high": hec5_metrics_raw["F1_CI_high"],
    }
    hec6_metrics = {
        "n": hec6_metrics_raw["N_samples"],
        "r2": hec6_metrics_raw["R2"],
        "r2_low": hec6_metrics_raw["R2_CI_low"],
        "r2_high": hec6_metrics_raw["R2_CI_high"],
        "mae": hec6_metrics_raw["MAE"],
        "mae_low": hec6_metrics_raw["MAE_CI_low"],
        "mae_high": hec6_metrics_raw["MAE_CI_high"],
        "acc": hec6_metrics_raw["Accuracy"],
        "acc_low": hec6_metrics_raw["Acc_CI_low"],
        "acc_high": hec6_metrics_raw["Acc_CI_high"],
        "precision": hec6_metrics_raw["Precision_Single"],
        "precision_low": hec6_metrics_raw["Precision_CI_low"],
        "precision_high": hec6_metrics_raw["Precision_CI_high"],
        "recall": hec6_metrics_raw["Recall_Single"],
        "recall_low": hec6_metrics_raw["Recall_CI_low"],
        "recall_high": hec6_metrics_raw["Recall_CI_high"],
        "f1": hec6_metrics_raw["F1_Single"],
        "f1_low": hec6_metrics_raw["F1_CI_low"],
        "f1_high": hec6_metrics_raw["F1_CI_high"],
    }

    fig = plt.figure(figsize=(7.2, 2.85))
    gs = fig.add_gridspec(
        1,
        3,
        left=0.075,
        right=0.988,
        bottom=0.20,
        top=0.92,
        wspace=0.50,
        width_ratios=[1.0, 1.0, 1.15],
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    draw_parity(
        axes[0],
        hec5_pred["y_true"].to_numpy(),
        hec5_pred["y_pred"].to_numpy(),
        "Reference HEC5 EFA",
        "Predicted HEC5 EFA",
        hec5_metrics,
        limits=(15, 125),
    )
    draw_parity(
        axes[1],
        hec6_pred["efa_actual"].to_numpy(),
        hec6_pred["efa_pred"].to_numpy(),
        "Reference HEC6 EFA",
        "Predicted HEC6 EFA",
        hec6_metrics,
        limits=(0, 90),
    )
    draw_classification_ci(axes[2], hec5_metrics, hec6_metrics)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["blue"], markersize=5, label="Correct"),
        Line2D([0], [0], marker="o", color=COLORS["red"], markerfacecolor="white", markersize=5, label="Error"),
        Line2D([0], [0], color=COLORS["dark"], linewidth=0.85, label="y = x"),
        Line2D([0], [0], color=COLORS["gray"], linestyle="--", linewidth=0.8, label="EFA = 45"),
    ]
    axes[0].legend(
        handles=handles,
        frameon=False,
        loc="lower right",
        handlelength=1.2,
        handletextpad=0.45,
        borderpad=0.1,
        labelspacing=0.35,
    )

    for ax, label in zip(axes, ["(a)", "(b)", "(c)"]):
        add_panel_label(ax, label)

    save_all(fig, outdir, args.stem)
    print(f"Saved Figure S10 files to {outdir.resolve()}")


if __name__ == "__main__":
    main()
