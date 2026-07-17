from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, precision_recall_fscore_support, r2_score


COL_HEC5 = "#31688e"  # viridis
COL_HEC6 = "#35b779"  # viridis
COL_DARK = "#222222"
COL_GRID = "#e6e6e6"
COL_ERR = "#b44e4e"


def setup_style() -> None:
    for font_path in [
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\timesbd.ttf",
        r"C:\Windows\Fonts\timesi.ttf",
        r"C:\Windows\Fonts\timesbi.ttf",
    ]:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10.5,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.8,
            "legend.fontsize": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="normal",
        color=COL_DARK,
    )


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    recall_manual = tp / (tp + fn) if (tp + fn) else np.nan
    bacc = np.nanmean([recall_manual, specificity])
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(bacc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_metrics(
    actual: np.ndarray,
    pred: np.ndarray,
    threshold: float = 45.0,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    y_true = (actual > threshold).astype(int)
    y_pred = (pred > threshold).astype(int)

    point = {
        "n": float(len(actual)),
        "r2": float(r2_score(actual, pred)),
        "mae": float(mean_absolute_error(actual, pred)),
    }
    point.update(classification_metrics(y_true, y_pred))

    rng = np.random.default_rng(seed)
    idx = np.arange(len(actual))
    rows: list[dict[str, float]] = []
    for _ in range(n_bootstrap):
        sample = rng.choice(idx, size=len(idx), replace=True)
        row = {
            "r2": float(r2_score(actual[sample], pred[sample]))
            if len(np.unique(actual[sample])) > 1
            else np.nan,
            "mae": float(mean_absolute_error(actual[sample], pred[sample])),
        }
        row.update(classification_metrics(y_true[sample], y_pred[sample]))
        rows.append(row)

    boot = pd.DataFrame(rows)
    for metric in ["r2", "mae", "acc", "precision", "recall", "f1"]:
        point[f"{metric}_low"] = float(np.nanpercentile(boot[metric], 2.5))
        point[f"{metric}_high"] = float(np.nanpercentile(boot[metric], 97.5))
    return point


def draw_two_point_ci(
    ax: plt.Axes,
    metric: str,
    hec5: dict[str, float],
    hec6: dict[str, float],
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    values = np.array([hec5[metric], hec6[metric]])
    lows = np.array([hec5[f"{metric}_low"], hec6[f"{metric}_low"]])
    highs = np.array([hec5[f"{metric}_high"], hec6[f"{metric}_high"]])
    x = np.array([0, 1])
    colors = [COL_HEC5, COL_HEC6]

    ax.bar(x, values, color=colors, width=0.52, edgecolor="none", alpha=0.88)
    ax.errorbar(
        x,
        values,
        yerr=np.vstack([values - lows, highs - values]),
        fmt="none",
        ecolor=COL_DARK,
        elinewidth=1.1,
        capsize=3.2,
        capthick=1.0,
    )
    yrange = ylim[1] - ylim[0] if ylim else float(np.nanmax(highs) - np.nanmin(lows))
    for xi, val, high in zip(x, values, highs):
        y_text = high + 0.035 * yrange
        if ylim:
            y_text = min(y_text, ylim[1] - 0.045 * yrange)
        ax.text(xi, y_text, f"{val:.3f}", ha="center", va="bottom", fontsize=9.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["HEC5", "HEC6"])
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color=COL_GRID, linewidth=0.6)


def draw_classification_panel(ax: plt.Axes, hec5: dict[str, float], hec6: dict[str, float]) -> None:
    metrics = [
        ("Accuracy", "acc"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 score", "f1"),
    ]
    y = np.arange(len(metrics))[::-1]
    offset = 0.12

    for row, color, label, dy in [(hec5, COL_HEC5, "HEC5", offset), (hec6, COL_HEC6, "HEC6", -offset)]:
        vals = np.array([row[key] for _, key in metrics])
        lows = np.array([row[f"{key}_low"] for _, key in metrics])
        highs = np.array([row[f"{key}_high"] for _, key in metrics])
        ax.errorbar(
            vals,
            y + dy,
            xerr=np.vstack([vals - lows, highs - vals]),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=4.8,
            elinewidth=1.1,
            capsize=3,
            label=label,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([name for name, _ in metrics])
    ax.set_xlim(0.30, 1.04)
    ax.set_xlabel("Metric value")
    ax.grid(axis="x", color=COL_GRID, linewidth=0.6)
    ax.legend(frameon=False, loc="lower left", handletextpad=0.4)


def draw_confusion(ax: plt.Axes, hec6: dict[str, float]) -> None:
    cm = np.array([[hec6["tn"], hec6["fp"]], [hec6["fn"], hec6["tp"]]], dtype=int)
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=max(1, int(cm.max())))
    for i in range(2):
        for j in range(2):
            color = "white" if (cm[i, j] > cm.max() * 0.55 or cm[i, j] <= 1) else COL_DARK
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13, color=color)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Multi", "Single"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Multi", "Single"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Reference class")
    ax.tick_params(length=0)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def draw_architecture_sweep(ax: plt.Axes, sweep_csv: Path) -> None:
    sweep = pd.read_csv(sweep_csv)
    retrained = sweep[sweep["source"] == "retrained_sweep"].copy()
    checkpoint = sweep[sweep["source"] == "manuscript_checkpoint"].copy()

    hidden_dims = sorted(retrained["hidden_dim"].dropna().astype(int).unique())
    colors = dict(zip(hidden_dims, ["#31688e", "#35b779", "#fde725"]))
    size_map = {8: 45, 16: 75, 32: 110}

    for hidden_dim in hidden_dims:
        sub = retrained[retrained["hidden_dim"].astype(int) == hidden_dim]
        ax.scatter(
            sub["hec5_accuracy"],
            sub["hec6_r2"],
            s=[size_map.get(int(h), 70) for h in sub["heads"]],
            c=colors[hidden_dim],
            edgecolor=COL_DARK,
            linewidth=0.45,
            alpha=0.9,
            label=fr"$h={hidden_dim}$",
        )

    if not checkpoint.empty:
        row = checkpoint.iloc[0]
        ax.scatter(
            row["hec5_accuracy"],
            row["hec6_r2"],
            marker="*",
            s=175,
            c=COL_ERR,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )

    ax.set_xlabel("HEC5 hold-out accuracy")
    ax.set_ylabel(r"HEC6 transfer $R^2$")
    ax.set_xlim(0.74, 0.95)
    ax.set_ylim(0.25, 0.85)
    ax.grid(color=COL_GRID, linewidth=0.6)

    color_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[h], markeredgecolor=COL_DARK, markersize=6, label=fr"$h={h}$")
        for h in hidden_dims
    ]
    size_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.75", markeredgecolor=COL_DARK, markersize=np.sqrt(size_map[h]) / 1.35, label=fr"$H={h}$")
        for h in [8, 16, 32]
    ]
    checkpoint_handle = Line2D([0], [0], marker="*", color="none", markerfacecolor=COL_ERR, markeredgecolor="white", markersize=9, label="checkpoint")
    ax.legend(
        handles=color_handles + size_handles + [checkpoint_handle],
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        handletextpad=0.35,
    )


def main() -> None:
    setup_style()
    deephec = Path(r"C:\Users\Ishan\OneDrive - Queen Mary, University of London\deepHEC-opt\fig2")
    outdir = Path("validation_outputs/figures")
    sweep_csv = Path("validation_outputs/gnn_architecture_sweep_long_20260715_132343/architecture_sweep_metrics.csv")
    outdir.mkdir(parents=True, exist_ok=True)

    hec5_pred = pd.read_csv(deephec / "fig2a_hec5_predictions.csv")
    hec5_raw = pd.read_csv(deephec / "fig2a_hec5_metrics.csv").iloc[0]
    hec6_pred = pd.read_csv(deephec / "HEC6_predictions.csv").drop_duplicates("composition", keep="first")

    hec5_actual = hec5_pred["y_true"].to_numpy()
    hec5_predicted = hec5_pred["y_pred"].to_numpy()
    hec5 = classification_metrics((hec5_actual > 45.0).astype(int), (hec5_predicted > 45.0).astype(int))
    hec5.update(
        {
            "n": float(hec5_raw["N_test"]),
            "r2": float(hec5_raw["R2"]),
            "r2_low": float(hec5_raw["R2_CI_low"]),
            "r2_high": float(hec5_raw["R2_CI_high"]),
            "mae": float(hec5_raw["MAE"]),
            "mae_low": float(hec5_raw["MAE_CI_low"]),
            "mae_high": float(hec5_raw["MAE_CI_high"]),
            "acc": float(hec5_raw["Accuracy"]),
            "acc_low": float(hec5_raw["Acc_CI_low"]),
            "acc_high": float(hec5_raw["Acc_CI_high"]),
            "precision": float(hec5_raw["Precision_Single"]),
            "precision_low": float(hec5_raw["Precision_CI_low"]),
            "precision_high": float(hec5_raw["Precision_CI_high"]),
            "recall": float(hec5_raw["Recall_Single"]),
            "recall_low": float(hec5_raw["Recall_CI_low"]),
            "recall_high": float(hec5_raw["Recall_CI_high"]),
            "f1": float(hec5_raw["F1_Single"]),
            "f1_low": float(hec5_raw["F1_CI_low"]),
            "f1_high": float(hec5_raw["F1_CI_high"]),
        }
    )
    hec6 = bootstrap_metrics(hec6_pred["efa_actual"].to_numpy(), hec6_pred["efa_pred"].to_numpy())

    metrics_out = pd.DataFrame(
        [
            {"set": "HEC5", **hec5},
            {"set": "HEC6", **hec6},
        ]
    )
    metrics_out.to_csv(outdir / "figure_S5_corrected_metrics.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    axes = axes.ravel()

    draw_two_point_ci(axes[0], "r2", hec5, hec6, r"$R^2$", ylim=(0, 1.02))
    draw_two_point_ci(axes[1], "mae", hec5, hec6, "MAE", ylim=(0, 15.5))
    draw_classification_panel(axes[2], hec5, hec6)
    draw_architecture_sweep(axes[3], sweep_csv)

    for ax, label in zip(axes, ["(a)", "(b)", "(c)", "(d)"]):
        add_panel_label(ax, label)

    for stem in ["figure_S5_gnn_transfer_corrected", "figure_S5_gnn_transfer_final", "figure_S5_gnn_transfer_refined"]:
        fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(outdir / f"{stem}.svg", bbox_inches="tight")
        fig.savefig(outdir / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved corrected Figure S5 to {outdir.resolve()}")


if __name__ == "__main__":
    main()
