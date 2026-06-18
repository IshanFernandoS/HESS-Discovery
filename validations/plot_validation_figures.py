from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


COLORS = {
    "blue": "#2F6B9A",
    "light_blue": "#DDEAF3",
    "orange": "#D9822B",
    "green": "#2E7D5B",
    "red": "#B54A4A",
    "purple": "#725A9C",
    "gray": "#666666",
    "light_gray": "#E8E8E8",
    "dark": "#222222",
}

MAIN_FIGSIZE = (2.8, 2.15)

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision_single": "Precision",
    "recall_single": "Recall",
    "f1_single": "F1",
}

SI_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision_single": "Precision",
    "recall_single": "Recall",
    "f1_single": "F1",
}

ABLATION_LABELS = {
    "baseline": "All descriptors",
    "drop_mismatch": "Without mismatch",
    "drop_bonding_carbon_affinity": "Without bonding/\nC affinity",
    "drop_refractory_floor": "Without refractory\nfloor",
    "drop_segregation_msi": "Without segregation/\nMSI",
    "drop_vec": "Without VEC",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_all(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def clean_comp_label(comp: str) -> str:
    text = str(comp)
    text = text.replace("(", "").replace(")", "")
    text = text.replace("C9", "").replace("C10", "").replace("C11", "")
    if len(text) > 22:
        return text[:19] + "..."
    return text


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=COLORS["dark"],
    )


def draw_confusion_matrix(ax: plt.Axes, base: Path) -> None:
    cm = read_csv(base / "gmm_literature_bootstrap" / "confusion_matrix.csv", index_col=0)
    values = cm.to_numpy(dtype=int)
    cmap = LinearSegmentedColormap.from_list("cm_blue", ["#F7FBFF", COLORS["blue"]])
    ax.imshow(values, cmap=cmap, vmin=0, vmax=max(35, int(values.max())))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] > values.max() * 0.55 else COLORS["dark"]
            ax.text(j, i, str(values[i, j]), ha="center", va="center", fontsize=10.5, color=color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred. multi", "Pred. single"])
    ax.set_yticklabels(["True multi", "True single"])
    ax.set_xlabel("Predicted label", labelpad=3)
    ax.set_ylabel("Literature label", labelpad=3)
    ax.tick_params(axis="both", labelsize=6.5, pad=2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_confusion_matrix(base: Path, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)
    draw_confusion_matrix(ax, base)
    save_all(fig, outdir, "01_confusion_matrix")


def draw_bootstrap_metrics(ax: plt.Axes, base: Path, metric_labels: dict[str, str] | None = None) -> None:
    if metric_labels is None:
        metric_labels = METRIC_LABELS
    df = read_csv(base / "gmm_literature_bootstrap" / "bootstrap_metrics.csv")
    df = df[df["metric"].isin(metric_labels)].copy()
    df["label"] = df["metric"].map(metric_labels)
    df = df.set_index("metric").loc[list(metric_labels)].reset_index()
    y = np.arange(len(df))[::-1]
    xerr = np.vstack([df["point"] - df["ci_low"], df["ci_high"] - df["point"]])

    ax.errorbar(
        df["point"],
        y,
        xerr=xerr,
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["blue"],
        elinewidth=1.2,
        capsize=3,
        markersize=4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlim(0.45, 1.03)
    ax.set_xlabel("Metric value")
    ax.grid(axis="x", color=COLORS["light_gray"], linewidth=0.6)


def plot_bootstrap_metrics(base: Path, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)
    draw_bootstrap_metrics(ax, base)
    save_all(fig, outdir, "02_bootstrap_metrics")


def draw_applicability_domain(ax: plt.Axes, base: Path) -> None:
    labelled_path = base / "applicability_domain" / "applicability_domain_literature_labelled.csv"
    selected_labelled_path = base / "applicability_domain" / "selected_applicability_literature_labelled_reference.csv"
    if labelled_path.exists() and selected_labelled_path.exists():
        df = read_csv(labelled_path)
        selected = read_csv(selected_labelled_path)
        group_styles = {
            9: ("HEC9", COLORS["blue"]),
            10: ("HEC10", COLORS["green"]),
            11: ("HEC11", COLORS["purple"]),
        }
        for n, (label, color) in group_styles.items():
            part = df[df["n_metals"] == n]
            if part.empty:
                continue
            ax.scatter(
                part["nn_percentile_vs_labelled_ref"],
                part["mahalanobis_percentile_vs_labelled_ref"],
                s=10,
                color=color,
                alpha=0.42,
                linewidths=0,
                label=label,
            )
        ax.scatter(
            selected["nn_percentile_vs_labelled_ref"],
            selected["mahalanobis_percentile_vs_labelled_ref"],
            s=34,
            facecolor="white",
            edgecolor=COLORS["red"],
            linewidth=1.2,
            zorder=6,
            label="Selected",
        )
        ax.axvline(95, color=COLORS["red"], linestyle="--", linewidth=0.8)
        ax.axhline(95, color=COLORS["red"], linestyle="--", linewidth=0.8)
        ax.text(
            2.0,
            96.5,
            "95th percentile",
            ha="left",
            va="bottom",
            fontsize=7.5,
            color=COLORS["red"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3, "alpha": 0.9},
        )
        ax.set_xlim(-2, 105)
        ax.set_ylim(-2, 105)
        ax.set_xlabel("NN percentile")
        ax.set_ylabel("Mahalanobis percentile")
        ax.grid(color=COLORS["light_gray"], linewidth=0.55)
        ax.legend(
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.02, 0.02),
            handletextpad=0.25,
            borderpad=0.1,
            labelspacing=0.25,
        )
        return

    df = read_csv(base / "applicability_domain" / "applicability_domain.csv")
    selected = read_csv(base / "applicability_domain" / "selected_applicability.csv")
    positions = []
    labels = []
    data = []
    for i, n in enumerate([9, 10, 11], start=1):
        vals = df.loc[df["n_metals"] == n, "nearest_neighbor_percentile_vs_reference"].dropna()
        positions.append(i)
        labels.append(f"HEC{n}")
        data.append(vals.to_numpy())
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["dark"], "linewidth": 1.0},
        boxprops={"edgecolor": COLORS["blue"], "linewidth": 0.9},
        whiskerprops={"color": COLORS["blue"], "linewidth": 0.9},
        capprops={"color": COLORS["blue"], "linewidth": 0.9},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(COLORS["light_blue"])
    rng = np.random.default_rng(7)
    for i, vals in zip(positions, data):
        jitter = rng.normal(0, 0.045, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=5, color=COLORS["blue"], alpha=0.28, linewidths=0)
    selected_y = selected["nearest_neighbor_percentile_vs_reference"].to_numpy(dtype=float)
    ax.scatter(
        np.full(len(selected_y), 1.0),
        selected_y,
        s=26,
        facecolor="white",
        edgecolor=COLORS["red"],
        linewidth=1.2,
        zorder=5,
    )
    ax.axhline(95, color=COLORS["red"], linestyle="--", linewidth=0.9)
    ax.text(
        3.43,
        96.4,
        "95th percentile",
        va="bottom",
        ha="right",
        fontsize=8.0,
        color=COLORS["red"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.9},
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(-1, 103)
    ax.set_ylabel("NN distance percentile")


def plot_applicability_domain(base: Path, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)
    draw_applicability_domain(ax, base)
    save_all(fig, outdir, "03_applicability_domain")


def draw_feature_ablation(ax: plt.Axes, base: Path) -> None:
    df = read_csv(base / "feature_ablation" / "ablation_metrics.csv")
    order = list(ABLATION_LABELS)
    df = df.set_index("variant").loc[order].reset_index()
    baseline = float(df.loc[df["variant"] == "baseline", "f1_single"].iloc[0])
    labels = [ABLATION_LABELS[v] for v in df["variant"]]
    y = np.arange(len(df))[::-1]
    xmin = max(0.0, float(df["f1_single"].min()) - 0.012)
    xmax = min(1.0, float(df["f1_single"].max()) + 0.012)
    ax.hlines(y, xmin=xmin, xmax=df["f1_single"], color=COLORS["light_gray"], linewidth=1.0, zorder=1)
    ax.scatter(df["f1_single"].iloc[0], y[0], s=28, color=COLORS["blue"], zorder=3)
    ax.scatter(df["f1_single"].iloc[1:], y[1:], s=28, color=COLORS["green"], zorder=3)
    ax.axvline(baseline, color=COLORS["dark"], linewidth=0.7, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("F1 score")
    ax.set_ylabel("Descriptor set", labelpad=4)
    ax.grid(axis="x", color=COLORS["light_gray"], linewidth=0.6)


def plot_feature_ablation(base: Path, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)
    draw_feature_ablation(ax, base)
    save_all(fig, outdir, "04_feature_ablation")


def plot_permutation_null(base: Path, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    draw_permutation_null(ax, base)
    save_all(fig, outdir, "05_label_permutation_null")


def draw_permutation_null(ax: plt.Axes, base: Path) -> None:
    null_df = read_csv(base / "label_permutation_null" / "null_metrics.csv")
    summary = read_csv(base / "label_permutation_null" / "permutation_summary.csv")
    metric = "f1_single"
    real = float(summary.loc[summary["metric"] == metric, "real"].iloc[0])
    p_val = float(summary.loc[summary["metric"] == metric, "empirical_p_value_greater_equal"].iloc[0])
    ax.hist(null_df[metric].dropna(), bins=25, color=COLORS["light_blue"], edgecolor=COLORS["blue"], linewidth=0.6)
    ax.axvline(real, color=COLORS["red"], linewidth=1.5)
    ax.text(
        real + 0.0015,
        ax.get_ylim()[1] * 0.88,
        f"Observed F1 = {real:.3f}\np = {p_val:.3f}",
        color=COLORS["red"],
        fontsize=9,
    )
    ax.set_xlabel("F1 after label shuffling")
    ax.set_ylabel("Count")


def plot_gmm_robustness(base: Path, outdir: Path) -> None:
    details = read_csv(base / "gmm_seed_component_robustness" / "selected_candidate_run_details.csv")
    details = details[np.isfinite(details["Rank"])].copy()
    comp_order = [
        "(HfMoNbReTaTiVWZr)C9",
        "(HfMoNbScTaTiVWZr)C9",
        "(HfMoNbTaTiVWYZr)C9",
    ]
    labels = ["Re candidate", "Sc candidate", "Y candidate"]
    data = [details.loc[details["Composition"] == comp, "Rank"].to_numpy() for comp in comp_order]
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.75)
    for body in parts["bodies"]:
        body.set_facecolor(COLORS["light_blue"])
        body.set_edgecolor(COLORS["blue"])
        body.set_alpha(0.9)
    for key in ["cmedians", "cbars", "cmins", "cmaxes"]:
        parts[key].set_color(COLORS["blue"])
        parts[key].set_linewidth(0.9)
    ax.axhline(10, color=COLORS["red"], linestyle="--", linewidth=0.8)
    ax.text(3.45, 10, "Top 10", va="center", color=COLORS["red"], fontsize=9)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Rank across GMM settings\n(lower is better)")
    ax.invert_yaxis()
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)
    save_all(fig, outdir, "06_gmm_rank_robustness")


def plot_mc_dropout_selected(base: Path, outdir: Path) -> None:
    path = base / "gnn_mc_dropout_uncertainty_selected" / "selected_mc_dropout_predictions.csv"
    if not path.exists():
        path = base / "gnn_mc_dropout_uncertainty_all_HEC9_HEC11_batch1" / "selected_mc_dropout_predictions.csv"
    if not path.exists():
        path = base / "gnn_mc_dropout_uncertainty_selected_batch1" / "selected_mc_dropout_predictions.csv"
    df = read_csv(path)
    labels = ["Re candidate", "Sc candidate", "Y candidate"]
    y = np.arange(len(df))[::-1]
    xerr = np.vstack([df["MC_mean"] - df["MC_q05"], df["MC_q95"] - df["MC_mean"]])
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.errorbar(
        df["MC_mean"],
        y,
        xerr=xerr,
        fmt="o",
        color=COLORS["purple"],
        ecolor=COLORS["purple"],
        elinewidth=1.2,
        capsize=3,
        markersize=4,
        label="MC mean, 5-95%",
    )
    ax.scatter(df["Predicted_EFA_eval_mode"], y, marker="D", s=24, color=COLORS["orange"], label="Eval-mode")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted EFA")
    ax.grid(axis="x", color=COLORS["light_gray"], linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    save_all(fig, outdir, "07_mc_dropout_selected")


def _mc_dropout_path(base: Path) -> Path:
    candidates = [
        base / "gnn_refined_s10" / "selected_candidate_mc_dropout_predictions.csv",
        base / "gnn_refined_s10" / "selected_hec9_mc_dropout_predictions.csv",
        base / "gnn_mc_dropout_uncertainty_selected" / "selected_mc_dropout_predictions.csv",
        base / "gnn_mc_dropout_uncertainty_all_HEC9_HEC11_batch1" / "selected_mc_dropout_predictions.csv",
        base / "gnn_mc_dropout_uncertainty_selected_batch1" / "selected_mc_dropout_predictions.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def draw_hec8_external_benchmark(ax: plt.Axes, base: Path) -> None:
    path = base / "external_hec8_benchmark" / "hec8_predictions.csv"
    df = read_csv(path)
    metrics_path = base / "external_hec8_benchmark" / "hec8_metrics.json"
    metrics = {}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

    ax.scatter(
        df["Actual_EFA"],
        df["Predicted_EFA"],
        s=12,
        color=COLORS["blue"],
        alpha=0.58,
        linewidths=0,
    )
    lo = float(min(df["Actual_EFA"].min(), df["Predicted_EFA"].min()))
    hi = float(max(df["Actual_EFA"].max(), df["Predicted_EFA"].max()))
    pad = 3.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=COLORS["dark"], linewidth=0.8)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Reference HEC8 EFA")
    ax.set_ylabel("Predicted HEC8 EFA")
    if metrics:
        ax.text(
            0.04,
            0.96,
            f"n = {int(metrics['n'])}\nMAE = {metrics['mae']:.1f}\n$R^2$ = {metrics['r2']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color=COLORS["dark"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.9},
        )


def draw_hec5_internal_holdout(ax: plt.Axes, base: Path) -> None:
    path = base / "gnn_internal_hec5_holdout" / "hec5_internal_holdout_predictions.csv"
    df = read_csv(path)
    metrics_path = base / "gnn_internal_hec5_holdout" / "hec5_internal_holdout_metrics.json"
    metrics = {}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

    ax.scatter(
        df["Actual_EFA"],
        df["Predicted_EFA"],
        s=12,
        color=COLORS["blue"],
        alpha=0.58,
        linewidths=0,
    )
    lo = float(min(df["Actual_EFA"].min(), df["Predicted_EFA"].min()))
    hi = float(max(df["Actual_EFA"].max(), df["Predicted_EFA"].max()))
    pad = 4.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=COLORS["dark"], linewidth=0.8)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Reference HEC5 EFA")
    ax.set_ylabel("Predicted HEC5 EFA")
    if metrics:
        ax.text(
            0.04,
            0.96,
            f"n = {int(metrics['n'])}\nMAE = {metrics['mae']:.2f}\n$R^2$ = {metrics['r2']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            color=COLORS["dark"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.9},
        )


def draw_gnn_literature_phase_association(ax: plt.Axes, base: Path) -> None:
    df = read_csv(base / "gnn_literature_phase_association" / "literature_gnn_predictions.csv")
    summary = read_csv(base / "gnn_literature_phase_association" / "summary.csv")
    all_summary = summary[summary["subset"] == "all_literature_labels"].iloc[0]
    non_train_summary = summary[summary["subset"] == "excluding_HEC5_train_split_overlap"].iloc[0]

    categories = [("Multi", 0, COLORS["orange"]), ("Single", 1, COLORS["blue"])]
    data = [df.loc[df["Label"] == label, "GNN_predicted_EFA"].to_numpy(dtype=float) for label, _, _ in categories]
    box = ax.boxplot(
        data,
        positions=[0, 1],
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["dark"], "linewidth": 1.0},
        boxprops={"edgecolor": COLORS["dark"], "linewidth": 0.8},
        whiskerprops={"color": COLORS["dark"], "linewidth": 0.8},
        capprops={"color": COLORS["dark"], "linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], [COLORS["light_gray"], COLORS["light_blue"]]):
        patch.set_facecolor(color)

    rng = np.random.default_rng(21)
    for label, x_pos, color in categories:
        part = df[df["Label"] == label].copy()
        jitter = rng.normal(0, 0.045, size=len(part))
        train_overlap = part["in_hec5_train_split"].astype(bool).to_numpy()
        ax.scatter(
            np.full(train_overlap.sum(), x_pos) + jitter[train_overlap],
            part.loc[train_overlap, "GNN_predicted_EFA"],
            s=18,
            color=color,
            alpha=0.70,
            linewidths=0,
            zorder=3,
        )
        ax.scatter(
            np.full((~train_overlap).sum(), x_pos) + jitter[~train_overlap],
            part.loc[~train_overlap, "GNN_predicted_EFA"],
            s=22,
            facecolor="white",
            edgecolor=color,
            alpha=0.95,
            linewidth=0.85,
            zorder=4,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Multi", "Single"])
    ax.set_xlabel("Literature label")
    ax.set_ylabel("GNN-predicted EFA")
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(30, 90)
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)
    ax.text(
        0.04,
        0.96,
        f"All AUC = {all_summary['roc_auc']:.3f}\nNon-train AUC = {non_train_summary['roc_auc']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color=COLORS["dark"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.9},
    )

def draw_mc_dropout_selected(ax: plt.Axes, base: Path, show_legend: bool = True) -> None:
    df = read_csv(_mc_dropout_path(base))
    order = [
        "(HfMoNbReTaTiVWZr)C9",
        "(HfMoNbScTaTiVWZr)C9",
        "(HfMoNbTaTiVWYZr)C9",
        "(HfMoNbReScTaTiVWZr)C10",
        "(HfMoNbScTaTiVWYZr)C10",
        "(HfMoNbReTaTiVWYZr)C10",
        "(HfMoNbReScTaTiVWYZr)C11",
    ]
    labels = {
        "(HfMoNbReTaTiVWZr)C9": "HEC9 Re",
        "(HfMoNbScTaTiVWZr)C9": "HEC9 Sc",
        "(HfMoNbTaTiVWYZr)C9": "HEC9 Y",
        "(HfMoNbReScTaTiVWZr)C10": "HEC10 Re+Sc",
        "(HfMoNbScTaTiVWYZr)C10": "HEC10 Sc+Y",
        "(HfMoNbReTaTiVWYZr)C10": "HEC10 Re+Y",
        "(HfMoNbReScTaTiVWYZr)C11": "HEC11 Re+Sc+Y",
    }
    available = [comp for comp in order if comp in set(df["Composition"])]
    df = df.set_index("Composition").loc[available].reset_index()
    y = np.arange(len(df))[::-1]
    xerr = np.vstack([df["MC_mean"] - df["MC_q05"], df["MC_q95"] - df["MC_mean"]])
    ax.errorbar(
        df["MC_mean"],
        y,
        xerr=xerr,
        fmt="o",
        color=COLORS["purple"],
        ecolor=COLORS["purple"],
        elinewidth=1.25,
        capsize=3,
        markersize=4.5,
        label="MC mean, 5-95%",
    )
    ax.scatter(
        df["Predicted_EFA_eval_mode"],
        y,
        marker="D",
        s=26,
        color=COLORS["orange"],
        label="Eval-mode",
        zorder=4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([labels[c] for c in df["Composition"]])
    ax.set_xlabel("Predicted EFA")
    xmin = float(min(df["MC_q05"].min(), df["Predicted_EFA_eval_mode"].min(), 45.0)) - 2.0
    xmax = float(max(df["MC_q95"].max(), df["Predicted_EFA_eval_mode"].max(), 45.0)) + 2.0
    ax.set_xlim(xmin, xmax)
    ax.grid(axis="x", color=COLORS["light_gray"], linewidth=0.6)
    if show_legend:
        ax.legend(frameon=False, loc="lower right", handlelength=1.1, handletextpad=0.4)


def plot_gnn_validation_figure(base: Path, outdir: Path) -> None:
    hec8_path = base / "external_hec8_benchmark" / "hec8_predictions.csv"
    mc_path = _mc_dropout_path(base)
    if not hec8_path.exists() or not mc_path.exists():
        return
    fig = plt.figure(figsize=(7.1, 2.85))
    gs = fig.add_gridspec(
        1,
        2,
        left=0.085,
        right=0.985,
        bottom=0.20,
        top=0.92,
        wspace=0.48,
        width_ratios=[1.0, 1.08],
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    draw_hec8_external_benchmark(axes[0], base)
    draw_mc_dropout_selected(axes[1], base)
    for ax, label in zip(axes, ["(a)", "(b)"]):
        add_panel_label(ax, label)
    save_all(fig, outdir, "gnn_validation_figure_ab")


def plot_gnn_validation_figure_abc(base: Path, outdir: Path) -> None:
    required = [
        base / "gnn_internal_hec5_holdout" / "hec5_internal_holdout_predictions.csv",
        base / "gnn_literature_phase_association" / "literature_gnn_predictions.csv",
        _mc_dropout_path(base),
    ]
    if not all(path.exists() for path in required):
        return
    fig = plt.figure(figsize=(7.3, 2.95))
    gs = fig.add_gridspec(
        1,
        3,
        left=0.075,
        right=0.988,
        bottom=0.20,
        top=0.91,
        wspace=0.55,
        width_ratios=[1.05, 0.92, 1.15],
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    draw_hec5_internal_holdout(axes[0], base)
    draw_gnn_literature_phase_association(axes[1], base)
    draw_mc_dropout_selected(axes[2], base, show_legend=False)
    for ax, label in zip(axes, ["(a)", "(b)", "(c)"]):
        add_panel_label(ax, label)
    save_all(fig, outdir, "gnn_validation_figure_abc")


def plot_gnn_literature_uncertainty_figure(base: Path, outdir: Path) -> None:
    required = [
        base / "gnn_literature_phase_association" / "literature_gnn_predictions.csv",
        _mc_dropout_path(base),
    ]
    if not all(path.exists() for path in required):
        return
    fig = plt.figure(figsize=(6.7, 2.95))
    gs = fig.add_gridspec(
        1,
        2,
        left=0.085,
        right=0.985,
        bottom=0.20,
        top=0.91,
        wspace=0.50,
        width_ratios=[0.90, 1.18],
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    draw_gnn_literature_phase_association(axes[0], base)
    draw_mc_dropout_selected(axes[1], base, show_legend=True)
    for ax, label in zip(axes, ["(a)", "(b)"]):
        add_panel_label(ax, label)
    save_all(fig, outdir, "gnn_literature_uncertainty_figure_ab")


def plot_hec8_benchmark(base: Path, outdir: Path) -> None:
    path = base / "external_hec8_benchmark" / "hec8_predictions.csv"
    if not path.exists():
        path = base / "external_hec8_benchmark_batch1" / "hec8_predictions.csv"
    if not path.exists():
        return
    df = read_csv(path)
    fig, ax = plt.subplots(figsize=(2.6, 2.45))
    ax.scatter(df["Actual_EFA"], df["Predicted_EFA"], s=13, color=COLORS["blue"], alpha=0.65, linewidths=0)
    lo = float(min(df["Actual_EFA"].min(), df["Predicted_EFA"].min()))
    hi = float(max(df["Actual_EFA"].max(), df["Predicted_EFA"].max()))
    ax.plot([lo, hi], [lo, hi], color=COLORS["dark"], linewidth=0.8)
    ax.set_xlabel("Reference HEC8 EFA")
    ax.set_ylabel("Predicted HEC8 EFA")
    ax.text(0.04, 0.94, "MAE = 17.79\n$R^2$ = -1.04", transform=ax.transAxes, va="top", fontsize=9)
    save_all(fig, outdir, "08_hec8_benchmark_diagnostic")


def plot_leave_out_diagnostic(base: Path, outdir: Path) -> None:
    elem_path = base / "gnn_leave_element_out_100epoch_seed0" / "leave_element_summary.csv"
    fam_path = base / "gnn_leave_family_out_atleast2_100epoch_seed0" / "leave_family_summary.csv"
    if not elem_path.exists() or not fam_path.exists():
        return
    elem = read_csv(elem_path).sort_values("mae_mean")
    fam = read_csv(fam_path).sort_values("mae_mean")
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.55), sharey=False)
    axes[0].bar(np.arange(len(elem)), elem["mae_mean"], color=COLORS["blue"], width=0.72)
    axes[0].set_xticks(np.arange(len(elem)))
    axes[0].set_xticklabels(elem["element"], rotation=45, ha="right")
    axes[0].set_ylabel("MAE")
    axes[0].grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)
    axes[1].bar(np.arange(len(fam)), fam["mae_mean"], color=COLORS["green"], width=0.72)
    axes[1].set_xticks(np.arange(len(fam)))
    axes[1].set_xticklabels(fam["family"], rotation=35, ha="right")
    axes[1].grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)
    axes[1].set_ylabel("MAE")
    save_all(fig, outdir, "09_leave_out_diagnostic")


def plot_main_validation_figure(base: Path, outdir: Path) -> None:
    fig = plt.figure(figsize=(7.1, 5.15))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.085,
        right=0.985,
        bottom=0.105,
        top=0.965,
        wspace=0.58,
        hspace=0.62,
        width_ratios=[1.0, 1.12],
        height_ratios=[1.0, 1.08],
    )
    axes = np.array(
        [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        ]
    )

    draw_confusion_matrix(axes[0, 0], base)
    draw_bootstrap_metrics(axes[0, 1], base)
    draw_applicability_domain(axes[1, 0], base)
    draw_feature_ablation(axes[1, 1], base)

    for ax, label in zip(axes.ravel(), ["(a)", "(b)", "(c)", "(d)"]):
        add_panel_label(ax, label)

    save_all(fig, outdir, "validation_figure_abcd")


def plot_revised_si_validation_figure(base: Path, outdir: Path) -> None:
    fig = plt.figure(figsize=(7.1, 5.15))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.095,
        right=0.985,
        bottom=0.105,
        top=0.965,
        wspace=0.62,
        hspace=0.64,
        width_ratios=[1.0, 1.08],
        height_ratios=[1.0, 1.08],
    )
    axes = np.array(
        [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        ]
    )

    draw_bootstrap_metrics(axes[0, 0], base, SI_METRIC_LABELS)
    draw_applicability_domain(axes[0, 1], base)
    draw_feature_ablation(axes[1, 0], base)
    draw_permutation_null(axes[1, 1], base)

    for ax, label in zip(axes.ravel(), ["(a)", "(b)", "(c)", "(d)"]):
        add_panel_label(ax, label)

    save_all(fig, outdir, "revised_si_validation_figure_abcd")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-style validation plots.")
    parser.add_argument("--validation-dir", default="validation_outputs")
    parser.add_argument("--outdir", default="validation_outputs/figures")
    args = parser.parse_args()

    setup_style()
    base = Path(args.validation_dir)
    outdir = ensure_dir(Path(args.outdir))
    plot_confusion_matrix(base, outdir)
    plot_bootstrap_metrics(base, outdir)
    plot_applicability_domain(base, outdir)
    plot_feature_ablation(base, outdir)
    plot_main_validation_figure(base, outdir)
    plot_revised_si_validation_figure(base, outdir)
    plot_permutation_null(base, outdir)
    plot_gmm_robustness(base, outdir)
    plot_mc_dropout_selected(base, outdir)
    plot_hec8_benchmark(base, outdir)
    plot_gnn_validation_figure(base, outdir)
    plot_gnn_validation_figure_abc(base, outdir)
    plot_gnn_literature_uncertainty_figure(base, outdir)
    plot_leave_out_diagnostic(base, outdir)
    print(f"Saved validation figures to {outdir.resolve()}")


if __name__ == "__main__":
    main()
