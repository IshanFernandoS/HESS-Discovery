from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from validations.common import apply_equation_selection, canonicalize


DISPLAY = {
    canonicalize("(HfMoNbReTaTiVWZr)C9"): "HEC9 Re",
    canonicalize("(HfMoNbScTaTiVWZr)C9"): "HEC9 Sc",
    canonicalize("(HfMoNbTaTiVWYZr)C9"): "HEC9 Y",
    canonicalize("(HfMoNbReScTaTiVWZr)C10"): "HEC10 Re+Sc",
    canonicalize("(HfMoNbScTaTiVWYZr)C10"): "HEC10 Sc+Y",
    canonicalize("(HfMoNbReTaTiVWYZr)C10"): "HEC10 Re+Y",
    canonicalize("(HfMoNbReScTaTiVWYZr)C11"): "HEC11 Re+Sc+Y",
}

DISPLAY_ORDER = list(DISPLAY)
ORDER_MARKERS = {9: "o", 10: "s", 11: "^"}
CANDIDATE_NUMBERS = {composition: i + 1 for i, composition in enumerate(DISPLAY_ORDER)}


def setup_style(font_path: Path | None = None) -> None:
    font_family = "Arial"
    if font_path is not None and font_path.exists():
        fm.fontManager.addfont(str(font_path))
        font_family = fm.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 9.3,
            "axes.labelsize": 9.4,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.06,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.8,
        fontweight="normal",
        color="#222222",
    )


def save_all(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=1200, bbox_inches="tight")
    plt.close(fig)


def load_inputs(base: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = base / "validation_outputs" / "equation_1_5_selection"
    basin = pd.read_csv(root / "equation_2_basin_scores.csv")
    efa = pd.read_csv(root / "equation_4_gnn_efa_predictions.csv")
    reference = pd.read_csv(root / "equations_1_5_full_selection.csv")
    audit = pd.read_csv(root / "experimental_candidate_audit.csv")
    for df in (basin, efa, reference, audit):
        df["Composition"] = df["Composition"].astype(str).map(canonicalize)
    return basin, efa, reference, audit


def compute_grid_selection(
    basin: pd.DataFrame,
    efa: pd.DataFrame,
    rho_grid: list[float],
    eta_grid: list[float],
) -> pd.DataFrame:
    rows = []
    for rho in rho_grid:
        for eta in eta_grid:
            selected = apply_equation_selection(basin, efa, rho=rho, eta=eta)
            selected = selected[selected["Composition"].isin(DISPLAY_ORDER)]
            for _, row in selected.iterrows():
                rows.append(
                    {
                        "Composition": row["Composition"],
                        "rho": rho,
                        "eta": eta,
                        "Eq5_selected": bool(row["Eq5_selected"]),
                    }
                )
    return pd.DataFrame(rows)


def plot_panel_a(ax: plt.Axes, reference: pd.DataFrame) -> None:
    orders = [9, 10, 11]
    pool_counts = reference.groupby("n_metals")["Composition"].count().to_dict()
    eq3_counts = reference.groupby("n_metals")["Eq3_selected"].sum().to_dict()
    eq5_counts = reference.groupby("n_metals")["Eq5_selected"].sum().to_dict()
    stages = [
        ("Pool", pool_counts, 0.14),
        ("Eq. (3)", eq3_counts, 0.52),
        ("Eq. (5)", eq5_counts, 0.90),
    ]
    x = np.arange(len(orders))
    width = 0.23
    cmap = plt.get_cmap("viridis")
    for i, (label, counts, cval) in enumerate(stages):
        vals = [int(counts.get(float(order), 0)) for order in orders]
        xpos = x + (i - 1) * width
        bars = ax.bar(
            xpos,
            vals,
            width=width,
            color=cmap(cval),
            edgecolor="white",
            linewidth=0.5,
            label=label,
        )
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.10,
                f"{value}",
                ha="center",
                va="bottom",
                fontsize=8.0,
            )
    ax.set_yscale("log")
    ax.set_ylim(0.7, 360)
    ax.set_xticks(x)
    ax.set_xticklabels([f"HEC{order}" for order in orders])
    ax.set_ylabel("Candidate count")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.13),
        ncol=3,
        handlelength=1.1,
        columnspacing=0.9,
        borderpad=0.2,
    )
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.75)
    add_panel_label(ax, "a")


def plot_panel_b(ax: plt.Axes, reference: pd.DataFrame, audit: pd.DataFrame) -> None:
    pool_sizes = {9: 220, 10: 66, 11: 12}
    table = audit.copy()
    table["_order"] = table["Composition"].map({c: i for i, c in enumerate(DISPLAY_ORDER)})
    table = table.sort_values("_order").reset_index(drop=True)
    table["order"] = table["n_metals"].astype(int)
    table["rank_label"] = table.apply(
        lambda row: f"{int(row['Basin_rank'])}/{pool_sizes[int(row['order'])]}",
        axis=1,
    )

    y = np.arange(len(table))
    point_color = plt.get_cmap("viridis")(0.55)
    ax.hlines(y, 0.823, table["Basin_score"], color="#D0D0D0", linewidth=1.0, zorder=1)
    ax.scatter(
        table["Basin_score"],
        y,
        s=78,
        color=point_color,
        edgecolor="#111111",
        linewidth=0.65,
        zorder=3,
    )
    for _, row in table.iterrows():
        yy = int(row.name)
        ax.text(
            row["Basin_score"] + 0.0012,
            yy,
            row["rank_label"],
            va="center",
            ha="left",
            fontsize=8.0,
            color="#222222",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([f"{CANDIDATE_NUMBERS[c]}. {DISPLAY[c]}" for c in table["Composition"]])
    ax.invert_yaxis()
    ax.set_xlim(0.822, 0.868)
    ax.set_xlabel(r"Basin score, $B(c)$")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.5, alpha=0.75)
    add_panel_label(ax, "b")


def plot_panel_c(ax: plt.Axes, audit: pd.DataFrame) -> None:
    pool_sizes = {9: 220, 10: 66, 11: 12}
    eq3_sizes = {9: 66, 10: 20, 11: 4}
    table = audit.copy()
    table["order"] = table["n_metals"].astype(int)
    table["basin_percentile"] = table.apply(
        lambda row: 1.0 - (row["Basin_rank"] - 1) / (pool_sizes[int(row["order"])] - 1),
        axis=1,
    )
    table["efa_percentile"] = table.apply(
        lambda row: 1.0 - (row["EFA_rank_within_Eq3"] - 1) / (eq3_sizes[int(row["order"])] - 1),
        axis=1,
    )
    table["_order"] = table["Composition"].map({c: i for i, c in enumerate(DISPLAY_ORDER)})
    table = table.sort_values("_order")
    y = np.arange(len(table))
    basin_color = plt.get_cmap("viridis")(0.42)
    efa_color = plt.get_cmap("viridis")(0.90)
    for yy, (_, row) in zip(y, table.iterrows()):
        ax.plot(
            [row["basin_percentile"], row["efa_percentile"]],
            [yy, yy],
            color="#C9C9C9",
            linewidth=0.8,
            zorder=1,
        )
        ax.scatter(
            row["basin_percentile"],
            yy,
            s=42,
            marker="o",
            color=basin_color,
            edgecolor="#111111",
            linewidth=0.55,
            zorder=2,
        )
        ax.scatter(
            row["efa_percentile"],
            yy,
            s=46,
            marker="D",
            color=efa_color,
            edgecolor="#111111",
            linewidth=0.55,
            zorder=3,
        )
    ax.set_xlim(0.25, 1.03)
    ax.set_yticks(y)
    ax.set_yticklabels([str(CANDIDATE_NUMBERS[c]) for c in table["Composition"]])
    ax.invert_yaxis()
    ax.set_xlabel("Rank percentile")
    ax.set_ylabel("Candidate")
    ax.grid(color="#D9D9D9", linewidth=0.5, alpha=0.75)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", color="none",
                   markerfacecolor=basin_color, markeredgecolor="#111111",
                   markersize=5.5, label="Basin rank"),
            Line2D([0], [0], marker="D", linestyle="none", color="none",
                   markerfacecolor=efa_color, markeredgecolor="#111111",
                   markersize=5.5, label="Conditional EFA rank"),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.58, 1.12),
        ncol=2,
        handletextpad=0.35,
        columnspacing=0.9,
        borderpad=0.2,
    )
    add_panel_label(ax, "c")


def plot_panel_d(ax: plt.Axes, audit: pd.DataFrame) -> None:
    table = audit.copy()
    table["_order"] = table["Composition"].map({c: i for i, c in enumerate(DISPLAY_ORDER)})
    table = table.sort_values("_order")
    freq = 100.0 * table["Eq5_frequency"].to_numpy(float)
    y = np.arange(len(table))
    colors = plt.get_cmap("viridis")(Normalize(vmin=0, vmax=100)(freq))
    bars = ax.barh(y, freq, height=0.58, color=colors, edgecolor="none")
    for bar, value in zip(bars, freq):
        x = max(value + 2.0, 3.0)
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:.0f}%", va="center", fontsize=8.4)
    ax.set_xlim(0, 108)
    ax.set_yticks(y)
    ax.set_yticklabels([str(CANDIDATE_NUMBERS[c]) for c in table["Composition"]])
    ax.invert_yaxis()
    ax.set_xlabel("Eq. (5) selection frequency (%)")
    ax.set_ylabel("Candidate")
    ax.axvline(50, color="#777777", linewidth=0.7, linestyle="--", dashes=(3, 2))
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.5, alpha=0.75)
    add_panel_label(ax, "d")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the revised main-text Figure 2.")
    parser.add_argument("--base", default=".")
    parser.add_argument("--outdir", default="validation_outputs/figures")
    parser.add_argument("--stem", default="figure_2_refined_equation_selection")
    parser.add_argument("--font-path", default="validation_outputs/fonts/EBGaramond-wght.ttf")
    args = parser.parse_args()

    base = Path(args.base)
    setup_style(base / args.font_path)
    outdir = Path(args.outdir)
    basin, efa, reference, audit = load_inputs(base)
    candidate_basin = basin[basin["n_metals"].isin([9.0, 10.0, 11.0])].copy()
    candidate_efa = efa[efa["Composition"].isin(candidate_basin["Composition"])].copy()
    # The full grid is retained in the CSV outputs and Table 2. The main figure
    # summarizes it as candidate-level Eq. (5) selection frequency.
    _ = compute_grid_selection(
        candidate_basin,
        candidate_efa,
        rho_grid=[0.10, 0.20, 0.30, 0.40, 0.50],
        eta_grid=[0.05, 0.10, 0.20, 0.30, 0.50],
    )

    fig = plt.figure(figsize=(7.8, 5.55), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.16],
        height_ratios=[1.0, 1.0],
        left=0.09,
        right=0.985,
        top=0.965,
        bottom=0.12,
        wspace=0.42,
        hspace=0.60,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_panel_a(ax_a, reference)
    plot_panel_b(ax_b, reference, audit)
    plot_panel_c(ax_c, audit)
    plot_panel_d(ax_d, audit)

    save_all(fig, outdir, args.stem)
    print(f"Wrote {outdir / (args.stem + '.pdf')}")
    print(f"Wrote {outdir / (args.stem + '.svg')}")
    print(f"Wrote {outdir / (args.stem + '.png')}")


if __name__ == "__main__":
    main()
