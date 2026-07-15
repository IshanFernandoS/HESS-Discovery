from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SELECTED = [
    "(HfMoNbReTaTiVWZr)C9",
    "(HfMoNbScTaTiVWZr)C9",
    "(HfMoNbTaTiVWYZr)C9",
    "(HfMoNbReScTaTiVWZr)C10",
    "(HfMoNbScTaTiVWYZr)C10",
    "(HfMoNbReTaTiVWYZr)C10",
    "(HfMoNbReScTaTiVWYZr)C11",
]

# Values from the current Table S2 and HEC9 score-decomposition export.
# HEC9 Pcal is not present in the decomposition CSV, so it is kept from Table S2.
HEC9_TABLE_S2 = {
    "(HfMoNbReTaTiVWZr)C9": {"P_single_cal": 0.916667, "Score": 1.140748},
    "(HfMoNbScTaTiVWZr)C9": {"P_single_cal": 0.916667, "Score": 1.162700},
    "(HfMoNbTaTiVWYZr)C9": {"P_single_cal": 0.916667, "Score": 1.071293},
}

COLORS = {
    "non_y": "#2F6B9A",
    "y_failed": "#B54A4A",
    "orange": "#D9822B",
    "purple": "#725A9C",
    "gray": "#666666",
    "light_gray": "#E6E6E6",
    "dark": "#222222",
}


def canonicalize(comp: str) -> str:
    s = str(comp).strip()
    match = re.match(r"^\s*\(([^)]+)\)\s*C?\s*(\d+)\s*$", s)
    if not match:
        return s
    metals = re.findall(r"[A-Z][a-z]?", match.group(1))
    return f"({''.join(sorted(metals))})C{int(match.group(2))}"


def parse_metals(comp: str) -> list[str]:
    match = re.match(r"^\(([^)]+)\)C(\d+)$", canonicalize(comp))
    if not match:
        return []
    return re.findall(r"[A-Z][a-z]?", match.group(1))


def latex_comp(comp: str) -> str:
    comp = canonicalize(comp)
    match = re.match(r"^\(([^)]+)\)C(\d+)$", comp)
    if not match:
        return comp
    return rf"\((\mathrm{{{match.group(1)}}})\mathrm{{C}}_{{{match.group(2)}}}\)"


def short_label(comp: str) -> str:
    comp = canonicalize(comp)
    metals = parse_metals(comp)
    suffix = re.search(r"C(\d+)$", comp).group(1)
    tags = [el for el in ["Re", "Sc", "Y"] if el in metals]
    return f"HEC{suffix} {'+'.join(tags)}"


def is_y_containing(comp: str) -> bool:
    return "Y" in parse_metals(comp)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
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


def load_gmm_values(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for comp, values in HEC9_TABLE_S2.items():
        rows.append({"Composition": canonicalize(comp), **values})
    for path in [Path(args.hec10_rank_csv), Path(args.hec11_rank_csv)]:
        df = pd.read_csv(path)
        df["Composition"] = df["Composition"].map(canonicalize)
        hit = df[df["Composition"].isin([canonicalize(c) for c in SELECTED])].copy()
        for _, row in hit.iterrows():
            rows.append(
                {
                    "Composition": row["Composition"],
                    "P_single_cal": float(row["P_single_cal"]),
                    "Score": float(row["Score"]),
                }
            )
    return pd.DataFrame(rows).drop_duplicates("Composition")


def load_table(args: argparse.Namespace) -> pd.DataFrame:
    order = {canonicalize(comp): i for i, comp in enumerate(SELECTED)}
    base = pd.DataFrame({"Composition": [canonicalize(comp) for comp in SELECTED]})

    gmm = load_gmm_values(args)

    mc = pd.read_csv(args.mc_dropout_csv)
    mc["Composition"] = mc["Composition"].map(canonicalize)
    mc = mc[
        [
            "Composition",
            "Predicted_EFA_eval_mode",
            "MC_mean",
            "MC_std",
            "MC_q05",
            "MC_q95",
        ]
    ]

    desc = pd.read_csv(args.descriptor_csv)
    desc["Composition"] = desc["Composition"].map(canonicalize)
    desc = desc[
        [
            "Composition",
            "Radius Mismatch",
            "Electronegativity Mismatch",
            "Minimum Carbide Melting Point",
        ]
    ]

    out = base.merge(gmm, on="Composition", how="left").merge(mc, on="Composition", how="left").merge(desc, on="Composition", how="left")
    out["Contains_Y"] = out["Composition"].map(is_y_containing)
    out["Experiment_outcome"] = np.where(out["Contains_Y"], "Failed (user-reported)", "Not assigned here")
    out["Short_label"] = out["Composition"].map(short_label)
    out["Risk_note"] = np.where(
        out["Contains_Y"],
        "Y-containing; GMM-positive but lower-confidence by GNN/mismatch diagnostics.",
        "Non-Y candidate; GMM-positive reference within selected set.",
    )
    out["order"] = out["Composition"].map(order)
    return out.sort_values("order").drop(columns="order")


def save_table(table: pd.DataFrame, outdir: Path, texdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    texdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / "y_failure_supporting_table.csv", index=False)

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Candidate-level diagnostics for Y-containing experimental failures.} GMM values are kept consistent with Table~S2 and the order-specific HEC10/HEC11 ranking files. GNN uncertainty values are from the manuscript-consistent checkpoint using 500 MC-dropout passes. Only Y-containing compositions are assigned a failure outcome here, based on the reported experimental observation; non-Y outcomes are intentionally not assigned unless separately confirmed.}",
        r"\label{tab:y_failure_supporting}",
        r"\begin{tabular}{lcccccccp{0.21\textwidth}}",
        r"\toprule",
        r"Composition & Y? & Experiment & \(P^{\mathrm{cal}}_{\mathrm{single}}\) & Score & MC EFA & MC 5--95\% & \(\delta_r\) & Interpretation \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        y_flag = "Yes" if row["Contains_Y"] else "No"
        experiment = str(row["Experiment_outcome"]).replace(" (user-reported)", "")
        interval = f"{row['MC_q05']:.1f}--{row['MC_q95']:.1f}"
        lines.append(
            f"{latex_comp(row['Composition'])} & {y_flag} & {experiment} & "
            f"{row['P_single_cal']:.6f} & {row['Score']:.6f} & {row['MC_mean']:.1f} & "
            f"{interval} & {row['Radius Mismatch']:.4f} & {row['Risk_note']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (texdir / "y_failure_supporting_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["dark"],
    )


def save_all(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_figure(table: pd.DataFrame, outdir: Path) -> None:
    setup_style()
    table = table.copy()
    labels = table["Short_label"].to_list()
    y = np.arange(len(table))[::-1]
    colors = np.where(table["Contains_Y"], COLORS["y_failed"], COLORS["non_y"])

    fig = plt.figure(figsize=(7.3, 5.0))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.095,
        right=0.985,
        bottom=0.115,
        top=0.965,
        wspace=0.50,
        hspace=0.56,
        width_ratios=[1.02, 1.18],
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    # (a) GMM/basin score.
    ax = axes[0]
    ax.barh(y, table["Score"], color=colors, height=0.62, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("GMM composite score")
    ax.set_xlim(0.70, max(1.30, float(table["Score"].max()) + 0.04))
    ax.grid(axis="x", color=COLORS["light_gray"], linewidth=0.6)
    add_panel_label(ax, "(a)")

    # (b) GNN uncertainty.
    ax = axes[1]
    xerr = np.vstack([table["MC_mean"] - table["MC_q05"], table["MC_q95"] - table["MC_mean"]])
    ax.errorbar(
        table["MC_mean"],
        y,
        xerr=xerr,
        fmt="o",
        color=COLORS["purple"],
        ecolor=COLORS["purple"],
        elinewidth=1.15,
        capsize=3,
        markersize=4.3,
        zorder=2,
    )
    ax.scatter(table["Predicted_EFA_eval_mode"], y, marker="D", s=24, color=COLORS["orange"], zorder=3)
    ax.axvline(45, color=COLORS["gray"], linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("Predicted EFA")
    ax.set_xlim(33, 52)
    ax.grid(axis="x", color=COLORS["light_gray"], linewidth=0.6)
    add_panel_label(ax, "(b)")

    # (c) mismatch descriptor map.
    ax = axes[2]
    markers = {9: "o", 10: "s", 11: "^"}
    for n_metals, marker in markers.items():
        part = table[table["Composition"].str.extract(r"C(\d+)$")[0].astype(int) == n_metals]
        if part.empty:
            continue
        ax.scatter(
            part["Radius Mismatch"],
            part["Electronegativity Mismatch"],
            s=42,
            marker=marker,
            c=np.where(part["Contains_Y"], COLORS["y_failed"], COLORS["non_y"]),
            edgecolors="white",
            linewidth=0.6,
            zorder=3,
            label=f"HEC{n_metals}",
        )
    panel_c_offsets = {
        "HEC9 Y": (3, 3),
        "HEC10 Sc+Y": (4, -5),
        "HEC10 Re+Y": (4, 5),
        "HEC11 Re+Sc+Y": (4, -5),
    }
    for _, row in table.iterrows():
        offset = panel_c_offsets.get(row["Short_label"], (3, 2))
        ax.annotate(
            row["Short_label"].replace("HEC", ""),
            (row["Radius Mismatch"], row["Electronegativity Mismatch"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.7,
        )
    ax.set_xlabel(r"Radius mismatch, $\delta_r$")
    ax.set_ylabel(r"Electronegativity mismatch, $\delta_\chi$")
    ax.grid(color=COLORS["light_gray"], linewidth=0.6)
    add_panel_label(ax, "(c)")

    # (d) integrated view: basin score vs GNN confidence.
    ax = axes[3]
    panel_d_offsets = {
        "HEC9 Y": (-21, -11),
        "HEC10 Sc+Y": (-2, 8),
        "HEC10 Re+Y": (5, -9),
        "HEC11 Re+Sc+Y": (5, 3),
    }
    for _, row in table.iterrows():
        ax.scatter(
            row["Score"],
            row["MC_mean"],
            s=54,
            marker="X" if row["Contains_Y"] else "o",
            color=COLORS["y_failed"] if row["Contains_Y"] else COLORS["non_y"],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        offset = panel_d_offsets.get(row["Short_label"], (3, 2))
        ax.annotate(
            row["Short_label"].replace("HEC", ""),
            (row["Score"], row["MC_mean"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.7,
        )
    ax.axhline(45, color=COLORS["gray"], linestyle="--", linewidth=0.8)
    ax.set_xlabel("GMM composite score")
    ax.set_ylabel("MC mean EFA")
    ax.set_xlim(0.70, max(1.30, float(table["Score"].max()) + 0.04))
    ax.set_ylim(39.5, 46.1)
    ax.grid(color=COLORS["light_gray"], linewidth=0.6)
    add_panel_label(ax, "(d)")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["non_y"], markeredgecolor="white", markersize=6, label="Non-Y / outcome not assigned"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=COLORS["y_failed"], markeredgecolor="white", markersize=6, label="Y-containing failed"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["orange"], markeredgecolor=COLORS["orange"], markersize=5, label="Eval-mode EFA"),
        Line2D([0], [0], color=COLORS["gray"], linestyle="--", linewidth=0.8, label="EFA = 45"),
    ]
    fig.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.55, 0.005), ncol=4, handlelength=1.2, columnspacing=1.2)
    fig.subplots_adjust(bottom=0.17)
    save_all(fig, outdir, "y_failure_supporting_figure_abcd")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Y-failure support table and figure.")
    parser.add_argument("--descriptor-csv", default=r"C:/Users/Ishan/Downloads/Compositions_Descriptros(Sheet1) (1).csv")
    parser.add_argument("--hec10-rank-csv", default=r"C:/Users/Ishan/Downloads/HEC10_rank_composite_noEFA.csv")
    parser.add_argument("--hec11-rank-csv", default=r"C:/Users/Ishan/Downloads/HEC11_rank_composite_noEFA.csv")
    parser.add_argument("--mc-dropout-csv", default="validation_outputs/gnn_refined_s10/selected_candidate_mc_dropout_predictions.csv")
    parser.add_argument("--table-outdir", default="validation_outputs/candidate_validation")
    parser.add_argument("--tex-outdir", default="validation_outputs/latex_tables")
    parser.add_argument("--figure-outdir", default="validation_outputs/figures")
    args = parser.parse_args()

    table = load_table(args)
    save_table(table, Path(args.table_outdir), Path(args.tex_outdir))
    make_figure(table, Path(args.figure_outdir))
    print(table.to_string(index=False))
    print(f"Saved table CSV to {Path(args.table_outdir).resolve() / 'y_failure_supporting_table.csv'}")
    print(f"Saved table TeX to {Path(args.tex_outdir).resolve() / 'y_failure_supporting_table.tex'}")
    print(f"Saved figure files to {Path(args.figure_outdir).resolve()}")


if __name__ == "__main__":
    main()
