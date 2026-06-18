from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import mannwhitneyu
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.composition.ion import IonProperty
from matminer.featurizers.composition.alloy import Miedema, WenAlloys, YangSolidSolution
from matminer.featurizers.composition.element import BandCenter, Stoichiometry, TMetalFraction
from matminer.featurizers.composition.orbital import AtomicOrbitals, ValenceOrbital

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, GlobalAttention, GraphNorm
from torch_geometric.utils import dense_to_sparse


COLORS = {
    "blue": "#2F6B9A",
    "light_blue": "#DDEAF3",
    "orange": "#D9822B",
    "light_orange": "#F3D8BC",
    "purple": "#725A9C",
    "red": "#B54A4A",
    "gray": "#666666",
    "light_gray": "#E6E6E6",
    "dark": "#222222",
}

SINGLE_PHASE_LIT = [
    "(CrHfMoNbTaTiVZr)C8",
    "(CrHfNbTaTiVWZr)C8",
    "(CrHfNbTaTi)C5",
    "(CrHfNbTaV)C5",
    "(CrHfTaTiZr)C5",
    "(CrMoNbTaW)C5",
    "(CrMoNbVW)C5",
    "(CrMoTaVW)C5",
    "(CrMoTiVW)C5",
    "(HfMoNbTaTi)C5",
    "(HfMoNbTaTiVWZr)C8",
    "(HfMoNbTaTiZr)C6",
    "(HfMoNbTaV)C5",
    "(HfMoNbTaZr)C5",
    "(HfMoTaTiZr)C5",
    "(HfNbTaTiV)C5",
    "(HfNbTaTiVZr)C6",
    "(HfNbTaTiW)C5",
    "(HfNbTaTiWZr)C6",
    "(HfNbTaTiZr)C5",
    "(HfNbTaVW)C5",
    "(HfTaTiWZr)C5",
    "(MoNbTaTiV)C5",
    "(MoNbTaTiW)C5",
    "(MoNbTaTiZr)C5",
    "(MoNbTaVW)C5",
    "(MoNbTaVZr)C5",
    "(NbTaTiVW)C5",
    "(NbTaTiWZr)C5",
    "(NbTaVWZr)C5",
    "(CrNbTaTiZr)C5",
    "(CrHfNbTaZr)C5",
    "(MoNbTaTiVW)C6",
    "(HfNbTaWZr)C5",
    "(MoNbTaWZr)C5",
]

MULTI_PHASE_LIT = [
    "(CrHfMoTiW)C5",
    "(CrHfMoVW)C5",
    "(CrHfTaVW)C5",
    "(CrHfTaWZr)C5",
    "(CrMoTiVZr)C5",
    "(CrMoTiWZr)C5",
    "(HfMoTaWZr)C5",
    "(HfMoTiWZr)C5",
    "(HfMoVWZr)C5",
    "(MoTiVWZr)C5",
]

SELECTED_CANDIDATES = [
    "(HfMoNbReTaTiVWZr)C9",
    "(HfMoNbScTaTiVWZr)C9",
    "(HfMoNbTaTiVWYZr)C9",
    "(HfMoNbReScTaTiVWZr)C10",
    "(HfMoNbScTaTiVWYZr)C10",
    "(HfMoNbReTaTiVWYZr)C10",
    "(HfMoNbReScTaTiVWYZr)C11",
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
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


def canonicalize(comp: str) -> str:
    s = str(comp).strip()
    match = re.match(r"^\s*\(([^)]+)\)\s*C?\s*(\d+)\s*$", s)
    if not match:
        return s
    metals = re.findall(r"[A-Z][a-z]?", match.group(1))
    return f"({''.join(sorted(metals))})C{int(match.group(2))}"


def metals_in_original_order(comp: str) -> list[str]:
    match = re.match(r"^\s*\(([^)]+)\)\s*C?\s*(\d+)\s*$", str(comp).strip())
    if not match:
        return []
    return re.findall(r"[A-Z][a-z]?", match.group(1))


def count_metals(comp: str) -> int:
    return len(metals_in_original_order(comp))


def build_element_feature_lookup(deephec_dir: Path) -> dict[str, np.ndarray]:
    feature_files = [
        deephec_dir / "data" / "final_features_train_updated.npy",
        deephec_dir / "data" / "final_features_test_updated.npy",
    ]
    composition_files = [
        deephec_dir / "data" / "HEC5_output.csv",
        deephec_dir / "data" / "HEC8_output.csv",
    ]
    lookup: dict[str, np.ndarray] = {}
    for feature_file, comp_file in zip(feature_files, composition_files):
        features = np.load(feature_file)
        df = pd.read_csv(comp_file)
        if len(features) != len(df):
            raise ValueError(f"Feature/CSV row mismatch: {feature_file} vs {comp_file}")
        for idx, comp in enumerate(df["Composition"].astype(str)):
            symbols = metals_in_original_order(comp) + ["C"]
            if len(symbols) != features.shape[1]:
                raise ValueError(f"Node count mismatch for {comp}")
            for symbol, vector in zip(symbols, features[idx]):
                vector = np.asarray(vector, dtype=np.float32)
                if symbol in lookup and not np.allclose(
                    lookup[symbol],
                    vector,
                    atol=1e-6,
                    rtol=1e-6,
                    equal_nan=True,
                ):
                    raise ValueError(f"Inconsistent node vector recovered for element {symbol}.")
                lookup[symbol] = vector
    return lookup


def build_matminer_element_feature_lookup(deephec_dir: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    """Build the exact element feature table used by the manuscript HEC6 checkpoint sweep."""
    featurizers = {
        "ElementProperty_Magpie": ElementProperty.from_preset("magpie"),
        "IonProperty": IonProperty(),
        "Miedema": Miedema(struct_types="all"),
        "WenAlloys": WenAlloys(),
        "YangSolidSolution": YangSolidSolution(),
        "BandCenter": BandCenter(),
        "Stoichiometry": Stoichiometry(),
        "TMetalFraction": TMetalFraction(),
        "AtomicOrbitals": AtomicOrbitals(),
        "ValenceOrbital": ValenceOrbital(),
    }
    target_elements = {
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Al",
        "Si",
        "C",
        "La",
    }
    element_features = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for element in target_elements:
            element_obj = Composition(element)
            combined_features = []
            for fz in featurizers.values():
                try:
                    feats = fz.featurize(element_obj)
                    numeric = [float(v) if isinstance(v, (int, float)) else float("nan") for v in feats]
                    combined_features.extend(numeric)
                except Exception:
                    combined_features.extend([float("nan")] * len(fz.feature_labels()))
            element_features[element] = combined_features

    all_feature_labels = []
    for name, fz in featurizers.items():
        all_feature_labels.extend([f"{name}_{label}" for label in fz.feature_labels()])

    feature_df = pd.DataFrame.from_dict(element_features, orient="index", columns=all_feature_labels)
    statistical_keywords = ["maximum", "minimum", "mode", "avg_dev", "range"]
    keep_cols = [
        col for col in feature_df.columns if not any(keyword in col.lower() for keyword in statistical_keywords)
    ]
    feature_df = feature_df[keep_cols]
    features_to_remove = [
        "IonProperty_compound possible",
        "WenAlloys_APE mean",
        "WenAlloys_Radii gamma",
        "WenAlloys_Mixing enthalpy",
        "Stoichiometry_0-norm",
        "Stoichiometry_2-norm",
        "Stoichiometry_3-norm",
        "Stoichiometry_5-norm",
        "Stoichiometry_7-norm",
        "Stoichiometry_10-norm",
        "WenAlloys_Atomic weight mean",
        "WenAlloys_Total weight",
        "ValenceOrbital_avg s valence electrons",
        "WenAlloys_Interant p electrons",
        "ValenceOrbital_avg p valence electrons",
        "ValenceOrbital_avg d valence electrons",
        "ValenceOrbital_avg f valence electrons",
        "WenAlloys_Interant s electrons",
    ]
    feature_df = feature_df.drop(columns=features_to_remove, errors="ignore")
    feature_df = feature_df.dropna(axis=1, how="any")
    feature_df = feature_df.loc[:, (feature_df != 0).any(axis=0)]
    feature_labels = feature_df.columns.tolist()
    lookup = {element: feature_df.loc[element].to_numpy(dtype=np.float32) for element in feature_df.index}

    bond_df = pd.read_excel(deephec_dir / "data" / "Bond.xlsx")
    bond_energy = {}
    for _, row in bond_df.iterrows():
        try:
            value = float(str(row["Formation Energy per atom"]).replace("eV/atom", "").strip())
        except Exception:
            value = float("nan")
        bond_energy[row["Name"]] = value
    feature_labels.append("Bond_Formation_Energy_per_atom")
    for element in lookup:
        lookup[element] = np.append(lookup[element], bond_energy.get(element, float("nan"))).astype(np.float32)
    return lookup, feature_labels


def create_edge_index(n_nodes: int) -> torch.Tensor:
    adj = torch.ones((n_nodes, n_nodes), dtype=torch.float32) - torch.eye(n_nodes, dtype=torch.float32)
    return dense_to_sparse(adj)[0]


def create_edge_attr(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    n_nodes = x.size(0)
    carbon_idx = n_nodes - 1
    edge_attr = torch.zeros(src.size(0), 1, dtype=x.dtype, device=x.device)
    mask_src = src == carbon_idx
    edge_attr[mask_src] = x[dst[mask_src], -1].unsqueeze(-1)
    mask_dst = (dst == carbon_idx) & ~mask_src
    edge_attr[mask_dst] = x[src[mask_dst], -1].unsqueeze(-1)
    mask_mm = ~(mask_src | mask_dst)
    edge_attr[mask_mm] = torch.abs(x[src[mask_mm], -1] - x[dst[mask_mm], -1]).unsqueeze(-1)
    return edge_attr


def graph_from_composition(comp: str, element_features: dict[str, np.ndarray]) -> Data:
    symbols = metals_in_original_order(comp) + ["C"]
    missing = [symbol for symbol in symbols if symbol not in element_features]
    if missing:
        raise KeyError(f"Missing element features for {missing} in {comp}")
    x = torch.tensor(np.stack([element_features[symbol] for symbol in symbols], axis=0), dtype=torch.float32)
    edge_index = create_edge_index(x.size(0))
    edge_attr = create_edge_attr(x, edge_index)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data


def graph_from_composition_matminer(comp: str, element_features: dict[str, np.ndarray]) -> Data:
    symbols = [element.symbol for element in Composition(comp).elements]
    missing = [symbol for symbol in symbols if symbol not in element_features]
    if missing:
        raise KeyError(f"Missing element features for {missing} in {comp}")
    x = torch.tensor(np.stack([element_features[symbol] for symbol in symbols], axis=0), dtype=torch.float32)
    edge_index = create_edge_index(x.size(0))
    edge_attr = create_edge_attr(x, edge_index)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data


class GCNEncoder(nn.Module):
    def __init__(self, in_channels=35, hidden_dim=4, num_layers=2, heads=32, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(
            GATv2Conv(
                in_channels=in_channels - 1,
                out_channels=hidden_dim,
                heads=heads,
                dropout=dropout,
                edge_dim=1,
                concat=False,
            )
        )
        for _ in range(1, num_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=False,
                )
            )
            self.norms.append(GraphNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.gate_nn = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.att_pool = GlobalAttention(gate_nn=self.gate_nn)
        self.readout_mlp = nn.Sequential(nn.Linear(hidden_dim, 1))

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        node_features = x[:, :-1].float()
        out = node_features
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = out
            if i == 0:
                out = conv(out, edge_index, edge_attr)
            else:
                out = conv(out, edge_index)
            out = norm(out)
            out = self.activation(out)
            out = self.dropout(out)
            if out.shape == residual.shape:
                out = out + residual
        graph_emb = self.att_pool(out, batch=batch)
        return self.readout_mlp(graph_emb)


def load_model(checkpoint: Path, device: torch.device, in_channels: int = 35) -> GCNEncoder:
    model = GCNEncoder(in_channels=in_channels).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_single_graphs(model: GCNEncoder, graphs: list[Data], device: torch.device) -> np.ndarray:
    preds = []
    model.eval()
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            preds.append(float(model(graph).view(-1).detach().cpu().numpy()[0]))
    return np.asarray(preds, dtype=float)


def mc_dropout_single_graphs(
    model: GCNEncoder,
    graphs: list[Data],
    device: torch.device,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    samples = []
    for _ in range(n_samples):
        model.train()
        preds = []
        with torch.no_grad():
            for graph in graphs:
                graph = graph.to(device)
                preds.append(float(model(graph).view(-1).detach().cpu().numpy()[0]))
        samples.append(preds)
    model.eval()
    return np.asarray(samples, dtype=float)


def make_literature_table(deephec_dir: Path) -> pd.DataFrame:
    rows = [{"Composition": canonicalize(c), "Exp_Phase": 1, "Label": "Single"} for c in SINGLE_PHASE_LIT]
    rows.extend({"Composition": canonicalize(c), "Exp_Phase": 0, "Label": "Multi"} for c in MULTI_PHASE_LIT)
    df = pd.DataFrame(rows).drop_duplicates("Composition").reset_index(drop=True)

    hec5 = pd.read_csv(deephec_dir / "data" / "HEC5_output.csv")
    hec5["Composition"] = hec5["Composition"].astype(str).map(canonicalize)
    indices = np.arange(len(hec5))
    train_idx, test_idx = train_test_split(indices, test_size=0.10, random_state=42)
    train_comps = set(hec5.loc[train_idx, "Composition"])
    test_comps = set(hec5.loc[test_idx, "Composition"])

    df["overlap_group"] = "Non-HEC5 order"
    df.loc[df["Composition"].isin(train_comps), "overlap_group"] = "HEC5 train-split overlap"
    df.loc[df["Composition"].isin(test_comps), "overlap_group"] = "HEC5 internal-test overlap"
    df["in_hec5_train_split"] = df["overlap_group"].eq("HEC5 train-split overlap")
    return df


def summarize_literature(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subset, part in [
        ("All literature labels", df),
        ("Excluding HEC5 train-overlap", df[~df["in_hec5_train_split"]].copy()),
    ]:
        y = part["Exp_Phase"].to_numpy(dtype=int)
        scores = part["GNN_predicted_EFA"].to_numpy(dtype=float)
        row = {
            "subset": subset,
            "n": int(len(part)),
            "n_single": int((y == 1).sum()),
            "n_multi": int((y == 0).sum()),
            "single_mean": float(part.loc[part["Exp_Phase"] == 1, "GNN_predicted_EFA"].mean()),
            "multi_mean": float(part.loc[part["Exp_Phase"] == 0, "GNN_predicted_EFA"].mean()),
        }
        if len(np.unique(y)) == 2:
            row["roc_auc"] = float(roc_auc_score(y, scores))
            row["average_precision"] = float(average_precision_score(y, scores))
            try:
                row["mannwhitney_p_single_greater"] = float(
                    mannwhitneyu(
                        part.loc[part["Exp_Phase"] == 1, "GNN_predicted_EFA"],
                        part.loc[part["Exp_Phase"] == 0, "GNN_predicted_EFA"],
                        alternative="greater",
                    ).pvalue
                )
            except Exception:
                row["mannwhitney_p_single_greater"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["dark"],
    )


def draw_literature_panel(ax: plt.Axes, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    groups = [("Multi", 0, COLORS["orange"], COLORS["light_orange"]), ("Single", 1, COLORS["blue"], COLORS["light_blue"])]
    data = [df.loc[df["Label"] == label, "GNN_predicted_EFA"].to_numpy(dtype=float) for label, _, _, _ in groups]
    box = ax.boxplot(
        data,
        positions=[0, 1],
        widths=0.44,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["dark"], "linewidth": 1.0},
        boxprops={"edgecolor": COLORS["dark"], "linewidth": 0.8},
        whiskerprops={"color": COLORS["dark"], "linewidth": 0.8},
        capprops={"color": COLORS["dark"], "linewidth": 0.8},
    )
    for patch, (_, _, _, fill) in zip(box["boxes"], groups):
        patch.set_facecolor(fill)

    rng = np.random.default_rng(17)
    for label, xpos, color, _ in groups:
        part = df[df["Label"] == label].copy()
        jitter = rng.normal(0, 0.042, size=len(part))
        ax.scatter(
            np.full(len(part), xpos) + jitter,
            part["GNN_predicted_EFA"],
            s=18,
            color=color,
            alpha=0.72,
            linewidths=0,
            zorder=3,
        )
    ax.axhline(45, color=COLORS["gray"], linestyle="--", linewidth=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Multi", "Single"])
    ax.set_xlabel("Literature phase label")
    ax.set_ylabel("GNN-predicted EFA")
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(10, 90)
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)

    all_row = summary.loc[summary["subset"] == "All literature labels"].iloc[0]
    text = (
        f"All AUC = {all_row['roc_auc']:.3f}\n"
        f"All AUPRC = {all_row['average_precision']:.3f}"
    )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color=COLORS["dark"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.90},
    )

    handles = [Line2D([0], [0], color=COLORS["gray"], linestyle="--", linewidth=0.8, label="EFA = 45")]
    ax.legend(handles=handles, frameon=False, loc="lower right", handlelength=1.2, handletextpad=0.45)


def draw_mc_panel(ax: plt.Axes, mc: pd.DataFrame) -> None:
    order = [canonicalize(c) for c in SELECTED_CANDIDATES]
    labels = {
        canonicalize("(HfMoNbReTaTiVWZr)C9"): "HEC9 Re",
        canonicalize("(HfMoNbScTaTiVWZr)C9"): "HEC9 Sc",
        canonicalize("(HfMoNbTaTiVWYZr)C9"): "HEC9 Y",
        canonicalize("(HfMoNbReScTaTiVWZr)C10"): "HEC10 Re+Sc",
        canonicalize("(HfMoNbScTaTiVWYZr)C10"): "HEC10 Sc+Y",
        canonicalize("(HfMoNbReTaTiVWYZr)C10"): "HEC10 Re+Y",
        canonicalize("(HfMoNbReScTaTiVWYZr)C11"): "HEC11 Re+Sc+Y",
    }
    mc = mc.set_index("Composition").loc[order].reset_index()
    y = np.arange(len(mc))[::-1]
    xerr = np.vstack([mc["MC_mean"] - mc["MC_q05"], mc["MC_q95"] - mc["MC_mean"]])
    ax.errorbar(
        mc["MC_mean"],
        y,
        xerr=xerr,
        fmt="o",
        color=COLORS["purple"],
        ecolor=COLORS["purple"],
        elinewidth=1.25,
        capsize=3,
        markersize=4.7,
        label="MC mean, 5-95%",
    )
    ax.scatter(
        mc["Predicted_EFA_eval_mode"],
        y,
        marker="D",
        s=29,
        color=COLORS["orange"],
        zorder=4,
        label="Eval-mode",
    )
    ax.axvline(45, color=COLORS["gray"], linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[c] for c in mc["Composition"]])
    ax.set_xlabel("Predicted EFA")
    xmin = float(min(mc["MC_q05"].min(), mc["Predicted_EFA_eval_mode"].min(), 45.0)) - 2.0
    xmax = float(max(mc["MC_q95"].max(), mc["Predicted_EFA_eval_mode"].max(), 45.0)) + 2.0
    ax.set_xlim(xmin, xmax)
    ax.grid(axis="x", color=COLORS["light_gray"], linewidth=0.6)
    ax.legend(frameon=False, loc="lower right", handlelength=1.2, handletextpad=0.45)


def save_all(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate refined, non-redundant Figure S10.")
    parser.add_argument(
        "--deephec-dir",
        default=".",
        help=(
            "Directory containing the manuscript data folder. The manuscript checkpoint "
            "best_gnn_model_4_2_32_88.0.pth is also expected here unless --checkpoint is supplied."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to best_gnn_model_4_2_32_88.0.pth. This script intentionally does not use the legacy GitHub checkpoint.",
    )
    parser.add_argument("--outdir", default="validation_outputs/figures")
    parser.add_argument("--data-outdir", default="validation_outputs/gnn_refined_s10")
    parser.add_argument("--stem", default="figure_S10_refined_gnn_validation")
    parser.add_argument("--mc-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_style()
    deephec = Path(args.deephec_dir)
    outdir = Path(args.outdir)
    data_outdir = Path(args.data_outdir)
    data_outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint) if args.checkpoint else deephec / "best_gnn_model_4_2_32_88.0.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing manuscript checkpoint: {checkpoint}. "
            "Use --checkpoint to point to best_gnn_model_4_2_32_88.0.pth. "
            "The legacy checkpoints/_model_26999_after_7l.pth is not used because it does not reproduce the manuscript GNN results."
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    element_features, feature_labels = build_matminer_element_feature_lookup(deephec)
    model = load_model(checkpoint, device, in_channels=len(feature_labels))

    lit = make_literature_table(deephec)
    lit_graphs = [graph_from_composition_matminer(comp, element_features) for comp in lit["Composition"]]
    lit["GNN_predicted_EFA"] = predict_single_graphs(model, lit_graphs, device)
    lit["Pred_single_at_45"] = (lit["GNN_predicted_EFA"] > 45).astype(int)
    lit_summary = summarize_literature(lit)

    selected = pd.DataFrame({"Composition": [canonicalize(c) for c in SELECTED_CANDIDATES]})
    selected["n_metals"] = selected["Composition"].map(count_metals)
    selected_graphs = [graph_from_composition_matminer(comp, element_features) for comp in selected["Composition"]]
    selected["Predicted_EFA_eval_mode"] = predict_single_graphs(model, selected_graphs, device)
    samples = mc_dropout_single_graphs(
        model,
        selected_graphs,
        device=device,
        n_samples=args.mc_samples,
        seed=args.seed,
    )
    selected["MC_mean"] = samples.mean(axis=0)
    selected["MC_std"] = samples.std(axis=0, ddof=1)
    selected["MC_q05"] = np.quantile(samples, 0.05, axis=0)
    selected["MC_q50"] = np.quantile(samples, 0.50, axis=0)
    selected["MC_q95"] = np.quantile(samples, 0.95, axis=0)
    selected["MC_n_samples"] = args.mc_samples

    lit.to_csv(data_outdir / "literature_gnn_predictions.csv", index=False)
    lit_summary.to_csv(data_outdir / "literature_gnn_summary.csv", index=False)
    selected.to_csv(data_outdir / "selected_candidate_mc_dropout_predictions.csv", index=False)
    selected.to_csv(data_outdir / "selected_hec9_mc_dropout_predictions.csv", index=False)
    with (data_outdir / "provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_architecture": "GCNEncoder(hidden_dim=4, num_layers=2, heads=32, dropout=0.2)",
                "feature_generation": "matminer/pymatgen path copied from HEC6_ablation_test.py",
                "bond_file": str(deephec / "data" / "Bond.xlsx"),
                "n_node_features": len(feature_labels),
                "edge_attr_rule": "carbon-edge feature plus metal-metal absolute formation-energy-channel difference",
                "mc_samples": args.mc_samples,
                "seed": args.seed,
            },
            handle,
            indent=2,
        )

    fig = plt.figure(figsize=(7.2, 3.65))
    gs = fig.add_gridspec(
        1,
        2,
        left=0.085,
        right=0.985,
        bottom=0.16,
        top=0.91,
        wspace=0.42,
        width_ratios=[0.90, 1.42],
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    draw_literature_panel(axes[0], lit, lit_summary)
    draw_mc_panel(axes[1], selected)
    for ax, label in zip(axes, ["(a)", "(b)"]):
        add_panel_label(ax, label)
    save_all(fig, outdir, args.stem)

    print(f"Saved refined Figure S10 to {outdir.resolve()}")
    print(lit_summary.to_string(index=False))
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
