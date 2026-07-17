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
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from validations.common import canonicalize, ensure_dir
from validations.generate_refined_figure_s10 import (
    GCNEncoder,
    build_matminer_element_feature_lookup,
    graph_from_composition_matminer,
    load_model,
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


COLORS = {
    "dark": "#222222",
    "grid": "#e6e6e6",
    "checkpoint": "#440154",
}


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
            "font.size": 10.0,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def attach_targets(graphs: list, targets: np.ndarray) -> list:
    for graph, target in zip(graphs, targets):
        graph.y = torch.tensor(float(target), dtype=torch.float32)
    return graphs


def make_graphs(compositions: list[str], element_features: dict[str, np.ndarray]) -> list:
    return [graph_from_composition_matminer(comp, element_features) for comp in compositions]


def predict_loader(model: nn.Module, graphs: list, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    pred = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred.append(model(batch).view(-1).detach().cpu().numpy())
    return np.concatenate(pred)


def load_hec6_graphs(path: Path, element_features: dict[str, np.ndarray]) -> tuple[pd.DataFrame, list]:
    df = pd.read_csv(path)
    if "Composition" not in df.columns or "Actual_EFA" not in df.columns:
        raise ValueError("HEC6 table must contain Composition and Actual_EFA columns.")
    df = df.copy()
    df["Composition"] = df["Composition"].astype(str).map(canonicalize)
    df = df.drop_duplicates("Composition", keep="first").reset_index(drop=True)
    graphs = attach_targets(
        make_graphs(df["Composition"].tolist(), element_features),
        df["Actual_EFA"].to_numpy(dtype=float),
    )
    return df, graphs


def load_hec5_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Composition" not in df.columns or "EFA" not in df.columns:
        raise ValueError("HEC5 table must contain Composition and EFA columns.")
    out = df[["Composition", "EFA"]].copy()
    out["Composition"] = out["Composition"].astype(str).map(canonicalize)
    out["EFA"] = pd.to_numeric(out["EFA"], errors="raise")
    return out


def class_metrics(actual: np.ndarray, pred: np.ndarray, threshold: float = 45.0) -> dict[str, float]:
    y_true = (np.asarray(actual) > threshold).astype(int)
    y_pred = (np.asarray(pred) > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    bacc = np.nanmean([recall, specificity])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(bacc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def regression_and_classification(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    out = {
        "n": int(len(actual)),
        "r2": float(r2_score(actual, pred)) if len(np.unique(actual)) > 1 else np.nan,
        "mae": float(mean_absolute_error(actual, pred)),
    }
    out.update(class_metrics(actual, pred))
    return out


def train_one_architecture(
    train_graphs: list,
    holdout_graphs: list,
    holdout_y: np.ndarray,
    *,
    hidden_dim: int,
    heads: int,
    layers: int,
    dropout: float,
    seed: int,
    epochs: int,
    eval_interval: int,
    patience_evals: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    in_channels: int,
) -> tuple[GCNEncoder, dict]:
    set_seed(seed)
    model = GCNEncoder(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        num_layers=layers,
        heads=heads,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6, amsgrad=False)
    loss_fn = nn.SmoothL1Loss()
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)

    best_state = None
    best_holdout_r2 = -np.inf
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n_graphs = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch).view(-1)
            target = batch.y.view(-1)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch.num_graphs
            n_graphs += batch.num_graphs
        train_loss = total / max(1, n_graphs)

        if epoch % eval_interval == 0 or epoch == epochs:
            holdout_pred = predict_loader(model, holdout_graphs, device=device, batch_size=batch_size)
            holdout_r2 = float(r2_score(holdout_y, holdout_pred))
            holdout_mae = float(mean_absolute_error(holdout_y, holdout_pred))
            history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "holdout_r2": holdout_r2,
                    "holdout_mae": holdout_mae,
                }
            )
            print(
                "progress "
                f"layers={layers} hidden_dim={hidden_dim} heads={heads} seed={seed} "
                f"epoch={epoch} train_loss={train_loss:.6f} "
                f"holdout_r2={holdout_r2:.6f} holdout_mae={holdout_mae:.6f}",
                flush=True,
            )
            if holdout_r2 > best_holdout_r2:
                best_holdout_r2 = holdout_r2
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if patience_evals and stale >= patience_evals:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_epoch": int(best_epoch),
        "best_holdout_r2_during_training": float(best_holdout_r2),
        "epochs_run": int(epoch),
        "history": history,
    }


def draw_sweep_figure(summary: pd.DataFrame, outdir: Path) -> None:
    setup_style()
    sweep = summary[summary["source"] == "retrained_sweep"].copy()
    if sweep.empty:
        return
    grouped = (
        sweep.groupby(["hidden_dim", "heads"], as_index=False)
        .agg(hec6_r2_mean=("hec6_r2", "mean"), hec6_r2_sd=("hec6_r2", "std"), hec5_r2_mean=("hec5_r2", "mean"))
        .sort_values(["hidden_dim", "heads"])
    )
    hidden = sorted(grouped["hidden_dim"].unique())
    heads = sorted(grouped["heads"].unique())
    matrix = grouped.pivot(index="hidden_dim", columns="heads", values="hec6_r2_mean").loc[hidden, heads]

    fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    im = ax.imshow(matrix.to_numpy(), cmap="viridis", vmin=-0.2, vmax=0.8, aspect="auto")
    ax.set_xticks(np.arange(len(heads)))
    ax.set_xticklabels([str(h) for h in heads])
    ax.set_yticks(np.arange(len(hidden)))
    ax.set_yticklabels([str(h) for h in hidden])
    ax.set_xlabel("Attention heads")
    ax.set_ylabel("Hidden dimension")
    for i, hd in enumerate(hidden):
        for j, head in enumerate(heads):
            val = matrix.loc[hd, head]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9.5, color="white" if val < 0.35 else COLORS["dark"])
    if 4 in hidden and 32 in heads:
        ax.scatter(
            [heads.index(32)],
            [hidden.index(4)],
            marker="*",
            s=120,
            c=COLORS["checkpoint"],
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"HEC6 transfer $R^2$")
    ax.set_title("Retrained architecture sweep")
    stem = outdir / "figure_S5d_gnn_architecture_sweep"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post hoc architecture sweep for the manuscript GNN. HEC5 hold-out is used "
            "for checkpoint selection within each retrained architecture; HEC6 is evaluated "
            "only after training and is not used to update weights."
        )
    )
    parser.add_argument("--hec6-csv", default="DFT_Calculations/DFT_calculated_EFA_for_HEC6.csv")
    parser.add_argument("--hec5-csv", default="data/HEC5_output.csv")
    parser.add_argument("--checkpoint", default="checkpoints/best_gnn_model_4_2_32_88.0.pth")
    parser.add_argument("--outdir", default="validation_outputs/gnn_architecture_sweep")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--heads", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--layers", nargs="+", type=int, default=[2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=1_000_000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--patience-evals", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--include-checkpoint", action="store_true")
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hec5_df = load_hec5_table(Path(args.hec5_csv))
    element_features, feature_labels = build_matminer_element_feature_lookup(Path("."))
    hec5_y = hec5_df["EFA"].to_numpy(dtype=float)
    indices = np.arange(len(hec5_df))
    train_idx, holdout_idx = train_test_split(indices, test_size=args.test_size, random_state=args.split_seed)
    train_df = hec5_df.iloc[train_idx].reset_index(drop=True)
    holdout_df = hec5_df.iloc[holdout_idx].reset_index(drop=True)
    train_graphs = attach_targets(
        make_graphs(train_df["Composition"].tolist(), element_features),
        train_df["EFA"].to_numpy(dtype=float),
    )
    holdout_graphs = attach_targets(
        make_graphs(holdout_df["Composition"].tolist(), element_features),
        holdout_df["EFA"].to_numpy(dtype=float),
    )
    holdout_y = hec5_y[holdout_idx]

    hec6_df, hec6_graphs = load_hec6_graphs(Path(args.hec6_csv), element_features)
    hec6_y = hec6_df["Actual_EFA"].to_numpy(dtype=float)
    hec6_df.to_csv(outdir / "hec6_unique_input.csv", index=False)

    rows = []
    histories = {}
    for layers in args.layers:
        for hidden_dim in args.hidden_dims:
            for heads in args.heads:
                for seed in args.seeds:
                    model, train_info = train_one_architecture(
                        train_graphs,
                        holdout_graphs,
                        holdout_y,
                        hidden_dim=hidden_dim,
                        heads=heads,
                        layers=layers,
                        dropout=args.dropout,
                        seed=seed,
                        epochs=args.epochs,
                        eval_interval=args.eval_interval,
                        patience_evals=args.patience_evals,
                        lr=args.lr,
                        batch_size=args.batch_size,
                        device=device,
                        in_channels=len(feature_labels),
                    )
                    hec5_pred = predict_loader(model, holdout_graphs, device=device, batch_size=args.batch_size)
                    hec6_pred = predict_loader(model, hec6_graphs, device=device, batch_size=args.batch_size)
                    hec5_metrics = regression_and_classification(holdout_y, hec5_pred)
                    hec6_metrics = regression_and_classification(hec6_y, hec6_pred)
                    row = {
                        "source": "retrained_sweep",
                        "hidden_dim": int(hidden_dim),
                        "heads": int(heads),
                        "layers": int(layers),
                        "dropout": float(args.dropout),
                        "seed": int(seed),
                        "epochs_run": int(train_info["epochs_run"]),
                        "best_epoch": int(train_info["best_epoch"]),
                        "best_holdout_r2_during_training": float(train_info["best_holdout_r2_during_training"]),
                    }
                    row.update({f"hec5_{k}": v for k, v in hec5_metrics.items()})
                    row.update({f"hec6_{k}": v for k, v in hec6_metrics.items()})
                    rows.append(row)
                    key = f"layers{layers}_hidden{hidden_dim}_heads{heads}_seed{seed}"
                    histories[key] = train_info["history"]
                    pd.DataFrame(rows).to_csv(outdir / "architecture_sweep_metrics.csv", index=False)
                    print(
                        f"{key}: HEC5 R2={hec5_metrics['r2']:.3f}, "
                        f"HEC6 R2={hec6_metrics['r2']:.3f}, HEC6 F1={hec6_metrics['f1']:.3f}"
                    )

    if args.include_checkpoint:
        checkpoint_model = load_model(Path(args.checkpoint), device, in_channels=len(feature_labels))
        checkpoint_device = device
        hec5_pred = predict_loader(checkpoint_model, holdout_graphs, device=checkpoint_device, batch_size=args.batch_size)
        hec6_pred = predict_loader(checkpoint_model, hec6_graphs, device=checkpoint_device, batch_size=args.batch_size)
        hec5_metrics = regression_and_classification(holdout_y, hec5_pred)
        hec6_metrics = regression_and_classification(hec6_y, hec6_pred)
        row = {
            "source": "manuscript_checkpoint",
            "hidden_dim": 4,
            "heads": 32,
            "layers": 2,
            "dropout": 0.2,
            "seed": np.nan,
            "epochs_run": np.nan,
            "best_epoch": np.nan,
            "best_holdout_r2_during_training": np.nan,
        }
        row.update({f"hec5_{k}": v for k, v in hec5_metrics.items()})
        row.update({f"hec6_{k}": v for k, v in hec6_metrics.items()})
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "architecture_sweep_metrics.csv", index=False)
    with (outdir / "architecture_sweep_history.json").open("w", encoding="utf-8") as handle:
        json.dump(histories, handle, indent=2)
    draw_sweep_figure(summary, outdir)

    print(f"Saved architecture sweep outputs to {outdir.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
