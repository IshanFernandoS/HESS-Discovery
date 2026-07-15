"""Manuscript-consistent GNN training/evaluation entry point.

The original repository script used an 80/20 HEC5 split, an 8-layer checkpoint
and MSE loss. Those settings do not reproduce the revised manuscript. This
script documents and runs the GNN path used for the current paper:

- HEC5 90/10 train/hold-out split with random_state=42;
- GCNEncoder(hidden_dim=4, num_layers=2, heads=32, dropout=0.2);
- Smooth L1 loss for training;
- checkpoint selection by best HEC5 hold-out R2;
- HEC6 evaluation on 22 unique POCC-DFT compositions.

The HEC5 hold-out subset is a model-selection subset, not an untouched final
test set. The HEC6 set is a targeted transfer/architecture-sensitivity
benchmark and is not used for GNN weight updates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from validations.common import canonicalize
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


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


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


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 45.0) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    true_label = (y_true > threshold).astype(int)
    pred_label = (y_pred > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_label,
        pred_label,
        average="binary",
        zero_division=0,
    )
    cm = confusion_matrix(true_label, pred_label, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "accuracy_at_45": float(accuracy_score(true_label, pred_label)),
        "precision_single_at_45": float(precision),
        "recall_single_at_45": float(recall),
        "f1_single_at_45": float(f1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "threshold": float(threshold),
    }


def load_hec5(hec5_path: str | Path) -> pd.DataFrame:
    hec5 = read_table(hec5_path)
    if "Composition" not in hec5.columns or "EFA" not in hec5.columns:
        raise ValueError("HEC5 table must contain Composition and EFA columns.")
    hec5 = hec5[["Composition", "EFA"]].copy()
    hec5["Composition"] = hec5["Composition"].astype(str).map(canonicalize)
    hec5["EFA"] = pd.to_numeric(hec5["EFA"], errors="raise")
    return hec5


def load_hec6_unique(hec6_path: str | Path) -> tuple[pd.DataFrame, int]:
    hec6 = read_table(hec6_path)
    if "Composition" not in hec6.columns or "Actual_EFA" not in hec6.columns:
        raise ValueError("HEC6 table must contain Composition and Actual_EFA columns.")
    hec6 = hec6[["Composition", "Actual_EFA"]].copy()
    hec6["Composition"] = hec6["Composition"].astype(str).map(canonicalize)
    hec6["Actual_EFA"] = pd.to_numeric(hec6["Actual_EFA"], errors="raise")
    duplicate_rows = int(hec6["Composition"].duplicated().sum())
    hec6 = hec6.drop_duplicates("Composition", keep="first").reset_index(drop=True)
    return hec6, duplicate_rows


def evaluate_checkpoint(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    element_features, feature_labels = build_matminer_element_feature_lookup(Path(args.repo_root))
    model = load_model(Path(args.checkpoint), device, in_channels=len(feature_labels))

    hec5 = load_hec5(args.hec5)
    indices = np.arange(len(hec5))
    _, holdout_idx = train_test_split(
        indices,
        test_size=args.holdout_fraction,
        random_state=args.split_seed,
    )
    hec5_holdout = hec5.iloc[holdout_idx].reset_index(drop=True)
    hec5_graphs = make_graphs(hec5_holdout["Composition"].tolist(), element_features)
    hec5_pred = predict_loader(model, hec5_graphs, device, args.batch_size)
    hec5_out = hec5_holdout.copy()
    hec5_out["Predicted_EFA"] = hec5_pred
    hec5_out["Residual"] = hec5_out["Predicted_EFA"] - hec5_out["EFA"]
    hec5_metrics = metric_summary(hec5_out["EFA"].to_numpy(), hec5_pred, args.threshold)
    hec5_metrics.update(
        {
            "split_seed": int(args.split_seed),
            "holdout_fraction": float(args.holdout_fraction),
            "evaluation_role": "HEC5 hold-out model-selection subset",
        }
    )

    hec6, duplicate_rows = load_hec6_unique(args.hec6)
    hec6_graphs = make_graphs(hec6["Composition"].tolist(), element_features)
    hec6_pred = predict_loader(model, hec6_graphs, device, args.batch_size)
    hec6_out = hec6.copy()
    hec6_out["Predicted_EFA"] = hec6_pred
    hec6_out["Residual"] = hec6_out["Predicted_EFA"] - hec6_out["Actual_EFA"]
    hec6_metrics = metric_summary(hec6_out["Actual_EFA"].to_numpy(), hec6_pred, args.threshold)
    hec6_metrics.update(
        {
            "duplicate_rows_removed": duplicate_rows,
            "evaluation_role": "22-unique-composition HEC6 POCC-DFT transfer benchmark",
        }
    )

    hec5_out.to_csv(outdir / "hec5_holdout_predictions.csv", index=False)
    hec6_out.to_csv(outdir / "hec6_unique_predictions.csv", index=False)
    provenance = {
        "checkpoint": str(Path(args.checkpoint)),
        "architecture": "GCNEncoder(hidden_dim=4, num_layers=2, heads=32, dropout=0.2)",
        "feature_generation": "matminer/pymatgen element features plus Bond.xlsx",
        "n_node_features": len(feature_labels),
        "edge_attr_rule": "carbon-edge feature plus metal-metal absolute formation-energy-channel difference",
        "hec5": hec5_metrics,
        "hec6": hec6_metrics,
    }
    with (outdir / "gnn_evaluation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)

    print("HEC5 hold-out metrics:")
    print(pd.DataFrame([hec5_metrics]).to_string(index=False))
    print("\nHEC6 unique-composition metrics:")
    print(pd.DataFrame([hec6_metrics]).to_string(index=False))
    print(f"\nOutputs: {outdir.resolve()}")


def train_checkpoint(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    element_features, feature_labels = build_matminer_element_feature_lookup(Path(args.repo_root))

    hec5 = load_hec5(args.hec5)
    indices = np.arange(len(hec5))
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=args.holdout_fraction,
        random_state=args.split_seed,
    )
    train = hec5.iloc[train_idx].reset_index(drop=True)
    holdout = hec5.iloc[holdout_idx].reset_index(drop=True)
    train_graphs = attach_targets(
        make_graphs(train["Composition"].tolist(), element_features),
        train["EFA"].to_numpy(dtype=float),
    )
    holdout_graphs = attach_targets(
        make_graphs(holdout["Composition"].tolist(), element_features),
        holdout["EFA"].to_numpy(dtype=float),
    )

    model = GCNEncoder(in_channels=len(feature_labels)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        amsgrad=False,
    )
    loss_fn = nn.SmoothL1Loss()
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)

    best_state = None
    best_r2 = -np.inf
    stale = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_graphs = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch).view(-1)
            loss = loss_fn(pred, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch.num_graphs
            n_graphs += batch.num_graphs

        if epoch % args.eval_every != 0:
            continue

        holdout_pred = predict_loader(model, holdout_graphs, device, args.batch_size)
        holdout_r2 = float(r2_score(holdout["EFA"].to_numpy(dtype=float), holdout_pred))
        row = {
            "epoch": int(epoch),
            "train_loss": total_loss / max(1, n_graphs),
            "holdout_r2": holdout_r2,
            "holdout_mae": float(mean_absolute_error(holdout["EFA"], holdout_pred)),
        }
        history.append(row)
        print(row)

        if holdout_r2 > best_r2:
            best_r2 = holdout_r2
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if args.patience and stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training ended before the first hold-out evaluation.")
    output_checkpoint = Path(args.output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_checkpoint)
    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)
    print(f"Saved best checkpoint to {output_checkpoint.resolve()} with hold-out R2={best_r2:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run manuscript-consistent HEC GNN training/evaluation.")
    parser.add_argument("--repo-root", default=".", help="Repository root containing data/ and checkpoints/.")
    parser.add_argument("--hec5", default="data/HEC5_output.csv")
    parser.add_argument("--hec6", default="DFT_Calculations/DFT_calculated_EFA_for_HEC6.csv")
    parser.add_argument("--checkpoint", default="checkpoints/best_gnn_model_4_2_32_88.0.pth")
    parser.add_argument("--outdir", default="validation_outputs/manuscript_gnn")
    parser.add_argument("--holdout-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=45.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--train", action="store_true", help="Retrain from HEC5 instead of only evaluating checkpoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--max-epochs", type=int, default=1_000_000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--output-checkpoint", default="checkpoints/retrained_manuscript_gnn.pth")
    args = parser.parse_args()

    if args.train:
        train_checkpoint(args)
    else:
        evaluate_checkpoint(args)


if __name__ == "__main__":
    main()
