from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from validations.common import canonicalize, parse_metals, regression_metrics


def _require_pyg():
    try:
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GATv2Conv, GlobalAttention, GraphNorm
        from torch_geometric.utils import dense_to_sparse
    except Exception as exc:
        raise ImportError(
            "This GNN validation requires torch_geometric. Install a PyTorch-compatible "
            "PyG build before running this script."
        ) from exc
    return Data, DataLoader, GATv2Conv, GlobalAttention, GraphNorm, dense_to_sparse


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metals_in_original_order(comp: str) -> list[str]:
    match = re.match(r"^\s*\(([^)]+)\)\s*C?\s*(\d+)\s*$", str(comp).strip())
    if not match:
        return parse_metals(comp)
    return re.findall(r"[A-Z][a-z]?", match.group(1))


def build_element_feature_lookup(
    feature_files: Sequence[str | Path] = ("data/final_features_train.npy", "data/final_features_test.npy"),
    composition_files: Sequence[str | Path] = ("data/HEC5_output.csv", "data/HEC8_output.csv"),
    strict: bool = True,
) -> dict[str, np.ndarray]:
    """Recover per-element node vectors from saved composition feature arrays."""
    lookup: dict[str, np.ndarray] = {}
    for feature_file, comp_file in zip(feature_files, composition_files):
        arr = np.load(feature_file)
        df = pd.read_csv(comp_file)
        if len(arr) != len(df):
            raise ValueError(f"Feature/CSV row mismatch for {feature_file} and {comp_file}.")
        for i, comp in enumerate(df["Composition"].astype(str)):
            symbols = metals_in_original_order(comp) + ["C"]
            if len(symbols) != arr.shape[1]:
                raise ValueError(
                    f"Node count mismatch for {comp}: parsed {len(symbols)}, feature rows {arr.shape[1]}"
                )
            for symbol, vector in zip(symbols, arr[i]):
                vector = np.asarray(vector, dtype=np.float32)
                if symbol in lookup:
                    if not np.allclose(lookup[symbol], vector, equal_nan=True, atol=1e-6, rtol=1e-6):
                        if strict:
                            raise ValueError(f"Inconsistent recovered feature vector for element {symbol}.")
                else:
                    lookup[symbol] = vector
    return lookup


def create_edge_index(n_nodes: int):
    _, _, _, _, _, dense_to_sparse = _require_pyg()
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


def graph_from_feature_array(x_array: np.ndarray, y: float | None = None):
    Data, _, _, _, _, _ = _require_pyg()
    x = torch.tensor(x_array, dtype=torch.float32)
    edge_index = create_edge_index(x.size(0))
    edge_attr = create_edge_attr(x, edge_index)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor(float(y), dtype=torch.float32)
    return data


def graph_from_composition(comp: str, element_features: dict[str, np.ndarray]):
    symbols = metals_in_original_order(comp) + ["C"]
    missing = [s for s in symbols if s not in element_features]
    if missing:
        raise KeyError(f"No recovered node feature vector for: {missing}")
    arr = np.stack([element_features[s] for s in symbols], axis=0)
    return graph_from_feature_array(arr)


class GCNEncoder(nn.Module):
    """Manuscript-consistent GATv2 EFA regressor.

    This architecture matches ``best_gnn_model_4_2_32_88.0.pth``. The older
    8-layer repository checkpoint is retained only for legacy comparison.
    """

    def __init__(self, in_channels=35, hidden_dim=4, num_layers=2, heads=32, dropout=0.2):
        super().__init__()
        _, _, GATv2Conv, GlobalAttention, GraphNorm, _ = _require_pyg()
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

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_features = x[:, :-1].float()
        if not hasattr(data, "edge_attr") or data.edge_attr is None:
            raise ValueError("GNN data objects must include precomputed edge_attr.")
        edge_attr = data.edge_attr.float()

        out = node_features
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = out
            if i == 0:
                out = conv(out, edge_index, edge_attr)
            else:
                out = conv(out, edge_index)
            out = norm(out, batch)
            out = self.activation(out)
            out = self.dropout(out)
            if out.size() == residual.size():
                out = out + residual
        graph_emb = self.att_pool(out, batch=batch)
        return self.readout_mlp(graph_emb)


def load_checkpoint_model(
    checkpoint: str | Path = "checkpoints/best_gnn_model_4_2_32_88.0.pth",
    device: str | torch.device | None = None,
) -> tuple[GCNEncoder, torch.device]:
    _require_pyg()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = GCNEncoder().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def make_dataset(features: np.ndarray, targets: Sequence[float] | None = None):
    if targets is None:
        return [graph_from_feature_array(features[i]) for i in range(len(features))]
    return [graph_from_feature_array(features[i], float(targets[i])) for i in range(len(features))]


def predict_graphs(model: nn.Module, graphs: Sequence, device: torch.device, batch_size: int = 16) -> np.ndarray:
    _, DataLoader, _, _, _, _ = _require_pyg()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds.append(model(batch).view(-1).detach().cpu().numpy())
    return np.concatenate(preds)


def train_regressor(
    train_graphs: Sequence,
    val_graphs: Sequence | None,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    device: str | torch.device | None = None,
) -> tuple[GCNEncoder, dict]:
    _require_pyg()
    _, DataLoader, _, _, _, _ = _require_pyg()
    set_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = GCNEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)
    loss_fn = nn.SmoothL1Loss()
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False) if val_graphs else None
    best_state = None
    best_val = float("inf")
    stale = 0
    history = []
    for epoch in range(1, int(epochs) + 1):
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
        val_loss = np.nan
        if val_loader is not None:
            model.eval()
            total = 0.0
            n_graphs = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred = model(batch).view(-1)
                    target = batch.y.view(-1)
                    loss = loss_fn(pred, target)
                    total += float(loss.item()) * batch.num_graphs
                    n_graphs += batch.num_graphs
            val_loss = total / max(1, n_graphs)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if patience and stale >= patience:
                break
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"epochs_run": len(history), "history": history, "best_val_loss": best_val}


def split_train_validation(indices: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(indices, dtype=int).copy()
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    if n_val < 1 or len(shuffled) - n_val < 2:
        return shuffled, np.array([], dtype=int)
    return shuffled[n_val:], shuffled[:n_val]


def load_hec5_feature_targets() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv("data/HEC5_output.csv")
    df["Composition"] = df["Composition"].astype(str).map(canonicalize)
    features = np.load("data/final_features_train.npy")
    targets = df["EFA"].to_numpy(dtype=float)
    if len(df) != len(features) or len(df) != len(targets):
        raise ValueError("HEC5 row/feature/target mismatch.")
    return df, features, targets


def evaluate_split(model, graphs: Sequence, y_true: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, dict]:
    y_pred = predict_graphs(model, graphs, device=device, batch_size=batch_size)
    return y_pred, regression_metrics(y_true, y_pred)
