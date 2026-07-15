from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from validations.common import (
    DEFAULT_DESCRIPTOR_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SELECTED_HEC9,
    canonicalize,
    ensure_dir,
    load_descriptor_table,
)
from validations.gnn_common import (
    _require_pyg,
    build_element_feature_lookup,
    graph_from_composition,
    load_checkpoint_model,
    predict_graphs,
    set_seed,
)


def parse_ints(values: list[str] | None, default: list[int]) -> list[int]:
    if not values:
        return default
    out = []
    for value in values:
        for part in str(value).split(","):
            if part.strip():
                out.append(int(part))
    return out


def mc_predict(model, graphs, device, n_samples: int, batch_size: int) -> np.ndarray:
    _, DataLoader, _, _, _, _ = _require_pyg()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    samples = []
    for _ in range(int(n_samples)):
        model.train()
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                preds.append(model(batch).view(-1).detach().cpu().numpy())
        samples.append(np.concatenate(preds))
    model.eval()
    return np.stack(samples, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MC-dropout uncertainty for HEC9-HEC11 EFA predictions."
    )
    parser.add_argument("--descriptor-csv", default=str(DEFAULT_DESCRIPTOR_PATH))
    parser.add_argument("--checkpoint", default="checkpoints/best_gnn_model_4_2_32_88.0.pth")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR / "gnn_mc_dropout_uncertainty"))
    parser.add_argument("--cardinalities", nargs="*", default=None, help="Default: 9 10 11")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selected-only", action="store_true")
    parser.add_argument("--selected", nargs="*", default=DEFAULT_SELECTED_HEC9)
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)
    cards = parse_ints(args.cardinalities, [9, 10, 11])
    selected = [canonicalize(x) for x in args.selected]

    df = load_descriptor_table(args.descriptor_csv)
    if args.selected_only:
        candidates = df[df["Composition"].isin(selected)].copy()
    else:
        candidates = df[df["n_metals"].isin(cards)].copy()
    candidates = candidates.sort_values(["n_metals", "Composition"]).reset_index(drop=True)
    if candidates.empty:
        raise ValueError("No candidate compositions selected for MC-dropout prediction.")

    element_features = build_element_feature_lookup()
    graphs = [graph_from_composition(comp, element_features) for comp in candidates["Composition"]]
    model, device = load_checkpoint_model(args.checkpoint)
    eval_pred = predict_graphs(model, graphs, device=device, batch_size=args.batch_size)
    samples = mc_predict(model, graphs, device=device, n_samples=args.n_samples, batch_size=args.batch_size)

    out = candidates[["Composition", "n_metals"]].copy()
    out["Predicted_EFA_eval_mode"] = eval_pred
    out["MC_mean"] = samples.mean(axis=0)
    out["MC_std"] = samples.std(axis=0, ddof=1) if samples.shape[0] > 1 else 0.0
    out["MC_q05"] = np.quantile(samples, 0.05, axis=0)
    out["MC_q50"] = np.quantile(samples, 0.50, axis=0)
    out["MC_q95"] = np.quantile(samples, 0.95, axis=0)
    out["MC_n_samples"] = int(args.n_samples)
    out.to_csv(outdir / "mc_dropout_predictions.csv", index=False)
    out[out["Composition"].isin(selected)].to_csv(outdir / "selected_mc_dropout_predictions.csv", index=False)

    print(f"Saved outputs to {outdir.resolve()}")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
