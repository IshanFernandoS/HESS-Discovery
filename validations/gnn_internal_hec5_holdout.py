from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split

from validations.common import DEFAULT_OUTPUT_DIR, bootstrap_regression_ci, ensure_dir, regression_metrics
from validations.gnn_common import load_checkpoint_model, make_dataset, predict_graphs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "HEC5 hold-out model-selection subset evaluation for the "
            "manuscript-consistent GNN checkpoint."
        )
    )
    parser.add_argument("--hec5-csv", default="data/HEC5_output.csv")
    parser.add_argument("--hec5-features", default="data/final_features_train.npy")
    parser.add_argument("--checkpoint", default="checkpoints/best_gnn_model_4_2_32_88.0.pth")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR / "gnn_internal_hec5_holdout"))
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    df = pd.read_csv(args.hec5_csv)
    features = np.load(args.hec5_features)
    if "EFA" not in df.columns:
        raise ValueError("HEC5 CSV must contain an 'EFA' target column.")
    if len(df) != len(features):
        raise ValueError("HEC5 CSV and feature array row counts do not match.")

    y = df["EFA"].to_numpy(dtype=float)
    indices = np.arange(len(df))
    _, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.split_seed,
    )

    graphs = make_dataset(features[test_idx], y[test_idx])
    model, device = load_checkpoint_model(args.checkpoint)
    pred = predict_graphs(model, graphs, device=device, batch_size=args.batch_size)

    metrics = regression_metrics(y[test_idx], pred)
    pearson = pearsonr(y[test_idx], pred)
    spearman = spearmanr(y[test_idx], pred)
    metrics.update(
        {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_r": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "test_size_fraction": float(args.test_size),
            "split_seed": int(args.split_seed),
            "evaluation_role": "hold-out model-selection subset",
        }
    )

    out = pd.DataFrame(
        {
            "Composition": df.loc[test_idx, "Composition"].to_numpy(),
            "Actual_EFA": y[test_idx],
            "Predicted_EFA": pred,
            "Residual": pred - y[test_idx],
        }
    )
    ci = bootstrap_regression_ci(
        y[test_idx],
        pred,
        n_bootstrap=args.n_bootstrap,
        seed=args.bootstrap_seed,
    )

    out.to_csv(outdir / "hec5_internal_holdout_predictions.csv", index=False)
    ci.to_csv(outdir / "hec5_internal_holdout_bootstrap_metrics.csv", index=False)
    with (outdir / "hec5_internal_holdout_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Saved outputs to {outdir.resolve()}")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(ci.to_string(index=False))


if __name__ == "__main__":
    main()
