from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from validations.common import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_regression_ci,
    ensure_dir,
    regression_metrics,
    write_json,
)
from validations.gnn_common import load_checkpoint_model, make_dataset, predict_graphs

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="External HEC8 EFA benchmark using HEC8_output.csv and final_features_test.npy."
    )
    parser.add_argument("--hec8-csv", default="data/HEC8_output.csv")
    parser.add_argument("--hec8-features", default="data/final_features_test.npy")
    parser.add_argument("--checkpoint", default="checkpoints/best_gnn_model_4_2_32_88.0.pth")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR / "external_hec8_benchmark"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    df = pd.read_csv(args.hec8_csv)
    features = np.load(args.hec8_features)
    if "EFA" not in df.columns:
        raise ValueError("HEC8 benchmark CSV must contain an 'EFA' target column.")
    if len(df) != len(features):
        raise ValueError("HEC8 CSV and feature array row counts do not match.")

    graphs = make_dataset(features, df["EFA"].to_numpy(dtype=float))
    model, device = load_checkpoint_model(args.checkpoint)
    pred = predict_graphs(model, graphs, device=device, batch_size=args.batch_size)
    y_true = df["EFA"].to_numpy(dtype=float)
    metrics = regression_metrics(y_true, pred)
    ci = bootstrap_regression_ci(y_true, pred, n_bootstrap=args.n_bootstrap, seed=args.seed)

    out = pd.DataFrame(
        {
            "Composition": df["Composition"],
            "Actual_EFA": y_true,
            "Predicted_EFA": pred,
            "Residual": pred - y_true,
        }
    )
    out.to_csv(outdir / "hec8_predictions.csv", index=False)
    ci.to_csv(outdir / "hec8_bootstrap_metrics.csv", index=False)
    write_json(outdir / "hec8_metrics.json", metrics)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=220)
        ax.scatter(y_true, pred, s=18, alpha=0.75)
        lo = min(float(y_true.min()), float(pred.min()))
        hi = max(float(y_true.max()), float(pred.max()))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
        ax.set_xlabel("Actual HEC8 EFA")
        ax.set_ylabel("Predicted HEC8 EFA")
        ax.set_title("External HEC8 EFA benchmark")
        fig.tight_layout()
        fig.savefig(outdir / "hec8_predicted_vs_actual.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"Plot skipped: {exc}")

    print(f"Saved outputs to {outdir.resolve()}")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(ci.to_string(index=False))


if __name__ == "__main__":
    main()
