from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from validations.common import DEFAULT_OUTPUT_DIR, ensure_dir, parse_metals
from validations.gnn_common import (
    evaluate_split,
    load_hec5_feature_targets,
    make_dataset,
    split_train_validation,
    train_regressor,
)


def parse_seeds(values: list[str] | None) -> list[int]:
    if not values:
        return [42]
    out = []
    for value in values:
        for part in str(value).split(","):
            if part.strip():
                out.append(int(part))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-element-out validation on HEC5 EFA data."
    )
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR / "gnn_leave_element_out"))
    parser.add_argument("--elements", nargs="*", default=None)
    parser.add_argument("--min-test", type=int, default=5)
    parser.add_argument("--min-train", type=int, default=50)
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=150)
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    seeds = parse_seeds(args.seeds)
    df, features, targets = load_hec5_feature_targets()
    metal_sets = df["Composition"].map(lambda c: set(parse_metals(c))).tolist()
    all_elements = sorted({element for metals in metal_sets for element in metals})
    elements = sorted(args.elements) if args.elements else all_elements

    metric_rows = []
    prediction_rows = []
    skipped_rows = []

    for element in elements:
        test_idx = np.array([i for i, metals in enumerate(metal_sets) if element in metals], dtype=int)
        train_pool = np.array([i for i, metals in enumerate(metal_sets) if element not in metals], dtype=int)
        if len(test_idx) < args.min_test or len(train_pool) < args.min_train:
            skipped_rows.append(
                {
                    "element": element,
                    "n_train": len(train_pool),
                    "n_test": len(test_idx),
                    "reason": "insufficient split size",
                }
            )
            continue
        test_graphs = make_dataset(features[test_idx], targets[test_idx])
        for seed in seeds:
            train_idx, val_idx = split_train_validation(train_pool, args.val_fraction, seed)
            train_graphs = make_dataset(features[train_idx], targets[train_idx])
            val_graphs = make_dataset(features[val_idx], targets[val_idx]) if len(val_idx) else None
            model, info = train_regressor(
                train_graphs,
                val_graphs,
                seed=seed,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                patience=args.patience,
            )
            y_pred, metrics = evaluate_split(
                model,
                test_graphs,
                targets[test_idx],
                device=next(model.parameters()).device,
                batch_size=args.batch_size,
            )
            metric_rows.append(
                {
                    "element": element,
                    "seed": seed,
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "n_test": len(test_idx),
                    "epochs_run": info["epochs_run"],
                    **metrics,
                }
            )
            for local_i, idx in enumerate(test_idx):
                prediction_rows.append(
                    {
                        "element": element,
                        "seed": seed,
                        "Composition": df.loc[idx, "Composition"],
                        "Actual_EFA": float(targets[idx]),
                        "Predicted_EFA": float(y_pred[local_i]),
                    }
                )

    pd.DataFrame(metric_rows).to_csv(outdir / "leave_element_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(outdir / "leave_element_predictions.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(outdir / "leave_element_skipped.csv", index=False)
    if metric_rows:
        summary = (
            pd.DataFrame(metric_rows)
            .groupby("element")
            .agg(
                n_runs=("seed", "count"),
                n_test=("n_test", "first"),
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                rmse_mean=("rmse", "mean"),
                r2_mean=("r2", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(outdir / "leave_element_summary.csv", index=False)
        print(summary.to_string(index=False))
    print(f"Saved outputs to {outdir.resolve()}")


if __name__ == "__main__":
    main()
