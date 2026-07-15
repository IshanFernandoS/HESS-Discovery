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


DEFAULT_FAMILIES = {
    "IVB": ["Ti", "Zr", "Hf"],
    "VB": ["V", "Nb", "Ta"],
    "VIB": ["Cr", "Mo", "W"],
    "late_transition": ["Mn", "Ir"],
    "p_block": ["Al", "Si"],
    "rare_earth": ["Sc", "Y"],
}


def parse_seeds(values: list[str] | None) -> list[int]:
    if not values:
        return [42]
    out = []
    for value in values:
        for part in str(value).split(","):
            if part.strip():
                out.append(int(part))
    return out


def family_holdout_mask(metal_sets: list[set[str]], family: set[str], mode: str, at_least: int) -> np.ndarray:
    counts = np.array([len(metals & family) for metals in metal_sets], dtype=int)
    if mode == "any":
        return counts >= 1
    if mode == "all":
        return counts == len(family)
    if mode == "atleast":
        return counts >= int(at_least)
    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-family-out validation on HEC5 EFA data.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR / "gnn_leave_family_out"))
    parser.add_argument("--families", nargs="*", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--mode", choices=["any", "all", "atleast"], default="any")
    parser.add_argument("--at-least", type=int, default=2)
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

    metric_rows = []
    prediction_rows = []
    skipped_rows = []

    for family_name in args.families:
        if family_name not in DEFAULT_FAMILIES:
            raise KeyError(f"Unknown family '{family_name}'. Options: {sorted(DEFAULT_FAMILIES)}")
        family = set(DEFAULT_FAMILIES[family_name])
        holdout = family_holdout_mask(metal_sets, family, args.mode, args.at_least)
        test_idx = np.flatnonzero(holdout)
        train_pool = np.flatnonzero(~holdout)
        if len(test_idx) < args.min_test or len(train_pool) < args.min_train:
            skipped_rows.append(
                {
                    "family": family_name,
                    "elements": ",".join(DEFAULT_FAMILIES[family_name]),
                    "mode": args.mode,
                    "at_least": args.at_least,
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
                    "family": family_name,
                    "elements": ",".join(DEFAULT_FAMILIES[family_name]),
                    "mode": args.mode,
                    "at_least": args.at_least,
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
                        "family": family_name,
                        "seed": seed,
                        "Composition": df.loc[idx, "Composition"],
                        "Actual_EFA": float(targets[idx]),
                        "Predicted_EFA": float(y_pred[local_i]),
                    }
                )

    pd.DataFrame(metric_rows).to_csv(outdir / "leave_family_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(outdir / "leave_family_predictions.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(outdir / "leave_family_skipped.csv", index=False)
    if metric_rows:
        summary = (
            pd.DataFrame(metric_rows)
            .groupby("family")
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
        summary.to_csv(outdir / "leave_family_summary.csv", index=False)
        print(summary.to_string(index=False))
    print(f"Saved outputs to {outdir.resolve()}")


if __name__ == "__main__":
    main()
