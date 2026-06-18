from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from validations.common import (
    DEFAULT_DESCRIPTOR_PATH,
    DEFAULT_OUTPUT_DIR,
    descriptor_feature_columns,
    ensure_dir,
    fit_gmm_posterior,
    load_descriptor_table,
    map_components_to_scores,
    metric_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Permutation/null test for literature phase labels.")
    parser.add_argument("--descriptor-csv", default=str(DEFAULT_DESCRIPTOR_PATH))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR / "label_permutation_null"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--calibrate", action="store_true", help="Apply isotonic calibration inside each label map.")
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    rng = np.random.default_rng(args.seed)

    df = load_descriptor_table(args.descriptor_csv)
    feature_cols = descriptor_feature_columns(df)
    result = fit_gmm_posterior(df, feature_cols, seed=args.seed, calibrate=False)

    is_labeled = result.df["Exp_Phase"].isin([0, 1]).to_numpy()
    y_all = result.df["Exp_Phase"].to_numpy()
    y_true = y_all[is_labeled].astype(int)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Need both classes for permutation test.")

    real_score, _ = map_components_to_scores(result.responsibilities, y_all, is_labeled, calibrate=args.calibrate)
    real_metrics = metric_summary(y_true, real_score[is_labeled], threshold=args.threshold)

    rows = []
    for i in range(int(args.n_permutations)):
        y_perm_all = y_all.copy()
        y_perm = rng.permutation(y_true)
        y_perm_all[is_labeled] = y_perm
        score, _ = map_components_to_scores(
            result.responsibilities,
            y_perm_all,
            is_labeled,
            calibrate=args.calibrate,
        )
        m = metric_summary(y_perm, score[is_labeled], threshold=args.threshold)
        rows.append({"permutation": i, **m})
    null_df = pd.DataFrame(rows)
    null_df.to_csv(outdir / "null_metrics.csv", index=False)

    summary_rows = []
    for metric in ["accuracy", "balanced_accuracy", "precision_single", "recall_single", "f1_single"]:
        values = null_df[metric].to_numpy(dtype=float)
        real = real_metrics[metric]
        p_value = (1.0 + np.nansum(values >= real)) / (len(values) + 1.0)
        summary_rows.append(
            {
                "metric": metric,
                "real": real,
                "null_mean": float(np.nanmean(values)),
                "null_ci_low": float(np.nanquantile(values, 0.025)),
                "null_ci_high": float(np.nanquantile(values, 0.975)),
                "empirical_p_value_greater_equal": float(p_value),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "permutation_summary.csv", index=False)

    print(f"Saved outputs to {outdir.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
