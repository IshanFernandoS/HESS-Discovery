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
    bootstrap_metric_ci,
    descriptor_feature_columns,
    ensure_dir,
    fit_gmm_posterior,
    load_descriptor_table,
    metric_summary,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confusion matrix and bootstrap CIs for literature single/multiphase labels."
    )
    parser.add_argument("--descriptor-csv", default=str(DEFAULT_DESCRIPTOR_PATH))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR / "gmm_literature_bootstrap"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--stratified-bootstrap", action="store_true")
    parser.add_argument("--no-calibration", action="store_true", help="Use raw component posterior.")
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    df = load_descriptor_table(args.descriptor_csv)
    feature_cols = descriptor_feature_columns(df)
    result = fit_gmm_posterior(
        df,
        feature_cols,
        seed=args.seed,
        calibrate=not args.no_calibration,
    )

    labelled = result.df[result.df["Exp_Phase"].isin([0, 1])].copy()
    if labelled.empty or labelled["Exp_Phase"].nunique() < 2:
        raise ValueError("Need both single and multiphase literature labels.")

    y_true = labelled["Exp_Phase"].astype(int).to_numpy()
    y_score = labelled["P_single"].astype(float).to_numpy()
    metrics = metric_summary(y_true, y_score, threshold=args.threshold)
    ci = bootstrap_metric_ci(
        y_true,
        y_score,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        threshold=args.threshold,
        stratified=args.stratified_bootstrap,
    )

    labelled["Pred_Phase"] = np.where(y_score >= args.threshold, 1, 0)
    labelled["Pred_Label"] = np.where(labelled["Pred_Phase"] == 1, "Single", "Multi")
    labelled["True_Label"] = np.where(labelled["Exp_Phase"].astype(int) == 1, "Single", "Multi")
    labelled[
        [
            "Composition",
            "True_Label",
            "Pred_Label",
            "Exp_Phase",
            "Pred_Phase",
            "P_single",
            "gmm_comp",
            "component_phase_label",
        ]
    ].to_csv(outdir / "labelled_predictions.csv", index=False)

    cm = pd.DataFrame(
        [
            [metrics["true_multi_pred_multi"], metrics["true_multi_pred_single"]],
            [metrics["true_single_pred_multi"], metrics["true_single_pred_single"]],
        ],
        index=["true_multi", "true_single"],
        columns=["pred_multi", "pred_single"],
    )
    cm.to_csv(outdir / "confusion_matrix.csv")
    ci.to_csv(outdir / "bootstrap_metrics.csv", index=False)
    write_json(
        outdir / "metrics.json",
        {
            **metrics,
            "feature_columns": feature_cols,
            "gmm": result.best,
            "calibrated": not args.no_calibration,
        },
    )

    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(3.2, 3.0), dpi=220)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm.to_numpy(),
            display_labels=["Multi", "Single"],
        )
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title("GMM vs literature labels")
        fig.tight_layout()
        fig.savefig(outdir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"Plot skipped: {exc}")

    print(f"Saved outputs to {outdir.resolve()}")
    print(cm.to_string())
    print(ci.to_string(index=False))


if __name__ == "__main__":
    main()
