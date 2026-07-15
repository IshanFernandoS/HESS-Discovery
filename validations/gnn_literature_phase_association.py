from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from validations.common import (
    DEFAULT_OUTPUT_DIR,
    MULTI_PHASE_LIT,
    SINGLE_PHASE_LIT,
    canonicalize,
    count_metals,
    ensure_dir,
)
from validations.gnn_common import (
    build_element_feature_lookup,
    graph_from_composition,
    load_checkpoint_model,
    predict_graphs,
)


def best_threshold_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    if len(np.unique(y_true)) < 2:
        return {"n": int(len(y_true)), "status": "one_class_only"}
    values = np.sort(np.unique(scores))
    thresholds = np.r_[[values[0] - 1e-9], (values[:-1] + values[1:]) / 2.0, [values[-1] + 1e-9]]
    best = None
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        candidate = {
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision_single": float(precision),
            "recall_single": float(recall),
            "f1_single": float(f1),
            "pred_single": int(y_pred.sum()),
        }
        key = (candidate["balanced_accuracy"], candidate["f1_single"], candidate["accuracy"])
        if best is None or key > (
            best["balanced_accuracy"],
            best["f1_single"],
            best["accuracy"],
        ):
            best = candidate
    assert best is not None
    y_pred = (scores >= best["threshold"]).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    best.update(
        {
            "n": int(len(y_true)),
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "average_precision": float(average_precision_score(y_true, scores)),
            "true_multi_pred_multi": int(cm[0, 0]),
            "true_multi_pred_single": int(cm[0, 1]),
            "true_single_pred_multi": int(cm[1, 0]),
            "true_single_pred_single": int(cm[1, 1]),
        }
    )
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Association between saved GNN EFA predictions and literature phase labels."
    )
    parser.add_argument("--hec5-csv", default="data/HEC5_output.csv")
    parser.add_argument("--checkpoint", default="checkpoints/best_gnn_model_4_2_32_88.0.pth")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR / "gnn_literature_phase_association"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.1)
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    rows = [{"Composition": canonicalize(c), "Exp_Phase": 1, "Label": "Single"} for c in SINGLE_PHASE_LIT]
    rows.extend({"Composition": canonicalize(c), "Exp_Phase": 0, "Label": "Multi"} for c in MULTI_PHASE_LIT)
    df = pd.DataFrame(rows).drop_duplicates("Composition").reset_index(drop=True)
    df["n_metals"] = df["Composition"].map(count_metals).astype(int)

    hec5 = pd.read_csv(args.hec5_csv)
    hec5["Composition"] = hec5["Composition"].astype(str).map(canonicalize)
    indices = np.arange(len(hec5))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.split_seed,
    )
    train_comps = set(hec5.loc[train_idx, "Composition"])
    test_comps = set(hec5.loc[test_idx, "Composition"])

    df["overlap_group"] = "Non-HEC5 order"
    df.loc[df["Composition"].isin(train_comps), "overlap_group"] = "HEC5 train-split overlap"
    df.loc[df["Composition"].isin(test_comps), "overlap_group"] = "HEC5 internal-test overlap"
    df["in_hec5_train_split"] = df["overlap_group"].eq("HEC5 train-split overlap")

    element_features = build_element_feature_lookup()
    graphs = [graph_from_composition(comp, element_features) for comp in df["Composition"]]
    model, device = load_checkpoint_model(args.checkpoint)
    pred = predict_graphs(model, graphs, device=device, batch_size=args.batch_size)
    df["GNN_predicted_EFA"] = pred

    y = df["Exp_Phase"].to_numpy(dtype=int)
    scores = df["GNN_predicted_EFA"].to_numpy(dtype=float)
    all_metrics = best_threshold_metrics(y, scores)
    all_metrics.update(
        {
            "subset": "all_literature_labels",
            "n_single": int((y == 1).sum()),
            "n_multi": int((y == 0).sum()),
            "single_mean": float(df.loc[df["Exp_Phase"] == 1, "GNN_predicted_EFA"].mean()),
            "single_median": float(df.loc[df["Exp_Phase"] == 1, "GNN_predicted_EFA"].median()),
            "multi_mean": float(df.loc[df["Exp_Phase"] == 0, "GNN_predicted_EFA"].mean()),
            "multi_median": float(df.loc[df["Exp_Phase"] == 0, "GNN_predicted_EFA"].median()),
        }
    )
    try:
        test = mannwhitneyu(
            df.loc[df["Exp_Phase"] == 1, "GNN_predicted_EFA"],
            df.loc[df["Exp_Phase"] == 0, "GNN_predicted_EFA"],
            alternative="greater",
        )
        all_metrics["mannwhitney_u_p_single_greater"] = float(test.pvalue)
    except Exception:
        all_metrics["mannwhitney_u_p_single_greater"] = np.nan

    summary_rows = [all_metrics]
    non_train = df[~df["in_hec5_train_split"]].copy()
    non_train_y = non_train["Exp_Phase"].to_numpy(dtype=int)
    non_train_scores = non_train["GNN_predicted_EFA"].to_numpy(dtype=float)
    non_train_metrics = best_threshold_metrics(non_train_y, non_train_scores)
    non_train_metrics.update(
        {
            "subset": "excluding_HEC5_train_split_overlap",
            "n_single": int((non_train_y == 1).sum()),
            "n_multi": int((non_train_y == 0).sum()),
            "single_mean": float(non_train.loc[non_train["Exp_Phase"] == 1, "GNN_predicted_EFA"].mean()),
            "single_median": float(non_train.loc[non_train["Exp_Phase"] == 1, "GNN_predicted_EFA"].median()),
            "multi_mean": float(non_train.loc[non_train["Exp_Phase"] == 0, "GNN_predicted_EFA"].mean()),
            "multi_median": float(non_train.loc[non_train["Exp_Phase"] == 0, "GNN_predicted_EFA"].median()),
        }
    )
    summary_rows.append(non_train_metrics)

    df["Pred_single_at_best_threshold"] = (scores >= all_metrics["threshold"]).astype(int)
    df.sort_values(["Exp_Phase", "GNN_predicted_EFA"], ascending=[True, False]).to_csv(
        outdir / "literature_gnn_predictions.csv",
        index=False,
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "summary.csv", index=False)
    with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)

    print(f"Saved outputs to {outdir.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
