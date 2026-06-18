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
    DEFAULT_SELECTED_HEC9,
    FEATURE_FAMILIES,
    canonicalize,
    compute_candidate_ranking,
    descriptor_feature_columns,
    ensure_dir,
    family_exclusion_columns,
    fit_gmm_posterior,
    load_descriptor_table,
    metric_summary,
)


def labelled_metrics(result, threshold: float) -> dict:
    labelled = result.df[result.df["Exp_Phase"].isin([0, 1])]
    y_true = labelled["Exp_Phase"].astype(int).to_numpy()
    y_score = labelled["P_single"].astype(float).to_numpy()
    return metric_summary(y_true, y_score, threshold=threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove one descriptor family at a time and rerun GMM ranking.")
    parser.add_argument("--descriptor-csv", default=str(DEFAULT_DESCRIPTOR_PATH))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR / "feature_ablation"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target-n-metals", type=int, default=9)
    parser.add_argument("--selected", nargs="*", default=DEFAULT_SELECTED_HEC9)
    parser.add_argument("--no-calibration", action="store_true")
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    selected = [canonicalize(x) for x in args.selected]
    df = load_descriptor_table(args.descriptor_csv)
    base_cols = descriptor_feature_columns(df)

    variants = {"baseline": base_cols}
    for family in FEATURE_FAMILIES:
        drop_cols = set(family_exclusion_columns(df, family))
        cols = [c for c in base_cols if c not in drop_cols]
        variants[f"drop_{family}"] = cols

    metric_rows = []
    selected_rows = []
    overlap_rows = []
    rankings = {}

    for name, cols in variants.items():
        if len(cols) < 2:
            metric_rows.append({"variant": name, "status": "failed", "error": "fewer than two features"})
            continue
        try:
            result = fit_gmm_posterior(
                df,
                cols,
                seed=args.seed,
                calibrate=not args.no_calibration,
            )
            ranked = compute_candidate_ranking(
                result.df,
                cols,
                target_n_metals=args.target_n_metals,
            )
            rankings[name] = ranked
            metrics = labelled_metrics(result, args.threshold)
            metric_rows.append(
                {
                    "variant": name,
                    "status": "ok",
                    "n_features": len(cols),
                    "dropped_features": ",".join(sorted(set(base_cols) - set(cols))),
                    "gmm_k": result.best["k"],
                    "gmm_covariance_type": result.best["covariance_type"],
                    "gmm_bic": result.best["bic"],
                    **metrics,
                }
            )
            rank_map = dict(zip(ranked["Composition"], ranked["Rank"])) if not ranked.empty else {}
            p_map = dict(zip(ranked["Composition"], ranked["P_single"])) if not ranked.empty else {}
            score_map = dict(zip(ranked["Composition"], ranked["Score"])) if not ranked.empty else {}
            for comp in selected:
                selected_rows.append(
                    {
                        "variant": name,
                        "Composition": comp,
                        "Rank": rank_map.get(comp, np.nan),
                        "P_single": p_map.get(comp, np.nan),
                        "Score": score_map.get(comp, np.nan),
                    }
                )
        except Exception as exc:
            metric_rows.append({"variant": name, "status": "failed", "error": str(exc)})

    baseline = rankings.get("baseline")
    if baseline is not None and not baseline.empty:
        for name, ranked in rankings.items():
            if name == "baseline" or ranked.empty:
                continue
            for k in [10, 20, 50]:
                base_top = set(baseline["Composition"].head(k))
                test_top = set(ranked["Composition"].head(k))
                overlap_rows.append(
                    {
                        "variant": name,
                        "top_k": k,
                        "overlap_count": len(base_top & test_top),
                        "overlap_fraction": len(base_top & test_top) / float(k),
                    }
                )

    pd.DataFrame(metric_rows).to_csv(outdir / "ablation_metrics.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(outdir / "selected_candidate_ablation_ranks.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(outdir / "topk_overlap_with_baseline.csv", index=False)
    for name, ranked in rankings.items():
        ranked.to_csv(outdir / f"ranking_{name}.csv", index=False)

    print(f"Saved outputs to {outdir.resolve()}")
    print(pd.DataFrame(metric_rows).to_string(index=False))


if __name__ == "__main__":
    main()
