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
    canonicalize,
    compute_candidate_ranking,
    descriptor_feature_columns,
    ensure_dir,
    fit_gmm_posterior,
    load_descriptor_table,
)


def parse_int_list(values: list[str] | None, default: list[int]) -> list[int]:
    if not values:
        return default
    out: list[int] = []
    for value in values:
        for part in str(value).split(","):
            if part.strip():
                out.append(int(part))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeat GMM across random seeds and component numbers and track candidate ranks."
    )
    parser.add_argument("--descriptor-csv", default=str(DEFAULT_DESCRIPTOR_PATH))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR / "gmm_seed_component_robustness"))
    parser.add_argument("--seeds", nargs="*", default=None, help="Seeds, e.g. --seeds 0 1 2 or 0,1,2")
    parser.add_argument("--k-values", nargs="*", default=None, help="Component counts, e.g. 2 3 4 or 2,3,4")
    parser.add_argument("--covariance-types", nargs="*", default=["full", "tied", "diag", "spherical"])
    parser.add_argument("--target-n-metals", type=int, default=9)
    parser.add_argument("--selected", nargs="*", default=DEFAULT_SELECTED_HEC9)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-calibration", action="store_true")
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    seeds = parse_int_list(args.seeds, list(range(10)))
    k_values = parse_int_list(args.k_values, list(range(2, 30)))
    selected = [canonicalize(x) for x in args.selected]

    df = load_descriptor_table(args.descriptor_csv)
    feature_cols = descriptor_feature_columns(df)

    run_rows = []
    selected_rows = []
    all_rank_rows = []

    for seed in seeds:
        for k in k_values:
            for cov in args.covariance_types:
                run_id = f"seed{seed}_k{k}_{cov}"
                try:
                    result = fit_gmm_posterior(
                        df,
                        feature_cols,
                        seed=seed,
                        k_values=[k],
                        covariance_types=[cov],
                        calibrate=not args.no_calibration,
                    )
                    ranked = compute_candidate_ranking(
                        result.df,
                        feature_cols,
                        target_n_metals=args.target_n_metals,
                    )
                except Exception as exc:
                    run_rows.append(
                        {
                            "run_id": run_id,
                            "seed": seed,
                            "k": k,
                            "covariance_type": cov,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
                    continue

                n_ranked = len(ranked)
                run_rows.append(
                    {
                        "run_id": run_id,
                        "seed": seed,
                        "k": k,
                        "covariance_type": cov,
                        "status": "ok",
                        "bic": result.best["bic"],
                        "n_ranked": n_ranked,
                    }
                )

                if not ranked.empty:
                    rr = ranked[
                        ["Composition", "Rank", "Score", "P_single", "cum_pmean", "hits_count"]
                    ].copy()
                    rr.insert(0, "run_id", run_id)
                    rr.insert(1, "seed", seed)
                    rr.insert(2, "k", k)
                    rr.insert(3, "covariance_type", cov)
                    all_rank_rows.append(rr)

                rank_map = dict(zip(ranked["Composition"], ranked["Rank"])) if not ranked.empty else {}
                p_map = dict(zip(ranked["Composition"], ranked["P_single"])) if not ranked.empty else {}
                score_map = dict(zip(ranked["Composition"], ranked["Score"])) if not ranked.empty else {}
                for comp in selected:
                    rank = rank_map.get(comp, np.nan)
                    selected_rows.append(
                        {
                            "run_id": run_id,
                            "seed": seed,
                            "k": k,
                            "covariance_type": cov,
                            "Composition": comp,
                            "Rank": rank,
                            "rank_percentile": float(rank / n_ranked) if n_ranked and np.isfinite(rank) else np.nan,
                            "P_single": p_map.get(comp, np.nan),
                            "Score": score_map.get(comp, np.nan),
                            f"in_top_{args.top_k}": bool(np.isfinite(rank) and rank <= args.top_k),
                        }
                    )

    run_df = pd.DataFrame(run_rows)
    sel_df = pd.DataFrame(selected_rows)
    run_df.to_csv(outdir / "gmm_runs.csv", index=False)
    sel_df.to_csv(outdir / "selected_candidate_run_details.csv", index=False)
    if all_rank_rows:
        pd.concat(all_rank_rows, ignore_index=True).to_csv(outdir / "all_rankings_by_run.csv", index=False)

    if not sel_df.empty:
        ok = sel_df[np.isfinite(sel_df["Rank"])].copy()
        summary = (
            ok.groupby("Composition")
            .agg(
                n_successful_runs=("Rank", "count"),
                rank_mean=("Rank", "mean"),
                rank_std=("Rank", "std"),
                rank_min=("Rank", "min"),
                rank_max=("Rank", "max"),
                rank_percentile_mean=("rank_percentile", "mean"),
                p_single_mean=("P_single", "mean"),
                p_single_std=("P_single", "std"),
                score_mean=("Score", "mean"),
                score_std=("Score", "std"),
                top_k_fraction=(f"in_top_{args.top_k}", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(outdir / "selected_candidate_summary.csv", index=False)
        print(summary.to_string(index=False))

    print(f"Saved outputs to {outdir.resolve()}")


if __name__ == "__main__":
    main()
