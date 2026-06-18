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
    DEFAULT_SELECTED_CANDIDATES,
    canonicalize,
    descriptor_feature_columns,
    ensure_dir,
    load_descriptor_table,
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


def pairwise_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    dist2 = np.maximum(aa + bb - 2.0 * a @ b.T, 0.0)
    return np.sqrt(dist2)


def percentile_against_reference(value: float, reference_values: np.ndarray) -> float:
    return float(100.0 * np.mean(reference_values <= value))


def compute_applicability_rows(
    df_ref: pd.DataFrame,
    x_ref: np.ndarray,
    df_query: pd.DataFrame,
    x_query: np.ndarray,
    percentile_cutoff: float,
) -> pd.DataFrame:
    dist_qr = pairwise_euclidean(x_query, x_ref)
    nearest_idx = dist_qr.argmin(axis=1)
    nearest_dist = dist_qr[np.arange(len(x_query)), nearest_idx]

    dist_rr = pairwise_euclidean(x_ref, x_ref)
    np.fill_diagonal(dist_rr, np.inf)
    ref_nn = dist_rr.min(axis=1)

    centroid = x_ref.mean(axis=0)
    cov = np.cov(x_ref, rowvar=False)
    inv_cov = np.linalg.pinv(cov)

    def mahalanobis(x: np.ndarray) -> np.ndarray:
        centered = x - centroid
        return np.sqrt(np.maximum(np.sum((centered @ inv_cov) * centered, axis=1), 0.0))

    ref_mahal = mahalanobis(x_ref)
    query_mahal = mahalanobis(x_query)

    rows = []
    for i, row in df_query.iterrows():
        nn_dist = float(nearest_dist[i])
        md = float(query_mahal[i])
        nn_pct = percentile_against_reference(nn_dist, ref_nn)
        md_pct = percentile_against_reference(md, ref_mahal)
        in_domain = nn_pct <= percentile_cutoff and md_pct <= percentile_cutoff
        nearest = df_ref.iloc[int(nearest_idx[i])]
        rows.append(
            {
                "Composition": row["Composition"],
                "n_metals": int(row["n_metals"]),
                "nearest_reference": nearest["Composition"],
                "nearest_reference_n_metals": int(nearest["n_metals"]),
                "nearest_reference_label": nearest.get("Exp_Phase", np.nan),
                "nearest_neighbor_distance": nn_dist,
                "nearest_neighbor_percentile_vs_reference": nn_pct,
                "mahalanobis_distance": md,
                "mahalanobis_percentile_vs_reference": md_pct,
                "in_domain_at_cutoff": bool(in_domain),
                "domain_call": "interpolation_like" if in_domain else "extrapolation_like",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Applicability-domain distances from a HEC5/HEC6 descriptor manifold."
    )
    parser.add_argument("--descriptor-csv", default=str(DEFAULT_DESCRIPTOR_PATH))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR / "applicability_domain"))
    parser.add_argument("--reference-cardinalities", nargs="*", default=None, help="Default: 5 6")
    parser.add_argument("--query-cardinalities", nargs="*", default=None, help="Default: 9 10 11")
    parser.add_argument("--selected", nargs="*", default=DEFAULT_SELECTED_CANDIDATES)
    parser.add_argument("--percentile-cutoff", type=float, default=95.0)
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    ref_cards = parse_ints(args.reference_cardinalities, [5, 6])
    query_cards = parse_ints(args.query_cardinalities, [9, 10, 11])
    selected = [canonicalize(x) for x in args.selected]

    df = load_descriptor_table(args.descriptor_csv)
    feature_cols = descriptor_feature_columns(df)
    x_all = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    complete = ~x_all.isna().any(axis=1)
    df = df.loc[complete].copy().reset_index(drop=True)
    x_all = x_all.loc[complete].to_numpy(dtype=float)

    ref_mask = df["n_metals"].isin(ref_cards).to_numpy()
    query_mask = df["n_metals"].isin(query_cards).to_numpy()
    if ref_mask.sum() < 5:
        raise ValueError("Too few reference compositions for applicability-domain analysis.")
    if query_mask.sum() < 1:
        raise ValueError("No query compositions for requested cardinalities.")

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_all[ref_mask])
    x_ref = scaler.transform(x_all[ref_mask])
    x_query = scaler.transform(x_all[query_mask])
    df_ref = df.loc[ref_mask].reset_index(drop=True)
    df_query = df.loc[query_mask].reset_index(drop=True)

    out = compute_applicability_rows(df_ref, x_ref, df_query, x_query, args.percentile_cutoff)
    out = out.drop(columns=["nearest_reference_label"])
    out.to_csv(outdir / "applicability_domain.csv", index=False)

    summary = (
        out.groupby("n_metals")
        .agg(
            n=("Composition", "count"),
            in_domain_fraction=("in_domain_at_cutoff", "mean"),
            nn_distance_median=("nearest_neighbor_distance", "median"),
            nn_percentile_median=("nearest_neighbor_percentile_vs_reference", "median"),
            mahalanobis_percentile_median=("mahalanobis_percentile_vs_reference", "median"),
        )
        .reset_index()
    )
    summary.to_csv(outdir / "summary_by_cardinality.csv", index=False)
    out[out["Composition"].isin(selected)].to_csv(outdir / "selected_applicability.csv", index=False)

    labelled_ref_mask = ref_mask & df["Exp_Phase"].isin([0, 1]).to_numpy()
    if labelled_ref_mask.sum() >= 5:
        labelled_scaler = StandardScaler().fit(x_all[labelled_ref_mask])
        x_labelled_ref = labelled_scaler.transform(x_all[labelled_ref_mask])
        x_labelled_query = labelled_scaler.transform(x_all[query_mask])
        df_labelled_ref = df.loc[labelled_ref_mask].reset_index(drop=True)
        labelled_out = compute_applicability_rows(
            df_labelled_ref,
            x_labelled_ref,
            df_query,
            x_labelled_query,
            args.percentile_cutoff,
        )
        labelled_out["nearest_reference_label"] = labelled_out["nearest_reference_label"].map(
            {0.0: "Multi", 1.0: "Single", 0: "Multi", 1: "Single"}
        )
        labelled_out = labelled_out.rename(
            columns={
                "nearest_reference": "nearest_labelled_reference",
                "nearest_reference_n_metals": "nearest_labelled_reference_n_metals",
                "nearest_neighbor_percentile_vs_reference": "nn_percentile_vs_labelled_ref",
                "mahalanobis_percentile_vs_reference": "mahalanobis_percentile_vs_labelled_ref",
            }
        )
        labelled_out.to_csv(outdir / "applicability_domain_literature_labelled.csv", index=False)
        labelled_out[labelled_out["Composition"].isin(selected)].to_csv(
            outdir / "selected_applicability_literature_labelled_reference.csv",
            index=False,
        )

    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42).fit(x_ref)
        ref_2 = pca.transform(x_ref)
        query_2 = pca.transform(x_query)
        fig, ax = plt.subplots(figsize=(6.0, 4.4), dpi=220)
        ax.scatter(ref_2[:, 0], ref_2[:, 1], s=12, c="#4c78a8", alpha=0.35, label="Reference HEC5/HEC6")
        colors = np.where(out["in_domain_at_cutoff"].to_numpy(), "#2ca02c", "#d62728")
        ax.scatter(query_2[:, 0], query_2[:, 1], s=18, c=colors, alpha=0.85, label="HEC9-HEC11 query")
        selected_mask = out["Composition"].isin(selected).to_numpy()
        if selected_mask.any():
            ax.scatter(
                query_2[selected_mask, 0],
                query_2[selected_mask, 1],
                s=70,
                facecolors="none",
                edgecolors="black",
                linewidths=1.2,
                label="Selected",
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Applicability domain in descriptor PCA space")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "applicability_domain_pca.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"PCA plot skipped: {exc}")

    print(f"Saved outputs to {outdir.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
