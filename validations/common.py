from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_DESCRIPTOR_PATH = Path("data/Compositions Descriptros.csv")
DEFAULT_OUTPUT_DIR = Path("validation_outputs")

DEFAULT_SELECTED_HEC9 = [
    "(HfMoNbReTaTiVWZr)C9",
    "(HfMoNbTaTiVWYZr)C9",
    "(HfMoNbScTaTiVWZr)C9",
]

DEFAULT_SELECTED_CANDIDATES = [
    "(HfMoNbReTaTiVWZr)C9",
    "(HfMoNbTaTiVWYZr)C9",
    "(HfMoNbScTaTiVWZr)C9",
    "(HfMoNbReScTaTiVWZr)C10",
    "(HfMoNbScTaTiVWYZr)C10",
    "(HfMoNbReTaTiVWYZr)C10",
    "(HfMoNbReScTaTiVWYZr)C11",
]

SINGLE_PHASE_LIT = [
    "(CrHfMoNbTaTiVZr)C8",
    "(CrHfNbTaTiVWZr)C8",
    "(CrHfNbTaTi)C5",
    "(CrHfNbTaV)C5",
    "(CrHfTaTiZr)C5",
    "(CrMoNbTaW)C5",
    "(CrMoNbVW)C5",
    "(CrMoTaVW)C5",
    "(CrMoTiVW)C5",
    "(HfMoNbTaTi)C5",
    "(HfMoNbTaTiVWZr)C8",
    "(HfMoNbTaTiZr)C6",
    "(HfMoNbTaV)C5",
    "(HfMoNbTaZr)C5",
    "(HfMoTaTiZr)C5",
    "(HfNbTaTiV)C5",
    "(HfNbTaTiVZr)C6",
    "(HfNbTaTiW)C5",
    "(HfNbTaTiWZr)C6",
    "(HfNbTaTiZr)C5",
    "(HfNbTaVW)C5",
    "(HfTaTiWZr)C5",
    "(MoNbTaTiV)C5",
    "(MoNbTaTiW)C5",
    "(MoNbTaTiZr)C5",
    "(MoNbTaVW)C5",
    "(MoNbTaVZr)C5",
    "(NbTaTiVW)C5",
    "(NbTaTiWZr)C5",
    "(NbTaVWZr)C5",
    "(CrNbTaTiZr)C5",
    "(CrHfNbTaZr)C5",
    "(MoNbTaTiVW)C6",
    "(HfNbTaWZr)C5",
    "(MoNbTaWZr)C5",
]

MULTI_PHASE_LIT = [
    "(CrHfMoTiW)C5",
    "(CrHfMoVW)C5",
    "(CrHfTaVW)C5",
    "(CrHfTaWZr)C5",
    "(CrMoTiVZr)C5",
    "(CrMoTiWZr)C5",
    "(HfMoTaWZr)C5",
    "(HfMoTiWZr)C5",
    "(HfMoVWZr)C5",
    "(MoTiVWZr)C5",
]

FEATURE_FAMILIES = {
    "mismatch": [
        "norm_rmis",
        "norm_xmis",
        "Radius Mismatch",
        "Electronegativity Mismatch",
    ],
    "bonding_carbon_affinity": [
        "norm_afe",
        "norm_fg",
        "norm_ca",
        "norm_cfdi",
        "Average Formation Enthalpy",
        "Formation Gap",
        "Carbon Affinity",
        "Carbide Formation Deviation Index",
    ],
    "refractory_floor": [
        "norm_mp",
        "Minimum Carbide Melting Point",
    ],
    "segregation_msi": [
        "norm_msi",
        "norm_mdri",
        "Metastable Segregation Index",
        "Magnetic Disorder Risk Index",
    ],
    "vec": [
        "norm_vec",
        "Average Valence Count",
    ],
}


def canonicalize(comp: str) -> str:
    """Return a sorted '(Metal...)Ck' representation when possible."""
    s = str(comp).strip()
    match = re.match(r"^\s*\(([^)]+)\)\s*C?\s*(\d+)\s*$", s)
    if not match:
        return s
    metals = re.findall(r"[A-Z][a-z]?", match.group(1))
    return f"({''.join(sorted(metals))})C{int(match.group(2))}"


def parse_metals(comp: str) -> list[str]:
    s = canonicalize(comp)
    match = re.match(r"^\(([^)]+)\)C(\d+)$", s)
    if not match:
        return []
    return re.findall(r"[A-Z][a-z]?", match.group(1))


def count_metals(comp: str) -> float:
    metals = parse_metals(comp)
    return float(len(metals)) if metals else np.nan


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def literature_label_map() -> dict[str, int]:
    labels = {canonicalize(c): 1 for c in SINGLE_PHASE_LIT}
    labels.update({canonicalize(c): 0 for c in MULTI_PHASE_LIT})
    return labels


def attach_literature_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Composition" not in out.columns:
        raise ValueError("Input table must contain a 'Composition' column.")
    out["Composition"] = out["Composition"].astype(str).map(canonicalize)
    out["n_metals"] = out["Composition"].map(count_metals)
    out["Exp_Phase"] = out["Composition"].map(literature_label_map())
    return out


def load_descriptor_table(path: str | Path = DEFAULT_DESCRIPTOR_PATH) -> pd.DataFrame:
    return attach_literature_labels(read_table(path))


def descriptor_feature_columns(
    df: pd.DataFrame,
    include_cfdi: bool = True,
    exclude: Iterable[str] | None = None,
) -> list[str]:
    """Return the GMM descriptor columns, following the corrected GMM script."""
    exclude_l = {str(x).lower() for x in (exclude or [])}
    norm_cols = [c for c in df.columns if str(c).startswith("norm_")]
    blocked = {"norm_efa", "norm_predicted_efa", "norm_mdri"}
    if not include_cfdi:
        blocked.add("norm_cfdi")
    cols = [c for c in norm_cols if c.lower() not in blocked and c.lower() not in exclude_l]
    if len(cols) >= 3:
        return cols

    fallback = [
        "Average Formation Enthalpy",
        "Formation Gap",
        "Radius Mismatch",
        "Electronegativity Mismatch",
        "Carbon Affinity",
        "Magnetic Disorder Risk Index",
        "Minimum Carbide Melting Point",
        "Average Valence Count",
        "Carbide Formation Deviation Index",
        "Metastable Segregation Index",
    ]
    return [c for c in fallback if c in df.columns and c.lower() not in exclude_l]


def family_exclusion_columns(df: pd.DataFrame, family_name: str) -> list[str]:
    if family_name not in FEATURE_FAMILIES:
        raise KeyError(f"Unknown family '{family_name}'. Options: {sorted(FEATURE_FAMILIES)}")
    wanted = set(FEATURE_FAMILIES[family_name])
    return [c for c in df.columns if c in wanted]


def zscore(values: Sequence[float]) -> np.ndarray:
    x = pd.to_numeric(pd.Series(values), errors="coerce").astype(float).to_numpy()
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        return x - mu
    return (x - mu) / sd


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


@dataclass
class GMMResult:
    df: pd.DataFrame
    feature_cols: list[str]
    scaler: object
    gmm: object
    responsibilities: np.ndarray
    x_scaled: np.ndarray
    component_is_single: np.ndarray
    best: dict


def _require_sklearn():
    try:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise ImportError(
            "This validation requires scikit-learn. Install with: "
            "python -m pip install scikit-learn"
        ) from exc
    return GaussianMixture, StandardScaler, IsotonicRegression


def map_components_to_scores(
    responsibilities: np.ndarray,
    y_labels: np.ndarray,
    is_labeled: np.ndarray,
    calibrate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Map unsupervised GMM components to single/multi using labelled mass."""
    _, _, IsotonicRegression = _require_sklearn()
    y_lab = y_labels[is_labeled].astype(int)
    r_lab = responsibilities[is_labeled]
    if len(y_lab) < 2 or len(np.unique(y_lab)) < 2:
        raise ValueError("Both single- and multiphase labels are required for component mapping.")
    single_mass = r_lab[y_lab == 1].sum(axis=0)
    multi_mass = r_lab[y_lab == 0].sum(axis=0)
    component_is_single = single_mass >= multi_mass
    p_single = responsibilities[:, component_is_single].sum(axis=1)
    if calibrate and len(y_lab) >= 25:
        p_all = responsibilities[:, component_is_single].sum(axis=1)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_all[is_labeled], y_lab)
        p_single = iso.transform(p_all)
    return p_single.astype(float), component_is_single


def fit_gmm_posterior(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    seed: int = 42,
    k_values: Sequence[int] | None = None,
    covariance_types: Sequence[str] = ("full", "tied", "diag", "spherical"),
    calibrate: bool = True,
) -> GMMResult:
    GaussianMixture, StandardScaler, _ = _require_sklearn()
    if not feature_cols:
        raise ValueError("No descriptor columns provided for GMM.")
    x = df.loc[:, list(feature_cols)].apply(pd.to_numeric, errors="coerce")
    row_ok = ~x.isna().any(axis=1)
    df_x = df.loc[row_ok].copy().reset_index(drop=True)
    x = x.loc[row_ok].to_numpy(dtype=float)
    if len(df_x) < 5:
        raise ValueError("Too few complete rows for GMM fitting.")

    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)

    best = None
    k_values = list(k_values) if k_values is not None else list(range(2, 30))
    for k in k_values:
        if k >= len(df_x):
            continue
        for cov in covariance_types:
            try:
                gmm = GaussianMixture(
                    n_components=int(k),
                    covariance_type=cov,
                    tol=1e-4,
                    reg_covar=1e-6,
                    max_iter=500,
                    n_init=5,
                    init_params="kmeans",
                    random_state=seed,
                )
                gmm.fit(x_scaled)
                bic = float(gmm.bic(x_scaled))
            except Exception:
                continue
            if best is None or bic < best["bic"]:
                best = {"gmm": gmm, "bic": bic, "k": int(k), "covariance_type": cov}
    if best is None:
        raise RuntimeError("GMM fit failed for all requested component/covariance settings.")

    responsibilities = best["gmm"].predict_proba(x_scaled)
    is_labeled = df_x["Exp_Phase"].isin([0, 1]).to_numpy()
    y_labels = df_x["Exp_Phase"].to_numpy()
    p_single, comp_is_single = map_components_to_scores(
        responsibilities,
        y_labels,
        is_labeled,
        calibrate=calibrate,
    )
    comp_id = responsibilities.argmax(axis=1)
    df_x["gmm_comp"] = comp_id
    df_x["P_single"] = p_single
    df_x["sample_phase_label"] = np.where(p_single >= 0.5, "Single-like", "Multi-like")
    df_x["component_phase_label"] = np.where(comp_is_single[comp_id], "Single-like", "Multi-like")
    return GMMResult(
        df=df_x,
        feature_cols=list(feature_cols),
        scaler=scaler,
        gmm=best["gmm"],
        responsibilities=responsibilities,
        x_scaled=x_scaled,
        component_is_single=comp_is_single,
        best={k: v for k, v in best.items() if k != "gmm"},
    )


def metric_summary(y_true: Sequence[int], y_score: Sequence[float], threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = int(len(y_true))
    acc = (tp + tn) / n if n else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    bacc = np.nanmean([recall, specificity])
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else np.nan
    return {
        "n": n,
        "threshold": float(threshold),
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "precision_single": float(precision) if np.isfinite(precision) else np.nan,
        "recall_single": float(recall) if np.isfinite(recall) else np.nan,
        "specificity_multi": float(specificity) if np.isfinite(specificity) else np.nan,
        "f1_single": float(f1) if np.isfinite(f1) else np.nan,
        "true_multi_pred_multi": tn,
        "true_multi_pred_single": fp,
        "true_single_pred_multi": fn,
        "true_single_pred_single": tp,
    }


def bootstrap_metric_ci(
    y_true: Sequence[int],
    y_score: Sequence[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
    threshold: float = 0.5,
    stratified: bool = False,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    metrics = ["accuracy", "balanced_accuracy", "precision_single", "recall_single", "f1_single"]
    draws = {m: [] for m in metrics}
    idx0 = np.flatnonzero(y_true == 0)
    idx1 = np.flatnonzero(y_true == 1)
    for _ in range(int(n_bootstrap)):
        if stratified:
            sample = np.concatenate(
                [
                    rng.choice(idx0, size=len(idx0), replace=True),
                    rng.choice(idx1, size=len(idx1), replace=True),
                ]
            )
        else:
            sample = rng.choice(np.arange(len(y_true)), size=len(y_true), replace=True)
        m = metric_summary(y_true[sample], y_score[sample], threshold=threshold)
        for name in metrics:
            draws[name].append(m[name])
    rows = []
    point = metric_summary(y_true, y_score, threshold=threshold)
    for name, values in draws.items():
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": name,
                "point": point[name],
                "bootstrap_mean": float(np.nanmean(arr)),
                "ci_low": float(np.nanquantile(arr, 0.025)),
                "ci_high": float(np.nanquantile(arr, 0.975)),
                "n_bootstrap": int(n_bootstrap),
                "stratified": bool(stratified),
            }
        )
    return pd.DataFrame(rows)


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum(err**2) / denom) if denom > 0 else np.nan
    return {"n": int(len(y_true)), "mae": mae, "rmse": rmse, "r2": r2}


def bootstrap_regression_ci(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(seed)
    names = ["mae", "rmse", "r2"]
    draws = {n: [] for n in names}
    for _ in range(int(n_bootstrap)):
        idx = rng.choice(np.arange(len(y_true)), size=len(y_true), replace=True)
        m = regression_metrics(y_true[idx], y_pred[idx])
        for name in names:
            draws[name].append(m[name])
    point = regression_metrics(y_true, y_pred)
    rows = []
    for name, values in draws.items():
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": name,
                "point": point[name],
                "bootstrap_mean": float(np.nanmean(arr)),
                "ci_low": float(np.nanquantile(arr, 0.025)),
                "ci_high": float(np.nanquantile(arr, 0.975)),
                "n_bootstrap": int(n_bootstrap),
            }
        )
    return pd.DataFrame(rows)


def qcut_bins(x: Sequence[float], q: int) -> pd.Series:
    return pd.qcut(pd.Series(x).astype(float), q=q, duplicates="drop")


def compute_candidate_ranking(
    df_x: pd.DataFrame,
    feature_cols: Sequence[str],
    target_n_metals: int = 9,
    n_bins_per_feature: int = 20,
    favor_top_q: float = 0.30,
    roll_smooth: int = 3,
    weights: tuple[float, float, float] = (0.33, 0.33, 0.34),
) -> pd.DataFrame:
    if "P_single" not in df_x.columns:
        raise ValueError("df_x must contain a 'P_single' column.")
    bands = []
    for feat in feature_cols:
        x = pd.to_numeric(df_x[feat], errors="coerce")
        p = pd.to_numeric(df_x["P_single"], errors="coerce")
        ok = ~(x.isna() | p.isna())
        if ok.sum() < 3:
            continue
        bins = qcut_bins(x[ok], q=n_bins_per_feature)
        tmp = pd.DataFrame({"x": x[ok].to_numpy(), "p": p[ok].to_numpy(), "bin": bins})
        grouped = tmp.groupby("bin", observed=True)
        bin_mid = grouped["x"].mean()
        bin_p = grouped["p"].mean()
        if roll_smooth and roll_smooth > 1:
            bin_p = bin_p.rolling(roll_smooth, center=True, min_periods=1).mean()
        n_top = max(1, int(math.ceil(len(bin_p) * favor_top_q)))
        top_intervals = set(bin_p.sort_values(ascending=False).index[:n_top])
        bands.append(
            pd.DataFrame(
                {
                    "feature": feat,
                    "bin_left": [iv.left for iv in bin_mid.index],
                    "bin_right": [iv.right for iv in bin_mid.index],
                    "bin_pmean": bin_p.to_numpy(dtype=float),
                    "is_favorable": [iv in top_intervals for iv in bin_mid.index],
                }
            )
        )
    if not bands:
        raise ValueError("No bands could be computed for ranking.")
    bands_df = pd.concat(bands, ignore_index=True)

    lookup = {}
    for feat, group in bands_df.groupby("feature"):
        intervals = pd.IntervalIndex.from_arrays(
            group["bin_left"].to_numpy(dtype=float),
            group["bin_right"].to_numpy(dtype=float),
            closed="right",
        )
        lookup[feat] = {
            "intervals": intervals,
            "pmean": group["bin_pmean"].to_numpy(dtype=float),
            "is_favorable": group["is_favorable"].to_numpy(dtype=bool),
        }

    rows = []
    for _, row in df_x.iterrows():
        pmeans = []
        hits = 0
        used = 0
        for feat in feature_cols:
            if feat not in lookup:
                continue
            value = row.get(feat, np.nan)
            if pd.isna(value):
                continue
            item = lookup[feat]
            idx = item["intervals"].get_indexer([float(value)])[0]
            if idx == -1:
                continue
            used += 1
            hits += int(item["is_favorable"][idx])
            pmeans.append(float(item["pmean"][idx]))
        rows.append(
            {
                "Composition": row["Composition"],
                "n_metals": float(row["n_metals"]),
                "P_single": float(row["P_single"]),
                "n_feat_used": int(used),
                "cum_pmean": float(np.nansum(pmeans)) if pmeans else np.nan,
                "bin_pmean_avg": float(np.nanmean(pmeans)) if pmeans else np.nan,
                "hits_count": int(hits),
            }
        )
    scores = pd.DataFrame(rows)
    pool = scores[scores["n_metals"] == float(target_n_metals)].copy()
    if pool.empty:
        return pool
    n_total = max(1, len(feature_cols))
    pool["coverage"] = pool["n_feat_used"] / float(n_total)
    w_cum, w_post, w_cov = weights
    pool["cum_pmean_z"] = zscore(pool["cum_pmean"])
    pool["P_single_z"] = zscore(pool["P_single"])
    pool["coverage_z"] = zscore(pool["coverage"])
    pool["Score"] = (
        w_cum * pool["cum_pmean_z"] + w_post * pool["P_single_z"] + w_cov * pool["coverage_z"]
    )
    ranked = pool.sort_values(
        by=["Score", "cum_pmean", "bin_pmean_avg"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranked["Rank"] = np.arange(1, len(ranked) + 1)
    return ranked
