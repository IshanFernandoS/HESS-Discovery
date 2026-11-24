"""Generate a transparent future-fabrication shortlist from frozen scores."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SCORES = ROOT / "inputs" / "final_scores_3301.csv"
RETENTION = ROOT / "inputs" / "retention_scoped_records.csv"
OUTPUT = ROOT / "future_fabrication_high_joint_index_candidates.csv"
EXPECTED_SCORES_SHA256 = (
    "6457c64b52eef1a1d387b74f3bbc9a9d0c65774c964f8928b5ed98d52d9fccdb"
)
EXPECTED_RETENTION_SHA256 = (
    "10769330e31132f4be84b887cdb0de9c362ff61527ff58c402175afbbfc478ed"
)
EXPECTED_OUTPUT_SHA256 = (
    "f6dea59c48cf3193b0300ecfacd99b397b14a6300a4cba79c90e5f2cf011ac3b"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def elements(formula):
    metal_text = formula.split(")C", 1)[0].lstrip("(")
    return re.findall(r"[A-Z][a-z]?", metal_text)


def application_tags(metals, order):
    metals = set(metals)
    tags = ["ultra-high-temperature structural ceramic", "hardness and wear"]
    if metals & {"Hf", "Ta", "Zr"}:
        tags.append("ablation and thermal-protection screening")
    if metals & {"Mo", "W", "Re"}:
        tags.append("thermal and electrical transport screening")
    if "Cr" in metals:
        tags.append("oxidation and corrosion screening")
    if metals & {"Sc", "Y"}:
        tags.append("grain-boundary chemistry and sintering study")
    if order >= 9:
        tags.append("electromagnetic-wave absorption and local-order study")
    return "; ".join(tags)


def attention_flags(metals, order):
    metals = set(metals)
    flags = []
    if "Y" in metals:
        flags.append("Y retention/process sensitivity; quantify matrix incorporation")
    if "Cr" in metals:
        flags.append("Cr depletion reported in related literature matrices")
    if "Sc" in metals:
        flags.append("Sc incorporation is composition dependent")
    if "Re" in metals:
        flags.append("Re cost and supply should be assessed")
    if order == 11:
        flags.append("HEC11 processing feasibility unresolved after prior extrusion")
    return "; ".join(flags) if flags else "no project-specific flag from current retention evidence"


assert sha256(SCORES) == EXPECTED_SCORES_SHA256
assert sha256(RETENTION) == EXPECTED_RETENTION_SHA256

scores = pd.read_csv(SCORES, float_precision="round_trip")
retention = pd.read_csv(RETENTION, float_precision="round_trip")
assert len(scores) == 3301 and scores["Composition"].nunique() == 3301
scores["n_metals"] = scores["n_metals"].astype(int)

scores["joint_rank_within_order"] = scores.groupby("n_metals")[
    "sequential_index"
].rank(method="min", ascending=False)
scores["joint_average_rank_within_order"] = scores.groupby("n_metals")[
    "sequential_index"
].rank(method="average", ascending=False)
order_sizes = scores.groupby("n_metals")["Composition"].transform("size")
scores["joint_percentile_within_order"] = 1 - (
    scores["joint_average_rank_within_order"] - 0.5
) / order_sizes
scores["joint_rank_across_3301"] = scores["sequential_index"].rank(
    method="average", ascending=False
)

retention_chemistries = set(retention["Composition"].dropna().astype(str))
scores["in_current_retention_cohort"] = scores["Composition"].isin(
    retention_chemistries
)

# Prespecified shortlist: positive I, top ten within its cation order, and no
# current retention record. This does not claim novelty against all literature.
shortlist = scores[
    (scores["sequential_index"] > 0)
    & (scores["joint_rank_within_order"] <= 10)
    & (~scores["in_current_retention_cohort"])
].copy()

shortlist["metal_elements"] = shortlist["Composition"].map(
    lambda value: ";".join(elements(value))
)
for element in ["Cr", "Hf", "Mo", "Nb", "Re", "Sc", "Ta", "Ti", "V", "W", "Y", "Zr"]:
    shortlist["contains_" + element] = shortlist["Composition"].map(
        lambda value, element=element: element in elements(value)
    )

shortlist["screening_tier"] = np.select(
    [
        shortlist["sequential_index"] >= 0.90,
        shortlist["sequential_index"] >= 0.70,
        shortlist["sequential_index"] >= 0.50,
    ],
    [
        "Tier 1: I >= 0.90",
        "Tier 2: 0.70 <= I < 0.90",
        "Tier 3: 0.50 <= I < 0.70",
    ],
    default="Tier 4: positive order-coverage candidate",
)
shortlist["project_role"] = np.where(
    shortlist["n_metals"] >= 9,
    "higher-order prospective fabrication",
    "lower-order control or benchmark",
)
shortlist["next_project_group"] = np.select(
    [
        (shortlist["n_metals"] >= 9) & (shortlist["sequential_index"] >= 0.50),
        shortlist["n_metals"] >= 9,
    ],
    [
        "core higher-order prospect (I >= 0.50)",
        "exploratory higher-order prospect (0 < I < 0.50)",
    ],
    default="lower-order experimental control",
)
shortlist["application_search_tags"] = shortlist.apply(
    lambda row: application_tags(elements(row["Composition"]), row["n_metals"]),
    axis=1,
)
shortlist["fabrication_attention_flags"] = shortlist.apply(
    lambda row: attention_flags(elements(row["Composition"]), row["n_metals"]),
    axis=1,
)
shortlist["recommended_initial_validation"] = (
    "replicated synthesis; density; XRD phase analysis; quantitative matrix-region EDS; "
    "minimum cation retention"
)
shortlist["application_tag_status"] = (
    "chemistry-derived literature/property-search direction; not a predicted application performance"
)
shortlist["selection_basis"] = (
    "positive frozen joint-selection index; top 10 within order; absent from current retention cohort"
)

shortlist = shortlist.sort_values(
    ["sequential_index", "n_metals", "Composition"],
    ascending=[False, False, True],
).reset_index(drop=True)
shortlist.insert(0, "shortlist_rank_by_joint_index", np.arange(1, len(shortlist) + 1))
shortlist["higher_order_shortlist_rank"] = pd.NA
higher_order = shortlist["n_metals"].ge(9)
shortlist.loc[higher_order, "higher_order_shortlist_rank"] = np.arange(
    1, int(higher_order.sum()) + 1
)

columns = [
    "shortlist_rank_by_joint_index",
    "Composition",
    "canonical_composition",
    "metal_elements",
    "n_metals",
    "screening_tier",
    "project_role",
    "next_project_group",
    "higher_order_shortlist_rank",
    "sequential_index",
    "sequential_raw_frequency",
    "joint_rank_within_order",
    "joint_average_rank_within_order",
    "joint_percentile_within_order",
    "joint_rank_across_3301",
    "B_raw",
    "Filter1_percentile",
    "Filter1_threshold_frequency",
    "EFA_raw",
    "EFA_percentile",
    "EFA_threshold_frequency",
    "EFA_matched_budget_index",
    "reference_Filter1_selected",
    "reference_sequential_selected",
    "application_search_tags",
    "fabrication_attention_flags",
    "recommended_initial_validation",
    "application_tag_status",
    "selection_basis",
] + [
    "contains_" + element
    for element in ["Cr", "Hf", "Mo", "Nb", "Re", "Sc", "Ta", "Ti", "V", "W", "Y", "Zr"]
]
export = shortlist[columns].rename(
    columns={
        "sequential_index": "joint_selection_index_I",
        "sequential_raw_frequency": "selection_frequency_f",
        "B_raw": "filter1_support_B",
        "Filter1_percentile": "filter1_percentile_within_order",
        "Filter1_threshold_frequency": "filter1_inclusion_frequency",
        "EFA_raw": "predicted_EFA_inverse_eV_per_atom",
        "EFA_percentile": "EFA_percentile_within_order",
        "EFA_threshold_frequency": "EFA_inclusion_frequency",
        "EFA_matched_budget_index": "EFA_only_matched_budget_index",
        "reference_Filter1_selected": "reference_filter1_selected",
        "reference_sequential_selected": "reference_joint_selected",
    }
)
export.to_csv(OUTPUT, index=False)
assert sha256(OUTPUT) == EXPECTED_OUTPUT_SHA256

assert len(shortlist) == 59
assert not shortlist["in_current_retention_cohort"].any()
assert shortlist["sequential_index"].gt(0).all()
assert shortlist["joint_rank_within_order"].le(10).all()
assert shortlist["Composition"].is_unique

summary = {
    "generated": "2026-11-20",
    "candidate_universe": len(scores),
    "shortlist_rows": len(shortlist),
    "rule": "positive joint index; top 10 within order; absent from current retention cohort",
    "counts_by_order": {
        str(key): int(value)
        for key, value in shortlist.groupby("n_metals").size().items()
    },
    "counts_by_tier": {
        str(key): int(value)
        for key, value in shortlist.groupby("screening_tier").size().items()
    },
    "higher_order_rows": int(shortlist["n_metals"].ge(9).sum()),
    "core_higher_order_rows_I_ge_0.50": int(
        ((shortlist["n_metals"] >= 9) & (shortlist["sequential_index"] >= 0.50)).sum()
    ),
    "exploratory_higher_order_rows_I_below_0.50": int(
        ((shortlist["n_metals"] >= 9) & (shortlist["sequential_index"] < 0.50)).sum()
    ),
    "input_scores": SCORES.relative_to(ROOT).as_posix(),
    "input_scores_sha256": sha256(SCORES),
    "input_retention_cohort": RETENTION.relative_to(ROOT).as_posix(),
    "input_retention_cohort_sha256": sha256(RETENTION),
    "output_csv": OUTPUT.relative_to(ROOT).as_posix(),
    "output_csv_sha256": sha256(OUTPUT),
    "models_retrained": False,
    "application_performance_predicted": False,
}
(ROOT / "provenance_and_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2))

