from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from validations.common import DEFAULT_OUTPUT_DIR, ensure_dir, load_descriptor_table, parse_metals, write_json
from validations.gnn_leave_family_out import DEFAULT_FAMILIES, family_holdout_mask


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="Report whether requested validations are feasible from repo data.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR / "feasibility"))
    parser.add_argument("--min-test", type=int, default=5)
    parser.add_argument("--min-train", type=int, default=50)
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    payload = {}

    hec5 = pd.read_csv("data/HEC5_output.csv")
    hec5_features = np.load("data/final_features_train.npy")
    metal_sets = hec5["Composition"].map(lambda c: set(parse_metals(c))).tolist()
    elements = sorted({e for s in metal_sets for e in s})

    element_rows = []
    for element in elements:
        n_test = sum(element in s for s in metal_sets)
        n_train = len(hec5) - n_test
        element_rows.append(
            {
                "validation": "leave_element_out",
                "group": element,
                "n_train": n_train,
                "n_test": n_test,
                "feasible_default": bool(n_train >= args.min_train and n_test >= args.min_test),
            }
        )
    pd.DataFrame(element_rows).to_csv(outdir / "leave_element_split_counts.csv", index=False)

    family_rows = []
    for family_name, members in DEFAULT_FAMILIES.items():
        for mode, at_least in [("any", 1), ("atleast", 2), ("all", len(members))]:
            holdout = family_holdout_mask(metal_sets, set(members), mode, at_least)
            n_test = int(holdout.sum())
            n_train = int((~holdout).sum())
            family_rows.append(
                {
                    "validation": "leave_family_out",
                    "group": family_name,
                    "elements": ",".join(members),
                    "mode": mode,
                    "at_least": at_least,
                    "n_train": n_train,
                    "n_test": n_test,
                    "feasible_default": bool(n_train >= args.min_train and n_test >= args.min_test),
                }
            )
    pd.DataFrame(family_rows).to_csv(outdir / "leave_family_split_counts.csv", index=False)

    descriptors = load_descriptor_table()
    hec8 = pd.read_csv("data/HEC8_output.csv")
    payload["data"] = {
        "hec5_rows": int(len(hec5)),
        "hec5_feature_shape": list(hec5_features.shape),
        "hec8_rows": int(len(hec8)),
        "hec8_has_efa": bool("EFA" in hec8.columns and hec8["EFA"].notna().all()),
        "descriptor_rows": int(len(descriptors)),
        "descriptor_cardinality_counts": {
            str(int(k)): int(v) for k, v in descriptors["n_metals"].value_counts().sort_index().items()
        },
        "literature_single_labels": int((descriptors["Exp_Phase"] == 1).sum()),
        "literature_multi_labels": int((descriptors["Exp_Phase"] == 0).sum()),
        "manuscript_checkpoint_exists": Path("checkpoints/best_gnn_model_4_2_32_88.0.pth").exists(),
        "legacy_checkpoint_exists": Path("checkpoints/_model_26999_after_7l.pth").exists(),
    }
    payload["dependencies"] = {
        "numpy": module_available("numpy"),
        "pandas": module_available("pandas"),
        "sklearn": module_available("sklearn"),
        "torch": module_available("torch"),
        "torch_geometric": module_available("torch_geometric"),
    }
    payload["feasibility"] = {
        "leave_element_out": "feasible; Sc and Y are underpowered with default min-test=5",
        "leave_family_out": "feasible; IVB/VB/VIB any-member holdouts are severe but valid",
        "mc_dropout_uncertainty": "feasible if torch_geometric is installed",
        "applicability_domain": "feasible from descriptor table",
        "external_hec8_benchmark": "technically feasible; manuscript claim depends on HEC8 EFA provenance",
        "literature_confusion_bootstrap": "feasible from curated labels",
        "gmm_seed_component_robustness": "feasible from descriptor table",
        "feature_ablation": "feasible from descriptor table",
        "label_permutation_null": "feasible from curated labels and descriptor table",
    }
    write_json(outdir / "feasibility_report.json", payload)

    print(f"Saved outputs to {outdir.resolve()}")
    print(pd.DataFrame(element_rows).to_string(index=False))
    print(pd.DataFrame(family_rows).to_string(index=False))
    print(payload)


if __name__ == "__main__":
    main()
