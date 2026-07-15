# Validation Scripts

These scripts add reviewer-facing validations without creating artificial labels, EFA values, or compositions. They only use data already present in this repository:

- `data/HEC5_output.csv` and the manuscript-consistent matminer/Bond feature generation path
- `DFT_Calculations/DFT_calculated_EFA_for_HEC6.csv` with 22 unique POCC-DFT HEC6 compositions
- `data/Compositions Descriptros.csv`
- `checkpoints/best_gnn_model_4_2_32_88.0.pth`
- the literature single-/multiphase label lists already used in `GMM_Clustering/GMM_Clustering.py`

Run commands from the repository root.

## Feasibility Summary

| Validation | Feasible from repo? | Notes |
|---|---:|---|
| Leave-element-out GNN validation | Yes | Uses 545 labelled HEC5 EFA rows. Sc and Y each have only 2 held-out rows, so the script skips them by default unless `--min-test` is lowered. |
| Leave-family-out GNN validation | Yes, but split size matters | Holding out any IVB/VB/VIB member leaves only 79/82/103 training rows respectively. The script also supports `--mode atleast --at-least 2` for a less extreme split. |
| MC-dropout uncertainty for HEC9-HEC11 | Yes | Uses the manuscript-consistent checkpoint. This is a model-uncertainty proxy, not a conformal guarantee. |
| Applicability-domain analysis | Yes | Uses descriptor-space distances from the HEC5/HEC6 reference manifold to HEC9-HEC11 candidates. |
| HEC6 POCC-DFT transfer benchmark | Yes | Uses 22 unique HEC6 compositions after removing the duplicate `(HfMoNbTaTiZr)C6` row. This is a targeted transfer/architecture-sensitivity benchmark, not a weight-update set. |
| Literature confusion matrix | Yes | Recomputes the actual matrix from the curated 35 single-phase and 10 multiphase labels. |
| Bootstrap labelled literature set | Yes | Bootstraps accuracy, balanced accuracy, precision, recall, and F1. |
| GMM seed/component robustness | Yes | Repeats GMM across seeds and component counts and tracks selected HEC9 ranks/posteriors. |
| Feature ablation | Yes | Drops mismatch, bonding/carbon-affinity, refractory-floor, segregation/MSI, and VEC descriptor families one at a time. |
| Label-permutation/null test | Yes | Shuffles literature labels and compares real-label metrics to null-label metrics. |

## Dependencies

Descriptor/GMM scripts:

```bash
python -m pip install scikit-learn pandas numpy matplotlib
```

GNN scripts:

```bash
python -m pip install torch
# Install torch_geometric using the wheel matching your local PyTorch build.
```

The manuscript-consistent GNN scripts require `matminer`, `pymatgen`, `torch`, and `torch_geometric`. Graphs contain the metal nodes plus carbon as the final node. The edge attribute is the carbon-edge formation-energy channel plus the metal-metal absolute formation-energy-channel difference.

## Commands

### Feasibility Report

```bash
python validations/feasibility_report.py
```

This writes split counts, label counts, cardinality counts, checkpoint presence, and dependency availability.

### Plot Validation Figures

After running the validations, generate separate publication-style subfigure panels:

```bash
python validations/plot_validation_figures.py
```

Figures are written to `validation_outputs/figures` as PDF, SVG, and 600 dpi PNG.

### Literature Confusion Matrix and Bootstrap

```bash
python validations/gmm_literature_confusion_bootstrap.py
```

Outputs:

- `validation_outputs/gmm_literature_bootstrap/confusion_matrix.csv`
- `validation_outputs/gmm_literature_bootstrap/bootstrap_metrics.csv`
- `validation_outputs/gmm_literature_bootstrap/labelled_predictions.csv`

### GMM Seed and Component Robustness

```bash
python validations/gmm_seed_component_robustness.py --seeds 0 1 2 3 4 5 6 7 8 9 --k-values 2 3 4 5 6 7 8 9 10 11
```

Outputs selected-candidate rank stability and all per-run rankings.

### Feature Ablation

```bash
python validations/feature_ablation.py
```

Outputs per-family metrics, selected-candidate ranks, and top-k overlap with the baseline ranking.

### Label-Permutation Null Test

```bash
python validations/label_permutation_null.py --n-permutations 1000
```

By default this uses raw mapped GMM posteriors. Add `--calibrate` only if you want the null test to include isotonic calibration.

### Applicability Domain

```bash
python validations/applicability_domain.py --reference-cardinalities 5 6 --query-cardinalities 9 10 11
```

Outputs nearest-neighbor and Mahalanobis distance percentiles. Candidates above the 95th percentile are marked `extrapolation_like`.

### Leave-Element-Out GNN

```bash
python validations/gnn_leave_element_out.py --epochs 2000 --seeds 0 1 2
```

For a quick dependency check:

```bash
python validations/gnn_leave_element_out.py --epochs 1 --elements Ti --min-test 1 --seeds 0
```

### Leave-Family-Out GNN

Strict family holdout, any member:

```bash
python validations/gnn_leave_family_out.py --families IVB VB VIB --mode any --epochs 2000 --seeds 0 1 2
```

Less extreme family-related chemistry holdout:

```bash
python validations/gnn_leave_family_out.py --families IVB VB VIB --mode atleast --at-least 2 --epochs 2000 --seeds 0 1 2
```

### Manuscript GNN Evaluation

```bash
python Model_Building.py --checkpoint checkpoints/best_gnn_model_4_2_32_88.0.pth
```

Outputs HEC5 hold-out model-selection subset metrics and HEC6 22-unique-composition transfer metrics.

### MC-Dropout Uncertainty for HEC9-HEC11

```bash
python validations/gnn_mc_dropout_uncertainty.py --n-samples 200 --cardinalities 9 10 11
```

For only the three selected HEC9 candidates:

```bash
python validations/gnn_mc_dropout_uncertainty.py --selected-only --n-samples 200
```

### HEC8 Legacy Benchmark

`validations/external_hec8_benchmark.py` is retained as a legacy exploratory script. Use it in the manuscript only if the `EFA` column in `data/HEC8_output.csv` is demonstrably independent of training/tuning and the checkpoint path is intentionally set.
