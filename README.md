# HESS Discovery

Code and data for the manuscript-consistent high-entropy carbide discovery
workflow. The current repository implements the sequential Eq. (1)-(5)
candidate-selection operator used in the revised paper, not the retired
posterior/coverage weighted composite score.

## Manuscript Workflow

### 1. GNN EFA Prediction

Candidate carbides are represented as variable-size PyTorch Geometric graphs.
For a `k`-metal carbide, the graph contains the `k` metal nodes plus carbon as
the final node. The manuscript-consistent GNN uses:

- `GCNEncoder(hidden_dim=4, num_layers=2, heads=32, dropout=0.2)`
- Smooth L1 training loss
- Adam optimizer with learning rate `5e-5` and weight decay `1e-6`
- 90/10 HEC5 train/hold-out split with `random_state=42`
- checkpoint selection by the best HEC5 hold-out `R2`

The HEC5 hold-out subset is therefore a model-selection subset, not a separate
untouched final test set. The HEC6 POCC-DFT calculations are used as a targeted
transfer and architecture-sensitivity benchmark for HEC5-trained checkpoints;
they are not used for GNN weight updates.

The manuscript checkpoint used for final prediction is:

```text
best_gnn_model_4_2_32_88.0.pth
```

Place this checkpoint in `checkpoints/` or pass its path with `--checkpoint`.
The older checkpoint `legacy/checkpoints/_model_26999_after_7l.pth` is retained
only for legacy comparison and should not be used for manuscript figures or
tables.

### 2. GMM Basin Evidence

Candidate carbides are embedded in a normalized descriptor space containing
thermodynamic competition, chemical mismatch, bonding, segregation and
processing descriptors. EFA-like variables and the magnetic-disorder risk index
are excluded from the GMM feature matrix to avoid circularity and reduce
collinearity.

The manuscript-consistent GMM path:

- fits candidate GMMs over `k=2..29` and covariance types
  `full`, `tied`, `diag`, and `spherical`
- selects component number and covariance form by BIC
- maps GMM components to single-like or multi-like classes using the curated
  literature labels
- calibrates the GMM single-like posterior with isotonic regression
- estimates descriptor evidence `s_j(c)` from empirical quantile-bin posterior
  means
- computes the basin score `B(c)` as the unweighted mean of descriptor evidence
- applies Eq. (3) by retaining the top `rho` fraction within each cation order
- applies Eq. (5) by ranking the retained subset using frozen GNN-predicted EFA

Selection frequency over a `rho`/`eta` grid is a sensitivity measure for the
shortlist size, not a calibrated probability of phase formation.

## Main Entry Points

Run commands from the repository root.

```bash
python Model_Building.py --checkpoint checkpoints/best_gnn_model_4_2_32_88.0.pth
```

Evaluates the manuscript-consistent GNN checkpoint on the HEC5 hold-out subset
and the 22 unique HEC6 POCC-DFT compositions.

```bash
python GMM_Clustering/GMM_Clustering.py --checkpoint checkpoints/best_gnn_model_4_2_32_88.0.pth
```

Runs the Eq. (1)-(5) GMM/EFA selection workflow and writes outputs to
`validation_outputs/equation_1_5_selection/`.

```bash
python validations/generate_equation_selection_table.py
```

Regenerates the manuscript Table 2 source files from the Eq. (1)-(5) audit.

```bash
python validations/generate_table_s4_gmm_cluster_diagnostics.py
python validations/gnn_architecture_sweep.py
python validations/plot_figure_s5_gnn_transfer_corrected.py
python validations/plot_figure_s7_gnn_rf_unique.py
```

Regenerates corrected Supplementary Table S4 and Figures S5/S7 from the
manuscript-consistent inputs and 22-unique-composition HEC6 benchmark.

## HEC6 Benchmark Convention

`DFT_Calculations/DFT_calculated_EFA_for_HEC6.csv` contains 22 unique HEC6
compositions. The duplicate `(HfMoNbTaTiZr)C6` entry present in earlier working
files was removed so that manuscript tables and code use the same unique-count
benchmark.

## Supplementary Data Archives

The submission-ready Supplementary Data 1-4 ZIP archives are mirrored in
`supplementary_data/` for repository-level reproducibility of the submitted
data packages.

Supplementary Data 4 contains the 22-composition HEC6 EFA summary and
representative AFLOW-POCC/VASP inputs/scripts. It does not contain the
configuration-level POCC/VASP structures, degeneracies, or absolute total
energies required to independently recompute every HEC6 EFA value. Reproducing
those values from first principles requires rerunning the AFLOW-POCC/VASP
workflow from the supplied inputs and scripts.

VASP PAW/POTCAR archives are not distributed in this repository. Users must
provide any required licensed VASP potential data locally.

## Validation Scripts

Additional reviewer-facing scripts are in `validations/`. See
`validations/README.md` for exact commands and scope. Scripts that mention the
legacy checkpoint or retired composite score are retained only for historical
comparison unless explicitly documented as manuscript-consistent.
