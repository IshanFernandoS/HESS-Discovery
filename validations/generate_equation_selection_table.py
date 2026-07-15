from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from validations.common import canonicalize


AUDIT_PATH = Path(
    "validation_outputs/equation_1_5_selection/experimental_candidate_audit.csv"
)
CSV_OUT = Path(
    "validation_outputs/candidate_validation/equation_1_5_experimental_table.csv"
)
TEX_OUT = Path(
    "validation_outputs/latex_tables/equation_1_5_experimental_table.tex"
)


EXPERIMENTAL_OUTCOMES = {
    canonicalize("(HfMoNbReTaTiVWZr)C9"): (
        "HESS main phase; all nominal cations detected in the HESS matrix; "
        "secondary phases present."
    ),
    canonicalize("(HfMoNbScTaTiVWZr)C9"): (
        "HESS main phase; all nominal cations detected in the HESS matrix; "
        "secondary phases present."
    ),
    canonicalize("(HfMoNbTaTiVWYZr)C9"): (
        "HESS main phase, but Y was not detected in the HESS matrix; "
        "secondary phases present."
    ),
    canonicalize("(HfMoNbReScTaTiVWZr)C10"): (
        "HESS main phase with all ten nominal cations detected; Re-rich carbide "
        "and Sc-rich oxide secondary phases."
    ),
    canonicalize("(HfMoNbScTaTiVWYZr)C10"): (
        "HESS diffraction pattern, but Y was not detected in the measured HESS matrix."
    ),
    canonicalize("(HfMoNbReTaTiVWYZr)C10"): (
        "HESS diffraction pattern; Y was not detected and Re was approximately "
        "4 at\% in the measured HESS matrix."
    ),
    canonicalize("(HfMoNbReScTaTiVWYZr)C11"): (
        "Not phase-assessed: the sample was extruded when pressure was applied."
    ),
}

DISPLAY_ORDER = [
    canonicalize("(HfMoNbReTaTiVWZr)C9"),
    canonicalize("(HfMoNbScTaTiVWZr)C9"),
    canonicalize("(HfMoNbTaTiVWYZr)C9"),
    canonicalize("(HfMoNbReScTaTiVWZr)C10"),
    canonicalize("(HfMoNbScTaTiVWYZr)C10"),
    canonicalize("(HfMoNbReTaTiVWYZr)C10"),
    canonicalize("(HfMoNbReScTaTiVWYZr)C11"),
]


def latex_composition(composition: str) -> str:
    match = re.match(r"^\(([^)]+)\)C(\d+)$", composition)
    if not match:
        return composition
    metals, order = match.groups()
    return rf"\((\mathrm{{{metals}}})\mathrm{{C}}_{{{order}}}\)"


def main() -> None:
    audit = pd.read_csv(AUDIT_PATH)
    audit["Composition"] = audit["Composition"].astype(str).map(canonicalize)
    audit["Order"] = audit["n_metals"].astype(int).map(lambda value: f"HEC{value}")
    pool_sizes = {9: 220, 10: 66, 11: 12}
    eq3_sizes = {9: 66, 10: 20, 11: 4}
    audit["Basin_rank_with_pool"] = audit.apply(
        lambda row: f"{int(row['Basin_rank'])}/{pool_sizes[int(row['n_metals'])]}", axis=1
    )
    audit["EFA_rank_with_Eq3_pool"] = audit.apply(
        lambda row: (
            f"{int(row['EFA_rank_within_Eq3'])}/{eq3_sizes[int(row['n_metals'])]}"
            if pd.notna(row["EFA_rank_within_Eq3"])
            else "--"
        ),
        axis=1,
    )
    audit["Eq5_selection_frequency_percent"] = 100.0 * audit["Eq5_frequency"]
    audit["Experimental_outcome"] = audit["Composition"].map(EXPERIMENTAL_OUTCOMES)
    if audit["Experimental_outcome"].isna().any():
        missing = audit.loc[audit["Experimental_outcome"].isna(), "Composition"].tolist()
        raise ValueError(f"Missing experimental outcome text for: {missing}")

    columns = [
        "Composition",
        "Order",
        "Basin_score",
        "Basin_rank_with_pool",
        "Predicted_EFA",
        "EFA_rank_with_Eq3_pool",
        "Eq5_selection_frequency_percent",
        "Experimental_outcome",
    ]
    table = audit[columns].copy()
    table["_display_order"] = table["Composition"].map(
        {composition: index for index, composition in enumerate(DISPLAY_ORDER)}
    )
    table = table.sort_values("_display_order").drop(columns="_display_order").reset_index(drop=True)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CSV_OUT, index=False)

    rows = []
    for _, row in table.iterrows():
        rows.append(
            " & ".join(
                [
                    latex_composition(row["Composition"]),
                    row["Order"],
                    f"{row['Basin_score']:.3f}",
                    row["Basin_rank_with_pool"],
                    f"{row['Predicted_EFA']:.2f}",
                    row["EFA_rank_with_Eq3_pool"],
                    f"{row['Eq5_selection_frequency_percent']:.0f}\%",
                    row["Experimental_outcome"],
                ]
            )
            + r" \\"
        )

    tex = r"""% Requires: \usepackage{booktabs,tabularx,array,rotating}
\begin{sidewaystable*}[p]
\centering
\small
\caption{\textbf{Equation-defined prioritization and experimental outcomes of selected higher-order compositions.}
The Eq.~(2) basin score, $B(c)$, is the unweighted mean of the available descriptor-level evidence values, with larger values indicating stronger multi-objective basin support. Basin rank is calculated within the complete candidate pool of the same cation order; lower numerical ranks indicate stronger basin support, and the denominator gives the corresponding pool size. Predicted EFA is the deterministic output of the manuscript-consistent GNN checkpoint. Conditional EFA rank is calculated only among candidates retained by Eq.~(3) at the reference value $\rho=0.30$, with lower ranks indicating higher predicted EFA. Eq.~(5) grid frequency is the percentage of 25 tested $(\rho,\eta)$ combinations for which a candidate is shortlisted, using $\rho\in\{0.10,0.20,0.30,0.40,0.50\}$ and $\eta\in\{0.05,0.10,0.20,0.30,0.50\}$; it quantifies prioritization robustness rather than a calibrated probability of experimental success. The experimental-outcome column distinguishes HESS-matrix formation, nominal-cation incorporation and processing-censored synthesis. The characterized Y-containing candidates have low Eq.~(5) grid frequencies and exclude Y from their measured HESS matrices, whereas the HEC10 Re+Sc candidate is retained across most settings and incorporates all ten nominal cations. The HEC9 Sc result further shows that the operator provides prioritization rather than deterministic success/failure classification.}
\label{tab:eq1_5_experimental_audit}
\setlength{\tabcolsep}{4.5pt}
\renewcommand{\arraystretch}{1.18}
\begin{tabularx}{\textwidth}{@{}l c c c c c c >{\raggedright\arraybackslash}X@{}}
\toprule
Candidate composition & Order & $B(c)$ & Basin rank & Predicted EFA & Conditional EFA rank ($\rho=0.30$) & Eq.~(5) grid frequency & Experimental outcome \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{sidewaystable*}
"""
    TEX_OUT.write_text(tex, encoding="utf-8")
    print(f"Wrote {CSV_OUT}")
    print(f"Wrote {TEX_OUT}")


if __name__ == "__main__":
    main()
