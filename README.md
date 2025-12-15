# HESS Discovery
**A screening-and-prediction framework for scalable discovery of high-order high-entropy carbides.**

HESS Discovery identifies promising transition-metal carbide compositions from a combinatorial design space using two complementary, order-agnostic signals: **graph-based EFA prediction** and **physics-informed multi-objective screening**.

## Workflow

<div style="text-align:center">
  <img src="assets/HEC_Paper_Main.png" alt="Project Workflow Diagram" width="700" style="max-width:100%;height:auto;">
</div>



### Graph neural network EFA prediction
Candidate compositions are represented as **permutation-invariant element graphs** (metal species plus a carbon node). A **graph-attention neural network** predicts **entropy-forming ability (EFA)** directly from the composition graph, enabling transfer from **quinary (HEC5)** training data to **higher-order** composition spaces.

### Physics-informed multi-objective screening
In parallel, each composition is embedded in a **multi-objective descriptor space** designed to capture competing constraints relevant to synthesizability and phase stability, including:
- **Thermodynamic competition and stability:** \(\Delta H_{\mathrm{carb}}\)
- **Hume–Rothery-type compatibility:** \(\delta r\), \(\delta\chi\), \(\langle \mathrm{VEC} \rangle\) window
- **Carbide bonding and carbon affinity (“weakest-link”):** \(H_{\mathrm{weak}}\)
- **Segregation/ordering and magnetic-disorder risk:** MSI, MDRI
- **Processing constraints:** e.g., minimum precursor carbide melting point \(T_m^{\min}\)

A **Gaussian mixture model (GMM)** over this objective space yields a **posterior solid-solution likelihood** and delineates a coherent **solid-solution basin**.

## Candidate selection principle
Final prioritization is based on the **intersection** of:
- **basin membership** (global posterior likelihood),
- **per-objective evidence** across screening criteria, and
- **high predicted EFA** from the GNN.

This combined decision rule reduces false positives that arise when candidates are selected using isolated metrics, and produces a **ranked shortlist** for experimental synthesis and characterization.






