!pip install pymatgen
!pip install matminer

!pip install torch_geometric

import re
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from pymatgen.core import Composition

# matminer featurizers
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.composition.ion import IonProperty
from matminer.featurizers.composition.alloy import Miedema, WenAlloys, YangSolidSolution
from matminer.featurizers.composition.element import BandCenter, Stoichiometry, TMetalFraction
from matminer.featurizers.composition.orbital import AtomicOrbitals, ValenceOrbital

import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GlobalAttention, GraphNorm

class GCNEncoder(nn.Module):
    def __init__(self, in_channels=35, hidden_dim=50, num_layers=2, heads=1, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Initial layer
        self.convs.append(
            GATv2Conv(
                in_channels=in_channels - 1,
                out_channels=hidden_dim,
                heads=heads,
                dropout=dropout,
                edge_dim=1,
                concat=False
            )
        )
        # self.norms.append(nn.LayerNorm(hidden_dim * heads))

        # Additional layers
        for _ in range(1, num_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=False
                )
            )
            self.norms.append(GraphNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        # Attention pooling
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.att_pool = GlobalAttention(gate_nn=self.gate_nn)

        # Final readout MLP
        self.readout_mlp = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_features = x[:, :-1].float()
        num_nodes = x.size(0)
        src, tgt = edge_index

        edge_attr = torch.zeros(src.size(0), 1, device=x.device, dtype=x.dtype)
        mask_src = (src == (num_nodes - 1))
        if mask_src.sum() > 0:
            edge_attr[mask_src] = x[tgt[mask_src], -1].unsqueeze(-1)
        mask_tgt = (tgt == (num_nodes - 1)) & ~(src == (num_nodes - 1))
        if mask_tgt.sum() > 0:
            edge_attr[mask_tgt] = x[src[mask_tgt], -1].unsqueeze(-1)
        edge_attr = edge_attr.float()

        out = node_features
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = out
            if i == 0:
                out = conv(out, edge_index,edge_attr)
            else:
                out = conv(out, edge_index)  # No edge_attr in subsequent layers

            out = norm(out)
            out = self.activation(out)
            out = self.dropout(out)

            if out.size() == residual.size():
                out = out + residual

        graph_emb = self.att_pool(out, batch=batch)
        out = self.readout_mlp(graph_emb)
        return out

# ------------------- Feature Preparation -------------------

# -------------------------------
# STEP 0: Define target elements and load compositions
# -------------------------------
target_elements = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Al", "Si", "C", "La"
}

input_file = "./data/HEC5_output.xlsx"
df_compositions = pd.read_excel(input_file)
compositions = df_compositions["Composition"].tolist()

# -------------------------------
# STEP 1: Initialize featurizers & generate element features
# -------------------------------
featurizers = {
    "ElementProperty_Magpie": ElementProperty.from_preset("magpie"),
    "IonProperty": IonProperty(),
    "Miedema": Miedema(struct_types="all"),
    "WenAlloys": WenAlloys(),
    "YangSolidSolution": YangSolidSolution(),
    "BandCenter": BandCenter(),
    "Stoichiometry": Stoichiometry(),
    "TMetalFraction": TMetalFraction(),
    "AtomicOrbitals": AtomicOrbitals(),
    "ValenceOrbital": ValenceOrbital()
}

# build element_features dict
element_features = {}
for el in target_elements:
    comp = Composition(el)
    feats = []
    for name, feat in featurizers.items():
        try:
            vals = feat.featurize(comp)
            # ensure numeric
            vals = [v if isinstance(v, (int, float)) else float("nan") for v in vals]
        except Exception:
            vals = [float("nan")] * len(feat.feature_labels())
        feats.extend(vals)
    element_features[el] = feats

# build labels list
all_feature_labels = []
for name, feat in featurizers.items():
    all_feature_labels += [f"{name}_{lbl}" for lbl in feat.feature_labels()]

# dataframe
element_features_df = pd.DataFrame.from_dict(
    element_features, orient="index", columns=all_feature_labels
)

# -------------------------------
# STEP 2: Clean up features
# -------------------------------
# remove statistical descriptors
statistical_keywords = ["maximum", "minimum", "mode", "avg_dev", "range"]
keep_cols = [
    c for c in element_features_df.columns
    if not any(k in c.lower() for k in statistical_keywords)
]
df_clean = element_features_df[keep_cols]

# drop specific unwanted columns
to_drop = [
    "IonProperty_compound possible", "WenAlloys_APE mean",
    "WenAlloys_Radii gamma", "WenAlloys_Mixing enthalpy",
    "Stoichiometry_0-norm", "Stoichiometry_2-norm",
    "Stoichiometry_3-norm", "Stoichiometry_5-norm",
    "Stoichiometry_7-norm", "Stoichiometry_10-norm",
    "WenAlloys_Atomic weight mean", "WenAlloys_Total weight",
    "ValenceOrbital_avg s valence electrons",
    "WenAlloys_Interant p electrons",
    "ValenceOrbital_avg p valence electrons",
    "ValenceOrbital_avg d valence electrons",
    "ValenceOrbital_avg f valence electrons",
    "WenAlloys_Interant s electrons"
]
df_clean = df_clean.drop(columns=to_drop, errors="ignore")

# drop any-with-NaN & all-zero cols
df_clean = df_clean.dropna(axis=1, how="any")
df_clean = df_clean.loc[:, (df_clean != 0).any(axis=0)]

# final labels & mapping
final_feature_labels = df_clean.columns.tolist()
print("--------------------------------------------------------------------------")
print(len(final_feature_labels))
element_to_features = {
    el: df_clean.loc[el].to_numpy()
    for el in df_clean.index
}

# -------------------------------
# STEP 3: Load Bond Information and Extend Element Features
# -------------------------------
bond_df = pd.read_excel("./data/Bond.xlsx")

def parse_formation_energy(s):
    try:
        return float(s.replace("eV/atom", "").strip())
    except:
        return float("nan")

# build raw dict
bond_energy = {
    row["Name"]: parse_formation_energy(str(row["Formation Energy per atom"]))
    for _, row in bond_df.iterrows()
}

# normalize to [0,1]
vals = [v for v in bond_energy.values() if not np.isnan(v)]
if vals:
    vmin, vmax = min(vals), max(vals)
    rng = vmax - vmin if vmax != vmin else 1.0
    for el in bond_energy:
        if not np.isnan(bond_energy[el]):
            bond_energy[el] = (bond_energy[el] - vmin) / rng

# append new label
new_label = "Bond_Formation_Energy_per_atom"
final_feature_labels.append(new_label)

# extend each element’s vector
for el in element_to_features:
    be = bond_energy.get(el, float("nan"))
    element_to_features[el] = np.append(element_to_features[el], be)

# -------------------------------
# STEP 4: Utility functions and prediction
# -------------------------------
def parse_sublattice_composition(comp_str):
    import re
    m = re.match(r'\((.*?)\)([A-Za-z]+)(\d*)', comp_str)
    if not m:
        raise ValueError(f"Invalid format: {comp_str}")
    cats, anion, cnt = m.groups()
    cations = re.findall(r'[A-Z][a-z]*', cats)
    comp_dict = {c: 1 for c in cations}
    comp_dict[anion] = int(cnt) if cnt else 1
    return Composition(comp_dict)

def create_edge_index(n):
    adj = torch.ones((n, n)) - torch.eye(n)
    return dense_to_sparse(adj)[0]

def create_edge_attr(x, edge_index):
    src, dst = edge_index
    n = x.size(0)
    last = n - 1
    ea = torch.zeros(src.size(0), 1, dtype=x.dtype, device=x.device)
    mask_s = (src == last)
    ea[mask_s] = x[dst[mask_s], -1].unsqueeze(-1)
    mask_d = (dst == last) & ~mask_s
    ea[mask_d] = x[src[mask_d], -1].unsqueeze(-1)
    return ea

def generate_features_from_composition(comp_str):
    """
    Build a [num_nodes × feature_size] tensor by stacking
    element_to_features[el] for each el in the sublattice.
    No atomic fractions appended here.
    """
    comp = parse_sublattice_composition(comp_str)
    els = [el.symbol for el in comp.elements]
    feat_list = []
    for el in els:
        if el not in element_to_features:
            raise KeyError(f"No features for element {el}")
        feat_list.append(element_to_features[el])
    arr = np.stack(feat_list, axis=0)  # shape: (num_nodes, feature_size)
    return torch.tensor(arr, dtype=torch.float32)


def predict_efa(comp_str, model, device):
    x = generate_features_from_composition(comp_str).to(device)
    edge_index = create_edge_index(x.size(0)).to(device)
    edge_attr  = create_edge_attr(x, edge_index)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        return model(data).item()

data = pd.read_excel("./data/Descriptors.xlsx")
data["Element"] = data["Binary Carbide"].str.extract(r"([A-Z][a-z]?)")
for col in ["Atomic Radius/ pm", "Valence e Count", "Electronegativity", "Formation Energy/ eV per atom"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data = data.dropna(subset=["Formation Energy/ eV per atom"])
formation_energy = data.groupby("Element")["Formation Energy/ eV per atom"].min().to_dict()
avg_props = data.groupby("Element")[["Atomic Radius/ pm", "Valence e Count", "Electronegativity"]].mean().to_dict("index")
magnetic_moment_lookup = {"Sc": 0.0, "Ti": 0.0, "V": 0.0, "Cr": 1.0, "Mn": 3.5, "Fe": 2.2, "Co": 1.7, "Ni": 0.6,
                          "Y": 0.0, "Zr": 0.0, "Nb": 0.0, "Mo": 0.0, "Hf": 0.0, "Ta": 0.0, "W": 0.0, "C": 0.0}
descriptor_data = {
    "formation_energy": formation_energy,
    "avg_props": avg_props,
    "magnetic_moment_lookup": magnetic_moment_lookup
}

print(descriptor_data)

# def multi_objective_score(composition_str, model, device, descriptor_data):
#     """
#     Predict EFA and calculate multiple physics-based descriptors
#     Return a multi-objective score based on all criteria
#     """

#     # 1. Predict EFA
#     predicted_efa = predict_efa(composition_str, model, device)

#     # 2. Calculate descriptors
#     comp = parse_sublattice_composition(composition_str)
#     elements = [el.symbol for el in comp.elements if el.symbol != "C"]
#     carbon_atoms = comp["C"]

#     metal_fractions = np.array([comp[el] for el in elements], dtype=float)
#     metal_fractions /= np.sum(metal_fractions)

#     element_formation_energy = descriptor_data["formation_energy"]
#     avg_props = descriptor_data["avg_props"]
#     magnetic_moment_lookup = descriptor_data["magnetic_moment_lookup"]

#     avg_formation = np.sum([
#         metal_fractions[i] * element_formation_energy[el]
#         for i, el in enumerate(elements)
#     ])
#     most_stable_binary = min([element_formation_energy[el] for el in elements])
#     formation_gap = avg_formation - most_stable_binary
#     radii = np.array([avg_props[el]["Atomic Radius/ pm"] for el in elements])
#     r_avg = np.sum(metal_fractions * radii)
#     radius_mismatch = np.sqrt(np.sum(metal_fractions * (1 - radii / r_avg) ** 2))
#     vec = np.sum([metal_fractions[i] * avg_props[el]["Valence e Count"] for i, el in enumerate(elements)])
#     vec += 4 * carbon_atoms
#     mdri = np.sum([metal_fractions[i] * abs(magnetic_moment_lookup.get(el, 0.0)) for i, el in enumerate(elements)])
#     electroneg = np.array([avg_props[el]["Electronegativity"] for el in elements])
#     x_avg = np.sum(metal_fractions * electroneg)
#     electronegativity_mismatch = np.sqrt(np.sum(metal_fractions * (electroneg - x_avg)**2))
#     carbon_affinity = max([element_formation_energy[el] for el in elements])

#     # 3. Multi-objective scoring
#     # Here you can define your *criteria* or *weighted scoring*
#     # You can make a simple hard-threshold filter or weighted formula

#     # Example: Normalize all objectives to [0,1] (better scoring when higher)
#     # These normalization ranges should be decided based on your dataset statistics
#     efa_score = predicted_efa / 150  # Assuming typical EFA ~ 0-150
#     formation_score = (-avg_formation + 2) / 2.5  # Normalize: more negative formation is better
#     gap_score = (1 - (formation_gap + 1) / 2)  # Smaller gap is better
#     radius_mismatch_score = 1 - min(radius_mismatch / 0.08, 1)  # Smaller mismatch better
#     mdri_score = 1 - min(mdri / 1.5, 1)  # Small MD_RI is better
#     electronegativity_mismatch_score = 1 - min(electronegativity_mismatch / 0.3, 1)  # Small mismatch better
#     carbon_affinity_score = (-carbon_affinity + 1) / 2  # More negative carbon affinity is better

#     # Weighted Sum
#     total_score = (
#         0.3 * efa_score +
#         0.2 * formation_score +
#         0.1 * gap_score +
#         0.1 * radius_mismatch_score +
#         0.1 * mdri_score +
#         0.1 * electronegativity_mismatch_score +
#         0.1 * carbon_affinity_score
#     )

#     result = {
#         "Predicted EFA": predicted_efa,
#         "Average Formation Enthalpy (eV/atom)": avg_formation,
#         "Formation Gap (eV/atom)": formation_gap,
#         "Atomic Radius Mismatch": radius_mismatch,
#         "Valence Electron Count": vec,
#         "Magnetic Disorder Risk Index": mdri,
#         "Electronegativity Mismatch": electronegativity_mismatch,
#         "Carbon Affinity Index": carbon_affinity,
#         "Total Multi-Objective Score": total_score
#     }

#     return result

# --- Safe imports and config ---
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")  # Safe backend for servers
import matplotlib.pyplot as plt
import numpy as np

def multi_objective_score_with_plot(composition_str, model, device, descriptor_data,
                                    predict_efa, parse_sublattice_composition,
                                    save_path="multi_objective_score.png"):
    """
    Predict EFA and calculate multiple physics-based descriptors.
    Visualize and save multi-objective weighted score breakdown.
    """

    # Predict EFA
    predicted_efa = predict_efa(composition_str, model, device)

    # Parse composition
    comp = parse_sublattice_composition(composition_str)
    elements = [el.symbol for el in comp.elements if el.symbol != "C"]
    carbon_atoms = comp["C"]

    metal_fractions = np.array([comp[el] for el in elements], dtype=float)
    metal_fractions = metal_fractions / np.sum(metal_fractions)

    # Load descriptor sources
    element_formation_energy = descriptor_data["formation_energy"]
    avg_props = descriptor_data["avg_props"]
    magnetic_moment_lookup = descriptor_data["magnetic_moment_lookup"]

    # Compute descriptors
    avg_formation = np.sum([
        metal_fractions[i] * element_formation_energy[el]
        for i, el in enumerate(elements)
    ])
    most_stable_binary = min([element_formation_energy[el] for el in elements])
    formation_gap = avg_formation - most_stable_binary

    radii = np.array([avg_props[el]["Atomic Radius/ pm"] for el in elements])
    r_avg = np.sum(metal_fractions * radii)
    radius_mismatch = np.sqrt(np.sum(metal_fractions * (1 - radii / r_avg) ** 2))

    vec = np.sum([metal_fractions[i] * avg_props[el]["Valence e Count"] for i, el in enumerate(elements)])
    vec += 4 * carbon_atoms

    mdri = np.sum([metal_fractions[i] * abs(magnetic_moment_lookup.get(el, 0.0)) for i, el in enumerate(elements)])

    electroneg = np.array([avg_props[el]["Electronegativity"] for el in elements])
    x_avg = np.sum(metal_fractions * electroneg)
    electronegativity_mismatch = np.sqrt(np.sum(metal_fractions * (electroneg - x_avg)**2))

    carbon_affinity = max([element_formation_energy[el] for el in elements])

    # Scaled scores
    # scores = {
    #     "EFA Score": predicted_efa / 150,
    #     "Formation Score": (-avg_formation + 2) / 2.5,
    #     "Gap Score": (1 - (formation_gap + 1) / 2),
    #     "Radius Mismatch Score": 1 - min(radius_mismatch / 0.08, 1),
    #     "MD_RI Score": 1 - min(mdri / 1.5, 1),
    #     "Electronegativity Mismatch Score": 1 - min(electronegativity_mismatch / 0.3, 1),
    #     "Carbon Affinity Score": (-carbon_affinity + 1) / 2
    # }

    # weights = {
    #     "EFA Score": 0.3,
    #     "Formation Score": 0.2,
    #     "Gap Score": 0.1,
    #     "Radius Mismatch Score": 0.1,
    #     "MD_RI Score": 0.1,
    #     "Electronegativity Mismatch Score": 0.1,
    #     "Carbon Affinity Score": 0.1
    # }

    efa_max = 150.0      # typical max EFA
    h_min, h_max = -1.5, 0.0   # formation enthalpy usually between -1.5 and 0 eV/atom
    gap_min, gap_max = 0.0, 1.0
    rad_min, rad_max = 0.0, 0.10
    vec_min, vec_max = 20.0, 40.0  # valence counts across your compositions
    mdri_max = 1.5
    elec_mis_max = 0.3
    ca_min, ca_max = -1.0, 1.0     # carbon affinity in eV

    # 2) build normalized “scores” in [0,1] (higher=better)
    scores = {
        "EFA Score": np.clip(predicted_efa / efa_max, 0, 1),
        "Formation Score": np.clip(( -avg_formation - h_min ) / (h_max - h_min), 0, 1),
        "Gap Score": np.clip(1 - (formation_gap - gap_min)/(gap_max - gap_min), 0, 1),
        "Radius Mismatch Score": np.clip(1 - (radius_mismatch - rad_min)/(rad_max - rad_min), 0, 1),
        "Valence Electron Count Score": np.clip((vec - vec_min)/(vec_max - vec_min), 0, 1),
        "MD_RI Score": np.clip(1 - mdri/mdri_max, 0, 1),
        "Electronegativity Mismatch Score": np.clip(1 - electronegativity_mismatch/elec_mis_max, 0, 1),
        "Carbon Affinity Score": np.clip(( -carbon_affinity - ca_min )/(ca_max - ca_min), 0, 1)
    }

    # 3) assign new weights (sum = 1.0)
    weights = {
        "EFA Score":                       0.25,
        "Formation Score":                 0.20,
        "Gap Score":                       0.10,
        "Radius Mismatch Score":           0.10,
        "Valence Electron Count Score":    0.10,
        "MD_RI Score":                     0.05,
        "Electronegativity Mismatch Score":0.10,
        "Carbon Affinity Score":           0.10,
    }

    weighted_scores = [scores[k] * weights[k] for k in scores]
    labels = list(scores.keys())

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.barh(labels, weighted_scores, color='royalblue')
    plt.xlabel("Weighted Score Contribution")
    plt.title(f"Multi-Objective Score Breakdown for {composition_str}")
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # --- Return descriptor summary ---
    return {
        "Predicted EFA": predicted_efa,
        "Average Formation Enthalpy (eV/atom)": avg_formation,
        "Formation Gap (eV/atom)": formation_gap,
        "Atomic Radius Mismatch": radius_mismatch,
        "Valence Electron Count": vec,
        "Magnetic Disorder Risk Index": mdri,
        "Electronegativity Mismatch": electronegativity_mismatch,
        "Carbon Affinity Index": carbon_affinity,
        "Total Multi-Objective Score": sum(weighted_scores)
    }


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCNEncoder(
    in_channels=35,
    hidden_dim=52,
    num_layers=8,
    heads=4,
    dropout=0.3
).to(device)



model.load_state_dict(torch.load("./checkpoints/_model_26999_after_7l.pth", map_location=device))
model.eval()

composition = "(HfMoNbTaTiV)C6"


score = multi_objective_score_with_plot(composition, model, device, descriptor_data,predict_efa, parse_sublattice_composition)
print("-----------------------------------------------------------------")
print(score)

from itertools import combinations
import pandas as pd

# Sample set of metal elements (can be customized)
all_metals = [
    "Ti","V","Cr","Zr","Nb","Mo","Hf","Ta","W"
]

# Function to generate HECn compositions
def generate_higher_order_carbides(elements, min_order=6, max_order=None):
    if max_order is None:
        max_order = len(elements)
    compositions = []
    for n in range(min_order, max_order + 1):
        for combo in combinations(sorted(elements), n):
            metal_part = ''.join(combo)
            comp = f"({metal_part})C{n}"
            compositions.append(comp)
    return compositions

# Generate HEC6 compositions
compositions = generate_higher_order_carbides(all_metals, min_order=6, max_order=9)

print(compositions)

score = []
for comp in compositions:
  score.append(multi_objective_score_with_plot(comp, model, device, descriptor_data,predict_efa, parse_sublattice_composition))

print(score)



def convert_scores_to_dataframe(scores, compositions):
    # Convert each dict to pandas DataFrame
    df_scores = pd.DataFrame(scores)

    # Insert compositions as a new column
    df_scores.insert(0, 'Composition', compositions)

    # Convert any np.float64 to native Python floats
    for col in df_scores.select_dtypes(include=[np.float64]).columns:
        df_scores[col] = df_scores[col].astype(float)

    return df_scores

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Example dataframe creation (user should input their actual dataset)
# data = pd.DataFrame({
#     "Composition": ["(CrHfMoNbTaTiVWZr)C9", "(CrHfMoNbTiVWZr)C8", "(CrHfMoNbTaTiVW)C7", "(CrHfMoNbTaTiW)C7", "(CrHfMoNbTaTiV)C7"],
#     "Predicted EFA": [99.90, 64.06, 74.77, 73.15, 73.67],
#     "Avg Formation Enthalpy": [-0.390, -0.839, -0.696, -0.616, -0.535],
#     "Formation Gap": [0.288, 0.284, 0.427, 0.507, 0.430],
#     "Atomic Radius Mismatch": [0.070, 0.074, 0.069, 0.051, 0.058],
#     "Multi-Objective Score": [0.583, 0.637, 0.577, 0.574, 0.564]
# })

data = convert_scores_to_dataframe(score, compositions)

sns.set_theme(style="whitegrid", font_scale=1.2)

# Plot 1: Predicted Entropy Forming Ability (EFA)
plt.figure(figsize=(10, 6))
sns.barplot(x="Predicted EFA", y="Composition", data=data.sort_values("Predicted EFA", ascending=False).head(20), palette="viridis")
plt.title("Top Compositions by Predicted EFA")
plt.xlabel("Predicted EFA")
plt.ylabel("Composition")
plt.tight_layout()
plt.show()

# Plot 2: Avg Formation Enthalpy
plt.figure(figsize=(10, 6))
sns.barplot(x="Average Formation Enthalpy (eV/atom)", y="Composition", data=data.sort_values("Average Formation Enthalpy (eV/atom)").head(20), palette="rocket")
plt.title("Most Thermodynamically Stable Compositions (ΔH)")
plt.xlabel("Formation Enthalpy (eV/atom)")
plt.ylabel("Composition")
plt.tight_layout()
plt.show()

# Plot 3: Formation Gap
plt.figure(figsize=(10, 6))
sns.barplot(x="Formation Gap (eV/atom)", y="Composition", data=data.sort_values("Formation Gap (eV/atom)").head(20), palette="crest")
plt.title("Top Compositions by Minimal Formation Gap")
plt.xlabel("Formation Gap (eV/atom)")
plt.ylabel("Composition")
plt.tight_layout()
plt.show()

# Plot 4: Atomic Radius Mismatch (δ)
plt.figure(figsize=(10, 6))
sns.barplot(x="Atomic Radius Mismatch", y="Composition", data=data.sort_values("Atomic Radius Mismatch").head(20), palette="mako")
plt.title("Compositions with Lowest Atomic Radius Mismatch")
plt.xlabel("Atomic Radius Mismatch (δ)")
plt.ylabel("Composition")
plt.tight_layout()
plt.show()

# Plot 5: Total Multi-Objective Score
plt.figure(figsize=(10, 6))
sns.barplot(x="Total Multi-Objective Score", y="Composition", data=data.sort_values("Total Multi-Objective Score", ascending=False).head(20), palette="flare")
plt.title("Compositions by Aggregate Multi-Objective Score")
plt.xlabel("Aggregate Stability Score")
plt.ylabel("Composition")
plt.tight_layout()
plt.show()

data=data.sort_values("Total Multi-Objective Score", ascending=False).head(20)
print(data)

def get_composition_ranks(df, comp_name):
    """
    Given a DataFrame `df` with a 'Composition' column and
    the metrics below, print the rank of `comp_name` for each metric
    (1 = best), along with the total number of compositions.
    """
    # Define your metrics and whether higher values are better
    metrics = {
    "Predicted EFA": {
        "ascending": False,
        "better": "higher"
    },
    "Average Formation Enthalpy (eV/atom)": {
        "ascending": True,
        "better": "more negative"
    },
    "Formation Gap (eV/atom)": {
        "ascending": True,
        "better": "lower"
    },
    "Atomic Radius Mismatch": {
        "ascending": True,
        "better": "lower"
    },
    "Valence Electron Count": {
        "ascending": False,
        "better": "higher"
    },
    "Magnetic Disorder Risk Index": {
        "ascending": True,
        "better": "lower"
    },
    "Electronegativity Mismatch": {
        "ascending": True,
        "better": "lower"
    },
    "Carbon Affinity Index": {
        "ascending": True,
        "better": "more negative"
    },
    "Total Multi-Objective Score": {
        "ascending": False,
        "better": "higher"
    },
}

    n = len(df)
    if comp_name not in df['Composition'].values:
        raise ValueError(f"Composition '{comp_name}' not found in DataFrame.")

    print(f"Ranking for {comp_name}")

    for metric, opts in metrics.items():
        # sort and reset index so rank = position + 1
        sorted_df = df.sort_values(metric, ascending=opts['ascending']).reset_index(drop=True)
        # find the row index of our composition
        rank = sorted_df.index[sorted_df['Composition'] == comp_name][0] + 1
        # print(f" • {metric}: rank {rank} / {n}    ({opts['better']} is better)")
        print(f" • {metric}: rank {rank} / {n} ")


get_composition_ranks(data, "(CrNbTaTiVZr)C6")

def convert_scores_to_dataframe(scores, compositions, selected_compositions=None):
    # Convert each dict to pandas DataFrame
    df_scores = pd.DataFrame(scores)

    # Insert compositions as a new column
    df_scores.insert(0, 'Composition', compositions)

    # Convert any np.float64 to native Python floats
    for col in df_scores.select_dtypes(include=[np.float64]).columns:
        df_scores[col] = df_scores[col].astype(float)

    # If selected_compositions is provided, filter the DataFrame
    if selected_compositions is not None:
        df_scores = df_scores[df_scores['Composition'].isin(selected_compositions)].reset_index(drop=True)

    return df_scores

# Re-import libraries due to reset state
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Manually recreating data based on user's provided example
# (only top 6 based on "Total Multi-Objective Score" provided explicitly here)
selected_compositions = [
    "(HfNbTaTiVZr)C6",
    "(HfMoNbTaTiZr)C6",
    "(MoNbTaTiVZr)C6",
    "(CrMoNbTaVW)C6",
    "(HfMoNbTaTiV)C6",
    "(CrMoNbTaTiV)C6"
]


top6_data = convert_scores_to_dataframe(score, compositions,selected_compositions)

# Normalize data for radar plot
metrics = [
    'Predicted EFA', 'Average Formation Enthalpy (eV/atom)', 'Formation Gap (eV/atom)',
    'Atomic Radius Mismatch', 'Valence Electron Count', 'Electronegativity Mismatch',
    'Carbon Affinity Index'
]

normalized_data = top6_data.copy()
for metric in metrics:
    min_val = normalized_data[metric].min()
    max_val = normalized_data[metric].max()
    normalized_data[metric] = (normalized_data[metric] - min_val) / (max_val - min_val)

# Radar plot function
def radar_plot(data, labels, title, ax):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    data = np.concatenate((data,[data[0]]))
    angles += angles[:1]

    ax.fill(angles, data, alpha=0.25)
    ax.plot(angles, data, linewidth=2, linestyle='solid')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title(title, size=12, color='blue', y=1.1)

# Plot radar charts
fig, axs = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(polar=True))

axs = axs.flatten()
for i, row in normalized_data.iterrows():
    radar_plot(row[metrics].values, metrics, row['Composition'], axs[i])

plt.tight_layout()
plt.show()

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 64}

matplotlib.rc('font', **font)

# Plot radar charts
fig, axs = plt.subplots(2, 3,figsize=(18, 10),subplot_kw=dict(polar=True))  # width=90 mm (~3.54 inches)


axs = axs.flatten()
for i, row in normalized_data.iterrows():
    radar_plot(row[metrics].values, metrics, row['Composition'], axs[i])

plt.tight_layout()

# Save figure with high resolution
fig.savefig('radar_plots_top6.png', dpi=600, bbox_inches='tight')

# Display plot
plt.show()

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming `scores` is your list of score dicts and `compositions` the matching list of strings
def convert_scores_to_dataframe(scores, compositions):
    df = pd.DataFrame(scores)
    df.insert(0, 'Composition', compositions)
    # Convert any np.float64 to native Python floats
    for col in df.select_dtypes(include=[np.float64]).columns:
        df[col] = df[col].astype(float)
    return df

# Create the DataFrame
df = convert_scores_to_dataframe(score, compositions)

# Parse element inclusion
all_elements = sorted({el for comp in compositions
                       for el in re.findall(r'[A-Z][a-z]?', comp.split(')')[0][1:])})
for el in all_elements:
    df[el] = df['Composition'].apply(lambda c: el in re.findall(r'[A-Z][a-z]?', c.split(')')[0][1:]))

# Metrics to analyze
metrics = [
    'Predicted EFA', 'Average Formation Enthalpy (eV/atom)', 'Formation Gap (eV/atom)',
    'Atomic Radius Mismatch', 'Valence Electron Count', 'Electronegativity Mismatch',
    'Carbon Affinity Index', 'Total Multi-Objective Score'
]

# Compute effect of inclusion vs exclusion for each element and metric
effect = pd.DataFrame(index=all_elements, columns=metrics, dtype=float)
for el in all_elements:
    included = df[df[el]]
    excluded = df[~df[el]]
    effect.loc[el] = included[metrics].mean() - excluded[metrics].mean()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.set(context='notebook', style='white')
ax = sns.heatmap(effect, cmap='coolwarm', center=0, annot=True, fmt=".2f",
                 cbar_kws={'label': 'Mean Difference'}, linewidths=0.5)
ax.set_title('Figure X | Effect of Element Inclusion on Key Metrics', fontsize=14, fontweight='bold', pad=16)
ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Element', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save high-resolution figure
plt.savefig('element_inclusion_effects.png', dpi=600, bbox_inches='tight')
plt.show()

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# assume df_scores is your full DataFrame created by convert_scores_to_dataframe()
# and contains columns ['Composition', <your metrics...>]

# 1) parse out element lists
df = df.copy()
df['Elements'] = df['Composition'].str.extract(r'^\((.*?)\)')[0] \
                   .apply(lambda s: re.findall(r'[A-Z][a-z]?', s))

# 2) define metrics of interest
metrics = [
    'Predicted EFA',
    'Average Formation Enthalpy (eV/atom)',
    'Formation Gap (eV/atom)',
    'Atomic Radius Mismatch',
    'Valence Electron Count',
    'Electronegativity Mismatch',
    'Total Multi-Objective Score'
]

# 3) build effect‐size and p‐value tables
elements = sorted(set(sum(df['Elements'], [])))
effect_df = pd.DataFrame(index=elements, columns=metrics, dtype=float)
pval_df   = pd.DataFrame(index=elements, columns=metrics, dtype=float)

for el in elements:
    inc = df[df['Elements'].apply(lambda L: el in L)]
    exc = df[df['Elements'].apply(lambda L: el not in L)]
    for m in metrics:
        effect_df.loc[el, m] = inc[m].mean() - exc[m].mean()
        _, pval_df.loc[el, m]   = ttest_ind(inc[m], exc[m], equal_var=False)

# 4) build annotation frame: two‐decimal Δ plus “*” if p<.05
annot = effect_df.round(2).astype(str)
for i in annot.index:
    for j in annot.columns:
        if pval_df.loc[i,j] < 0.05:
            annot.loc[i,j] += '*'

# 5) plot a clustered, diverging heatmap
sns.set(context='paper',
        style='white',
        font='serif',
        font_scale=1.1,
       )
cmap = sns.diverging_palette(220, 20, as_cmap=True)

g = sns.clustermap(effect_df,
                   cmap=cmap,
                   center=0,
                   row_cluster=True,
                   col_cluster=False,
                   annot=annot,
                   fmt='',
                   linewidths=0.5,
                   figsize=(12,10))

# 6) polish labels & title
g.ax_heatmap.set_xlabel('Metrics',    fontsize=12)
g.ax_heatmap.set_ylabel('Alloying Elements',    fontsize=12)
# g.fig.suptitle('Impact of Element Inclusion on Stability Metrics (Δ = mean_in – mean_out)',
#                fontsize=16, y=1.02)

# 7) save high-quality figure
plt.savefig('element_effects_heatmap.png', dpi=600, bbox_inches='tight')
plt.savefig('element_effects_heatmap.pdf', bbox_inches='tight')  # vector version for publication

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# assume df_scores is your full DataFrame created by convert_scores_to_dataframe()
df = df.copy()

# 1) parse out element lists
df['Elements'] = df['Composition'].str.extract(r'^\((.*?)\)')[0] \
                   .apply(lambda s: re.findall(r'[A-Z][a-z]?', s))

# 2) define metrics of interest
metrics = [
    'Predicted EFA',
    'Average Formation Enthalpy (eV/atom)',
    'Formation Gap (eV/atom)',
    'Atomic Radius Mismatch',
    'Valence Electron Count',
    'Electronegativity Mismatch',
    'Total Multi-Objective Score'
]

# 3) build effect‐size and p‐value tables
elements = sorted(set(sum(df['Elements'], [])))
effect_df = pd.DataFrame(index=elements, columns=metrics, dtype=float)
pval_df   = pd.DataFrame(index=elements, columns=metrics, dtype=float)

for el in elements:
    inc = df[df['Elements'].apply(lambda L: el in L)]
    exc = df[df['Elements'].apply(lambda L: el not in L)]
    for m in metrics:
        effect_df.loc[el, m] = inc[m].mean() - exc[m].mean()
        _, pval_df.loc[el, m]   = ttest_ind(inc[m], exc[m], equal_var=False)

# 4) build annotation frame: two‐decimal Δ plus “*” if p<.05
annot = effect_df.round(2).astype(str)
for i in annot.index:
    for j in annot.columns:
        if pval_df.loc[i,j] < 0.05:
            annot.loc[i,j] += '*'

# 5) plot a clustered, diverging heatmap
sns.set(context='paper',
        style='white',
        font='serif',
        font_scale=1.2,
       )
cmap = sns.diverging_palette(220, 20, as_cmap=True)

g = sns.clustermap(
    effect_df,
    cmap=cmap,
    center=0,
    row_cluster=False,
    col_cluster=False,
    annot=annot,
    fmt='',
    linewidths=0.5,
    figsize=(12, 10),
    cbar_kws={
        'orientation': 'horizontal'
    }
)

# 6) move colorbar to top and style it
# g.cbar_kws places the bar below by default; reposition it:
cbar = g.cax
cbar.set_position([0.30, 0.90, 0.60, 0.02])
cbar.xaxis.set_ticks_position('top')
cbar.xaxis.set_label_position('top')
cbar.tick_params(labelsize=10)

# 7) polish labels & title
g.ax_heatmap.set_xlabel('Metric',     fontsize=14, labelpad=10)
g.ax_heatmap.set_ylabel('Alloying Elements',     fontsize=14, labelpad=10)
g.ax_heatmap.set_xticklabels(
    g.ax_heatmap.get_xticklabels(),
    rotation=45,
    ha='right',
    fontsize=11
)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=11)
g.cax.set_title('Δ Mean Metric (Inclusion – Exclusion)', fontsize=12, pad=12)

# add a super‐title
# g.fig.suptitle(
#     'Impact of Element Inclusion on Stability Metrics',
#     fontsize=16,
#     y=1.02
# )

# 8) save high-quality figure
plt.savefig('element_effects_heatmap.png', dpi=600, bbox_inches='tight')
plt.savefig('element_effects_heatmap.pdf', bbox_inches='tight')

import pandas as pd
import numpy as np

def compute_descriptor_scores(df):
    """
    Compute individual normalized scores for each descriptor.
    Assumes df has columns:
      - 'Predicted EFA'
      - 'Average Formation Enthalpy (eV/atom)'
      - 'Formation Gap (eV/atom)'
      - 'Atomic Radius Mismatch'
      - 'Magnetic Disorder Risk Index'
      - 'Electronegativity Mismatch'
      - 'Carbon Affinity Index'
    """
    scores = pd.DataFrame(index=df.index)
    scores['EFA_score'] = df['Predicted EFA'] / 150.0
    scores['Formation_score'] = (-df['Average Formation Enthalpy (eV/atom)'] + 2) / 2.5
    scores['Gap_score'] = 1 - (df['Formation Gap (eV/atom)'] + 1) / 2
    scores['Radius_mismatch_score'] = 1 - np.minimum(df['Atomic Radius Mismatch'] / 0.08, 1)
    scores['MDRI_score'] = 1 - np.minimum(df['Magnetic Disorder Risk Index'] / 1.5, 1)
    scores['Elec_mismatch_score'] = 1 - np.minimum(df['Electronegativity Mismatch'] / 0.3, 1)
    scores['Carbon_affinity_score'] = (-df['Carbon Affinity Index'] + 1) / 2
    return scores

# Example: assume `df_all` is your full DataFrame with raw descriptors plus a 'Composition' column
# df_all = convert_scores_to_dataframe(scores_list, compositions_list)

# Baseline weights
baseline_weights = {
    'EFA_score': 0.30,
    'Formation_score': 0.20,
    'Gap_score': 0.10,
    'Radius_mismatch_score': 0.10,
    'MDRI_score': 0.10,
    'Elec_mismatch_score': 0.10,
    'Carbon_affinity_score': 0.10
}

# Compute raw descriptor scores
df_scores = compute_descriptor_scores(df)

# Compute baseline total and Top 10
baseline_total = df_scores.dot(pd.Series(baseline_weights))
baseline_top10 = df.loc[baseline_total.nlargest(10).index, 'Composition'].tolist()

# Sensitivity analysis: ±10% perturbation of each weight
results = {}
for metric, w in baseline_weights.items():
    for change in (-0.1, 0.1):
        # Perturb weight
        w_new = baseline_weights.copy()
        w_new[metric] = w * (1 + change)
        # Re-normalize to sum to 1
        total_w = sum(w_new.values())
        w_new = {k: v/total_w for k, v in w_new.items()}

        # Compute new total scores and Top 10
        total_scores = df_scores.dot(pd.Series(w_new))
        top10 = df.loc[total_scores.nlargest(10).index, 'Composition'].tolist()
        overlap = len(set(top10).intersection(baseline_top10))

        results[(metric, change)] = {
            'weights': w_new,
            'top10': top10,
            'overlap_with_baseline': overlap
        }

# Display overlap results
print("Sensitivity of Top 10 to ±10% weight perturbations:")
for (metric, change), info in results.items():
    sign = '+' if change > 0 else '-'
    print(f" {metric} {sign}10%: Overlap = {info['overlap_with_baseline']}/10 compositions")

print(baseline_top10)

