!pip install matminer pymatgen

import numpy as np
import pandas as pd
from pymatgen.core import Composition
import torch

# Import matminer featurizers
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.composition.ion import IonProperty
from matminer.featurizers.composition.alloy import Miedema, WenAlloys, YangSolidSolution
from matminer.featurizers.composition.element import BandCenter, Stoichiometry, TMetalFraction
from matminer.featurizers.composition.orbital import AtomicOrbitals, ValenceOrbital

# -------------------------------
# STEP 0: Define target elements and load compositions
# -------------------------------

target_elements = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Al", "Si", "C", "La"
}

# Load HEC5 compositions
input_file = "./data/HEC5_output.xlsx"
df_compositions = pd.read_excel(input_file)
compositions = df_compositions["Composition"].tolist()

# -------------------------------
# STEP 1: Initialize Matminer featurizers and generate element features
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

# Dictionary to hold each target element’s features
element_features = {}
for element in target_elements:
    element_obj = Composition(element)
    combined_features = []
    for featurizer_name, featurizer in featurizers.items():
        try:
            features = featurizer.featurize(element_obj)
            numeric_features = [f if isinstance(f, (int, float)) else float('nan') for f in features]
            combined_features.extend(numeric_features)
        except Exception as e:
            print(f"Error processing {featurizer_name} for element {element}: {e}")
            combined_features.extend([float('nan')] * len(featurizer.feature_labels()))
    element_features[element] = combined_features

# Combine feature labels from all featurizers
all_feature_labels = []
for featurizer_name, featurizer in featurizers.items():
    try:
        all_feature_labels.extend([f"{featurizer_name}_{label}" for label in featurizer.feature_labels()])
    except Exception as e:
        print(f"Error processing labels for {featurizer_name}: {e}")

# Convert element features to DataFrame
element_features_df = pd.DataFrame.from_dict(element_features, orient="index", columns=all_feature_labels)

# -------------------------------
# STEP 2: Clean up features
# -------------------------------

# Remove statistical features
statistical_keywords = ["maximum", "minimum", "mode", "avg_dev", "range"]
filtered_columns = [
    col for col in element_features_df.columns
    if not any(keyword in col.lower() for keyword in statistical_keywords)
]
filtered_element_features_df = element_features_df[filtered_columns]

# Remove specific features
features_to_remove = [
    "IonProperty_compound possible", "WenAlloys_APE mean", "WenAlloys_Radii gamma", "WenAlloys_Mixing enthalpy",
    "Stoichiometry_0-norm", "Stoichiometry_2-norm", "Stoichiometry_3-norm", "Stoichiometry_5-norm",
    "Stoichiometry_7-norm", "Stoichiometry_10-norm", "WenAlloys_Atomic weight mean","WenAlloys_Total weight", 
    "ValenceOrbital_avg s valence electrons", "WenAlloys_Interant p electrons", 
    "ValenceOrbital_avg p valence electrons", "ValenceOrbital_avg d valence electrons", 
    "ValenceOrbital_avg f valence electrons", "WenAlloys_Interant s electrons"
]

filtered_element_features_df = filtered_element_features_df.drop(columns=features_to_remove, errors='ignore')

# Remove features with any NaN values
filtered_element_features_df = filtered_element_features_df.dropna(axis=1, how="any")

# Remove features that are all zeros
non_zero_columns = (filtered_element_features_df != 0).any(axis=0)
filtered_element_features_df = filtered_element_features_df.loc[:, non_zero_columns]

# Save final feature labels
final_feature_labels = filtered_element_features_df.columns.tolist()

# Map elements to their numeric-only features
element_to_features = {
    element: filtered_element_features_df.loc[element].to_numpy()
    for element in filtered_element_features_df.index
}


# -------------------------------
# STEP 3: Load Bond Information and Extend Element Features
# -------------------------------

# Load the bond information Excel file. Make sure the filename matches your file.
# -------------------------------
# STEP 3: Load Bond Information and Extend Element Features
# -------------------------------

# Load the bond information Excel file. Make sure the filename matches your file.
bond_info_file = "./data/Bond.xlsx"
bond_df = pd.read_excel(bond_info_file)

# Create a dictionary mapping element symbol to the formation energy (as a float)
# The "Formation Energy per atom" column might include text like "-0.423 eV/atom",
# so we strip off the non-numeric part.
def parse_formation_energy(energy_str):
    try:
        # Remove any 'eV/atom' and whitespace, then convert to float.
        return float(energy_str.replace("eV/atom", "").strip())
    except Exception as e:
        print(f"Error parsing formation energy '{energy_str}': {e}")
        return float('nan')

# Create dictionary: key = element symbol, value = formation energy (or nan if parsing fails)
bond_energy_dict = {}
for idx, row in bond_df.iterrows():
    element = row["Name"]
    energy_val = parse_formation_energy(str(row["Formation Energy per atom"]))
    bond_energy_dict[element] = energy_val

# --- Normalize the formation energy values to the range [0, 1] ---
# First, extract all valid (non-NaN) energy values.
valid_values = [v for v in bond_energy_dict.values() if not np.isnan(v)]
if valid_values:
    min_energy = min(valid_values)
    max_energy = max(valid_values)
    range_energy = max_energy - min_energy if max_energy != min_energy else 1.0
    # Update each energy value with its normalized version.
    for element in bond_energy_dict:
        if not np.isnan(bond_energy_dict[element]):
            # bond_energy_dict[element] = (bond_energy_dict[element] - min_energy) / range_energy
            bond_energy_dict[element] = bond_energy_dict[element]
# ------------------------------------------------------------

# Option: if you want to include additional bond data (like the "Bond" string), you might encode it here.
# For this example, we only add the normalized formation energy value.

# Now, extend each element’s feature vector by appending the normalized formation energy value.
# Also update the final feature labels to include the new bond feature.
new_feature_label = "Bond_Formation_Energy_per_atom"
final_feature_labels.append(new_feature_label)

for element in element_to_features.keys():
    # If the element is present in the bond info, append its normalized formation energy;
    # otherwise, append NaN.
    bond_energy = bond_energy_dict.get(element, float('nan'))
    original_vector = element_to_features[element]
    # Use np.append to add the bond energy as an additional feature
    extended_vector = np.append(original_vector, bond_energy)
    element_to_features[element] = extended_vector


# -------------------------------
# STEP 4: Generate the Composition Features Array (3D) including Bond Information
# -------------------------------

# For each composition, we create a list of feature vectors (one per element) padded to the maximum number of elements.
composition_features = []
max_elements = max(len(Composition(c).elements) for c in compositions)
# The updated feature size now includes the bond feature.
feature_size = len(final_feature_labels)

for comp in compositions:
    try:
        parsed_composition = Composition(comp)
        # Extract element symbols in the composition
        elements = [el.symbol for el in parsed_composition.elements]

        # For each element, fetch its extended feature vector (including bond info)
        element_vectors = [
            element_to_features[el] if el in element_to_features else [float('nan')] * feature_size
            for el in elements
        ]
        # Pad with NaNs if the composition has fewer than max_elements
        pad_length = max_elements - len(elements)
        if pad_length > 0:
            padding = [[float('nan')] * feature_size] * pad_length
            padded_features = element_vectors + padding
        else:
            padded_features = element_vectors
        composition_features.append(padded_features)
    except Exception as e:
        print(f"Error processing composition {comp}: {e}")
        composition_features.append([[float('nan')] * feature_size] * max_elements)

# Convert to a 3D NumPy array
composition_features_array = np.array(composition_features, dtype=np.float64)

# # Save the 3D array and final feature labels (which now include the bond feature)
# output_file_features = "composition_features_cleaned.npy"
# output_file_labels = "final_feature_labels_cleaned.npy"
# np.save(output_file_features, composition_features_array)
# np.save(output_file_labels, np.array(final_feature_labels))
# print(f"3D array saved to {output_file_features}. Shape: {composition_features_array.shape}")
# print(f"Final feature labels saved to {output_file_labels}.")

# Optionally, assign the training features to a variable
final_features_train = composition_features_array

# -------------------------------
# (Optional) Process HEC8 Compositions Similarly
# -------------------------------
# Load HEC8 compositions
input_file_test = "./data/HEC8_output.xlsx"
df_compositions_test = pd.read_excel(input_file_test)
compositions_test = df_compositions_test["Composition"].tolist()

composition_features_test = []
max_elements_test = max(len(Composition(c).elements) for c in compositions_test)
for comp in compositions_test:
    try:
        parsed_composition = Composition(comp)
        elements = [el.symbol for el in parsed_composition.elements]
        element_vectors = [
            element_to_features[el] if el in element_to_features else [float('nan')] * feature_size
            for el in elements
        ]
        pad_length = max_elements_test - len(elements)
        if pad_length > 0:
            padding = [[float('nan')] * feature_size] * pad_length
            padded_features = element_vectors + padding
        else:
            padded_features = element_vectors
        composition_features_test.append(padded_features)
    except Exception as e:
        print(f"Error processing composition {comp}: {e}")
        composition_features_test.append([[float('nan')] * feature_size] * max_elements_test)

composition_features_test_array = np.array(composition_features_test, dtype=np.float64)
final_features_test = composition_features_test_array
print(f"Test 3D array shape: {composition_features_test_array.shape}")

import numpy as np

# Define file paths for saving
train_features_file = "./data/final_features_train.npy"
test_features_file = "./data/final_features_test.npy"

# Save the arrays
# np.save(train_features_file, final_features_train)
# np.save(test_features_file, final_features_test)

# print(f"Training features saved to {train_features_file}")
# print(f"Test features saved to {test_features_file}")

# --- Loading the arrays later ---

# Load the saved numpy arrays
final_features_train = np.load(train_features_file)
final_features_test = np.load(test_features_file)

print(f"Loaded training features shape: {final_features_train.shape}")
print(f"Loaded test features shape: {final_features_test.shape}")