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

import pandas as pd

# Assuming element_to_features and final_feature_labels are already defined

# Convert the dictionary to a DataFrame
data = {element: features for element, features in element_to_features.items()}
df = pd.DataFrame(data).T  # Transpose to have elements as rows

# Assign feature labels to the columns
df.columns = final_feature_labels

# Identify duplicate features
duplicate_features = []

# Compare each column with the others to find duplicates
for i in range(len(df.columns)):
    for j in range(i + 1, len(df.columns)):
        if df.iloc[:, i].equals(df.iloc[:, j]):
            duplicate_features.append((df.columns[i], df.columns[j]))

# Display duplicate features
if duplicate_features:
    print("Duplicate Features Found:")
    for pair in duplicate_features:
        print(f"{pair[0]} and {pair[1]} are duplicates")
else:
    print("No duplicate features found.")