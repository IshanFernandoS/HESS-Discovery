# Descriptor generation only (no NSGA/GNN).
# Assumes you've already handled any duplicate compositions upstream.

import json
from collections import Counter

import numpy as np
import pandas as pd

# --------------------------
# SD1 — per-element summary
# --------------------------
# Source sheet with per-binary-carbide properties
src_path = "/content/Descriptors-2.xlsx"

# Columns required downstream (names kept as-is)
KEEP_COLS = [
    "Binary Carbide",
    "Atomic Radius/ pm",
    "Valence e Count",
    "Electronegativity",
    "Formation Energy/ eV per atom",
    "Carbide Melting Point",
]

df_raw = pd.read_excel(src_path)[KEEP_COLS].copy()

# Coerce numerics; leave string column intact
for col in KEEP_COLS[1:]:
    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

# Pull metal symbol from the left part of "Binary Carbide" (e.g., "TiC" -> "Ti")
df_raw["Element"] = df_raw["Binary Carbide"].str.extract(r"^([A-Z][a-z]?)")

# Aggregate to one row per element
# - Formation energy: min (more negative => stronger M–C)
# - Other properties: mean
agg_rules = {
    "Formation Energy/ eV per atom": "min",
    "Carbide Melting Point": "mean",
    "Atomic Radius/ pm": "mean",
    "Electronegativity": "mean",
    "Valence e Count": "mean",
}
sd1_df = df_raw.groupby("Element", as_index=False).agg(agg_rules)

# Save SD1
sd1_df.to_csv("SD1_descriptors_clean.csv", index=False)

# --------------------------
# SD2 — mixed-metal pairs
# --------------------------
# Hand-curated proxy for mixed-metal carbide formation enthalpy (eV/atom)
binary_carbide_formation_energy = {
    ('Sc','Ti'):None, ('Sc','V'):0.179, ('Sc','Cr'):None, ('Sc','Y'):0.096,
    ('Sc','Zr'):-0.015, ('Sc','Nb'):0.1, ('Sc','Mo'):None, ('Sc','Hf'):-0.002,
    ('Sc','Ta'):None, ('Sc','W'):None, ('Sc','Re'):-0.261, ('Ti','V'):0.093,
    ('Ti','Cr'):0.084, ('Ti','Y'):0.364, ('Ti','Zr'):0.038, ('Ti','Nb'):0.054,
    ('Ti','Mo'):-0.134, ('Ti','Hf'):0.029, ('Ti','Ta'):0.084, ('Ti','W'):0.023,
    ('Ti','Re'):-0.352, ('V','Cr'):0.045, ('V','Nb'):0.091, ('V','Mo'):-0.059,
    ('V','Hf'):0.034, ('V','Ta'):-0.111, ('V','W'):-0.061, ('V','Re'):-0.309,
    ('Cr','Y'):None, ('Cr','Zr'):0.004, ('Cr','Nb'):0.037, ('Cr','Mo'):0.106,
    ('Cr','Hf'):0.085, ('Cr','Ta'):0.060, ('Cr','W'):0.121, ('Cr','Re'):0.133,
    ('Y','Zr'):0.113, ('Y','Nb'):None, ('Y','Mo'):None, ('Y','Hf'):None,
    ('Y','Ta'):None, ('Y','W'):None, ('Y','Re'):-0.255, ('Zr','Nb'):0.111,
    ('Zr','Mo'):-0.139, ('Zr','Hf'):0.002, ('Zr','Ta'):0.153, ('Zr','W'):-0.140,
    ('Zr','Re'):-0.347, ('Nb','Mo'):-0.069, ('Nb','Hf'):0.114, ('Nb','Ta'):0.011,
    ('Nb','W'):0.027, ('Nb','Re'):-0.017, ('Mo','Hf'):-0.173, ('Mo','Ta'):-0.104,
    ('Mo','W'):-0.0, ('Mo','Re'):-0.029, ('Hf','Ta'):0.148, ('Hf','W'):-0.169,
    ('Hf','Re'):-0.409, ('Ta','W'):-0.028, ('Ta','Re'):-0.247, ('W','Re'):0.043
}

pair_rows = []
for (a, b), val in binary_carbide_formation_energy.items():
    pair_rows.append({
        "metal_i": a,
        "metal_j": b,
        "DeltaH_MM_proxy_eV_per_atom": (np.nan if val is None else float(val)),
        "source": "Materials Project (most stable mixed-metal carbide) / code list",
    })
sd2_df = pd.DataFrame(pair_rows)

# Save SD2
sd2_df.to_csv("SD2_mm_pairs.csv", index=False)

# --------------------------
# SD2 — quick QC snapshot
# --------------------------
finite_vals = sd2_df["DeltaH_MM_proxy_eV_per_atom"].dropna().values

# Elements present in the pair dictionary
elements_in_pairs = sorted({e for pair in binary_carbide_formation_energy for e in pair})
n = len(elements_in_pairs)
n_pairs_possible = n * (n - 1) // 2  # should be 66 for 12 elements

qc_summary = {
    "n_elements": int(sd1_df["Element"].nunique()),
    "n_pairs_possible": int(n_pairs_possible),
    "n_pairs_present": int(np.isfinite(finite_vals).sum()),
    # Keep original denominator (66.0) to avoid changing any downstream expectations
    "coverage_fraction": float(np.isfinite(finite_vals).sum() / 66.0),
    "min": float(np.nanmin(finite_vals)) if finite_vals.size else None,
    "median": float(np.nanmedian(finite_vals)) if finite_vals.size else None,
    "max": float(np.nanmax(finite_vals)) if finite_vals.size else None,
    "sign_counts": dict(Counter(np.sign(finite_vals))),
}

with open("SD2_mm_pairs_qc.json", "w") as f:
    json.dump(qc_summary, f, indent=2)

print("Wrote SD1_descriptors_clean.csv, SD2_mm_pairs.csv, and SD2_mm_pairs_qc.json")