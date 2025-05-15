#!/bin/bash
# Script: calculate_EFA.sh
# This script processes each ARUN.POCC_* directory to:
# 1. Extract the total energy from the VASP output file (last "F=" occurrence)
# 2. Extract the degeneracy (DG value) from the first line of the POSCAR
# 3. Assume the number of atoms is 10 (fixed)
# 4. Convert the total energy to energy per atom (eV/atom)
# 5. Write directory, energy per atom, and degeneracy to a temporary file (efa_data.txt)
# 6. Use AWK to compute the weighted mixed enthalpy (H_mix), weighted standard deviation (sigma),
#    and EFA = 1/sigma.
#
# If no vasp.sh.o* file is found, if the output file contains an error,
# or if the total energy is effectively 0, that directory is skipped (and the file removed).

temp_file="efa_data.txt"
> "$temp_file"  # Clear the temporary file

# Loop over directories starting with ARUN.POCC_
for dir in ARUN.POCC_*; do
  if [ -d "$dir" ]; then
    echo "Processing directory: $dir"
    
    # Locate the VASP output file (e.g., vasp.sh.o*)
    out_file=$(find "$dir" -maxdepth 1 -type f -name "vasp.sh.o*" | tail -n 1)
    
    # If no output file is found, skip this directory
    if [ -z "$out_file" ]; then
      echo "No vasp.sh.o* file found in $dir, skipping..."
      continue
    fi

    # Check if the output file contains error messages (case-insensitive)
    if grep -qiE "ERROR|FATAL" "$out_file"; then
      echo "Error found in $out_file. Removing file and skipping directory $dir..."
      rm -f "$out_file"
      continue
    fi
    
    poscar_file="$dir/POSCAR"

    # Extract total energy from the VASP output (last occurrence of "F=")
    energy_line=$(grep "F=" "$out_file" | tail -n 1)
    energy=$(echo "$energy_line" | awk '{for(i=1;i<=NF;i++){ if($i=="F="){ print $(i+1); exit } }}')

    # Check if the total energy is effectively 0 (using a threshold of 1e-10)
    if awk "BEGIN {if (sqrt(($energy)*($energy)) < 1e-10) exit 0; else exit 1}"; then
      echo "Total energy is effectively 0 in $dir. Removing $out_file and skipping..."
      rm -f "$out_file"
      continue
    fi

    # Extract degeneracy from the first line of POSCAR (look for "DG=###")
    if [ -f "$poscar_file" ]; then
      degeneracy=$(head -n 1 "$poscar_file" | sed -n 's/.*DG=\([0-9]\+\).*/\1/p')
      # If not found, assume degeneracy = 1
      [ -z "$degeneracy" ] && degeneracy=1
    else
      degeneracy="NaN"
    fi

    # Fixed number of atoms
    num_atoms=10

    # Convert total energy to energy per atom using awk (handling scientific notation)
    if [[ "$num_atoms" != "NaN" && "$energy" != "NaN" ]]; then
      energy_per_atom=$(awk -v E="$energy" -v natoms="$num_atoms" 'BEGIN {printf "%.10f", E/natoms}')
    else
      energy_per_atom="NaN"
    fi

    # Write collected data (directory, energy per atom, degeneracy) to the temporary file
    echo "$dir $energy_per_atom $degeneracy" >> "$temp_file"
  fi
done

echo "Collected data in $temp_file:"
cat "$temp_file"
echo ""

# Now, compute H_mix, sigma, and EFA using AWK.
result=$(awk '
{
  energy = $2 + 0;   # Energy per atom (eV/atom)
  g = $3 + 0;        # Degeneracy
  sum_g += g;
  sum_gh += g * energy;
  energies[NR] = energy;
  g_arr[NR] = g;
  count = NR;
}
END {
  if(sum_g <= 0){
    print "No data or sum_g=0. Cannot compute EFA.";
    exit;
  }
  H_mix = sum_gh / sum_g;
  sum_sq = 0;
  for(i = 1; i <= count; i++){
    diff = energies[i] - H_mix;
    sum_sq += g_arr[i] * diff * diff;
  }
  sigma = sqrt(sum_sq / sum_g);
  if(sigma == 0){
    EFA = "Inf";
  } else {
    EFA = 1 / sigma;
  }
  printf("H_mix (eV/atom) = %e\n", H_mix);
  printf("Total weighted degeneracy (sum_i g_i) = %f\n", sum_g);
  printf("sum_i g_i*(H_i - H_mix)^2 = %e\n", sum_sq);
  printf("sigma (eV/atom) = %e\n", sigma);
  printf("EFA (1/(eV/atom)) = %e\n", EFA);
}' "$temp_file")

echo "Calculated values:"
echo "$result"
