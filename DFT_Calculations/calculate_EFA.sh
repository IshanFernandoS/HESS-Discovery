#!/bin/bash
# Script: calculate_EFA.sh
# This script processes each ARUN.POCC_* directory to:
# 1. Extract the total energy from the VASP output file (last "F=" occurrence)
# 2. Extract the degeneracy (DG value) from the first line of the POSCAR
# 3. Assume the number of atoms is 10 (fixed)
# 4. Convert the total energy to energy per atom (eV/atom)
# 5. Write directory, energy per atom, and degeneracy to a temporary file (efa_data.txt)
# 6. Use AWK to compute:
#      H_mix = (sum_i g_i * H_i) / (sum_i g_i)
#      sigma = sqrt( sum_i g_i*(H_i - H_mix)^2 / sum_i g_i )
#      EFA   = 1 / sigma
#
# The energy per atom is computed using awk (which supports scientific notation)
# to avoid errors in decimal places.

temp_file="efa_data.txt"
> "$temp_file"  # Clear the temporary file

# Loop over directories starting with ARUN.POCC_
for dir in ARUN.POCC_*; do
  if [ -d "$dir" ]; then
    echo "Processing directory: $dir"
    
    # Locate the VASP output file (e.g., vasp.sh.o*)
    out_file=$(find "$dir" -maxdepth 1 -type f -name "vasp.sh.o*" | tail -n 1)
    poscar_file="$dir/POSCAR"

    # Extract total energy from the VASP output (last occurrence of "F=")
    if [ -f "$out_file" ]; then
      energy_line=$(grep "F=" "$out_file" | tail -n 1)
      # Assume the energy is the number immediately following "F="
      energy=$(echo "$energy_line" | awk '{for(i=1;i<=NF;i++){ if($i=="F="){ print $(i+1); exit } }}')
    else
      energy="NaN"
    fi

    # Extract degeneracy from the first line of POSCAR (look for "DG=###")
    if [ -f "$poscar_file" ]; then
      degeneracy=$(head -n 1 "$poscar_file" | sed -n 's/.*DG=\([0-9]\+\).*/\1/p')
      # If not found, assume degeneracy = 1
      [ -z "$degeneracy" ] && degeneracy=1
    else
      degeneracy="NaN"
    fi

    # Fix the number of atoms to 10
    num_atoms=10

    # Convert energy to energy per atom using awk (handles scientific notation)
    if [[ "$num_atoms" != "NaN" && "$energy" != "NaN" ]]; then
      energy_per_atom=$(awk -v E="$energy" -v natoms="$num_atoms" 'BEGIN {printf "%.10f", E/natoms}')
    else
      energy_per_atom="NaN"
    fi

    # Write the collected data (directory, energy per atom, degeneracy) to the temporary file
    echo "$dir $energy_per_atom $degeneracy" >> "$temp_file"
  fi
done

echo "Collected data in $temp_file:"
cat "$temp_file"
echo ""

# Now, compute H_mix, sigma (population std. dev.), and EFA using AWK.
result=$(awk '
{
  # Convert energy (per atom) and degeneracy to numbers.
  energy = $2 + 0;
  g = $3 + 0;
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
  # Calculate the weighted mixed enthalpy: H_mix = (sum_i g_i * H_i) / sum_i g_i
  H_mix = sum_gh / sum_g;
  
  # Calculate the weighted sum of squared deviations S = sum_i g_i*(H_i - H_mix)^2
  sum_sq = 0;
  for(i = 1; i <= count; i++){
    diff = energies[i] - H_mix;
    sum_sq += g_arr[i] * diff * diff;
  }
  
  # Population standard deviation: sigma = sqrt(sum_sq / sum_g)
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
