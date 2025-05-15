#!/bin/bash
# Script: create_POTCAR_all.sh
# This script loops over all directories beginning with ARUN.POCC*,
# and in each directory it creates a combined POTCAR file by concatenating
# the POTCAR files from the subdirectories: C, Mo_pv, Nb_sv, Ta_pv, V_sv, and W_pv.

for dir in ARUN.POCC*; do
  if [ -d "$dir" ]; then
    echo "Processing directory: $dir"
    cd "$dir" || { echo "Cannot enter $dir"; continue; }
    
    # Check if all expected subdirectories exist
    if [ -d "C" ] && [ -d "Mo_pv" ] && [ -d "Nb_sv" ] && [ -d "Ta_pv" ] && [ -d "V_sv" ] && [ -d "W_pv" ]; then
      echo "Creating combined POTCAR in $dir"
      cat C/POTCAR \
          Mo_pv/POTCAR \
          Nb_sv/POTCAR \
          Ta_pv/POTCAR \
          V_sv/POTCAR \
          W_pv/POTCAR > POTCAR
      echo "POTCAR created. TITEL lines:"
      grep TITEL POTCAR
    else
      echo "Warning: One or more required pseudopotential subdirectories are missing in $dir."
    fi
    cd ..
  fi
done
