#!/bin/bash

# Files to be copied
FILES_TO_COPY=("INCAR" "KPOINTS")

# Loop over every directory whose name starts with ARUN.POCC_
for d in ARUN.POCC_*; do
  # Check if this is actually a directory
  if [ -d "$d" ]; then
    echo "Copying files to $d"
    # Copy each file, overwriting if it exists
    for f in "${FILES_TO_COPY[@]}"; do
      cp -f "$f" "$d"
    done
  fi
done
