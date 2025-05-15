#!/bin/bash

# Script name: run_all_vasp.sh

for d in ARUN.POCC*; do
  if [ -d "$d" ]; then
    # Only process approximately 10% of directories
    if [ $(( RANDOM % 2 )) -eq 0 ]; then
      echo "Entering directory: $d"
      cd "$d" || exit 1

      if [ -f "vasp.sh" ]; then
        echo "Submitting vasp.sh in $d"
        qsub vasp.sh
      else
        echo "No vasp.sh found in $d; skipping..."
      fi

      cd ..
    else
      echo "Skipping directory: $d"
    fi
  fi
done

