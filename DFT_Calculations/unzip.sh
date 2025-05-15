#!/bin/bash

# Name of the tar archive
TARFILE="potpaw_PBE.64.tgz"

# Loop over every directory that starts with ARUN.POCC_
for d in ARUN.POCC_*; do
  # Check if it's actually a directory (and not a file)
  if [ -d "$d" ]; then
    echo "Extracting $TARFILE into $d"
    # Extract the .tgz contents into that directory
    tar -xzf "$TARFILE" -C "$d"
  fi
done
