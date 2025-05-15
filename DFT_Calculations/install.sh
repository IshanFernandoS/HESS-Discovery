#!/bin/bash
# Download the compiled AFLOW file and default settings file
wget https://s4e.ai/workshops/2024_ai_materials/DAY_1/0_installation/AFLOW-src.zip
wget https://s4e.ai/workshops/2024_ai_materials/DAY_1/0_installation/aflow.rc

# Unzip the downloaded AFLOW source zip file
unzip AFLOW-src.zip

# Add executable permission to the AFLOW binary
chmod +x ./AFLOW-src/src/aflow

# Update PATH to include the AFLOW binary directory; assumes current directory
export PATH=$PATH:"$(pwd)/AFLOW-src/src"

# Copy the aflow.rc file to the default configuration location in your home directory
cp aflow.rc "$HOME/.aflow.rc"

# Create working directories
mkdir -p aflow_pocc/test1
cd aflow_pocc/test1

# Run the AFLOW command
aflow --aflow_proto=AB_cF8_225_a_b.BA:S:Se:Zn --params=-1 --pocc_params=S0-0.75xA-0.25xB_S1-1.0xC --generate_aflowin_only

# Print the current directory
pwd
