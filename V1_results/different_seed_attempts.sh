#!/bin/bash

# # Get version number
# version=$1

# Initialise for conda to not be weird
source ~/.bash_profile

current_dir=$(pwd)

# Store current environment
current_env=$(conda info --envs | grep '*' | awk '{print $1}')

# Activate the desired environment (baler)
conda activate baler

# # Run your commands
poetry run baler --project higgs higgs_project --mode train
poetry run baler --project higgs higgs_project --mode compress
poetry run baler --project higgs higgs_project --mode decompress
poetry run baler --project higgs higgs_project --mode plot

# Copy results to the desired directory
cp "$current_dir/workspaces/higgs/higgs_project/output/decompressed_output/decompressed.npz" "$current_dir/V4_results/V4/decompressed.npz"
cp "$current_dir/workspaces/higgs/higgs_project/output/plotting/Loss_plot.pdf" "$current_dir/V4_results/V4/Loss_plot.pdf"
cp "$current_dir/workspaces/higgs/higgs_project/output/plotting/comparison.pdf" "$current_dir/V4_results/V4/comparison.pdf"

# Deactivate and revert to the original environment
conda deactivate
conda activate higgs_analysis

python "$current_dir/V4_results/V4_calc_masses_and_plot.py" --v 4


