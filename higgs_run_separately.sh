#!/bin/bash

# Train - compress - decompress
poetry run baler --project higgs higgs_project --mode train
poetry run baler --project higgs higgs_project --mode compress
poetry run baler --project higgs higgs_project --mode decompress
poetry run baler --project higgs higgs_project --mode plot

# Copy decompressed output to save separately
cp workspaces/higgs/higgs_project/output/decompressed_output/decompressed.npz workspaces/higgs/higgs_project/output/decompressed_output/decompressed_photon_pt.npz 
# Copy compressed output to save separately
cp workspaces/higgs/higgs_project/output/compressed_output/compressed.npz workspaces/higgs/higgs_project/output/compressed_output/compressed_photon_pt.npz 
# Copy loss plot to save separately
cp workspaces/higgs/higgs_project/output/plotting/Loss_plot.pdf workspaces/higgs/higgs_project/output/plotting/Loss_plot_pt.pdf 

# Change the input path from photon_pt to photon_eta
sed -i '' '2s|    c.input_path = ".*"|    c.input_path = "workspaces/higgs/data/13TeV_photon_eta.npz"|' "/Users/oscarfuentes/masters_project/baler_sem1/workspaces/higgs/higgs_project/config/higgs_project_config.py"

# Repeat this process, cycling through eta -> phi -> E -> pt (only change input path back to pt)

poetry run baler --project higgs higgs_project --mode train
poetry run baler --project higgs higgs_project --mode compress
poetry run baler --project higgs higgs_project --mode decompress
poetry run baler --project higgs higgs_project --mode plot

cp workspaces/higgs/higgs_project/output/decompressed_output/decompressed.npz workspaces/higgs/higgs_project/output/decompressed_output/decompressed_photon_eta.npz 
cp workspaces/higgs/higgs_project/output/compressed_output/compressed.npz workspaces/higgs/higgs_project/output/compressed_output/compressed_photon_eta.npz 
cp workspaces/higgs/higgs_project/output/plotting/Loss_plot.pdf workspaces/higgs/higgs_project/output/plotting/Loss_plot_eta.pdf 

sed -i '' '2s|    c.input_path = ".*"|    c.input_path = "workspaces/higgs/data/13TeV_photon_phi.npz"|' "/Users/oscarfuentes/masters_project/baler_sem1/workspaces/higgs/higgs_project/config/higgs_project_config.py"


poetry run baler --project higgs higgs_project --mode train
poetry run baler --project higgs higgs_project --mode compress
poetry run baler --project higgs higgs_project --mode decompress
poetry run baler --project higgs higgs_project --mode plot

cp workspaces/higgs/higgs_project/output/decompressed_output/decompressed.npz workspaces/higgs/higgs_project/output/decompressed_output/decompressed_photon_phi.npz 
cp workspaces/higgs/higgs_project/output/compressed_output/compressed.npz workspaces/higgs/higgs_project/output/compressed_output/compressed_photon_phi.npz 
cp workspaces/higgs/higgs_project/output/plotting/Loss_plot.pdf workspaces/higgs/higgs_project/output/plotting/Loss_plot_phi.pdf 

sed -i '' '2s|    c.input_path = ".*"|    c.input_path = "workspaces/higgs/data/13TeV_photon_E.npz"|' "/Users/oscarfuentes/masters_project/baler_sem1/workspaces/higgs/higgs_project/config/higgs_project_config.py"


poetry run baler --project higgs higgs_project --mode train
poetry run baler --project higgs higgs_project --mode compress
poetry run baler --project higgs higgs_project --mode decompress
poetry run baler --project higgs higgs_project --mode plot

cp workspaces/higgs/higgs_project/output/decompressed_output/decompressed.npz workspaces/higgs/higgs_project/output/decompressed_output/decompressed_photon_E.npz 
cp workspaces/higgs/higgs_project/output/compressed_output/compressed.npz workspaces/higgs/higgs_project/output/compressed_output/compressed_photon_E.npz 
cp workspaces/higgs/higgs_project/output/plotting/Loss_plot.pdf workspaces/higgs/higgs_project/output/plotting/Loss_plot_E.pdf 

sed -i '' '2s|    c.input_path = ".*"|    c.input_path = "workspaces/higgs/data/13TeV_photon_pt.npz"|' "/Users/oscarfuentes/masters_project/baler_sem1/workspaces/higgs/higgs_project/config/higgs_project_config.py"


