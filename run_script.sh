#!/bin/bash

BATCH_SIZE=$1

export PATH="/gluster/home/ofrebato/.cache/pypoetry/virtualenvs/baler-O7HYjMIZ-py3.9/bin:/gluster/home/ofrebato/.local/bin:/bin:/gluster/home/ofrebato/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/puppetlabs/bin:$PATH"
echo $PATH

# poetry environment
mkdir -p .cache/pypoetry/virtualenvs/
cp -r /gluster/home/ofrebato/.cache/pypoetry/virtualenvs/baler-O7HYjMIZ-py3.9 .cache/pypoetry/virtualenvs/baler-O7HYjMIZ-py3.9

# Copy baler
cp -r /gluster/home/ofrebato/baler baler_${BATCH_SIZE}
cd baler_${BATCH_SIZE}

# Set up Poetry to use scratch directory
export POETRY_HOME="$PWD/.poetry"
export POETRY_VIRTUALENVS_PATH="$PWD/.venvs"
export POETRY_VIRTUALENVS_IN_PROJECT=true

# Install Poetry and dependencies
poetry install --no-root
source /gluster/home/ofrebato/.cache/pypoetry/virtualenvs/baler-O7HYjMIZ-py3.9/bin/activate

# Run for a given batch size
python run_multiple.py -n 32 -s $PWD/run_multiple_results/result_${BATCH_SIZE}.npz -b ${BATCH_SIZE} > job_${BATCH_SIZE}.out 2>&1

# Copy results to home directory
cp run_multiple_results/result_${BATCH_SIZE}.npz /gluster/home/ofrebato/baler/run_multiple_results/results_${BATCH_SIZE}.npz
cp Hist_mean_distances.png /gluster/home/ofrebato/baler/hist_mean_distance/hist_mean_distances_${BATCH_SIZE}.png
cp Mean_mean_distances.png /gluster/home/ofrebato/baler/mean_mean_distance/mean_mean_distances_${BATCH_SIZE}.png
cp All_mean_distances.png /gluster/home/ofrebato/baler/all_mean_distance/all_mean_distances_${BATCH_SIZE}.png
cat job_${BATCH_SIZE}.out >> /gluster/home/ofrebato/baler/job.all_output

echo "COMPLETED RUN_SCRIPT.SH"
