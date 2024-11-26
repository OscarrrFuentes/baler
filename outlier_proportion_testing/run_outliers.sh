#!/bin/bash

OUTLIER_PROPORTION=$1

export PATH="/gluster/home/ofrebato/.cache/pypoetry/virtualenvs/baler-O7HYjMIZ-py3.9/bin:/gluster/home/ofrebato/.local/bin:/bin:/gluster/home/ofrebato/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/puppetlabs/bin:$PATH"
echo $PATH

# poetry environment
mkdir -p .cache/pypoetry/virtualenvs/
cp -r /gluster/home/ofrebato/.cache/pypoetry/virtualenvs/baler-O7HYjMIZ-py3.9 .cache/pypoetry/virtualenvs/baler-O7HYjMIZ-py3.9

#Logging
mkdir logs
PARENT_DIR=$PWD

# Copy baler
cp -r /gluster/home/ofrebato/baler baler
cd baler

# Set up Poetry to use scratch directory
export POETRY_HOME="$PWD/.poetry"
export POETRY_VIRTUALENVS_PATH="$PWD/.venvs"
export POETRY_VIRTUALENVS_IN_PROJECT=true

# Install Poetry and dependencies
poetry install --no-root
source /gluster/home/ofrebato/.cache/pypoetry/virtualenvs/baler-O7HYjMIZ-py3.9/bin/activate

# Path to your config file
CONFIG_FILE="$PWD/workspaces/MNIST/MNIST_project/config/MNIST_project_config.py"

>"/gluster/home/ofrebato/baler/outlier_proportion_testing/full_results.txt"

mkdir -p "$PWD/outlier_proportion_testing/results"
rm -rf "$PWD/outlier_proportion_testing/results/*"
echo $PWD

for ((i=0; i<=9; i+=1))
do
    >"$PWD/outlier_proportion_testing/results/sep_$i.txt"
    >"$PWD/outlier_proportion_testing/results/inc_$i.txt"
done

# Append the batch size to the config file
echo -e "\n    c.outlier_proportion = $OUTLIER_PROPORTION" >> $CONFIG_FILE

for ((i=0; i<=60; i+=1))
do
# Run the training and compression with the current batch size
    poetry run baler --project MNIST MNIST_project --mode train
    poetry run baler --project MNIST MNIST_project --mode compress
    python outlier_proportion_testing/calc_distance.py
    poetry run baler --project MNIST MNIST_project --mode decompress
    python outlier_proportion_testing/calc_distance.py --sep
done

python outlier_proportion_testing/outlier_proportion_analysis.py --proportion $OUTLIER_PROPORTION

cp -r $PARENT_DIR/logs /gluster/home/ofrebato/baler/logs

mkdir -p /gluster/home/ofrebato/baler/outlier_proportion_testing/results/$OUTLIER_PROPORTION
cp -r $PWD/outlier_proportion_testing/results/* /gluster/home/ofrebato/baler/outlier_proportion_testing/results/$OUTLIER_PROPORTION



