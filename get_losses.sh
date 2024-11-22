#!/bin/bash

BATCH_SIZE=$1

export PATH="/gluster/home/ofrebato/.cache/pypoetry/virtualenvs/baler-O7HYjMIZ-py3.9/bin:/gluster/home/ofrebato/.local/bin:/bin:/gluster/home/ofrebato/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/puppetlabs/bin:$PATH"
echo $PATH

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

# Path to your config file
CONFIG_FILE="workspaces/MNIST/MNIST_project/config/MNIST_project_config.py"

>"/gluster/home/ofrebato/baler/loss_datas/inc_loss_data.txt"
>"/gluster/home/ofrebato/baler/loss_datas/sep_loss_data.txt"

# Append the batch size to the config file
echo -e "\n    c.batch_size = $BATCH_SIZE" >> $CONFIG_FILE

inc_values=()
sep_values=()
for ((i=0; i<=32; i+=1))
do
    # Run the training and compression with the current batch size
    poetry run baler --project MNIST MNIST_project --mode train
    min_loss=$(python get_min_loss.py)
    inc_values+=("$min_loss")
    poetry run baler --project MNIST MNIST_project --mode compress
    min_loss=$(python get_min_loss.py)
    sep_values+=("$min_loss")
done

python calc_mean_loss.py --values "${inc_values[*]}" --batch_size $BATCH_SIZE
python calc_mean_loss.py --values "${sep_values[*]}" --batch_size $BATCH_SIZE --sep

    