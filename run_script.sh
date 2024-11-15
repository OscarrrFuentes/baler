#!/bin/bash
export PATH="/gluster/home/ofrebato/.cache/pypoetry/virtualenvs/baler-O7HYjMIZ-py3.9/bin:/gluster/home/ofrebato/.local/bin:/bin:/gluster/home/ofrebato/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/puppetlabs/bin:$PATH"
echo $PATH

# Python setup
# python -m ensurepip --upgrade
# python -m pip install --upgrade pip
# pip install ipython ipykernel 
# pip install numpy matplotlib torch scikit-learn tqdm numba pytest hls4ml
# python -m pip install --user pipx
# python -m pipx ensurepath
# pipx install poetry
# poetry config virtualenvs.options.system-site-packages true

# logging
touch job.out
touch job.err
touch job.log

# poetry environment
mkdir -p .cache/pypoetry/virtualenvs/

# baler
cp -r /gluster/home/ofrebato/baler baler
cp -r /gluster/home/ofrebato/.cache/pypoetry/virtualenvs/baler-O7HYjMIZ-py3.9 .cache/pypoetry/virtualenvs/baler-O7HYjMIZ-py3.9
cd baler
#copy results
cp job.out /gluster/home/ofrebato/baler/job.out
cp job.err /gluster/home/ofrebato/baler/job.err
cp job.log /gluster/home/ofrebato/baler/job.log

# # poetry
# poetry install
#copy results
cp job.out /gluster/home/ofrebato/baler/job.out
cp job.err /gluster/home/ofrebato/baler/job.err
cp job.log /gluster/home/ofrebato/baler/job.log

# run
python run_multiple.py -n 2 -s

#copy results
cp job.out /gluster/home/ofrebato/baler/job.out
cp job.err /gluster/home/ofrebato/baler/job.err
cp job.log /gluster/home/ofrebato/baler/job.log
cp run_multiple_results/results.npz /gluster/home/ofrebato/baler/run_multiple_results/results.npz
echo "COMPLETED RUN_SCRIPT.SH :)"
