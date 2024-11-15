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

# poetry environment
mkdir -p .cache/pypoetry/virtualenvs/

# baler
cp -r /gluster/home/ofrebato/baler baler
cp -r /gluster/home/ofrebato/.cache/pypoetry/virtualenvs/baler-O7HYjMIZ-py3.9 .cache/pypoetry/virtualenvs/baler-O7HYjMIZ-py3.9
cd baler

# logging
touch job.out
touch job.err
touch job.log
touch job.all_output
echo "hello world" > job.all_output

#copy results
cp job.out /gluster/home/ofrebato/baler/job.out
cp job.err /gluster/home/ofrebato/baler/job.err
cp job.log /gluster/home/ofrebato/baler/job.log
cp job.all_output /gluster/home/ofrebato/job.all_output

# poetry
poetry install

#copy results
cp job.out /gluster/home/ofrebato/baler/job.out
cp job.err /gluster/home/ofrebato/baler/job.err
cp job.log /gluster/home/ofrebato/baler/job.log
cp job.all_output /gluster/home/ofrebato/job.all_output

# run
python run_multiple.py -n 1 -s

#copy results
cp job.out /gluster/home/ofrebato/baler/job.out
cp job.err /gluster/home/ofrebato/baler/job.err
cp job.log /gluster/home/ofrebato/baler/job.log
cp job.all_output /gluster/home/ofrebato/job.all_output
cp run_multiple_results/results.npz /gluster/home/ofrebato/baler/run_multiple_results/results.npz

cat job.all_output
echo "COMPLETED RUN_SCRIPT.SH"
