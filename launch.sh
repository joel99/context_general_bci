#!/bin/bash
#SBATCH --job-name=ndt2
#SBATCH --gres gpu:1
#SBATCH -p gpu
#SBATCH -c 6
#SBATCH -t 48:00:00
#SBATCH -x mind-0-3
#SBATCH --mem 40G
#SBATCH --output=slurm_logs/%j.out

echo $@
hostname
source ~/.bashrc # Note bashrc has been modified to allow slurm jobs
source ~/load_env.sh
module unload cudnn-11.6-v8.4.1.50 # not sure where these are coming from but they're failing the rnn runs
module unload cuda-11.6 # not sure where these are coming from but they're failing the rnn runs
mamba activate ndt2
python -u run.py $@

