#!/bin/bash
#SBATCH --job-name=ndt2
#SBATCH --cluster gpu
#SBATCH -p a100_nvlink
#SBATCH --gres gpu:1
#SBATCH -c 16
#SBATCH -t 24:00:00
#SBATCH --mem 90G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --constraint=80g

echo $@
hostname
source ~/.bashrc # Note bashrc has been modified to allow slurm jobs
source ~/load_env.sh
python -u run.py $@

