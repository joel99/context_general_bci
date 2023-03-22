#!/bin/bash
#SBATCH --job-name=ndt2
#SBATCH --gres gpu:1
#SBATCH -p gpu
#SBATCH -c 6
#SBATCH -t 48:00:00
#SBATCH --mem 20G
#SBATCH -x mind-1-23,mind-1-26
#SBATCH --output=slurm_logs/%j.out

echo $@
hostname
source ~/.bashrc # Note bashrc has been modified to allow slurm jobs
source ~/load_env.sh
python -u run.py $@

