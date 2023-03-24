#!/bin/bash
#SBATCH --job-name=ndt2
#SBATCH --cluster gpu
#SBATCH -p gtx1080
#SBATCH --gres gpu:1
#SBATCH -c 3
#SBATCH -t 24:00:00
#SBATCH --mem 40G
#SBATCH --output=slurm_logs/%j.out

echo $@
hostname
source ~/.bashrc # Note bashrc has been modified to allow slurm jobs
source ~/load_env.sh
python -u run.py $@

