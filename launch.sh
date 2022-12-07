#!/bin/bash
#SBATCH --job-name=ndt2
#SBATCH --gres gpu:1
#SBATCH -p gpu
#SBATCH -c 12
#SBATCH -t 96:00:00
#SBATCH --mem 30G
#SBATCH --output=slurm_logs/%j.out


echo $@
hostname
source ~/.bashrc # Note bashrc has been modified to allow slurm jobs
module rm cuda-11.3
module rm cudnn-11.3-v8.2.0.53
module add cuda-11.6
module add cudnn-11.6-v8.4.1.50
conda activate py10_2
python -u run.py +exp=$1

