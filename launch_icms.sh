#!/bin/bash
#SBATCH --job-name=icms
#SBATCH --gres gpu:1
#SBATCH -p gpu
#SBATCH -c 6
#SBATCH -t 96:00:00
#SBATCH --mem 40G
#SBATCH --output=slurm_logs/%j.out

# We may end up getting A5000s which is a waste, we should look into autoscaling...
module load cuda-11.3
module load cudnn-11.3-v8.2.0.53

echo $@
hostname
source ~/.bashrc # Note bashrc has been modified to allow slurm jobs
conda activate rlpyt
python -u run.py -n $SLURM_JOB_ID -c $@