#!/bin/bash
#SBATCH --job-name=ndt2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH --mem 40G
#SBATCH --output=slurm_logs/%j.out

echo 'tasks'
echo $SLURM_NTASKS
echo 'per node'
echo $SLURM_NTASKS_PER_NODE
hostname
source ~/.bashrc # Note bashrc has been modified to allow slurm jobs
module add cuda-11.6
module add cudnn-11.6-v8.4.1.50
conda activate py10_2
python -u run.py +exp=$1

