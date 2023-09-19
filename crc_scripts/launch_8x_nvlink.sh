#!/bin/bash
#SBATCH --job-name=ndt2_8x
#SBATCH --cluster gpu
#SBATCH -p a100_nvlink
#SBATCH -t 16:00:00
#SBATCH --mem 800G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --constraint=80g

echo 'tasks'
echo $SLURM_NTASKS
echo 'per node'
export SLURM_NTASKS_PER_NODE=8
hostname
source ~/.bashrc # Note bashrc has been modified to allow slurm jobs
source ~/load_env.sh
srun python -u run.py $1

