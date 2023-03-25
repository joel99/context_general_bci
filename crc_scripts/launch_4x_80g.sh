#!/bin/bash
#SBATCH --job-name=ndt2_4x
#SBATCH --cluster gpu
#SBATCH -p a100_nvlink
#SBATCH -t 4:00:00
#SBATCH --mem 400G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --constraint=80g

echo 'tasks'
echo $SLURM_NTASKS
echo 'per node'
export SLURM_NTASKS_PER_NODE=4
hostname
source ~/.bashrc # Note bashrc has been modified to allow slurm jobs
source ~/load_env.sh
srun python -u run.py $1

