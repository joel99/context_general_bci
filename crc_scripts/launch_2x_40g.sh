#!/bin/bash
#SBATCH --job-name=ndt2_2x
#SBATCH --cluster gpu
#SBATCH -p a100
#SBATCH -t 16:00:00
#SBATCH --mem 150G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
echo 'tasks'
echo $SLURM_NTASKS
echo 'per node'
export SLURM_NTASKS_PER_NODE=2
hostname
source ~/.bashrc # Note bashrc has been modified to allow slurm jobs
source ~/load_env.sh
srun python -u run.py $1

