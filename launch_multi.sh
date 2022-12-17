#!/bin/bash
#SBATCH --job-name=ndt2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH --mem 50G
#SBATCH --output=slurm_logs/%j.out

# Multinode notes
# From https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster.html
# # debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# # on your cluster you might need these:
# # set the network interface

# # Currently (7/24/22) IB isn't working well on mind due to memlock limits
# export NCCL_IB_DISABLE=1
# # export NCCL_SOCKET_IFNAME=^docker0,lo

# # might need the latest CUDA
# # module load NCCL/2.4.7-1-cuda.10.0

# srun python3 artifacts/artifact_estimator.py --num-nodes 1 -c $@ # Multinode

hostname
source ~/.bashrc # Note bashrc has been modified to allow slurm jobs
module rm cuda-11.3
module rm cudnn-11.3-v8.2.0.53
module add cuda-11.6
module add cudnn-11.6-v8.4.1.50
conda activate py10_2
python -u run.py +exp=$1

