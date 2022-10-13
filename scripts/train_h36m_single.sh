#!/bin/bash

#SBATCH -J voxelpose_h36m_train
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH -o /mntnfs/med_data5/wangjiong/3dpose_school/voxelpose-pytorch/slurm_logs/train_%j.out
#SBATCH -e /mntnfs/med_data5/wangjiong/3dpose_school/voxelpose-pytorch/slurm_logs/train_%j.out
#SBATCH --mail-type=ALL  # BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yandq2020@mail.sustech.edu.cn

# export MASTER_PORT=$((12000 + $RANDOM % 2000))
set -x
# CONFIG=configs/h36m/train_h36m.yaml
CONFIG=configs/h36m/train_h36m_bs32.yaml

# PYTHONPATH="$(dirname ./scripts/train_h36m_single.sh)/..":$PYTHONPATH \
which python

python -m torch.distributed.launch --nproc_per_node=1 --use_env run/train_3d.py --cfg $CONFIG
