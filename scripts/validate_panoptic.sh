#!/bin/bash

#SBATCH -J VOXELPOSE_PAN_VAL
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH -o /mntnfs/med_data5/wangjiong/3dpose_school/voxelpose-pytorch/slurm_logs/validate_%j.out
#SBATCH -e /mntnfs/med_data5/wangjiong/3dpose_school/voxelpose-pytorch/slurm_logs/validate_%j.out
#SBATCH --mail-type=ALL  # BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yandq2020@mail.sustech.edu.cn

# export MASTER_PORT=$((12000 + $RANDOM % 2000))
set -x
CONFIG=configs/panoptic/resnet50/prn64_cpn80x80x20_960x512_cam5.yaml

# PYTHONPATH="$(dirname ./scripts/validate_panoptic.sh)/..":$PYTHONPATH \
which python

python -m torch.distributed.launch --nproc_per_node=1 --use_env run/validate_3d.py --cfg $CONFIG
