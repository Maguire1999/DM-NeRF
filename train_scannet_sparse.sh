#!/bin/bash

GPU=$1
scene=$2
# 010 113 38
sparse_inv=$3
configs=./configs/scannet/train/scene0${scene}_00_sparse.txt
NOHUP_FILE=../log/dm-nerf/scannet/scene0${scene}_00_sparse${sparse_inv}-GPU_${GPU}.out
export CUDA_VISIBLE_DEVICES=$1
nohup python3 train_scannet.py --config ${configs} --label_sparse_inv ${sparse_inv} >$NOHUP_FILE 2>&1 &

