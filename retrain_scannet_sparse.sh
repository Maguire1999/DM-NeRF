#!/bin/bash

GPU=$1
scene=88
# 010 113 38
sparse_inv=1
configs=./configs/scannet/train/scene0${scene}_00_sparse.txt
NOHUP_FILE=../log/dm-nerf/scannet/scene0${scene}_00_sparse${sparse_inv}-GPU_${GPU}.out
export CUDA_VISIBLE_DEVICES=$1

nohup python3 reload_train_scannet.py --config configs/scannet/train/scene0088_00_sparse.txt --log_time 202304090218 --label_sparse_inv 1 --test_model 200000.tar  >$NOHUP_FILE 2>&1 &