#!/bin/bash

GPU=$1
sparse=$2
# 010 113 38

NOHUP_FILE=../log/dm-nerf/scannet/test_0038_sparse${sparse}-GPU_${GPU}.out
export CUDA_VISIBLE_DEVICES=$1
nohup python3 test_scannet.py --config configs/scannet/train/scene0038_00_sparse.txt --log_time 202304080154 --label_sparse_inv ${sparse} --test_model 300000.tar >$NOHUP_FILE 2>&1 &
#nohup python3 test_scannet.py --config configs/scannet/train/scene0038_00_sparse.txt --log_time 202304080154 --label_sparse_inv 20 --test_model 300000.tar >$NOHUP_FILE 2>&1 &
